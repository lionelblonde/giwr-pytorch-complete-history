from collections import defaultdict
import os.path as osp

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch import autograd

import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from helpers.math_util import LRScheduler
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import perception_stack_parser, ActorPhi, ActorVAE, TanhGaussActor, Critic, RewardAverager


CWPQ_TEMP = 10.0
BCP_TEMP = 1.0
ADV_ESTIM_SAMPLES = 4
ONE_SIDED_PEN = True


class TSPOAgent(object):

    def __init__(self, env, device, hps, to_load_in_memory):
        self.env = env
        self.ob_space = self.env.observation_space
        self.ob_shape = self.ob_space.shape
        self.ac_space = self.env.action_space
        self.ac_shape = self.ac_space.shape

        log_env_info(logger, self.env)

        self.ob_dim = self.ob_shape[0]  # num dims
        self.ac_dim = self.ac_shape[0]  # num dims
        self.device = device
        self.hps = hps
        assert self.hps.lookahead > 1 or not self.hps.n_step_returns
        assert self.hps.rollout_len <= self.hps.batch_size
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))

        # Define action clipping range
        self.max_ac = max(np.abs(np.amax(self.ac_space.high.astype('float32'))),
                          np.abs(np.amin(self.ac_space.low.astype('float32'))))

        # Parse the noise types
        self.param_noise, self.ac_noise = None, None  # keep this, needed in orchestrator

        # Create observation normalizer that maintains running statistics
        if self.hps.obs_norm:
            self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=not self.hps.cuda)  # no mpi sharing when using cuda
        else:
            self.rms_obs = None

        assert self.hps.ret_norm or not self.hps.popart
        if self.hps.ret_norm:
            # Create return normalizer that maintains running statistics
            self.rms_ret = RunMoms(shape=(1,), use_mpi=False)

        # Create online and target nets, and initialize the target nets
        hidden_dims = perception_stack_parser(self.hps.perception_stack)
        self.actr = ActorPhi(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        sync_with_root(self.actr)

        self.bcp_actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[3]).to(self.device)
        sync_with_root(self.actr)
        self.main_eval_bcp_actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[3]).to(self.device)
        self.maxq_eval_bcp_actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[3]).to(self.device)
        self.cwpq_eval_bcp_actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[3]).to(self.device)
        self.main_eval_bcp_actr.load_state_dict(self.bcp_actr.state_dict())
        self.maxq_eval_bcp_actr.load_state_dict(self.bcp_actr.state_dict())
        self.cwpq_eval_bcp_actr.load_state_dict(self.bcp_actr.state_dict())

        self.crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
        self.targ_crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
        self.targ_crit.load_state_dict(self.crit.state_dict())
        if self.hps.clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
            self.targ_twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())

        # Create VAE actor, "batch-constrained" by construction
        self.vae = ActorVAE(self.env, self.hps, self.rms_obs, hidden_dims[2]).to(self.device)
        sync_with_root(self.vae)
        self.main_eval_vae = ActorVAE(self.env, self.hps, self.rms_obs, hidden_dims[2]).to(self.device)
        self.maxq_eval_vae = ActorVAE(self.env, self.hps, self.rms_obs, hidden_dims[2]).to(self.device)
        self.cwpq_eval_vae = ActorVAE(self.env, self.hps, self.rms_obs, hidden_dims[2]).to(self.device)
        self.main_eval_vae.load_state_dict(self.vae.state_dict())
        self.maxq_eval_vae.load_state_dict(self.vae.state_dict())
        self.cwpq_eval_vae.load_state_dict(self.vae.state_dict())
        # Note: why do we create another VAE model for evaluation?
        # because in the official implementation, BCQ uses the trained VAE to decode the state:
        # https://github.com/sfujim/BCQ/blob/cc139e296ce117f9c8e2bfccf7e568a14baf8892/continuous_BCQ/BCQ.py#L127

        # Create learnable Lagrangian multiplier
        self.log_alpha_ent = torch.tensor(0.).to(self.device)
        self.log_alpha_ent.requires_grad = True

        # Set target entropy to minus action dimension
        self.targ_ent = -self.ac_dim

        # Set up replay buffer
        shapes = {
            'obs0': (self.ob_dim,),
            'obs1': (self.ob_dim,),
            'acs': (self.ac_dim,),
            'acs1': (self.ac_dim,),  # SARSA
            'rews': (1,),
            'dones1': (1,),
            'rets': (1,),
        }
        self.replay_buffer = self.setup_replay_buffer(shapes)

        # Load the offline dataset in memory
        assert to_load_in_memory is not None
        self.replay_buffer.load(to_load_in_memory)

        # Set up the optimizers
        self.actr_opt = torch.optim.Adam(self.actr.parameters(),
                                         lr=self.hps.actor_lr)
        self.crit_opt = torch.optim.Adam(self.crit.parameters(),
                                         lr=self.hps.critic_lr,
                                         weight_decay=self.hps.wd_scale)
        if self.hps.clipped_double:
            self.twin_opt = torch.optim.Adam(self.twin.parameters(),
                                             lr=self.hps.critic_lr,
                                             weight_decay=self.hps.wd_scale)

        self.bcp_actr_opt = torch.optim.Adam(self.bcp_actr.parameters(),
                                             lr=self.hps.actor_lr)

        self.vae_opt = torch.optim.Adam(self.vae.parameters(),
                                        lr=self.hps.behavior_lr)

        self.log_alpha_ent_opt = torch.optim.Adam([self.log_alpha_ent],
                                                  lr=self.hps.actor_lr)

        # Set up lr scheduler
        self.actr_sched = LRScheduler(
            optimizer=self.actr_opt,
            initial_lr=self.hps.actor_lr,
            lr_schedule=self.hps.lr_schedule,
            total_num_steps=self.hps.num_steps,
        )

        if self.hps.use_reward_averager:
            self.reward_averager = RewardAverager(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
            self.ra_opt = torch.optim.Adam(self.reward_averager.parameters(), lr=self.hps.ra_lr)

        log_module_info(logger, 'actr', self.actr)
        log_module_info(logger, 'crit', self.crit)
        if self.hps.clipped_double:
            log_module_info(logger, 'twin', self.crit)

        log_module_info(logger, 'bcp_actr', self.bcp_actr)

        log_module_info(logger, 'vae', self.vae)

    @property
    def alpha_ent(self):
        return self.log_alpha_ent.exp()

    def norm_rets(self, x):
        """Standardize if return normalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.standardize(x)
        else:
            return x

    def denorm_rets(self, x):
        """Standardize if return denormalization is used, do nothing otherwise"""
        if self.hps.ret_norm:
            return self.rms_ret.destandardize(x)
        else:
            return x

    def setup_replay_buffer(self, shapes):
        """Setup experiental memory unit"""
        logger.info(">>>> setting up replay buffer")
        # Create the buffer
        if self.hps.prioritized_replay:
            if self.hps.unreal:  # Unreal prioritized experience replay
                replay_buffer = UnrealReplayBuffer(
                    self.hps.mem_size,
                    shapes,
                    save_full_history=False,
                )
            else:  # Vanilla prioritized experience replay
                replay_buffer = PrioritizedReplayBuffer(
                    self.hps.mem_size,
                    shapes,
                    alpha=self.hps.alpha,
                    beta=self.hps.beta,
                    ranked=self.hps.ranked,
                    save_full_history=False,
                )
        else:  # Vanilla experience replay
            replay_buffer = ReplayBuffer(
                self.hps.mem_size,
                shapes,
                save_full_history=False,
            )
        # Summarize replay buffer creation (relies on `__repr__` method)
        logger.info("{} configured".format(replay_buffer))
        return replay_buffer

    def store_transition(self, transition):
        """Store the transition in memory and update running moments"""
        assert not self.hps.offline, "this method should not be used in this setting."
        # Store transition in the replay buffer
        self.replay_buffer.append(transition)
        if self.hps.obs_norm:
            # Update the observation normalizer
            self.rms_obs.update(transition['obs0'])

    def sample_batch(self):
        """Sample a batch of transitions from the replay buffer"""

        # def _patcher(x, y, z):
        #     return .detach().cpu().numpy()  # redundant detach

        # Get a batch of transitions from the replay buffer
        if self.hps.n_step_returns:
            batch = self.replay_buffer.lookahead_sample(
                self.hps.batch_size,
                self.hps.lookahead,
                self.hps.gamma,
                # _patcher,  # no need
                None,
            )
        else:
            batch = self.replay_buffer.sample(
                self.hps.batch_size,
                # _patcher,  # no need
                None,
            )
        return batch

    def predict(self, ob, apply_noise, which):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated QZ value.
        Note: keep 'apply_noise' even if unused, to preserve the unified signature.
        """
        if apply_noise:
            _actr = self.bcp_actr
            _vae = self.vae
        else:
            if which in ['maxq', 'cwpq']:

                if which == 'maxq':
                    _actr = self.maxq_eval_bcp_actr
                    _vae = self.maxq_eval_vae
                else:  # which == 'cwpq'
                    _actr = self.cwpq_eval_bcp_actr
                    _vae = self.cwpq_eval_vae

                ob = torch.Tensor(ob[None]).to(self.device).repeat(100, 1)  # duplicate 100 times
                ac_from_vae = _vae.decode(ob)
                ac = self.actr.act(ob, ac_from_vae)
                # Among the 100 values, take the one with the highest Q value (or Z value)
                q_value = self.crit.QZ(ob, ac).mean(dim=1)

                if which == 'maxq':
                    index = q_value.argmax(0)
                else:  # which == 'cwpq'
                    adv_value = q_value - q_value.mean(dim=0)
                    weight = F.softplus(adv_value,
                                        beta=1. / CWPQ_TEMP,
                                        threshold=20.).clamp(min=0.01, max=1e12)
                    index = torch.multinomial(weight, num_samples=1, generator=_actr.gen).squeeze()

                ac = ac[index]

            else:  # which == 'main'
                _actr = self.main_eval_bcp_actr
                ob = torch.Tensor(ob[None]).to(self.device)
                ac = float(self.max_ac) * _actr.mode(ob, sg=True)
                # Gaussian, so mode == mean, can use either interchangeably

        # Place on cpu and collapse into one dimension
        ac = ac.cpu().detach().numpy().flatten()
        # Clip the action
        ac = ac.clip(-self.max_ac, self.max_ac)
        return ac

    def ac_factory_1(self, ob, inflate):
        _ob = ob.unsqueeze(1).repeat(1, inflate, 1).view(ob.shape[0] * inflate, ob.shape[1])
        _ac = self.vae.decode(_ob)
        ac = self.actr.act(_ob, _ac)
        return ac

    def ac_factory_2(self, ob, inflate):
        _ob = ob.unsqueeze(1).repeat(1, inflate, 1).view(ob.shape[0] * inflate, ob.shape[1])
        ac = float(self.max_ac) * self.bcp_actr.sample(_ob, sg=False)
        return ac

    def q_factory(self, crit, ob, ac):
        ob_dim = ob.shape[0]
        ac_dim = ac.shape[0]
        num_repeat = int(ac_dim / ob_dim)
        _ob = ob.unsqueeze(1).repeat(1, num_repeat, 1).view(ob.shape[0] * num_repeat, ob.shape[1])
        q_value = crit.QZ(_ob, ac)
        q_value = q_value.view(ob.shape[0], num_repeat, 1)
        return q_value

    def update_actor_critic(self, batch, update_actor, iters_so_far):
        """Train the actor and critic networks
        Note, 'update_actor' is here to keep the unified signature.
        """

        # Container for all the metrics
        metrics = defaultdict(list)

        # Transfer to device
        state = torch.Tensor(batch['obs0']).to(self.device)
        action = torch.Tensor(batch['acs']).to(self.device)
        next_state = torch.Tensor(batch['obs1']).to(self.device)
        reward = torch.Tensor(batch['rews']).to(self.device)
        done = torch.Tensor(batch['dones1'].astype('float32')).to(self.device)
        if self.hps.prioritized_replay:
            iws = torch.Tensor(batch['iws']).to(self.device)
        if self.hps.n_step_returns:
            td_len = torch.Tensor(batch['td_len']).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        if self.hps.use_reward_averager:
            # Update the reward averager
            ra_loss = F.smooth_l1_loss(self.reward_averager(state, action), reward)  # Huber loss
            ra_grad_pen = self.grad_pen(
                fa=self.reward_averager,
                state=state,
                action=action,
            )
            ra_loss += self.hps.scale_ra_grad_pen * ra_grad_pen
            self.ra_opt.zero_grad()
            ra_loss.backward()
            self.ra_opt.step()
            # Override the reward tensor
            reward = self.reward_averager(state, action)

        # Calculate log-probability of the action predicted by the BCP actor
        action_from_bcp_actr = float(self.max_ac) * self.bcp_actr.sample(state, sg=False)
        log_prob = self.bcp_actr.logp(state, action_from_bcp_actr)

        # Train the behavioral cloning actor
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        kl_loss = -0.5 * (1 + std.pow(2).log() - mean.pow(2) - std.pow(2)).mean()
        # Note, the previous is just the closed form kl divergence for the normal distribution
        vae_loss = recon_loss + (0.5 * kl_loss)

        self.vae_opt.zero_grad()
        vae_loss.backward()
        average_gradients(self.vae, self.device)
        self.vae_opt.step()

        # Compute QZ estimate
        q = self.denorm_rets(self.crit.QZ(state, action))
        if self.hps.clipped_double:
            twin_q = self.denorm_rets(self.twin.QZ(state, action))

        # Compute target QZ estimate
        next_state = torch.repeat_interleave(next_state, 10, 0)  # duplicate 10 times
        next_action_from_vae = self.vae.decode(next_state)
        next_action = self.actr.act(next_state, next_action_from_vae)
        q_prime = self.targ_crit.QZ(next_state, next_action)
        if self.hps.clipped_double:
            # Define QZ' as the minimum QZ value between TD3's twin QZ's
            twin_q_prime = self.targ_twin.QZ(next_state, next_action)
            q_prime = (self.hps.ensemble_q_lambda * torch.min(q_prime, twin_q_prime) +
                       (1. - self.hps.ensemble_q_lambda) * torch.max(q_prime, twin_q_prime))
        # Take max over each action sampled
        q_prime = q_prime.reshape(self.hps.batch_size, -1).max(1)[0].reshape(-1, 1)
        # Assemble the target
        targ_q = (reward +
                  (self.hps.gamma ** td_len) * (1. - done) *
                  self.denorm_rets(q_prime))

        # Add target bonus
        if self.hps.targ_q_bonus == 'al':
            al_q = self.targ_crit.QZ(state, action)
            al_emp_adv_ac = self.ac_factory_2(state, ADV_ESTIM_SAMPLES)
            al_emp_adv_from_actr = self.q_factory(self.targ_crit, state, al_emp_adv_ac).mean(dim=1)
            al_adv = al_q - al_emp_adv_from_actr
            if self.hps.clipped_double:
                twin_al_q = self.targ_twin.QZ(state, action)
                twin_al_emp_adv_from_actr = self.q_factory(self.targ_twin, state, al_emp_adv_ac).mean(dim=1)
                twin_al_adv = twin_al_q - twin_al_emp_adv_from_actr
                al_adv = torch.min(al_adv, twin_al_adv)
            targ_q += self.hps.scale_targ_q_bonus * al_adv

        targ_q = self.norm_rets(targ_q).detach()

        if self.hps.ret_norm:
            if self.hps.popart:
                # Apply Pop-Art, https://arxiv.org/pdf/1602.07714.pdf
                # Save the pre-update running stats
                old_mean = torch.Tensor(self.rms_ret.mean).to(self.device)
                old_std = torch.Tensor(self.rms_ret.std).to(self.device)
                # Update the running stats
                self.rms_ret.update(targ_q)
                # Get the post-update running statistics
                new_mean = torch.Tensor(self.rms_ret.mean).to(self.device)
                new_std = torch.Tensor(self.rms_ret.std).to(self.device)
                # Preserve the output from before the change of normalization old->new
                # for both online and target critic(s)
                outs = [self.crit.out_params, self.targ_crit.out_params]
                if self.hps.clipped_double:
                    outs.extend([self.twin.out_params, self.targ_twin.out_params])
                for out in outs:
                    w, b = out
                    w.data.copy_(w.data * old_std / new_std)
                    b.data.copy_(((b.data * old_std) + old_mean - new_mean) / new_std)
            else:
                # Update the running stats
                self.rms_ret.update(targ_q)

        # Critic loss
        mse_td_errors = F.mse_loss(q, targ_q, reduction='none')
        if self.hps.clipped_double:
            twin_mse_td_errors = F.mse_loss(twin_q, targ_q, reduction='none')

        if self.hps.prioritized_replay:
            # Adjust with importance weights
            mse_td_errors *= iws
            if self.hps.clipped_double:
                twin_mse_td_errors *= iws
            # Update priorities
            new_priorities = np.abs((q - targ_q).detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(batch['idxs'].reshape(-1), new_priorities)

        crit_loss = mse_td_errors.mean()
        if self.hps.clipped_double:
            twin_loss = twin_mse_td_errors.mean()

        self.crit_opt.zero_grad()
        crit_loss.backward()
        if self.hps.clipped_double:
            self.twin_opt.zero_grad()
            twin_loss.backward()
        self.crit_opt.step()
        if self.hps.clipped_double:
            self.twin_opt.step()

        # Actor loss
        action_from_vae = self.vae.decode(state)
        # First stream
        _actr_loss = -self.crit.QZ(state, self.actr.act(state, action_from_vae))
        actr_loss = _actr_loss.mean()
        self.actr_opt.zero_grad()
        actr_loss.backward(retain_graph=True)
        average_gradients(self.actr, self.device)
        if self.hps.clip_norm > 0:
            U.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
        self.actr_opt.step()
        # Second stream
        if iters_so_far >= self.hps.warm_start:

            if self.hps.targ_q_bonus in ['al', 'pal'] and self.hps.use_temp_corr:
                # Apply temperature correction
                bcp_temp = (1. + self.hps.scale_targ_q_bonus) * BCP_TEMP
            else:
                bcp_temp = BCP_TEMP

            emp_adv_ac = self.ac_factory_2(state, ADV_ESTIM_SAMPLES)
            emp_adv_from_bcp_actr = self.q_factory(self.crit, state, emp_adv_ac).mean(dim=1)

            crr_q = self.crit.QZ(state, action)
            crr_adv = crr_q - emp_adv_from_bcp_actr
            crr_adv = torch.exp(crr_adv / bcp_temp).clamp(max=20.)
            bcp_actr_loss = -self.bcp_actr.logp(state, action) * crr_adv.detach()

            _state = torch.repeat_interleave(state, 10, 0)  # duplicate 10 times
            _action_from_vae = self.vae.decode(_state)
            _action = self.actr.act(_state, _action_from_vae)
            _q_prime = self.crit.QZ(_state, _action)
            if self.hps.clipped_double:
                # Define QZ' as the minimum QZ value between TD3's twin QZ's
                _twin_q_prime = self.targ_twin.QZ(_state, _action)
                _q_prime = (self.hps.ensemble_q_lambda * torch.min(_q_prime, _twin_q_prime) +
                            (1. - self.hps.ensemble_q_lambda) * torch.max(_q_prime, _twin_q_prime))
            # Take argmax over each action sampled
            _argmax_action_index = q_prime.reshape(self.hps.batch_size, -1).argmax(1).reshape(-1, 1)
            _argmax_action = torch.gather(_action.reshape(self.hps.batch_size, 10, -1),
                                          1,
                                          _argmax_action_index.unsqueeze(-1).repeat(1, 1, self.ac_dim))
            _argmax_action = _argmax_action.squeeze(dim=1)

            bcp_q = self.crit.QZ(state, _argmax_action)
            bcp_adv = bcp_q - emp_adv_from_bcp_actr
            bcp_adv = torch.exp(bcp_adv / bcp_temp).clamp(max=20.)
            bcp_actr_loss -= (self.hps.scale_second_stream_loss *
                              self.bcp_actr.logp(state, _argmax_action) * bcp_adv.detach())
        else:
            # Use behavioral cloning loss
            bcp_actr_loss = ((self.alpha_ent * log_prob) - self.bcp_actr.logp(state, action)).mean()
        bcp_actr_loss = bcp_actr_loss.mean()
        self.bcp_actr_opt.zero_grad()
        bcp_actr_loss.backward()
        average_gradients(self.bcp_actr, self.device)
        self.bcp_actr_opt.step()

        # Update the alpha coefficient of the entropy regularizer via dual gradient descent
        alpha_ent_loss = (self.log_alpha_ent * (-log_prob - self.targ_ent).detach()).mean()
        self.log_alpha_ent_opt.zero_grad()
        alpha_ent_loss.backward()
        self.log_alpha_ent_opt.step()

        _lr = self.actr_sched.step(steps_so_far=iters_so_far)
        logger.info(f"lr is {_lr} after {iters_so_far} iters")

        # Update target nets
        self.update_target_net()

        # Log metrics
        metrics['crit_loss'].append(crit_loss)
        if self.hps.clipped_double:
            metrics['twin_loss'].append(twin_loss)
        if self.hps.prioritized_replay:
            metrics['iws'].append(iws)
        metrics['actr_loss'].append(actr_loss)

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        lrnows = {'actr': _lr}

        return metrics, lrnows

    def grad_pen(self, fa, state, action, std=10.0):
        """Define the gradient penalty regularizer"""
        # Create the states to apply the contraint on
        eps_s = state.clone().detach().data.normal_(0, std)
        zeta_state = state + eps_s
        zeta_state.requires_grad = True
        # Create the actions to apply the contraint on
        eps_a = action.clone().detach().data.normal_(0, std)
        zeta_action = action + eps_a
        zeta_action.requires_grad = True
        # Define the input(s) w.r.t. to take the gradient
        inputs = [zeta_state, zeta_action]
        # Create the operation of interest
        score = fa(*inputs)
        # Get the gradient of this operation with respect to its inputs
        grads = autograd.grad(
            outputs=score,
            inputs=inputs,
            only_inputs=True,
            grad_outputs=[torch.ones_like(score)],
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        # Return the gradient penalties
        grads = torch.cat(list(grads), dim=-1)
        grads_norm = grads.norm(2, dim=-1)
        if ONE_SIDED_PEN:
            # Penalize the gradient for having a norm GREATER than 1
            _grad_pen = torch.max(torch.zeros_like(grads_norm), grads_norm - 1.).pow(2)
        else:
            # Penalize the gradient for having a norm LOWER OR GREATER than 1
            _grad_pen = (grads_norm - 1.).pow(2)
        grad_pen = _grad_pen.mean()
        return grad_pen

    def update_target_net(self):
        """Update the target networks"""
        for param, targ_param in zip(self.crit.parameters(), self.targ_crit.parameters()):
            targ_param.data.copy_(self.hps.polyak * param.data +
                                  (1. - self.hps.polyak) * targ_param.data)
        if self.hps.clipped_double:
            for param, targ_param in zip(self.twin.parameters(), self.targ_twin.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)

    def update_eval_nets(self):
        for param, eval_param in zip(self.bcp_actr.parameters(), self.main_eval_bcp_actr.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.bcp_actr.parameters(), self.maxq_eval_bcp_actr.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.bcp_actr.parameters(), self.cwpq_eval_bcp_actr.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.vae.parameters(), self.main_eval_vae.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.vae.parameters(), self.maxq_eval_vae.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.vae.parameters(), self.cwpq_eval_vae.parameters()):
            eval_param.data.copy_(param.data)

    def save(self, path, iters_so_far):
        torch.save(self.actr.state_dict(), osp.join(path, f"actr_{iters_so_far}.pth"))
        torch.save(self.crit.state_dict(), osp.join(path, f"crit_{iters_so_far}.pth"))
        if self.hps.clipped_double:
            torch.save(self.twin.state_dict(), osp.join(path, f"twin_{iters_so_far}.pth"))
        torch.save(self.bcp_actr.state_dict(), osp.join(path, f"bcp_actr_{iters_so_far}.pth"))
        torch.save(self.vae.state_dict(), osp.join(path, f"vae_{iters_so_far}.pth"))

    def load(self, path, iters_so_far):
        self.actr.load_state_dict(torch.load(osp.join(path, f"actr_{iters_so_far}.pth")))
        self.crit.load_state_dict(torch.load(osp.join(path, f"crit_{iters_so_far}.pth")))
        if self.hps.clipped_double:
            self.twin.load_state_dict(torch.load(osp.join(path, f"twin_{iters_so_far}.pth")))
        self.bcp_actr.load_state_dict(torch.load(osp.join(path, f"bcp_actr_{iters_so_far}.pth")))
        self.vae.load_state_dict(torch.load(osp.join(path, f"vae_{iters_so_far}.pth")))
