from collections import deque, defaultdict
import os
import os.path as osp

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F

import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from helpers.math_util import LRScheduler
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import perception_stack_parser, TanhGaussActor, Critic


ALPHA_DIV_CLAMPS = [0., 500.]
TARG_DIV = 0.
BC_TRAINING_STEPS_PER_BATCH = 10
CWPQ_TEMP = 10.0

debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class BRACAgent(object):

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
        self.actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        sync_with_root(self.actr)
        self.main_eval_actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.maxq_eval_actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.cwpq_eval_actr = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.main_eval_actr.load_state_dict(self.actr.state_dict())
        self.maxq_eval_actr.load_state_dict(self.actr.state_dict())
        self.cwpq_eval_actr.load_state_dict(self.actr.state_dict())

        self.crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
        self.targ_crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
        self.targ_crit.load_state_dict(self.crit.state_dict())
        if self.hps.clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
            self.targ_twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())

        # Create another actor, trained via BC to estimate the distribution of the behavior policy
        self.actr_b = TanhGaussActor(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        sync_with_root(self.actr_b)

        self.hps.use_adaptive_alpha = None  # unused in this algorithm, make sure it can not interfere
        # Common trick: rewrite the Lagrange multiplier alpha as log(w), and optimize for w
        if self.hps.brac_use_adaptive_alpha_ent:
            # Create learnable Lagrangian multiplier
            self.log_alpha_ent = torch.tensor(self.hps.brac_init_temp_log_alpha_ent).to(self.device)
            self.log_alpha_ent.requires_grad = True
        else:
            self.log_alpha_ent = self.hps.brac_init_temp_log_alpha_ent
        if self.hps.brac_use_adaptive_alpha_div:
            # Create learnable Lagrangian multiplier
            self.log_alpha_div = torch.tensor(self.hps.brac_init_temp_log_alpha_div).to(self.device)
            self.log_alpha_div.requires_grad = True
        else:
            self.log_alpha_div = self.hps.brac_init_temp_log_alpha_div

        # Set target entropy to minus action dimension
        self.targ_ent = -self.ac_dim
        # Set target divergence to hard-coded value (advised in BRAC codebase: 0.)
        self.targ_div = TARG_DIV

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

        self.actr_b_opt = torch.optim.Adam(self.actr_b.parameters(),
                                           lr=self.hps.behavior_lr)

        if self.hps.brac_use_adaptive_alpha_ent:  # choice: same lr for both alphas
            self.log_alpha_ent_opt = torch.optim.Adam([self.log_alpha_ent],
                                                      lr=self.hps.log_alpha_lr)
        if self.hps.brac_use_adaptive_alpha_div:  # choice: same lr for both alphas
            self.log_alpha_div_opt = torch.optim.Adam([self.log_alpha_div],
                                                      lr=self.hps.log_alpha_lr)

        # Set up lr scheduler
        self.actr_sched = LRScheduler(
            optimizer=self.actr_opt,
            initial_lr=self.hps.actor_lr,
            lr_schedule=self.hps.lr_schedule,
            total_num_steps=self.hps.num_steps,
        )

        log_module_info(logger, 'actr', self.actr)
        log_module_info(logger, 'crit', self.crit)
        if self.hps.clipped_double:
            log_module_info(logger, 'twin', self.crit)

        log_module_info(logger, 'actr_b', self.actr_b)

    @property
    def alpha_ent(self):
        if self.hps.brac_use_adaptive_alpha_ent:
            return self.log_alpha_ent.exp()
        else:
            return self.hps.brac_init_temp_log_alpha_ent

    @property
    def alpha_div(self):
        if self.hps.brac_use_adaptive_alpha_div:
            return self.log_alpha_div.exp().clamp(*ALPHA_DIV_CLAMPS)
        else:
            return self.hps.brac_init_temp_log_alpha_div

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
        # Predict the action
        if apply_noise:
            _actr = self.actr
            ob = torch.Tensor(ob[None]).to(self.device)
            ac = float(self.max_ac) * _actr.sample(ob, sg=True)
        else:
            if which in ['maxq', 'cwpq']:

                if which == 'maxq':
                    _actr = self.maxq_eval_actr
                else:  # which == 'cwpq'
                    _actr = self.cwpq_eval_actr

                ob = torch.Tensor(ob[None]).to(self.device).repeat(100, 1)  # duplicate 100 times
                ac = float(self.max_ac) * _actr.sample(ob, sg=True)
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
                _actr = self.main_eval_actr
                ob = torch.Tensor(ob[None]).to(self.device)
                ac = float(self.max_ac) * _actr.mode(ob, sg=True)
                # Gaussian, so mode == mean, can use either interchangeably

        # Place on cpu and collapse into one dimension
        ac = ac.cpu().detach().numpy().flatten()
        # Clip the action
        ac = ac.clip(-self.max_ac, self.max_ac)
        return ac

    def update_behavior_policy_clone(self, batch):
        """Train the behavior policy."""

        # Transfer to device
        state = torch.Tensor(batch['obs0']).to(self.device)
        action = torch.Tensor(batch['acs']).to(self.device)

        bc_loss_deque = deque(maxlen=BC_TRAINING_STEPS_PER_BATCH)

        for _ in range(BC_TRAINING_STEPS_PER_BATCH):
            actr_b_loss = -self.actr_b.logp(state, action).mean()
            bc_loss_deque.append(actr_b_loss.exp().detach().cpu().numpy())

            self.actr_b_opt.zero_grad()
            actr_b_loss.backward()
            average_gradients(self.actr_b, self.device)
            self.actr_b_opt.step()

        return np.mean(bc_loss_deque)

    def update_actor_critic(self, batch, update_actor, iters_so_far):
        """Train the actor and critic networks
        Note, 'update_actor' is here to keep the unified signature.
        """

        # Note, at this point, the behavior policy clone is alreay pre-trained

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

        if update_actor:

            # Train the actor
            action_from_actr = float(self.max_ac) * self.actr.sample(state, sg=False)
            log_prob = self.actr.logp(state, action_from_actr)
            q_from_actr = self.crit.QZ(state, action_from_actr)
            if self.hps.clipped_double:
                twin_q_from_actr = self.twin.QZ(state, action_from_actr)
                q_from_actr = torch.min(q_from_actr, twin_q_from_actr)
                q_from_actr = (self.hps.ensemble_q_lambda * torch.min(q_from_actr, twin_q_from_actr) +
                               (1. - self.hps.ensemble_q_lambda) * torch.max(q_from_actr, twin_q_from_actr))

            # Compute the divergence between the actor and the behavior policy
            # Note, we only care about the KL divergence: best reported results in BRAC
            # In BRAC, the KL considered is KL(policy || behavior policy)
            log_prob_1 = self.actr.logp(state, action_from_actr)
            log_prob_2 = self.actr_b.logp(state, action_from_actr)
            div = log_prob_1 - log_prob_2

            # Actor loss
            # Only update the policy after a certain number of iteration (BRAC codebase: 20000)
            if iters_so_far >= self.hps.warm_start:
                actr_loss = ((self.alpha_ent * log_prob) - q_from_actr).mean()
                actr_loss += self.alpha_div * div.mean()
            else:
                actr_loss = (self.alpha_ent * log_prob).mean()
                actr_loss += (self.alpha_div * div).mean()

            metrics['actr_loss'].append(actr_loss)

            self.actr_opt.zero_grad()
            if self.hps.brac_use_adaptive_alpha_ent or self.hps.brac_use_adaptive_alpha_div:
                actr_loss.backward(retain_graph=True)  # double-checked: OK
            else:
                actr_loss.backward()
            average_gradients(self.actr, self.device)
            if self.hps.clip_norm > 0:
                U.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
            self.actr_opt.step()

            _lr = self.actr_sched.step(steps_so_far=iters_so_far)
            if DEBUG:
                logger.info(f"lr is {_lr} after {iters_so_far} iters")

            if self.hps.brac_use_adaptive_alpha_ent:
                alpha_ent_loss = (self.log_alpha_ent * (-log_prob - self.targ_ent).detach()).mean()
                self.log_alpha_ent_opt.zero_grad()
                alpha_ent_loss.backward()
                self.log_alpha_ent_opt.step()

            if self.hps.brac_use_adaptive_alpha_div:
                alpha_div_loss = (-self.log_alpha_div * (div - self.targ_div).detach()).mean()
                self.log_alpha_div_opt.zero_grad()
                alpha_div_loss.backward()
                self.log_alpha_div_opt.step()

            if DEBUG:
                logger.info(f"alpha_ent: {self.alpha_ent}")  # leave this here, for sanity checks
                logger.info(f"alpha_div: {self.alpha_div}")  # leave this here, for sanity checks

        # Compute QZ estimate
        q = self.denorm_rets(self.crit.QZ(state, action))
        if self.hps.clipped_double:
            twin_q = self.denorm_rets(self.twin.QZ(state, action))

        # Compute target QZ estimate
        next_action = float(self.max_ac) * self.actr.sample(next_state)
        q_prime = self.targ_crit.QZ(next_state, next_action)
        if self.hps.clipped_double:
            # Define QZ' as the minimum QZ value between TD3's twin QZ's
            twin_q_prime = self.targ_twin.QZ(next_state, next_action)
            q_prime = (self.hps.ensemble_q_lambda * torch.min(q_prime, twin_q_prime) +
                       (1. - self.hps.ensemble_q_lambda) * torch.max(q_prime, twin_q_prime))
        # Take max over each action sampled from the VAE
        q_prime = q_prime.reshape(self.hps.batch_size, -1).max(1)[0].reshape(-1, 1)

        # Add the causal entropy regularization term
        next_log_prob = self.actr.logp(next_state, next_action)
        q_prime -= self.alpha_ent * next_log_prob

        if self.hps.brac_value_kl_pen:
            # Add value penalty regularizater
            next_log_prob_1 = self.actr.logp(next_state, next_action)
            next_log_prob_2 = self.actr_b.logp(next_state, next_action)
            div = next_log_prob_1 - next_log_prob_2
            q_prime -= self.alpha_div * div

        # Assemble the target
        targ_q = (reward +
                  (self.hps.gamma ** td_len) * (1. - done) *
                  self.denorm_rets(q_prime))
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

        # Log metrics
        metrics['crit_loss'].append(crit_loss)
        if self.hps.clipped_double:
            metrics['twin_loss'].append(twin_loss)
        if self.hps.prioritized_replay:
            metrics['iws'].append(iws)

        # Update target nets
        self.update_target_net()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        lrnows = {'actr': _lr}

        return metrics, lrnows

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
        for param, eval_param in zip(self.actr.parameters(), self.main_eval_actr.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.actr.parameters(), self.maxq_eval_actr.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.actr.parameters(), self.cwpq_eval_actr.parameters()):
            eval_param.data.copy_(param.data)

    def save(self, path, iters_so_far):
        if self.hps.obs_norm:
            torch.save(self.rms_obs.state_dict(), osp.join(path, f"rms_obs_{iters_so_far}.pth"))
        torch.save(self.actr.state_dict(), osp.join(path, f"actr_{iters_so_far}.pth"))
        torch.save(self.crit.state_dict(), osp.join(path, f"crit_{iters_so_far}.pth"))
        if self.hps.clipped_double:
            torch.save(self.twin.state_dict(), osp.join(path, f"twin_{iters_so_far}.pth"))
        torch.save(self.actr_b.state_dict(), osp.join(path, f"actr_b_{iters_so_far}.pth"))

    def load(self, path, iters_so_far):
        if self.hps.obs_norm:
            self.rms_obs.load_state_dict(torch.load(osp.join(path, f"rms_obs_{iters_so_far}.pth")))
        self.actr.load_state_dict(torch.load(osp.join(path, f"actr_{iters_so_far}.pth")))
        self.crit.load_state_dict(torch.load(osp.join(path, f"crit_{iters_so_far}.pth")))
        if self.hps.clipped_double:
            self.twin.load_state_dict(torch.load(osp.join(path, f"twin_{iters_so_far}.pth")))
        self.actr_b.load_state_dict(torch.load(osp.join(path, f"actr_b_{iters_so_far}.pth")))
