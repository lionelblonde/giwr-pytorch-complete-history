from collections import defaultdict
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
from agents.nets import perception_stack_parser, TanhGaussActor, ActorVAE, Critic


EXPANSION = 4
LOG_ALPHA_CLAMPS = [-5., 10.]
CWPQ_TEMP = 10.0

debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


class BEARAgent(object):

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

        # Create VAE actor, "batch-constrained" by construction
        self.vae = ActorVAE(self.env, self.hps, self.rms_obs, hidden_dims[2]).to(self.device)
        sync_with_root(self.vae)

        # Common trick: rewrite the Lagrange multiplier alpha as log(w), and optimize for w
        if self.hps.use_adaptive_alpha:
            # Create learnable Lagrangian multiplier
            self.log_alpha = torch.tensor(self.hps.init_temp_log_alpha).to(self.device)
            self.log_alpha.requires_grad = True
        else:
            self.log_alpha = self.hps.init_temp_log_alpha

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

        self.vae_opt = torch.optim.Adam(self.vae.parameters(),
                                        lr=self.hps.behavior_lr)

        if self.hps.use_adaptive_alpha:
            self.log_alpha_opt = torch.optim.Adam([self.log_alpha],
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

        log_module_info(logger, 'vae', self.vae)

    @property
    def alpha(self):
        if self.hps.use_adaptive_alpha:
            return self.log_alpha.clamp(*LOG_ALPHA_CLAMPS).exp()
        else:
            return self.hps.init_temp_log_alpha

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

        if update_actor:

            # Train the actor
            expansion = EXPANSION  # original hp in BEAR codebase
            squashed_action_from_vae, action_from_vae = self.vae.decodex(state, expansion=expansion)
            expanded_state = state.unsqueeze(1).repeat(1, expansion, 1).view(-1, state.shape[1])
            squashed_action_from_actr, action_from_actr = self.actr.act(expanded_state)
            new_shape = [action.shape[0], expansion, action.shape[1]]
            squashed_action_from_actr = squashed_action_from_actr.view(*new_shape)
            action_from_actr = action_from_actr.view(*new_shape)

            squashed_action_loss = F.mse_loss(  # for monitoring purposes, never optimized
                squashed_action_from_vae,
                squashed_action_from_actr,
                reduction='none',
            ).sum(-1)
            action_loss = F.mse_loss(  # for monitoring purposes, never optimized
                action_from_vae,
                action_from_actr,
                reduction='none',
            ).sum(-1)
            metrics['squashed_action_loss'].append(squashed_action_loss)
            metrics['action_loss'].append(action_loss)

            action_from_actr = float(self.max_ac) * squashed_action_from_actr
            collapsed_action = action_from_actr[:, 0, :]  # along the first mmd-expanded dimension
            assert collapsed_action.shape == action.shape
            q_from_actr = self.crit.QZ(state, collapsed_action)
            if self.hps.clipped_double:
                twin_q_from_actr = self.twin.QZ(state, collapsed_action)
                q_from_actr = torch.min(q_from_actr, twin_q_from_actr)[:, 0]

            # Deal with the MMD divergence
            inputs = dict(input_a=action_from_vae,
                          input_b=action_from_actr,
                          sigma=self.hps.bear_mmd_sigma)
            if self.hps.bear_mmd_kernel == 'laplacian':
                mmd_loss = self.mmd_loss_laplacian(**inputs)
            elif self.hps.bear_mmd_kernel == 'gaussian':
                mmd_loss = self.mmd_loss_gaussian(**inputs)
            else:
                raise NotImplementedError("invalid kernel.")

            # Actor loss
            if iters_so_far >= self.hps.warm_start:
                if self.hps.use_adaptive_alpha:
                    actr_loss = (-q_from_actr + (self.alpha * (mmd_loss - self.hps.bear_mmd_epsilon))).mean()
                else:
                    actr_loss = (-q_from_actr + (100. * mmd_loss)).mean()
            else:
                if self.hps.use_adaptive_alpha:
                    actr_loss = (self.alpha * (mmd_loss - self.hps.bear_mmd_epsilon)).mean()
                else:
                    actr_loss = 100. * mmd_loss.mean()
            metrics['actr_loss'].append(actr_loss)

            self.actr_opt.zero_grad()
            if self.hps.use_adaptive_alpha:
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

            if self.hps.use_adaptive_alpha:
                alpha_loss = -(self.alpha * (mmd_loss - self.hps.bear_mmd_epsilon).detach()).mean()
                self.log_alpha_opt.zero_grad()
                alpha_loss.backward()
                self.log_alpha_opt.step()

            if DEBUG:
                logger.info(f"alpha: {self.alpha}")  # leave this here, for sanity checks

        # Compute QZ estimate
        q = self.denorm_rets(self.crit.QZ(state, action))
        if self.hps.clipped_double:
            twin_q = self.denorm_rets(self.twin.QZ(state, action))

        # Compute target QZ estimate
        next_state = torch.repeat_interleave(next_state, 10, 0)  # duplicate 10 times
        next_action = float(self.max_ac) * self.actr.sample(next_state)
        q_prime = self.targ_crit.QZ(next_state, next_action)
        if self.hps.clipped_double:
            # Define QZ' as the minimum QZ value between TD3's twin QZ's
            twin_q_prime = self.targ_twin.QZ(next_state, next_action)
            q_prime = (self.hps.ensemble_q_lambda * torch.min(q_prime, twin_q_prime) +
                       (1. - self.hps.ensemble_q_lambda) * torch.max(q_prime, twin_q_prime))
        # Take max over each action sampled from the VAE
        q_prime = q_prime.reshape(self.hps.batch_size, -1).max(1)[0].reshape(-1, 1)
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

    def mmd_loss_laplacian(self, input_a, input_b, sigma):
        """Assemble the MMD constraint with Laplacian kernel for support matching"""
        diff_x_x = input_a.unsqueeze(2) - input_a.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) /
                               (2.0 * sigma)).exp(), dim=(1, 2))
        diff_x_y = input_a.unsqueeze(2) - input_b.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) /
                               (2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = input_b.unsqueeze(2) - input_b.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) /
                               (2.0 * sigma)).exp(), dim=(1, 2))
        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

    def mmd_loss_gaussian(self, input_a, input_b, sigma):
        """Assemble the MMD constraint with Gaussian Kernel support matching"""
        diff_x_x = input_a.unsqueeze(2) - input_a.unsqueeze(1)  # B x N x N x d
        diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) /
                               (2.0 * sigma)).exp(), dim=(1, 2))
        diff_x_y = input_a.unsqueeze(2) - input_b.unsqueeze(1)
        diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) /
                               (2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = input_b.unsqueeze(2) - input_b.unsqueeze(1)  # B x N x N x d
        diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) /
                               (2.0 * sigma)).exp(), dim=(1, 2))
        overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
        return overall_loss

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
        torch.save(self.vae.state_dict(), osp.join(path, f"vae_{iters_so_far}.pth"))

    def load(self, path, iters_so_far):
        if self.hps.obs_norm:
            self.rms_obs.load_state_dict(torch.load(osp.join(path, f"rms_obs_{iters_so_far}.pth")))
        self.actr.load_state_dict(torch.load(osp.join(path, f"actr_{iters_so_far}.pth")))
        self.crit.load_state_dict(torch.load(osp.join(path, f"crit_{iters_so_far}.pth")))
        if self.hps.clipped_double:
            self.twin.load_state_dict(torch.load(osp.join(path, f"twin_{iters_so_far}.pth")))
        self.vae.load_state_dict(torch.load(osp.join(path, f"vae_{iters_so_far}.pth")))
