from collections import namedtuple, defaultdict
import os.path as osp

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F

from helpers import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import perception_stack_parser, TanhGaussActor, Critic


class CQLAgent(object):

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
        assert self.hps.clip_norm >= 0
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))
        assert not self.hps.use_c51 and not self.hps.use_qr

        # Define action clipping range
        self.max_ac = max(np.abs(np.amax(self.ac_space.high.astype('float32'))),
                          np.abs(np.amin(self.ac_space.low.astype('float32'))))

        # Parse the noise types
        self.param_noise, self.ac_noise = None, None  # keep this, needed in orchestrator

        # Create online and target nets, and initialize the target nets
        hidden_dims = perception_stack_parser(self.hps.perception_stack)
        self.actr = TanhGaussActor(self.env, self.hps, hidden_dims=hidden_dims[0]).to(self.device)
        sync_with_root(self.actr)
        self.targ_actr = TanhGaussActor(self.env, self.hps, hidden_dims=hidden_dims[0]).to(self.device)
        self.targ_actr.load_state_dict(self.actr.state_dict())

        self.crit = Critic(self.env, self.hps, hidden_dims=hidden_dims[1]).to(self.device)
        sync_with_root(self.crit)
        self.targ_crit = Critic(self.env, self.hps, hidden_dims=hidden_dims[1]).to(self.device)
        self.targ_crit.load_state_dict(self.crit.state_dict())
        if self.hps.clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(self.env, self.hps, hidden_dims=hidden_dims[1]).to(self.device)
            sync_with_root(self.twin)
            self.targ_twin = Critic(self.env, self.hps, hidden_dims=hidden_dims[1]).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())

        # Common trick: rewrite the Lagrange multiplier alpha as log(w), and optimize for w
        if self.hps.use_adaptive_alpha:
            # Create learnable Lagrangian multiplier
            self.w = torch.tensor(self.hps.init_temperature).to(self.device)
            self.w.requires_grad = True
        else:
            self.w = self.hps.init_temperature

        # Set target entropy to minus action dimension
        self.targ_ent = -self.ac_dim

        # Set up replay buffer
        shapes = {
            'obs0': (self.ob_dim,),
            'obs1': (self.ob_dim,),
            'acs': (self.ac_dim,),
            'rews': (1,),
            'dones1': (1,),
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

        if self.hps.use_adaptive_alpha:
            self.w_opt = torch.optim.Adam([self.w], lr=self.hps.alpha_lr)

        # Set up the learning rate schedule
        def _lr(t):  # flake8: using a def instead of a lambda
            if self.hps.with_scheduler:
                return (1.0 - ((t - 1.0) / (self.hps.num_timesteps //
                                            self.hps.rollout_len)))
            else:
                return 1.0

        # Set up lr scheduler
        self.actr_sched = torch.optim.lr_scheduler.LambdaLR(self.actr_opt, _lr)

        assert self.hps.ret_norm or not self.hps.popart
        if self.hps.ret_norm:
            # Create return normalizer that maintains running statistics
            self.rms_ret = RunMoms(shape=(1,), use_mpi=False)  # Careful, set to False here

        log_module_info(logger, 'actr', self.actr)
        log_module_info(logger, 'crit', self.crit)
        if self.hps.clipped_double:
            log_module_info(logger, 'twin', self.crit)

    @property
    def alpha(self):
        if self.hps.use_adaptive_alpha:
            return self.w.exp()
        else:
            return self.hps.init_temperature

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
        """Setup experimental memory unit"""
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
        # Store transition in the replay buffer
        self.replay_buffer.append(transition)
        # Update the running moments for all the networks (online and targets)
        _state = transition['obs0']
        self.actr.rms_obs.update(_state)
        self.crit.rms_obs.update(_state)
        self.targ_actr.rms_obs.update(_state)
        self.targ_crit.rms_obs.update(_state)
        if self.hps.clipped_double:
            self.twin.rms_obs.update(_state)
            self.targ_twin.rms_obs.update(_state)

    # def patcher(self):
    #     raise NotImplementedError  # no need

    def sample_batch(self):
        """Sample a batch of transitions from the replay buffer"""

        def _patcher(x, y, z):
            return self.patcher(x, y, z).detach().cpu().numpy()  # redundant detach

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

    def predict(self, ob, apply_noise):
        """Predict an action, with or without perturbation,
        and optionaly compute and return the associated QZ value.
        Note: keep 'apply_noise' even if unused, to preserve the unified signature.
        """
        # Create tensor from the state (`require_grad=False` by default)
        ob = torch.Tensor(ob[None]).to(self.device)
        # Predict the action
        if apply_noise:
            _ac = self.actr.sample(ob)
        else:
            _ac = self.actr.mode(ob)
        # Gaussian, so mode == mean, can use either interchangeably
        ac = float(self.max_ac) * _ac
        # Place on cpu and collapse into one dimension
        ac = ac.cpu().detach().numpy().flatten()
        # Clip the action
        ac = ac.clip(-self.max_ac, self.max_ac)
        return ac

    def ac_factory(self, actr):

        def _predict_ac(ob, stride):
            _ob = ob.unsqueeze(1).repeat(1, stride, 1).view(ob.shape[0] * stride, ob.shape[1])
            _ac = self.actr.sample(_ob, sg=False)
            _logp = self.actr.logp(_ob, _ac)
            return _ac, _logp.view(ob.shape[0], stride, 1)

        return _predict_ac

    def q_factory(self, crit):

        def _predict_q(ob, ac):
            ob_dim = ob.shape[0]
            ac_dim = ac.shape[0]
            num_repeat = int(ac_dim / ob_dim)
            _ob = ob.unsqueeze(1).repeat(1, num_repeat, 1).view(ob.shape[0] * num_repeat, ob.shape[1])
            q = crit(_ob, ac).view(ob.shape[0], num_repeat, 1)
            return q

        return _predict_q

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

        if update_actor:
            # Actor loss
            action_from_actr = float(self.max_ac) * self.actr.sample(state, sg=False)
            log_prob = self.actr.logp(state, action_from_actr)
            q_from_actr = self.crit.QZ(state, action_from_actr)
            if self.hps.clipped_double:
                twin_q_from_actr = self.twin.QZ(state, action_from_actr)
                q_from_actr = torch.min(q_from_actr, twin_q_from_actr)
            actr_loss = ((self.alpha * log_prob) - q_from_actr).mean()
            # Log metrics
            metrics['actr_loss'].append(actr_loss)

            self.actr_opt.zero_grad()
            actr_loss.backward(retain_graph=False)
            average_gradients(self.actr, self.device)
            if self.hps.clip_norm > 0:
                U.clip_grad_norm_(self.actr.parameters(), self.hps.clip_norm)
            self.actr_opt.step()
            self.actr_sched.step(iters_so_far)

            if self.hps.use_adaptive_alpha:
                alpha_loss = (self.w * (-log_prob - self.targ_ent).detach()).mean()
                self.w_opt.zero_grad()
                alpha_loss.backward(retain_graph=False)
                self.w_opt.step()
                # Log metrics
                metrics['alpha_loss'].append(alpha_loss)

        # Compute QZ estimate
        q = self.denorm_rets(self.crit.QZ(state, action))
        if self.hps.clipped_double:
            twin_q = self.denorm_rets(self.twin.QZ(state, action))

        # Compute target QZ estimate
        next_action = float(self.max_ac) * self.targ_actr.sample(next_state, sg=True)
        # Note, here, always stochastic selection of the target action
        q_prime = self.targ_crit.QZ(next_state, next_action)
        if self.hps.clipped_double:
            # Define QZ' as the minimum QZ value between TD3's twin QZ's
            twin_q_prime = self.targ_twin.QZ(next_state, next_action)
            q_prime = torch.min(q_prime, twin_q_prime)

        # Add the causal entropy regularization term
        next_log_prob = self.targ_actr.logp(next_state, next_action)
        q_prime -= self.alpha * next_log_prob

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

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Add CQL contribution (rest is pretty much exactly SAC)
        # Actions and log-probabilities
        stride = 10
        temperature = 1.0
        min_q_weight = 5.0  # or 10.0 apparently: https://github.com/aviralkumar2907/CQL
        min_q_version = 3

        cql_ac, cql_logp = self.ac_factory(self.actr)(state, stride)
        cql_next_ac, cql_next_logp = self.ac_factory(self.actr)(next_state, stride=stride)
        cql_rand_ac = torch.Tensor(self.hps.batch_size * stride, self.ac_dim).uniform_(-1, 1).to(self.device)
        # Q-values
        cql_q = self.q_factory(self.crit)(state, cql_ac)
        cql_next_q = self.q_factory(self.crit)(state, cql_next_ac)
        # XXX: probably wrong -> next state
        cql_rand_q = self.q_factory(self.crit)(state, cql_rand_ac)
        if self.hps.clipped_double:
            cql_twin_q = self.q_factory(self.twin)(state, cql_ac)
            cql_next_twin_q = self.q_factory(self.twin)(state, cql_next_ac)
            # XXX: probably wrong -> next state
            cql_rand_twin_q = self.q_factory(self.twin)(state, cql_rand_ac)
        # Concatenate
        # dim set to 1 not -1, ensure the size is not 1

        if min_q_version == 3:
            # Importance-sampled version
            weird_stuff = np.log(0.5 ** cql_rand_ac.shape[-1])
            cql_cat_q = torch.cat(
                [cql_rand_q - weird_stuff,
                 cql_next_q - cql_next_logp.detach(),
                 cql_q - cql_logp.detach()],
                dim=1,
            )
            if self.hps.clipped_double:
                cql_cat_twin_q = torch.cat(
                    [cql_rand_twin_q - weird_stuff,
                     cql_next_twin_q - cql_next_logp.detach(),
                     cql_twin_q - cql_logp.detach()],
                    dim=1,
                )
        else:
            cql_cat_q = torch.cat(
                [q,
                 cql_q,
                 cql_next_q,
                 cql_rand_q],
                dim=1,
            )
            if self.hps.clipped_double:
                cql_cat_twin_q = torch.cat(
                    [twin_q,
                     cql_twin_q,
                     cql_next_twin_q,
                     cql_rand_twin_q],
                    dim=1,
                )

        # Compute the standard deviation (only for monitoring purposes)
        cql_std_q = torch.std(cql_cat_q, dim=1)
        if self.hps.clipped_double:
            cql_std_twin_q = torch.std(cql_cat_twin_q, dim=1)

        min_crit_loss = torch.logsumexp(cql_cat_q / temperature, dim=1).mean()
        min_crit_loss *= min_q_weight * temperature
        if self.hps.clipped_double:
            min_twin_loss = torch.logsumexp(cql_cat_twin_q / temperature, dim=1).mean()
            min_twin_loss *= min_q_weight * temperature

        min_crit_loss -= q.mean() * min_q_weight
        if self.hps.clipped_double:
            min_twin_loss -= twin_q.mean() * min_q_weight

        # Add the new losses to the vanilla ones
        crit_loss += min_crit_loss
        if self.hps.clipped_double:
            twin_loss += min_twin_loss

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.crit_opt.zero_grad()
        crit_loss.backward(retain_graph=True)
        average_gradients(self.crit, self.device)
        if self.hps.clipped_double:
            self.twin_opt.zero_grad()
            twin_loss.backward(retain_graph=True)
            average_gradients(self.twin, self.device)
        self.crit_opt.step()
        if self.hps.clipped_double:
            self.twin_opt.step()

        # Log metrics
        metrics['crit_loss'].append(crit_loss)
        if self.hps.clipped_double:
            metrics['twin_loss'].append(twin_loss)
        if self.hps.prioritized_replay:
            metrics['iws'].append(iws)

        if iters_so_far % self.hps.crit_targ_update_freq == 0:
            self.update_target_net(iters_so_far)

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        lrnows = {'actr': self.actr_sched.get_last_lr()}

        return metrics, lrnows

    def update_target_net(self, iters_so_far):
        """Update the target networks"""
        for param, targ_param in zip(self.actr.parameters(), self.targ_actr.parameters()):
            targ_param.data.copy_(self.hps.polyak * param.data +
                                  (1. - self.hps.polyak) * targ_param.data)
        for param, targ_param in zip(self.crit.parameters(), self.targ_crit.parameters()):
            targ_param.data.copy_(self.hps.polyak * param.data +
                                  (1. - self.hps.polyak) * targ_param.data)
        if self.hps.clipped_double:
            for param, targ_param in zip(self.twin.parameters(), self.targ_twin.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)

    def save(self, path, iters):
        SaveBundle = namedtuple('SaveBundle', ['model', 'optimizer', 'scheduler'])
        actr_bundle = SaveBundle(
            model=self.actr.state_dict(),
            optimizer=self.actr_opt.state_dict(),
            scheduler=self.actr_sched.state_dict(),
        )
        crit_bundle = SaveBundle(
            model=self.crit.state_dict(),
            optimizer=self.crit_opt.state_dict(),
            scheduler=None,
        )
        torch.save(actr_bundle._asdict(), osp.join(path, "model_actr_iter{}.pth".format(iters)))
        torch.save(crit_bundle._asdict(), osp.join(path, "model_crit_iter{}.pth".format(iters)))
        if self.hps.clipped_double:
            twin_bundle = SaveBundle(
                model=self.twin.state_dict(),
                optimizer=self.twin_opt.state_dict(),
                scheduler=None,
            )
            torch.save(twin_bundle._asdict(), osp.join(path, "model_twin_iter{}.pth".format(iters)))

    def load(self, path, iters):
        actr_bundle = torch.load(osp.join(path, "model_actr_iter{}.pth".format(iters)))
        self.actr.load_state_dict(actr_bundle['model'])
        self.actr_opt.load_state_dict(actr_bundle['optimizer'])
        self.actr_sched.load_state_dict(actr_bundle['scheduler'])
        crit_bundle = torch.load(osp.join(path, "model_crit_iter{}.pth".format(iters)))
        self.crit.load_state_dict(crit_bundle['model'])
        self.crit_opt.load_state_dict(crit_bundle['optimizer'])
        if self.hps.clipped_double:
            twin_bundle = torch.load(osp.join(path, "model_twin_iter{}.pth".format(iters)))
            self.twin.load_state_dict(twin_bundle['model'])
            self.twin_opt.load_state_dict(twin_bundle['optimizer'])
