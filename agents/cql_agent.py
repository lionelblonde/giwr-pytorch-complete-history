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
from agents.nets import perception_stack_parser, TanhGaussActor, Critic


ALPHA_PRI_CLAMPS = [0., 1_000_000.]
CQL_TEMP = 1.0
CWPQ_TEMP = 10.0

debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 2)


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
        if self.hps.clip_norm <= 0:
            logger.info("clip_norm={} <= 0, hence disabled.".format(self.hps.clip_norm))
        assert not self.hps.use_c51 and not self.hps.use_qr

        # Override with environment-specific hyper-parameter values, in line with CQL's codebase
        self.hps.cql_targ_lower_bound = 5.0 if 'antmaze' in self.hps.env_id else 1.0

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

        self.hps.use_adaptive_alpha = None  # unused in this algorithm, make sure it can not interfere
        # Common trick: rewrite the Lagrange multiplier alpha as log(w), and optimize for w
        if self.hps.cql_use_adaptive_alpha_ent:
            # Create learnable Lagrangian multiplier
            self.log_alpha_ent = torch.tensor(self.hps.cql_init_temp_log_alpha_ent).to(self.device)
            self.log_alpha_ent.requires_grad = True
        else:
            self.log_alpha_ent = self.hps.cql_init_temp_log_alpha_ent
        if self.hps.cql_use_adaptive_alpha_pri:
            # Create learnable Lagrangian multiplier
            self.log_alpha_pri = torch.tensor(self.hps.cql_init_temp_log_alpha_pri).to(self.device)
            self.log_alpha_pri.requires_grad = True
        else:
            self.log_alpha_pri = self.hps.cql_init_temp_log_alpha_pri

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

        if self.hps.cql_use_adaptive_alpha_ent:  # cql choice: same lr as actor
            self.log_alpha_ent_opt = torch.optim.Adam([self.log_alpha_ent],
                                                      lr=self.hps.actor_lr)
        if self.hps.cql_use_adaptive_alpha_pri:  # cql choice: same lr as critic
            self.log_alpha_pri_opt = torch.optim.Adam([self.log_alpha_pri],
                                                      lr=self.hps.critic_lr)

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

    @property
    def alpha_ent(self):
        if self.hps.cql_use_adaptive_alpha_ent:
            return self.log_alpha_ent.exp()
        else:
            return self.hps.cql_init_temp_log_alpha_ent

    @property
    def alpha_pri(self):
        if self.hps.cql_use_adaptive_alpha_pri:
            return self.log_alpha_pri.exp().clamp(*ALPHA_PRI_CLAMPS)
        else:
            return self.hps.cql_init_temp_log_alpha_pri

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

    def ac_factory(self, actr, ob, inflate):
        _ob = ob.unsqueeze(1).repeat(1, inflate, 1).view(ob.shape[0] * inflate, ob.shape[1])
        _ac = float(self.max_ac) * actr.sample(_ob, sg=False)
        _logp = actr.logp(_ob, _ac)
        return _ac, _logp.view(ob.shape[0], inflate, 1)

    def q_factory(self, crit, ob, ac):
        ob_dim = ob.shape[0]
        ac_dim = ac.shape[0]
        num_repeat = int(ac_dim / ob_dim)
        _ob = ob.unsqueeze(1).repeat(1, num_repeat, 1).view(ob.shape[0] * num_repeat, ob.shape[1])
        q = crit.QZ(_ob, ac).view(ob.shape[0], num_repeat, 1)
        return q

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

        action_from_actr = float(self.max_ac) * self.actr.sample(state, sg=False)
        log_prob = self.actr.logp(state, action_from_actr)

        # Only update the policy after a certain number of iteration (CQL codebase: 20000)
        # Note, as opposed to BEAR and BRAC, after the warm start, the BC loss is not used anymore

        # Actor loss
        if iters_so_far >= self.hps.warm_start:
            # Use full-blown loss
            q_from_actr = self.crit.QZ(state, action_from_actr)
            if self.hps.clipped_double:
                twin_q_from_actr = self.twin.QZ(state, action_from_actr)
                q_from_actr = torch.min(q_from_actr, twin_q_from_actr)
            actr_loss = ((self.alpha_ent * log_prob) - q_from_actr).mean()
        else:
            # Use behavioral cloning loss
            actr_loss = ((self.alpha_ent * log_prob) - self.actr.logp(state, action)).mean()

        metrics['actr_loss'].append(actr_loss)

        self.actr_opt.zero_grad()
        if self.hps.cql_use_adaptive_alpha_ent:
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

        if self.hps.cql_use_adaptive_alpha_ent:
            alpha_ent_loss = (self.log_alpha_ent * (-log_prob - self.targ_ent).detach()).mean()
            self.log_alpha_ent_opt.zero_grad()
            alpha_ent_loss.backward()
            self.log_alpha_ent_opt.step()
            metrics['alpha_ent_loss'].append(alpha_ent_loss)

        if DEBUG:
            logger.info(f"alpha_ent: {self.alpha_ent}")  # leave this here, for sanity checks

        # Compute QZ estimate
        q = self.denorm_rets(self.crit.QZ(state, action))
        if self.hps.clipped_double:
            twin_q = self.denorm_rets(self.twin.QZ(state, action))

        # Compute target QZ estimate
        next_action = float(self.max_ac) * self.actr.sample(next_state, sg=True)
        # Note, here, always stochastic selection of the target action
        q_prime = self.targ_crit.QZ(next_state, next_action)
        if self.hps.clipped_double:
            # Define QZ' as the minimum QZ value between TD3's twin QZ's
            twin_q_prime = self.targ_twin.QZ(next_state, next_action)
            q_prime = (self.hps.ensemble_q_lambda * torch.min(q_prime, twin_q_prime) +
                       (1. - self.hps.ensemble_q_lambda) * torch.max(q_prime, twin_q_prime))

        if not self.hps.cql_deterministic_backup:
            # Add the causal entropy regularization term
            next_log_prob = self.actr.logp(next_state, next_action)
            q_prime -= self.alpha_ent * next_log_prob

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

        # Assemble the 3 pieces of the CQL loss
        # (cf. slide 16 in: https://docs.google.com/presentation/d/
        # 1F-dNg2LT75z9vJiPqASHayiZ3ewB6HE0KPgnZnY2rTg/edit#slide=id.g80c29cc4d2_0_101)

        min_crit_loss = 0.
        if self.hps.clipped_double:
            min_twin_loss = 0.

        assert self.hps.base_pe_loss in ['pure_td', 'cql_1', 'cql_2'], "invalid base loss for policy evaluation."

        if self.hps.base_pe_loss in ['cql_1', 'cql_2']:
            # Piece #1: minimize the Q-function everywhere (consequently, the erroneously big Q-values
            # will be the first to be shrinked)

            # Add CQL contribution (rest is pretty much exactly SAC)
            # Actions and log-probabilities
            cql_ac, cql_logp = self.ac_factory(self.actr, state, self.hps.cql_state_inflate)
            cql_next_ac, cql_next_logp = self.ac_factory(self.actr, next_state, self.hps.cql_state_inflate)
            cql_rand_ac = torch.Tensor(
                self.hps.batch_size * self.hps.cql_state_inflate, self.ac_dim
            ).uniform_(-self.max_ac, self.max_ac).to(self.device)
            # Q-values
            cql_q = self.q_factory(self.crit, state, cql_ac)
            cql_next_q = self.q_factory(self.crit, state, cql_next_ac)
            cql_rand_q = self.q_factory(self.crit, state, cql_rand_ac)
            if self.hps.clipped_double:
                cql_twin_q = self.q_factory(self.twin, state, cql_ac)
                cql_next_twin_q = self.q_factory(self.twin, state, cql_next_ac)
                cql_rand_twin_q = self.q_factory(self.twin, state, cql_rand_ac)

            # Concatenate every Q-values estimates into one big vector that we'll later try to shrink
            # The answer to "why are so many Q-values are evaluated here?" is:
            # "we want to cover the maximum amount of ground, so we consider all the Q-values we can afford."
            # Note, `dim` is set to 1 not -1, ensure the size is not 1

            # Note, some importance-sampling sprinkled in there

            weird_stuff = np.log(0.5 ** cql_rand_ac.shape[-1])
            cql_cat_q = torch.cat([
                cql_rand_q - weird_stuff,
                cql_next_q - cql_next_logp.detach(),
                cql_q - cql_logp.detach(),
            ], dim=1)
            if self.hps.clipped_double:
                cql_cat_twin_q = torch.cat([
                    cql_rand_twin_q - weird_stuff,
                    cql_next_twin_q - cql_next_logp.detach(),
                    cql_twin_q - cql_logp.detach(),
                ], dim=1)

            min_crit_loss += (torch.logsumexp(cql_cat_q / CQL_TEMP, dim=1).mean() *
                              self.hps.cql_min_q_weight * CQL_TEMP)
            if self.hps.clipped_double:
                min_twin_loss += (torch.logsumexp(cql_cat_twin_q / CQL_TEMP, dim=1).mean() *
                                  self.hps.cql_min_q_weight * CQL_TEMP)

        if self.hps.base_pe_loss == 'cql_2':
            # Piece #2: maximize the Q-function on points in the offline dataset
            min_crit_loss -= (q.mean() * self.hps.cql_min_q_weight)
            if self.hps.clipped_double:
                min_twin_loss -= (twin_q.mean() * self.hps.cql_min_q_weight)

        if self.hps.base_pe_loss in ['cql_1', 'cql_2']:
            # Introduce the (learnable) coefficient alpha prime to scale the previous loss(es)
            min_crit_loss = self.alpha_pri * (min_crit_loss - self.hps.cql_targ_lower_bound)
            if self.hps.clipped_double:
                min_twin_loss = self.alpha_pri * (min_twin_loss - self.hps.cql_targ_lower_bound)

            if self.hps.cql_use_adaptive_alpha_pri:
                if self.hps.clipped_double:
                    alpha_pri_loss = -0.5 * (min_crit_loss + min_twin_loss)
                else:
                    alpha_pri_loss = -min_crit_loss
                self.log_alpha_pri_opt.zero_grad()
                alpha_pri_loss.backward(retain_graph=True)
                self.log_alpha_pri_opt.step()
                metrics['alpha_pri_loss'].append(alpha_pri_loss)

            if DEBUG:
                logger.info(f"alpha_pri: {self.alpha_pri}")  # leave this here, for sanity checks

        # Piece #3: Add the new losses to the vanilla ones, i.e. the traditional TD errors to minimize
        crit_loss += min_crit_loss
        if self.hps.clipped_double:
            twin_loss += min_twin_loss

        self.crit_opt.zero_grad()
        crit_loss.backward(retain_graph=True)
        # It is here necessary to retain the graph because the other Q function(s) in the ensemble re-use it
        # There is an overlap because the dependency graph in cql does not stop at the Q function input,
        # but goes futher in the actor because of the extra components in the loss of the Q's that contain
        # actions from the agent at the current and next states. Since these generated actions are
        # shared by all the Q's in the ensemble, we need to retain the graph after the first backward.
        # Note, we would not have to do this if we had summed the losses of the Q's of the ensemble,
        # and optimized it with a single optimizer.
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

        if iters_so_far % self.hps.crit_targ_update_freq == 0:
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

    def load(self, path, iters_so_far):
        if self.hps.obs_norm:
            self.rms_obs.load_state_dict(torch.load(osp.join(path, f"rms_obs_{iters_so_far}.pth")))
        self.actr.load_state_dict(torch.load(osp.join(path, f"actr_{iters_so_far}.pth")))
        self.crit.load_state_dict(torch.load(osp.join(path, f"crit_{iters_so_far}.pth")))
        if self.hps.clipped_double:
            self.twin.load_state_dict(torch.load(osp.join(path, f"twin_{iters_so_far}.pth")))
