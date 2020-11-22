from collections import defaultdict
import os.path as osp

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch import autograd

from helpers import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from helpers.math_util import huber_quant_reg_loss
from helpers.math_util import LRScheduler
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import perception_stack_parser, TanhGaussActor, MixtureTanhGaussActor, Critic
from agents.rnd import RandomNetworkDistillation


ALPHA_PRI_CLAMPS = [0., 1_000_000.]
CQL_TEMP = 1.0
EPS_CE = 1e-6
U_ESTIM_SAMPLES = 10
CRR_TEMP = 1.0
ADV_ESTIM_SAMPLES = 4


class PTSOAgent(object):

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

        # Define critic to use
        assert sum([self.hps.use_c51, self.hps.use_qr]) <= 1
        if self.hps.use_c51:
            assert not self.hps.clipped_double
            c51_supp_range = (self.hps.c51_vmin,
                              self.hps.c51_vmax,
                              self.hps.c51_num_atoms)
            self.c51_supp = torch.linspace(*c51_supp_range).to(self.device)
            self.c51_delta = ((self.hps.c51_vmax - self.hps.c51_vmin) /
                              (self.hps.c51_num_atoms - 1))
            c51_offset_range = (0,
                                (self.hps.batch_size - 1) * self.hps.c51_num_atoms,
                                self.hps.batch_size)
            c51_offset = torch.linspace(*c51_offset_range).to(self.device)
            self.c51_offset = c51_offset.long().unsqueeze(1).expand(self.hps.batch_size,
                                                                    self.hps.c51_num_atoms)
        elif self.hps.use_qr:
            assert not self.hps.clipped_double
            qr_cum_density = np.array([((2 * i) + 1) / (2.0 * self.hps.num_tau)
                                       for i in range(self.hps.num_tau)])
            qr_cum_density = torch.Tensor(qr_cum_density).to(self.device)
            self.qr_cum_density = qr_cum_density.view(1, 1, -1, 1).expand(self.hps.batch_size,
                                                                          self.hps.num_tau,
                                                                          self.hps.num_tau,
                                                                          -1).to(self.device)

        # Parse the noise types
        self.param_noise, self.ac_noise = None, None  # keep this, needed in orchestrator

        # Create observation normalizer that maintains running statistics
        if self.hps.obs_norm:
            self.rms_obs = RunMoms(shape=self.ob_shape, use_mpi=not self.hps.cuda)  # no mpi sharing when using cuda
        else:
            self.rms_obs = None

        assert self.hps.ret_norm or not self.hps.popart
        assert not (self.hps.use_c51 and self.hps.ret_norm)
        assert not (self.hps.use_qr and self.hps.ret_norm)
        if self.hps.ret_norm:
            # Create return normalizer that maintains running statistics
            self.rms_ret = RunMoms(shape=(1,), use_mpi=False)

        # Create online and target nets, and initialize the target nets
        hidden_dims = perception_stack_parser(self.hps.perception_stack)
        Actr = MixtureTanhGaussActor if self.hps.gauss_mixture else TanhGaussActor

        self.actr = Actr(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        sync_with_root(self.actr)
        self.main_eval_actr = Actr(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.maxq_eval_actr = Actr(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.main_eval_actr.load_state_dict(self.actr.state_dict())
        self.maxq_eval_actr.load_state_dict(self.actr.state_dict())

        self.crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=True).to(self.device)
        self.targ_crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=True).to(self.device)
        self.targ_crit.load_state_dict(self.crit.state_dict())
        if self.hps.clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=False).to(self.device)
            self.targ_twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=False).to(self.device)
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

        # Initialize the precision matrix
        self.precision = 5. * torch.eye(hidden_dims[1][-1]).to(self.device)

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
        self.crit_opt = torch.optim.Adam(self.crit.q_trainable_params,
                                         lr=self.hps.critic_lr,
                                         weight_decay=self.hps.wd_scale)
        if self.hps.clipped_double:
            self.twin_opt = torch.optim.Adam(self.twin.q_trainable_params,
                                             lr=self.hps.critic_lr,
                                             weight_decay=self.hps.wd_scale)

        self.u_opt = torch.optim.Adam(self.crit.u_trainable_params, lr=1e-3)

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

        if self.hps.ptso_use_rnd_monitoring:
            # Create RND networks
            self.rnd = RandomNetworkDistillation(self.env, self.device, self.hps, self.rms_obs)

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
            return self.log_alpha_pri.exp().data.clamp_(*ALPHA_PRI_CLAMPS)
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
        # Store transition in the replay buffer
        self.replay_buffer.append(transition)
        # Update the observation normalizer
        self.rms_obs.update(transition['obs0'])

    def patcher(self):
        raise NotImplementedError  # no need

    def sample_batch(self):
        """Sample a batch of transitions from the replay buffer"""

        # def _patcher(x, y, z):
        #     return self.patcher(x, y, z).detach().cpu().numpy()  # redundant detach

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

    def predict(self, ob, apply_noise, max_q):
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
            if max_q:
                _actr = self.maxq_eval_actr
                ob = torch.Tensor(ob[None]).to(self.device).repeat(100, 1)  # duplicate 100 times
                ac = float(self.max_ac) * _actr.sample(ob, sg=True)
                # Among the 100 values, take the one with the highest Q value (or Z value)
                q_value = self.crit.QZ(ob, ac).mean(dim=1)

                if self.hps.ptso_use_u_inference_time:
                    u_value = self.crit.wrap_with_u_head(self.crit.phi(ob, ac)).mean(dim=1).clamp(min=1e-6).sqrt()
                    q_value -= self.hps.ptso_u_scale_p_i * u_value

                index = q_value.argmax(0)
                ac = ac[index]
            else:
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
        q_value = crit.QZ(_ob, ac)
        if self.hps.use_c51:
            q_value = q_value.matmul(self.c51_supp).unsqueeze(-1)
        elif self.hps.use_qr:
            q_value = q_value.mean(dim=1, keepdim=True)
        return q_value.view(ob.shape[0], num_repeat, 1)

    def u_factory(self, crit, ob, ac):
        ob_dim = ob.shape[0]
        ac_dim = ac.shape[0]
        num_repeat = int(ac_dim / ob_dim)
        _ob = ob.unsqueeze(1).repeat(1, num_repeat, 1).view(ob.shape[0] * num_repeat, ob.shape[1])
        u_value = crit.wrap_with_u_head(self.crit.phi(_ob, ac)).view(ob.shape[0], num_repeat, 1)
        return u_value

    def q_cb_factory(self, crit, ob, ac, actr, ucb_or_lcb, u_scale):  # ac here just for the shape
        # By construction, 'ob' and 'ac' have the same shape
        _ob_tiled = ob.unsqueeze(1).repeat(1, self.hps.cql_state_inflate, 1).view(
            ob.shape[0] * self.hps.cql_state_inflate, ob.shape[1]
        )
        _ac_tiled = float(self.max_ac) * actr.sample(_ob_tiled, sg=True)
        _ac_untiled = _ac_tiled.view(ac.shape[0], self.hps.cql_state_inflate, ac.shape[1])
        with torch.no_grad():
            _q_value = crit.QZ(_ob_tiled, _ac_tiled)
            if self.hps.use_c51:
                _q_value = _q_value.matmul(self.c51_supp).unsqueeze(-1)
            elif self.hps.use_qr:
                _q_value = _q_value.mean(dim=1, keepdim=True)
            _q_value = _q_value.mean(dim=1)
            if hasattr(crit, 'u_head'):
                _u_value = crit.wrap_with_u_head(crit.phi(_ob_tiled, _ac_tiled)).mean(dim=1).clamp(min=1e-6).sqrt()
                if ucb_or_lcb:
                    _q_value += u_scale * _u_value
                else:
                    _q_value -= u_scale * _u_value
            _q_value = _q_value.view(ob.shape[0], self.hps.cql_state_inflate, 1)
            if ucb_or_lcb:
                index = _q_value.argmax(1)
            else:
                index = _q_value.argmin(1)
            optimistic_ac = torch.gather(_ac_untiled, 1, index.unsqueeze(-1).expand(-1, -1, ac.shape[1])).view(
                ac.shape[0], ac.shape[1]
            )

            # # Sanity-check, to assert that the gather operation does what we want
            # _optimistic_ac = torch.empty(ac.shape[0], ac.shape[1]).to(self.device)
            # for i in range(ac.shape[0]):
            #     _optimistic_ac[i, :] = _ac_untiled[i, index[i], :]
            # assert torch.all(torch.eq(_optimistic_ac, optimistic_ac))

        q_value = crit.QZ(ob, optimistic_ac)
        if self.hps.use_c51:
            q_value = q_value.matmul(self.c51_supp).unsqueeze(-1)
        elif self.hps.use_qr:
            q_value = q_value.mean(dim=1, keepdim=True)
        return q_value.view(ob.shape[0], 1, 1)

    def update_actor_critic(self, batch, update_actor, iters_so_far):
        """Train the actor and critic networks
        Note, 'update_actor' is here to keep the unified signature.
        """

        # Container for all the metrics
        metrics = defaultdict(list)

        if self.hps.ptso_use_rnd_monitoring:
            # Update the RND network
            self.rnd.update(batch)
            logger.info("just updated the rnd estimate")

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

        next_action = float(self.max_ac) * self.actr.sample(next_state, sg=True)
        # Note, here, always stochastic selection of the target action

        if self.hps.use_c51:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> C51.

            # Compute QZ estimate
            z = self.crit.QZ(state, action).unsqueeze(-1)
            # Compute target QZ estimate
            z_prime = self.targ_crit.QZ(next_state, next_action).detach()
            # Project on the c51 support
            gamma_mask = (self.hps.gamma ** td_len) * (1 - done)
            p_targ_z = reward + (gamma_mask * self.c51_supp.unsqueeze(0))
            # Clamp when the projected value goes under the min or over the max
            p_targ_z = p_targ_z.clamp(self.hps.c51_vmin, self.hps.c51_vmax)
            # Define the translation (bias) to map the projection to [0, num_atoms - 1]
            bias = (p_targ_z - self.hps.c51_vmin) / self.c51_delta
            # How to assign mass at atoms while the values are over the continuous
            # interval [0, num_atoms - 1]? Calculate the lower and upper atoms.
            # Note, the integers of the interval coincide exactly with the atoms.
            # The considered atoms are therefore calculated simply using floor and ceil.
            l_atom, u_atom = bias.floor().long(), bias.ceil().long()
            # Deal with the case where bias is an integer (exact atom), bias = l_atom = u_atom,
            # in which case we offset l by 1 to the left and u by 1 to the right when applicable.
            l_atom[(u_atom > 0) * (l_atom == u_atom)] -= 1
            u_atom[(l_atom < (self.hps.c51_num_atoms - 1)) * (l_atom == u_atom)] += 1
            # Calculate the gaps between the bias and the lower and upper atoms respectively
            l_gap = bias - l_atom.float()
            u_gap = u_atom.float() - bias
            # Create the translated and projected target, with the right size, but zero-filled
            t_p_targ_z = z_prime.detach().clone().zero_()  # detach just in case
            t_p_targ_z.view(-1).index_add_(
                0,
                (l_atom + self.c51_offset).view(-1),
                (z_prime * u_gap).view(-1)
            )
            t_p_targ_z.view(-1).index_add_(
                0,
                (u_atom + self.c51_offset).view(-1),
                (z_prime * l_gap).view(-1)
            )
            # Reshape target to be of shape [batch_size, self.hps.c51_num_atoms, 1]
            t_p_targ_z = t_p_targ_z.view(-1, self.hps.c51_num_atoms, 1)

            # Critic loss
            ce_losses = -(t_p_targ_z.detach() * torch.log(z.clamp(EPS_CE, 1. - EPS_CE))).sum(dim=1)
            # shape is batch_size x 1

            if self.hps.prioritized_replay:
                # Update priorities
                new_priorities = np.abs(ce_losses.detach().cpu().numpy()) + 1e-6
                self.replay_buffer.update_priorities(batch['idxs'].reshape(-1), new_priorities)
                # Adjust with importance weights
                ce_losses *= iws

            crit_loss = ce_losses.mean()

        elif self.hps.use_qr:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> QR.

            # Compute QZ estimate
            z = self.crit.QZ(state, action).unsqueeze(-1)

            # Compute target QZ estimate
            z_prime = self.targ_crit.QZ(next_state, next_action)
            # Reshape rewards to be of shape [batch_size x num_tau, 1]
            reward = reward.repeat(self.hps.num_tau, 1)
            # Reshape product of gamma and mask to be of shape [batch_size x num_tau, 1]
            gamma_mask = ((self.hps.gamma ** td_len) * (1 - done)).repeat(self.hps.num_tau, 1)
            z_prime = z_prime.view(-1, 1)
            targ_z = reward + (gamma_mask * z_prime)
            # Reshape target to be of shape [batch_size, num_tau, 1]
            targ_z = targ_z.view(-1, self.hps.num_tau, 1)

            # Critic loss
            # Compute the TD error loss
            # Note: online version has shape [batch_size, num_tau, 1],
            # while the target version has shape [batch_size, num_tau, 1].
            td_errors = targ_z[:, :, None, :].detach() - z[:, None, :, :]  # broadcasting
            # The resulting shape is [batch_size, num_tau, num_tau, 1]

            # Assemble the Huber Quantile Regression loss
            huber_td_errors = huber_quant_reg_loss(td_errors, self.qr_cum_density)
            # The resulting shape is [batch_size, num_tau_prime, num_tau, 1]

            if self.hps.prioritized_replay:
                # Adjust with importance weights
                huber_td_errors *= iws
                # Update priorities
                new_priorities = np.abs(td_errors.sum(dim=2).mean(dim=1).detach().cpu().numpy())
                new_priorities += 1e-6
                self.replay_buffer.update_priorities(batch['idxs'].reshape(-1), new_priorities)

            # Sum over current quantile value (tau, N in paper) dimension, and
            # average over target quantile value (tau prime, N' in paper) dimension.
            crit_loss = huber_td_errors.sum(dim=2)
            # Resulting shape is [batch_size, num_tau_prime, 1]
            crit_loss = crit_loss.mean(dim=1)
            # Resulting shape is [batch_size, 1]
            # Average across the minibatch
            crit_loss = crit_loss.mean()

        else:

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> VANILLA.

            # Compute QZ estimate
            q = self.denorm_rets(self.crit.QZ(state, action))
            if self.hps.clipped_double:
                twin_q = self.denorm_rets(self.twin.QZ(state, action))

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

        if self.hps.cql_use_min_q_loss:
            # Add CQL contribution (rest is pretty much exactly SAC)
            # Actions and log-probabilities
            cql_ac, cql_logp = self.ac_factory(self.actr, state, self.hps.cql_state_inflate)
            if self.hps.ptso_q_min_scale > 0 or self.hps.ptso_q_max_scale > 0:
                ptso_ac, ptso_logp = self.ac_factory(self.actr, state, 1)  # do not inflate the state here
            cql_next_ac, cql_next_logp = self.ac_factory(self.actr, next_state, self.hps.cql_state_inflate)
            cql_rand_ac = torch.Tensor(self.hps.batch_size * self.hps.cql_state_inflate,
                                       self.ac_dim).uniform_(-self.max_ac, self.max_ac).to(self.device)
            # Q-values
            cql_q = self.q_factory(self.crit, state, cql_ac)
            cql_next_q = self.q_factory(self.crit, state, cql_next_ac)
            cql_rand_q = self.q_factory(self.crit, state, cql_rand_ac)
            if self.hps.clipped_double:
                cql_twin_q = self.q_factory(self.twin, state, cql_ac)
                cql_next_twin_q = self.q_factory(self.twin, state, cql_next_ac)
                cql_rand_twin_q = self.q_factory(self.twin, state, cql_rand_ac)

            # Q-values associated with actions to emphasize on
            if self.hps.ptso_q_min_scale > 0:
                ptso_q_min = self.q_cb_factory(
                    self.crit, state, ptso_ac, self.actr, ucb_or_lcb=True, u_scale=self.hps.ptso_u_scale_q_min
                )
            if self.hps.ptso_q_max_scale > 0:
                ptso_q_max = self.q_cb_factory(
                    self.crit, state, ptso_ac, self.actr, ucb_or_lcb=False, u_scale=self.hps.ptso_u_scale_q_max
                )
            if self.hps.clipped_double:
                if self.hps.ptso_q_min_scale > 0:
                    ptso_twin_q_min = self.q_cb_factory(
                        self.twin, state, ptso_ac, self.actr, ucb_or_lcb=True, u_scale=self.hps.ptso_u_scale_q_min
                    )
                if self.hps.ptso_q_max_scale > 0:
                    ptso_twin_q_max = self.q_cb_factory(
                        self.twin, state, ptso_ac, self.actr, ucb_or_lcb=False, u_scale=self.hps.ptso_u_scale_q_max
                    )

            # Concatenate every Q-values estimates into one big vector that we'll later try to shrink
            # The answer to "why are so many Q-values are evaluated here?" is:
            # "we want to cover the maximum amount of ground, so we consider all the Q-values we can afford."
            # Note, `dim` is set to 1 not -1, ensure the size is not 1
            if self.hps.cql_use_version_3:
                # Importance-sampled version
                weird_stuff = np.log(0.5 ** cql_rand_ac.shape[-1])
                cql_cat_q = torch.cat([
                    cql_rand_q - weird_stuff,
                    cql_next_q - cql_next_logp.detach(),
                    cql_q - cql_logp.detach(),
                ], dim=1)
                if self.hps.ptso_q_min_scale > 0:
                    cql_cat_q = torch.cat([
                        cql_cat_q,
                        self.hps.ptso_q_min_scale * (ptso_q_min - ptso_logp.detach())
                    ], dim=1)
                if self.hps.clipped_double:
                    cql_cat_twin_q = torch.cat([
                        cql_rand_twin_q - weird_stuff,
                        cql_next_twin_q - cql_next_logp.detach(),
                        cql_twin_q - cql_logp.detach(),
                    ], dim=1)
                    if self.hps.ptso_q_min_scale > 0:
                        cql_cat_twin_q = torch.cat([
                            cql_cat_twin_q,
                            self.hps.ptso_q_min_scale * (ptso_twin_q_min - ptso_logp.detach())
                        ], dim=1)
            else:
                # Here for posterity, we always are in the clause above
                raise ValueError("uncomment the code below to use this clause.")
                # cql_cat_q = torch.cat([
                #     q.unsqueeze(1),
                #     cql_q,
                #     cql_next_q,
                #     cql_rand_q,
                # ], dim=1)
                # if self.hps.clipped_double:
                #     cql_cat_twin_q = torch.cat([
                #         twin_q.unsqueeze(1),
                #         cql_twin_q,
                #         cql_next_twin_q,
                #         cql_rand_twin_q,
                #     ], dim=1)
            # weirdly, the version 3 does not use the stock Q networks, but no questions asked for now

        # Assemble the 3 pieces of the CQL loss
        # (cf. slide 16 in: https://docs.google.com/presentation/d/
        # 1F-dNg2LT75z9vJiPqASHayiZ3ewB6HE0KPgnZnY2rTg/edit#slide=id.g80c29cc4d2_0_101)

        min_crit_loss = 0.
        if self.hps.clipped_double:
            min_twin_loss = 0.

        if self.hps.cql_use_min_q_loss:
            # Piece #1: minimize the Q-function everywhere (consequently, the erroneously big Q-values
            # will be the first to be shrinked)
            min_crit_loss += (torch.logsumexp(cql_cat_q / CQL_TEMP, dim=1).mean() *
                              self.hps.cql_min_q_weight * CQL_TEMP)
            if self.hps.clipped_double:
                min_twin_loss += (torch.logsumexp(cql_cat_twin_q / CQL_TEMP, dim=1).mean() *
                                  self.hps.cql_min_q_weight * CQL_TEMP)

        if self.hps.cql_use_max_q_loss:
            # Piece #2: maximize the Q-function on points in the offline dataset

            # When using distributional critics, define q from z
            if self.hps.use_c51:
                q = z.squeeze(-1).matmul(self.c51_supp).unsqueeze(-1)
            elif self.hps.use_qr:
                q = z.squeeze(-1).mean(dim=1, keepdim=True)

            min_crit_loss -= (q.mean() * self.hps.cql_min_q_weight)
            if self.hps.ptso_q_max_scale > 0:
                min_crit_loss -= (ptso_q_max.mean() * self.hps.ptso_q_max_scale)
            if self.hps.clipped_double:
                min_twin_loss -= (twin_q.mean() * self.hps.cql_min_q_weight)
                if self.hps.ptso_q_max_scale > 0:
                    min_twin_loss -= (ptso_twin_q_max.mean() * self.hps.ptso_q_max_scale)

            if self.hps.cql_use_adaptive_alpha_pri:
                min_crit_loss = self.alpha_pri * (min_crit_loss - self.hps.cql_targ_lower_bound)
                if self.hps.clipped_double:
                    min_twin_loss = self.alpha_pri * (min_twin_loss - self.hps.cql_targ_lower_bound)

                if self.hps.clipped_double:
                    alpha_pri_loss = -0.5 * (min_crit_loss + min_twin_loss)
                else:
                    alpha_pri_loss = -min_crit_loss
                self.log_alpha_pri_opt.zero_grad()
                alpha_pri_loss.backward(retain_graph=True)
                self.log_alpha_pri_opt.step()
                metrics['alpha_pri_loss'].append(alpha_pri_loss)

        # Piece #3: Add the new losses to the vanilla ones, i.e. the traditional TD errors to minimize
        crit_loss += min_crit_loss
        if self.hps.clipped_double:
            twin_loss += min_twin_loss

        if self.hps.ptso_use_or_monitor_grad_pen:
            # Add (or just monitor) gradient penalties to regularize the phi embedding

            # Note, we usually dump the metrics even if it's not an eval round (legibility),
            # but this bit it too computationally expensive
            if self.hps.ptso_grad_pen_scale_s > 0 or self.hps.ptso_grad_pen_scale_a > 0:
                # Use one or two gradient penalties in training
                crit_grad_pen_s, crit_grad_pen_a, crit_grads_norm_s, crit_grads_norm_a = self.grad_pen(
                    fa=self.crit.phi,
                    state=state,
                    action_1=action,
                    action_2=action_from_actr.detach(),
                )
                metrics['phi_gradnorm_s'].append(crit_grads_norm_s.mean())
                metrics['phi_gradnorm_a'].append(crit_grads_norm_a.mean())
                if self.hps.ptso_grad_pen_scale_s > 0:
                    crit_loss += self.hps.ptso_grad_pen_scale_s * crit_grad_pen_s.mean()
                if self.hps.ptso_grad_pen_scale_a > 0:
                    crit_loss += self.hps.ptso_grad_pen_scale_a * crit_grad_pen_a.mean()

                if self.hps.clipped_double:
                    if self.hps.ptso_grad_pen_scale_s > 0 or self.hps.ptso_grad_pen_scale_a > 0:
                        twin_grad_pen_s, twin_grad_pen_a, twin_grads_norm_s, twin_grads_norm_a = self.grad_pen(
                            fa=self.twin.phi,
                            state=state,
                            action_1=action,
                            action_2=action_from_actr.detach(),
                        )
                    if self.hps.ptso_grad_pen_scale_s > 0:
                        twin_loss += self.hps.ptso_grad_pen_scale_s * twin_grad_pen_s.mean()
                    if self.hps.ptso_grad_pen_scale_a > 0:
                        twin_loss += self.hps.ptso_grad_pen_scale_a * twin_grad_pen_a.mean()
            else:
                if iters_so_far % self.hps.eval_frequency == 0:
                    # Just monitor the gradient norms' values
                    _, _, crit_grads_norm_s, crit_grads_norm_a = self.grad_pen(
                        fa=self.crit.phi,
                        state=state,
                        action_1=action,
                        action_2=action_from_actr.detach(),
                    )
                    metrics['phi_gradnorm_s'].append(crit_grads_norm_s.mean())
                    metrics['phi_gradnorm_a'].append(crit_grads_norm_a.mean())

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

        # Piece #4 (PTSO): uncertainty via Uncertainty Bellman Equation
        # The target of the Uncertainty Bellman Equation is the variance estimate of the error epsilon (paper)

        # We don't use the twin critic past this point, to same on evalutations and matrix multiplications

        assert not self.hps.n_step_returns
        # Depending on a hyper-parameter, use the target critics in `targ_u` calculation, or not at all here
        crit_in_ube_targ = self.targ_crit if self.hps.ptso_use_targ_for_u else self.crit
        # Depending on a hyper-parameter, use the action from the behavior policy, or from the actor
        action_in_phi = action if self.hps.ptso_use_behav_ac_in_phi else action_from_actr.detach()

        # Create the UBE target
        # Computations done with batch dimension
        with torch.no_grad():
            # Assemble the uncertainty bellman target
            _new_phi = self.crit.phi(state, action_in_phi).unsqueeze(-1)
            _new_phi_t = torch.transpose(_new_phi, -1, -2)
            inflated_precision = self.precision.unsqueeze(0).repeat(self.hps.batch_size, 1, 1)
            v_q = torch.bmm(torch.bmm(_new_phi_t,
                                      inflated_precision),
                            _new_phi).squeeze(-1)
            u_prime = crit_in_ube_targ.wrap_with_u_head(crit_in_ube_targ.phi(next_state, next_action))
            targ_u = (v_q + ((self.hps.gamma ** 2) * (1. - done) * u_prime)).clamp(min=0.)
            # Note, we clamp for the target to be non-negative, to urge the prediction to always be non-negative
        u_pred = self.crit.wrap_with_u_head(self.crit.phi(state, action).detach())  # can only update the output head
        u_loss = F.mse_loss(u_pred, targ_u.detach())  # detach, just in case
        metrics['v_q'].append(v_q.mean())
        metrics['u_pred'].append(u_pred.mean())
        metrics['u_loss'].append(u_loss)

        self.u_opt.zero_grad()
        u_loss.backward(retain_graph=True)
        self.u_opt.step()

        if self.hps.ptso_use_rnd_monitoring:
            # Monitor associated rnd estimate
            metrics['rnd_score'].append(torch.exp(self.rnd.get_int_rew(state)) - 1.)

        # Monitoring uncertainty values with features evaluated at other actions
        if iters_so_far % self.hps.eval_frequency == 0:
            # Note, we usually dump the metrics even if it's not an eval round (legibility),
            # but this bit it too computationally expensive
            with torch.no_grad():
                # Compute the entities of interest with the action in phi sampled uniformly
                action_uniform = torch.Tensor(self.hps.batch_size, self.ac_dim).uniform_(-self.max_ac,
                                                                                         self.max_ac).to(self.device)
                _new_phi_uniform = self.crit.phi(state, action_uniform).unsqueeze(-1)
                _new_phi_t_uniform = torch.transpose(_new_phi_uniform, -1, -2)
                v_q_uniform = torch.bmm(torch.bmm(_new_phi_t_uniform,
                                                  inflated_precision),
                                        _new_phi_uniform).squeeze(-1)
                # Keep in the 'no_grad' context, this is just for monitoring
                u_pred_uniform = self.crit.wrap_with_u_head(self.crit.phi(state, action_uniform).detach()).mean()
                metrics['v_q_uniform'].append(v_q_uniform.mean())
                metrics['u_pred_uniform'].append(u_pred_uniform.mean())

        # Update the global uncertainty matrix (sigma in UBE, the covariance matrix) according to equation 14 in UBE
        # Computations done without the batch dimension (non-batchable iterative process)
        n = int(np.clip(self.hps.ptso_num_mat_updates_per_iter, 1, self.hps.batch_size))
        # Note, pick an n << batch_size to save compute time
        # Note, we adapt the global precision estimate with actions from the actor, not from the behavioral policy
        with torch.no_grad():
            for i in range(n):
                denom = (torch.matmul(torch.matmul(_new_phi_t[i, ...],
                                                   self.precision),
                                      _new_phi[i, ...])
                         + 1.).squeeze()  # scalar
                numer = torch.matmul(torch.matmul(torch.matmul(self.precision,
                                                               _new_phi[i, ...]),
                                                  _new_phi_t[i, ...]),
                                     self.precision)
                self.precision -= (numer / denom)

        if iters_so_far % self.hps.crit_targ_update_freq == 0:
            self.update_target_net(iters_so_far)

        # Only update the policy after a certain number of iteration (CQL codebase: 20000)
        # Note, as opposed to BEAR and BRAC, after the warm start, the BC loss is not used anymore

        # Actor loss
        if iters_so_far >= self.hps.warm_start:
            # Use offline RL loss

            # For monitoring purposes, evaluate Q at the agent's action
            # Even if unused by the selected loss thereafter, it is worth the evaluation cost
            q_from_actr = self.crit.QZ(state, action_from_actr)
            if self.hps.use_c51:
                q_from_actr = q_from_actr.matmul(self.c51_supp).unsqueeze(-1)
            elif self.hps.use_qr:
                q_from_actr = q_from_actr.mean(dim=1, keepdim=True)
            if self.hps.clipped_double:
                twin_q_from_actr = self.twin.QZ(state, action_from_actr)
                q_from_actr = torch.min(q_from_actr, twin_q_from_actr)
            metrics['q_mean'].append(q_from_actr.mean())
            metrics['q_std'].append(q_from_actr.std())
            metrics['q_min'].append(q_from_actr.min())
            metrics['q_max'].append(q_from_actr.max())
            # Same goes for the uncertainty
            phi_from_actr = self.crit.phi(state, action_from_actr)
            u_from_actr = self.crit.wrap_with_u_head(phi_from_actr).clamp(min=1e-8).sqrt()
            # Note, we clamp for the value to be non-negative, for the square-root to always be defined
            # We clip with an epsilon strictly greater than zero because of sqrt's derivative at zero
            metrics['u_mean'].append(u_from_actr.mean())
            metrics['u_std'].append(u_from_actr.std())
            metrics['u_min'].append(u_from_actr.min())
            metrics['u_max'].append(u_from_actr.max())
            # # Sanity-check
            # assert not u_from_actr.isnan().any()

            # Assemble base loss
            if self.hps.base_pi_loss in ['sac', 'cql']:
                actr_loss = (self.alpha_ent * log_prob) - q_from_actr
            elif self.hps.base_pi_loss in ['crr_exp', 'crr_binary', 'crr_binary_max']:
                crr_q = self.crit.QZ(state, action)
                if self.hps.use_c51:
                    crr_q = crr_q.matmul(self.c51_supp).unsqueeze(-1)
                elif self.hps.use_qr:
                    crr_q = crr_q.mean(dim=1, keepdim=True)
                emp_adv_ac, _ = self.ac_factory(self.actr, state, ADV_ESTIM_SAMPLES)
                if 'max' in self.hps.base_pi_loss:
                    emp_adv_from_actr = self.q_factory(self.crit, state, emp_adv_ac).max(dim=1).values.detach()
                else:
                    emp_adv_from_actr = self.q_factory(self.crit, state, emp_adv_ac).mean(dim=1).detach()
                crr_adv = crr_q - emp_adv_from_actr
                if 'binary' in self.hps.base_pi_loss:
                    crr_adv = 1. * crr_adv.gt(0.)
                else:  # only other case: exp
                    crr_adv = torch.exp(crr_adv / CRR_TEMP).clamp(max=20.)
                actr_loss = -self.actr.logp(state, action) * crr_adv
            else:
                raise NotImplementedError("invalid base loss for policy improvement.")

            # If opted in, add the uncertainty contribution in the policy improvement loss
            if self.hps.ptso_u_scale_p_i > 0:

                if self.hps.ptso_use_unexpected_uncertainty:
                    # Create an 'advantage-like' estimator for the uncertainty, the unexpected uncertainty
                    u_u_ac, _ = self.ac_factory(self.actr, state, U_ESTIM_SAMPLES)
                    expected_u_from_actr = self.u_factory(self.crit, state, u_u_ac).mean(dim=1).detach()
                    u_from_actr -= expected_u_from_actr

                if self.hps.ptso_use_v_and_u:
                    _new_phi = phi_from_actr.unsqueeze(-1)
                    _new_phi_t = torch.transpose(_new_phi, -1, -2)
                    inflated_precision = self.precision.unsqueeze(0).repeat(self.hps.batch_size, 1, 1)
                    v_q = torch.bmm(torch.bmm(_new_phi_t,
                                              inflated_precision),
                                    _new_phi).squeeze(-1)
                    actr_loss += self.hps.ptso_u_scale_p_i * (v_q + ((self.hps.gamma ** 2) * u_from_actr))
                else:
                    actr_loss += self.hps.ptso_u_scale_p_i * u_from_actr

            actr_loss = actr_loss.mean()
        else:
            # Use behavioral cloning losses
            actr_loss = F.mse_loss(action, action_from_actr)
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
        logger.info(f"lr is {_lr} after {iters_so_far} iters")

        if self.hps.cql_use_adaptive_alpha_ent:
            alpha_ent_loss = (self.log_alpha_ent * (-log_prob - self.targ_ent).detach()).mean()
            self.log_alpha_ent_opt.zero_grad()
            alpha_ent_loss.backward()
            self.log_alpha_ent_opt.step()
            metrics['alpha_ent_loss'].append(alpha_ent_loss)

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        lrnows = {'actr': _lr}

        return metrics, lrnows

    def grad_pen(self, fa, state, action_1, action_2):
        """Define the gradient penalty regularizer"""
        # Create the states to apply the contraint on
        eps_s = state.clone().detach().data.normal_(0, 10)
        zeta_state = state + eps_s
        zeta_state.requires_grad = True
        # Create the actions to apply the contraint on
        eps_a = torch.rand(action_1.size(0), 1).to(self.device)
        zeta_action = eps_a * action_1 + ((1. - eps_a) * action_2)
        zeta_action.requires_grad = True
        # Create the operation of interest
        score = fa(zeta_state, zeta_action)
        # Define the input(s) w.r.t. to take the gradient
        inputs = [zeta_state, zeta_action]
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
        grads_norm_s = list(grads)[0].norm(2, dim=-1)
        grads_norm_a = list(grads)[1].norm(2, dim=-1)
        grad_pen_s = (grads_norm_s - self.hps.ptso_grad_pen_targ_s).pow(2).mean()
        grad_pen_a = (grads_norm_a - self.hps.ptso_grad_pen_targ_a).pow(2).mean()
        return grad_pen_s, grad_pen_a, grads_norm_s, grads_norm_a

    def update_target_net(self, iters_so_far):
        """Update the target networks"""
        if sum([self.hps.use_c51, self.hps.use_qr]) == 0:
            # If non-distributional, targets slowly track their non-target counterparts
            for param, targ_param in zip(self.crit.parameters(), self.targ_crit.parameters()):
                targ_param.data.copy_(self.hps.polyak * param.data +
                                      (1. - self.hps.polyak) * targ_param.data)
            if self.hps.clipped_double:
                for param, targ_param in zip(self.twin.parameters(), self.targ_twin.parameters()):
                    targ_param.data.copy_(self.hps.polyak * param.data +
                                          (1. - self.hps.polyak) * targ_param.data)
        else:
            # If distributional, periodically set target weights with online's
            if iters_so_far % self.hps.targ_up_freq == 0:
                self.targ_crit.load_state_dict(self.crit.state_dict())

    def update_eval_nets(self):
        for param, eval_param in zip(self.actr.parameters(), self.main_eval_actr.parameters()):
            eval_param.data.copy_(param.data)
        for param, eval_param in zip(self.actr.parameters(), self.maxq_eval_actr.parameters()):
            eval_param.data.copy_(param.data)

    def save(self, path, iters_so_far):
        torch.save(self.actr.state_dict(), osp.join(path, f"actr_{iters_so_far}.pth"))
        torch.save(self.crit.state_dict(), osp.join(path, f"crit_{iters_so_far}.pth"))
        if self.hps.clipped_double:
            torch.save(self.twin.state_dict(), osp.join(path, f"twin_{iters_so_far}.pth"))

    def load(self, path, iters_so_far):
        self.actr.load_state_dict(torch.load(osp.join(path, f"actr_{iters_so_far}.pth")))
        self.crit.load_state_dict(torch.load(osp.join(path, f"crit_{iters_so_far}.pth")))
        if self.hps.clipped_double:
            self.twin.load_state_dict(torch.load(osp.join(path, f"twin_{iters_so_far}.pth")))
