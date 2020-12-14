from collections import defaultdict
import os.path as osp
from copy import copy

import numpy as np
import torch
import torch.nn.utils as U
import torch.nn.functional as F
from torch import autograd

import logger
from helpers.console_util import log_env_info, log_module_info
from helpers.distributed_util import average_gradients, sync_with_root, RunMoms
from helpers.math_util import huber_quant_reg_loss
from helpers.math_util import LRScheduler
from agents.memory import ReplayBuffer, PrioritizedReplayBuffer, UnrealReplayBuffer
from agents.nets import perception_stack_parser, TanhGaussActor, MixtureTanhGaussActor, Critic, RewardAverager
from agents.rnd import RandomNetworkDistillation


ALPHA_PRI_CLAMPS = [0., 1_000_000.]
CQL_TEMP = 1.0
EPS_CE = 1e-6
U_ESTIM_SAMPLES = 4
CWPQ_TEMP = 10.0
CRR_TEMP = 1.0
ADV_ESTIM_SAMPLES = 4
RND_INNER_SCALE = 100.0


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

        # Override with environment-specific hyper-parameter values, in line with CQL's codebase
        self.hps.cql_targ_lower_bound = 5.0 if 'antmaze' in self.hps.env_id else 1.0

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
        Actr_ = MixtureTanhGaussActor if self.hps.gauss_mixture else TanhGaussActor

        self.actr = Actr_(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        sync_with_root(self.actr)
        self.main_eval_actr = Actr_(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.maxq_eval_actr = Actr_(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.cwpq_eval_actr = Actr_(self.env, self.hps, self.rms_obs, hidden_dims[0]).to(self.device)
        self.main_eval_actr.load_state_dict(self.actr.state_dict())
        self.maxq_eval_actr.load_state_dict(self.actr.state_dict())
        self.cwpq_eval_actr.load_state_dict(self.actr.state_dict())

        self.crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=True).to(self.device)
        self.targ_crit = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=True).to(self.device)
        self.targ_crit.load_state_dict(self.crit.state_dict())
        if self.hps.clipped_double:
            # Create second ('twin') critic and target critic
            # TD3, https://arxiv.org/abs/1802.09477
            self.twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=False).to(self.device)
            self.targ_twin = Critic(self.env, self.hps, self.rms_obs, hidden_dims[1], ube=False).to(self.device)
            self.targ_twin.load_state_dict(self.twin.state_dict())
        # First create another set of hyper-parameters in which the critic is non-distributional
        # and then create a critic to be trained via Monte Carlo estimation
        _hps = copy(self.hps)
        _hps.use_c51 = False
        _hps.use_qr = False
        self.mc_crit = Critic(self.env, _hps, self.rms_obs, hidden_dims[1], ube=False).to(self.device)

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
        self.crit_opt = torch.optim.Adam(self.crit.q_trainable_params,
                                         lr=self.hps.critic_lr,
                                         weight_decay=self.hps.wd_scale)
        if self.hps.clipped_double:
            self.twin_opt = torch.optim.Adam(self.twin.q_trainable_params,
                                             lr=self.hps.critic_lr,
                                             weight_decay=self.hps.wd_scale)

        self.u_opt = torch.optim.Adam(self.crit.u_trainable_params, lr=1e-3)

        self.mc_crit_opt = torch.optim.Adam(self.mc_crit.q_trainable_params, lr=3e-4)

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

        if self.hps.ptso_use_rnd_monitoring or self.hps.base_pe_loss in ['htg_1', 'htg_2']:
            # Create RND networks
            self.rnd = RandomNetworkDistillation(self.env, self.device, self.hps, self.rms_obs)

        if self.hps.ptso_use_reward_averager:
            self.reward_averager = RewardAverager(self.env, self.hps, self.rms_obs, hidden_dims[1]).to(self.device)
            self.ra_opt = torch.optim.Adam(self.reward_averager.parameters(), lr=self.hps.ptso_ra_lr)

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
        # Update the observation normalizer
        self.rms_obs.update(transition['obs0'])

    def sample_batch(self):
        """Sample a batch of transitions from the replay buffer"""
        # Create patcher if needed
        _patcher = None
        if self.hps.ptso_use_reward_averager:
            logger.info("we'bout2patch")

            def _patcher(x, y, z):
                x = torch.Tensor(x).to(self.device)
                y = torch.Tensor(y).to(self.device)
                z = torch.Tensor(z).to(self.device)
                return self.reward_averager(x, y, z).detach().cpu().numpy()  # redundant detach

        # Get a batch of transitions from the replay buffer
        if self.hps.n_step_returns:
            batch = self.replay_buffer.lookahead_sample(
                self.hps.batch_size,
                self.hps.lookahead,
                self.hps.gamma,
                patcher=_patcher,
            )
        else:
            batch = self.replay_buffer.sample(
                self.hps.batch_size,
                patcher=_patcher,
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

                u_value = self.crit.wrap_with_u_head(self.crit.phi(ob, ac)).mean(dim=1).clamp(min=1e-6).sqrt()

                if self.hps.ptso_use_u_inference_time:
                    q_value -= self.hps.ptso_u_scale_p_i * u_value

                if which == 'maxq':
                    q_mean = q_value.mean(dim=0)  # scalar
                    mc_q_mean = self.mc_crit.QZ(ob, ac).mean()  # scalar
                    u_mean = u_value.mean(dim=0)  # scalar
                    index = q_value.argmax(0)
                else:  # which == 'cwpq'
                    weight = torch.exp(q_value / CWPQ_TEMP).clamp(min=0.01, max=1_000_000_000.)
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

        if which == 'maxq':
            return {'ac': ac,
                    'q_mean': q_mean.cpu().detach().numpy().flatten(),
                    'mc_q_mean': mc_q_mean.cpu().detach().numpy().flatten(),
                    'u_mean': u_mean.cpu().detach().numpy().flatten()}
        else:
            return ac

    def ac_factory(self, actr, ob, inflate):
        _ob = ob.unsqueeze(1).repeat(1, inflate, 1).view(ob.shape[0] * inflate, ob.shape[1])
        _ac = float(self.max_ac) * actr.sample(_ob, sg=False)
        _logp = actr.logp(_ob, _ac)
        return _ac, _logp.view(ob.shape[0], inflate, 1)

    def q_factory(self, crit, ob, ac, mc=False, compute_rnd=False):
        ob_dim = ob.shape[0]
        ac_dim = ac.shape[0]
        num_repeat = int(ac_dim / ob_dim)
        _ob = ob.unsqueeze(1).repeat(1, num_repeat, 1).view(ob.shape[0] * num_repeat, ob.shape[1])
        q_value = crit.QZ(_ob, ac)
        if compute_rnd:
            _rnd_score = self.rnd.get_int_rew(_ob, ac).view(ob.shape[0], num_repeat, 1)
            rnd_score = 1.0 - torch.exp(-_rnd_score * RND_INNER_SCALE)
        if not mc:
            if self.hps.use_c51:
                q_value = q_value.matmul(self.c51_supp).unsqueeze(-1)
            elif self.hps.use_qr:
                q_value = q_value.mean(dim=1, keepdim=True)
        q_value = q_value.view(ob.shape[0], num_repeat, 1)
        if compute_rnd:
            return q_value, rnd_score
        else:
            return q_value

    def u_factory(self, crit, ob, ac):
        ob_dim = ob.shape[0]
        ac_dim = ac.shape[0]
        num_repeat = int(ac_dim / ob_dim)
        _ob = ob.unsqueeze(1).repeat(1, num_repeat, 1).view(ob.shape[0] * num_repeat, ob.shape[1])
        u_value = crit.wrap_with_u_head(self.crit.phi(_ob, ac)).view(ob.shape[0], num_repeat, 1)
        return u_value

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
            next_state_td1 = torch.Tensor(batch['obs1_td1']).to(self.device)
        else:
            td_len = torch.ones_like(done).to(self.device)

        if self.hps.ptso_use_rnd_monitoring or self.hps.base_pe_loss in ['htg_1', 'htg_2']:
            # Update the RND network
            self.rnd.update(batch)
            logger.info("just updated the rnd estimate")
            # Monitor associated rnd estimate
            metrics['rnd_score'].append(1.0 - torch.exp(-self.rnd.get_int_rew(state, action)))

        if self.hps.ptso_use_reward_averager:
            # Update the reward averager
            ra_loss = F.mse_loss(self.reward_averager(state, action, next_state), reward)
            ra_grad_pen_s, ra_grad_pen_a, ra_grad_pen_ns, _, _, _ = self.grad_pen(
                fa=self.reward_averager,
                state=state,
                action=action,
                next_state=next_state,
                targ_gn_s=self.hps.ptso_ra_grad_pen_targ_s,
                targ_gn_a=self.hps.ptso_ra_grad_pen_targ_a,
            )
            ra_loss += self.hps.ptso_ra_grad_pen_scale_s * (0.5 * (ra_grad_pen_s + ra_grad_pen_ns))
            ra_loss += self.hps.ptso_ra_grad_pen_scale_a * ra_grad_pen_a
            self.ra_opt.zero_grad()
            ra_loss.backward()
            self.ra_opt.step()

        # Checking the validity of the policy evaluation technique
        assert self.hps.base_pe_loss in ['pure_td', 'cql_1', 'cql_2', 'htg_1', 'htg_2'], "invalid PE technique."

        action_from_actr = float(self.max_ac) * self.actr.sample(state, sg=False)
        log_prob = self.actr.logp(state, action_from_actr)

        if self.hps.ptso_use_sarsa:
            logger.info("using SARSA")
            next_action_behave = torch.Tensor(batch['acs1']).to(self.device)
            next_action_policy = float(self.max_ac) * self.actr.sample(next_state, sg=True)
            with torch.no_grad():
                _u = self.crit.wrap_with_u_head(self.crit.phi(next_state, next_action_policy))
                # Create an 'advantage-like' estimator for the uncertainty, the unexpected uncertainty
                u_u_ac, _ = self.ac_factory(self.actr, next_state, U_ESTIM_SAMPLES)
                u_u_from_actr = self.u_factory(self.crit, next_state, u_u_ac)
                _u = 1. * (_u - u_u_from_actr.max(dim=1).values).gt(0.)
                logger.info(f"number of 1's in the unexpected uncertainty vector: {_u.sum()}/{_u.shape[0]}")
            next_action = (_u * next_action_behave) + ((1 - _u) * next_action_policy)
        else:
            logger.info("NOT using SARSA")
            next_action = float(self.max_ac) * self.actr.sample(next_state, sg=True)
            # Note, here, always stochastic selection of the target action

        if self.hps.use_c51:

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

        # When using distributional critics, define q from z
        if self.hps.use_c51:
            q = z.squeeze(-1).matmul(self.c51_supp).unsqueeze(-1)
        elif self.hps.use_qr:
            q = z.squeeze(-1).mean(dim=1, keepdim=True)

        # Assemble the 3 pieces of the CQL loss
        # (cf. slide 16 in: https://docs.google.com/presentation/d/
        # 1F-dNg2LT75z9vJiPqASHayiZ3ewB6HE0KPgnZnY2rTg/edit#slide=id.g80c29cc4d2_0_101)

        min_crit_loss = 0.
        if self.hps.clipped_double:
            min_twin_loss = 0.

        if self.hps.base_pe_loss in ['cql_1', 'cql_2', 'htg_1', 'htg_2']:
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
            if self.hps.base_pe_loss in ['htg_1', 'htg_2']:
                cql_q, cql_ac_rnd = self.q_factory(self.crit, state, cql_ac, compute_rnd=True)
                cql_next_q, cql_next_ac_rnd = self.q_factory(self.crit, state, cql_next_ac, compute_rnd=True)
                cql_rand_q, cql_rand_ac_rnd = self.q_factory(self.crit, state, cql_rand_ac, compute_rnd=True)
            else:
                cql_q, cql_ac_rnd = self.q_factory(self.crit, state, cql_ac), 1.0
                cql_next_q, cql_next_ac_rnd = self.q_factory(self.crit, state, cql_next_ac), 1.0
                cql_rand_q, cql_rand_ac_rnd = self.q_factory(self.crit, state, cql_rand_ac), 1.0
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
                (cql_rand_q - weird_stuff) * cql_rand_ac_rnd,
                (cql_next_q - cql_next_logp.detach()) * cql_next_ac_rnd,
                (cql_q - cql_logp.detach()) * cql_ac_rnd,
            ], dim=1)
            if self.hps.clipped_double:
                cql_cat_twin_q = torch.cat([
                    (cql_rand_twin_q - weird_stuff) * cql_rand_ac_rnd,
                    (cql_next_twin_q - cql_next_logp.detach()) * cql_next_ac_rnd,
                    (cql_twin_q - cql_logp.detach()) * cql_ac_rnd,
                ], dim=1)

            min_crit_loss += (torch.logsumexp(cql_cat_q / CQL_TEMP, dim=1).mean() *
                              self.hps.cql_min_q_weight * CQL_TEMP)
            if self.hps.clipped_double:
                min_twin_loss += (torch.logsumexp(cql_cat_twin_q / CQL_TEMP, dim=1).mean() *
                                  self.hps.cql_min_q_weight * CQL_TEMP)

        if self.hps.base_pe_loss in ['cql_2', 'htg_2']:
            # Piece #2: maximize the Q-function on points in the offline dataset
            min_crit_loss -= (q.mean() * self.hps.cql_min_q_weight)
            if self.hps.clipped_double:
                min_twin_loss -= (twin_q.mean() * self.hps.cql_min_q_weight)

        if self.hps.base_pe_loss in ['cql_1', 'cql_2', 'htg_1', 'htg_2']:
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

        logger.info(f"alpha_pri: {self.alpha_pri}")  # leave this here, for sanity checks
        metrics['alpha_pri'].append(self.alpha_pri)

        # Piece #3: Add the new losses to the vanilla ones, i.e. the traditional TD errors to minimize
        crit_loss += min_crit_loss
        if self.hps.clipped_double:
            twin_loss += min_twin_loss

        if self.hps.ptso_use_or_monitor_grad_pen:
            # Add (or just monitor) gradient penalties to regularize the phi embedding

            # Note, we usually dump the metrics even if it's not an eval round (legibility),
            # but this bit it too computationally expensive
            if self.hps.ptso_phi_grad_pen_scale_s > 0 or self.hps.ptso_phi_grad_pen_scale_a > 0:
                # Use one or two gradient penalties in training
                crit_grad_pen_s, crit_grad_pen_a, crit_grads_norm_s, crit_grads_norm_a = self.grad_pen(
                    fa=self.crit.phi,
                    state=state,
                    action=action,
                    targ_gn_s=self.hps.ptso_phi_grad_pen_targ_s,
                    targ_gn_a=self.hps.ptso_phi_grad_pen_targ_a,
                )
                metrics['phi_gradnorm_s'].append(crit_grads_norm_s.mean())
                metrics['phi_gradnorm_a'].append(crit_grads_norm_a.mean())
                if self.hps.ptso_phi_grad_pen_scale_s > 0:
                    crit_loss += self.hps.ptso_phi_grad_pen_scale_s * crit_grad_pen_s.mean()
                if self.hps.ptso_phi_grad_pen_scale_a > 0:
                    crit_loss += self.hps.ptso_phi_grad_pen_scale_a * crit_grad_pen_a.mean()

                if self.hps.clipped_double:
                    if self.hps.ptso_phi_grad_pen_scale_s > 0 or self.hps.ptso_phi_grad_pen_scale_a > 0:
                        twin_grad_pen_s, twin_grad_pen_a, twin_grads_norm_s, twin_grads_norm_a = self.grad_pen(
                            fa=self.twin.phi,
                            state=state,
                            action=action,
                            targ_gn_s=self.hps.ptso_phi_grad_pen_targ_s,
                            targ_gn_a=self.hps.ptso_phi_grad_pen_targ_a,
                        )
                    if self.hps.ptso_phi_grad_pen_scale_s > 0:
                        twin_loss += self.hps.ptso_phi_grad_pen_scale_s * twin_grad_pen_s.mean()
                    if self.hps.ptso_phi_grad_pen_scale_a > 0:
                        twin_loss += self.hps.ptso_phi_grad_pen_scale_a * twin_grad_pen_a.mean()
            else:
                if iters_so_far % self.hps.eval_frequency == 0:
                    # Just monitor the gradient norms' values
                    _, _, crit_grads_norm_s, crit_grads_norm_a = self.grad_pen(
                        fa=self.crit.phi,
                        state=state,
                        action=action,
                        targ_gn_s=self.hps.ptso_phi_grad_pen_targ_s,
                        targ_gn_a=self.hps.ptso_phi_grad_pen_targ_a,
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

        # Piece #4 (PTSO): uncertainty via Uncertainty Bellman Equation, with its own optimizer
        # The target of the Uncertainty Bellman Equation is the variance estimate of the error epsilon (paper)

        # We don't use the twin critic past this point, to same on evalutations and matrix multiplications

        # Depending on a hyper-parameter, use the target critics in `targ_u` calculation, or not at all here
        crit_in_ube_targ = self.targ_crit if self.hps.ptso_use_targ_for_u else self.crit

        # Create the UBE target
        # Computations done with batch dimension
        with torch.no_grad():
            # Assemble the uncertainty bellman target
            _new_phi = self.crit.phi(state, action).unsqueeze(-1)
            _new_phi_t = torch.transpose(_new_phi, -1, -2)
            inflated_precision = self.precision.unsqueeze(0).repeat(self.hps.batch_size, 1, 1)
            v_q = torch.bmm(torch.bmm(_new_phi_t,
                                      inflated_precision),
                            _new_phi).squeeze(-1)
            if self.hps.n_step_returns:
                _next_state = next_state_td1
                _next_action = float(self.max_ac) * self.actr.sample(_next_state, sg=True)
            else:
                _next_state = next_state
                _next_action = float(self.max_ac) * self.actr.sample(next_state, sg=True)  # re-compute just in case
            u_prime = crit_in_ube_targ.wrap_with_u_head(crit_in_ube_targ.phi(_next_state, _next_action))
            targ_u = (v_q + ((self.hps.gamma ** 2) * (1. - done) * u_prime)).clamp(min=0.)
            # Note, we clamp for the target to be non-negative, to urge the prediction to always be non-negative
        u_pred = self.crit.wrap_with_u_head(self.crit.phi(state, action).detach())  # can only update the output head
        u_loss = F.mse_loss(u_pred, targ_u.detach())  # detach, just in case
        metrics['v_fb_mean'].append(v_q.mean())
        metrics['u_fb_mean'].append(u_pred.mean())
        metrics['u_loss'].append(u_loss)

        self.u_opt.zero_grad()
        u_loss.backward(retain_graph=True)
        self.u_opt.step()

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
                v_uniform = torch.bmm(torch.bmm(_new_phi_t_uniform,
                                                inflated_precision),
                                      _new_phi_uniform).squeeze(-1)
                # Keep in the 'no_grad' context, this is just for monitoring
                u_uniform = self.crit.wrap_with_u_head(self.crit.phi(state, action_uniform).detach()).mean()
                metrics['v_fu_mean'].append(v_uniform.mean())
                metrics['u_fu_mean'].append(u_uniform.mean())

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

        # Update the Monte-Carlo critic
        rets = torch.Tensor(batch['rets']).to(self.device)
        mc_crit_loss = F.mse_loss(self.mc_crit.QZ(state, action), rets)

        self.mc_crit_opt.zero_grad()
        mc_crit_loss.backward()
        self.mc_crit_opt.step()

        # Log the Monte-Carlo Q estimate at the state and action from the minibatch
        mc_q_from_behav = self.mc_crit.QZ(state, action)
        metrics['mcq_fb_mean'].append(mc_q_from_behav.mean())
        metrics['q_fb_mean'].append(q.mean())
        metrics['q_fb_std'].append(q.std())
        metrics['q_fb_min'].append(q.min())
        metrics['q_fb_max'].append(q.max())

        # Log the gap as defined in eq 10 of the CQL paper to monitor the gap-expanding property of CQL
        rand_ac = torch.Tensor(
            self.hps.batch_size * self.hps.cql_state_inflate, self.ac_dim
        ).uniform_(-self.max_ac, self.max_ac).to(self.device)
        rand_q = self.q_factory(self.crit, state, rand_ac)
        gap = (rand_q.max(dim=1).values - q)
        metrics['gap'].append(gap.mean())

        # Only update the policy after a certain number of iteration (CQL codebase: 20000)
        # Note, as opposed to BEAR and BRAC, after the warm start, the BC loss is not used anymore

        # Actor loss
        if iters_so_far >= self.hps.warm_start:
            # Use offline RL loss

            # Log the Monte-Carlo Q estimate at the state from the minibatch and predicted action
            mc_q_from_actr = self.mc_crit.QZ(state, action_from_actr)
            metrics['mcq_fa_mean'].append(mc_q_from_actr.mean())

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
            metrics['q_fa_mean'].append(q_from_actr.mean())
            metrics['q_fa_std'].append(q_from_actr.std())
            metrics['q_fa_min'].append(q_from_actr.min())
            metrics['q_fa_max'].append(q_from_actr.max())
            # Same goes for the uncertainty
            phi_from_actr = self.crit.phi(state, action_from_actr)
            u_from_actr = self.crit.wrap_with_u_head(phi_from_actr).clamp(min=1e-8).sqrt()
            # Note, we clamp for the value to be non-negative, for the square-root to always be defined
            # We clip with an epsilon strictly greater than zero because of sqrt's derivative at zero
            metrics['u_fa_mean'].append(u_from_actr.mean())
            metrics['u_fa_std'].append(u_from_actr.std())
            metrics['u_fa_min'].append(u_from_actr.min())
            metrics['u_fa_max'].append(u_from_actr.max())
            # # Sanity-check
            # assert not u_from_actr.isnan().any()

            # Assemble base loss
            if self.hps.base_pi_loss in ['sac', 'cql']:
                actr_loss = (self.alpha_ent * log_prob) - q_from_actr
            elif self.hps.base_pi_loss in ['crr_exp', 'crr_binary', 'crr_binary_max',]:
                crr_q = self.crit.QZ(state, action)
                if self.hps.use_c51:
                    crr_q = crr_q.matmul(self.c51_supp).unsqueeze(-1)
                elif self.hps.use_qr:
                    crr_q = crr_q.mean(dim=1, keepdim=True)
                emp_adv_ac, _ = self.ac_factory(self.actr, state, ADV_ESTIM_SAMPLES)
                if 'max' in self.hps.base_pi_loss:
                    emp_adv_from_actr = self.q_factory(self.crit, state, emp_adv_ac).max(dim=1).values
                else:
                    emp_adv_from_actr = self.q_factory(self.crit, state, emp_adv_ac).mean(dim=1)
                crr_adv = crr_q - emp_adv_from_actr
                if 'binary' in self.hps.base_pi_loss:
                    crr_adv = 1. * crr_adv.gt(0.)  # trick to easily cast to float
                elif self.hps.base_pi_loss == 'crr_exp':
                    crr_adv = torch.exp(crr_adv / CRR_TEMP).clamp(max=20.)
                actr_loss = -self.actr.logp(state, action) * crr_adv.detach()
            elif self.hps.base_pi_loss == 'qprop_exp':
                qprop_q = self.mc_crit.QZ(state, action)
                emp_adv_ac, _ = self.ac_factory(self.actr, state, ADV_ESTIM_SAMPLES)
                emp_adv_from_actr = self.q_factory(self.mc_crit, state, emp_adv_ac, mc=True).mean(dim=1)
                qprop_adv = qprop_q - emp_adv_from_actr
                # Compute the gradients involved in the control variate
                _grads = autograd.grad(
                    outputs=q_from_actr,
                    inputs=[action_from_actr],
                    only_inputs=True,
                    grad_outputs=[torch.ones_like(q_from_actr)],
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False,
                )
                grads = list(_grads)[0]
                # Compute the expected action from the policy, re-using the ones previously predicted
                expected_action = emp_adv_ac.view(-1, ADV_ESTIM_SAMPLES, self.ac_dim).mean(dim=1)
                # Assemble the control variate
                control_variate = (grads * (action - expected_action)).sum(dim=1, keepdim=True)
                # Compute covariance between advantage estimate (MC in Q-prop) and control variate
                covariance = qprop_adv * control_variate
                # Compute eta
                if self.hps.ptso_qprop_aggressive_eta:
                    eta = 1. * torch.sign(covariance).detach()
                else:
                    eta = 1. * covariance.gt(0.).detach()
                # Augment the log-likelihood weight with the weighted control variate
                qprop_adv -= eta * control_variate
                # Assemble the loss
                actr_loss = -self.actr.logp(state, action) * qprop_adv.detach().exp().clamp(max=20.)
                # Note, a major difference with the original qprop update is the use of the exponential
                # which does not align with their derivations but is still intuitively consistent
                actr_loss -= eta * q_from_actr
            elif self.hps.base_pi_loss == 'awr_exp':
                awr_q = self.mc_crit.QZ(state, action)
                emp_adv_ac, _ = self.ac_factory(self.actr, state, ADV_ESTIM_SAMPLES)
                emp_adv_from_actr = self.q_factory(self.mc_crit, state, emp_adv_ac, mc=True).mean(dim=1)
                awr_adv = awr_q - emp_adv_from_actr.exp().clamp(max=20.)
                actr_loss = -self.actr.logp(state, action) * awr_adv.detach()  # detach just in case
            else:
                raise NotImplementedError("invalid base loss for policy improvement.")

            # If opted in, add the uncertainty contribution in the policy improvement loss
            if self.hps.ptso_u_scale_p_i > 0:

                # Create an 'advantage-like' estimator for the uncertainty, the unexpected uncertainty
                u_u_ac, _ = self.ac_factory(self.actr, state, U_ESTIM_SAMPLES)
                u_u_from_actr = self.u_factory(self.crit, state, u_u_ac).detach()

                actr_loss += (self.hps.ptso_u_scale_p_i *
                              self.actr.logp(state, action_from_actr.detach()) *
                              (u_from_actr - u_u_from_actr.max(dim=1).values).gt(0.).detach())

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

        logger.info(f"alpha_ent: {self.alpha_ent}")  # leave this here, for sanity checks

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        lrnows = {'actr': _lr}

        return metrics, lrnows

    def grad_pen(self, fa, state, action, targ_gn_s, targ_gn_a, std=10.0, next_state=None):
        """Define the gradient penalty regularizer"""
        # Create the states to apply the contraint on
        eps_s = state.clone().detach().data.normal_(0, std)
        zeta_state = state + eps_s
        zeta_state.requires_grad = True
        if next_state is not None:
            eps_ns = next_state.clone().detach().data.normal_(0, std)
            zeta_next_state = next_state + eps_ns
            zeta_next_state.requires_grad = True
        # Create the actions to apply the contraint on
        eps_a = action.clone().detach().data.normal_(0, std)
        zeta_action = action + eps_a
        zeta_action.requires_grad = True
        # Define the input(s) w.r.t. to take the gradient
        inputs = [zeta_state, zeta_action]
        if next_state is not None:
            inputs.append(zeta_next_state)
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
        grads_norm_s = list(grads)[0].norm(2, dim=-1)
        grads_norm_a = list(grads)[1].norm(2, dim=-1)
        grad_pen_s = (grads_norm_s - targ_gn_s).pow(2).mean()
        grad_pen_a = (grads_norm_a - targ_gn_a).pow(2).mean()
        if next_state is None:
            return grad_pen_s, grad_pen_a, grads_norm_s, grads_norm_a
        else:
            grads_norm_ns = list(grads)[2].norm(2, dim=-1)
            grad_pen_ns = (grads_norm_ns - targ_gn_s).pow(2).mean()  # same target as with state
            return grad_pen_s, grad_pen_a, grad_pen_ns, grads_norm_s, grads_norm_a, grads_norm_ns

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
        for param, eval_param in zip(self.actr.parameters(), self.cwpq_eval_actr.parameters()):
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
