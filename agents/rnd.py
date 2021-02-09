import math
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import logger
from helpers.console_util import log_module_info
from agents.nets import init
from helpers.distributed_util import RunMoms


STANDARDIZED_OB_CLAMPS = [-5., 5.]
HIDDEN_SIZE = 256


class PredNet(nn.Module):

    def __init__(self, env, hps, rms_obs):
        super(PredNet, self).__init__()
        self.hps = hps
        self.leak = 0.1
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, HIDDEN_SIZE)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_3', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

    def forward(self, obs, acs):
        if self.rms_obs is not None:
            obs = self.rms_obs.standardize(obs).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(torch.cat([obs, acs], dim=-1))
        return x


class TargNet(PredNet):

    def __init__(self, env, hps, rms_obs):
        super(TargNet, self).__init__(env, hps, rms_obs)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        self.leak = 0.1
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, HIDDEN_SIZE)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_3', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

        # Prevent the weights from ever being updated
        for param in self.fc_stack.parameters():
            param.requires_grad = False


class RandomNetworkDistillation(object):

    """Difference compared to original RND:
    we add the intrinsic reward to the main reward and learn a single advantage,
    as opposed to learning one advantage for each, since we do not play with different horizons.
    """

    def __init__(self, env, device, hps, rms_obs):
        self.env = env
        self.device = device
        self.hps = hps
        self.rms_obs = rms_obs

        # Create nets
        self.pred_net = PredNet(self.env, self.hps, self.rms_obs).to(self.device)
        self.targ_net = TargNet(self.env, self.hps, self.rms_obs).to(self.device)  # fixed, not trained

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=3e-4)

        # Define reward normalizer
        self.rms_int_rews = RunMoms(shape=(1,), use_mpi=False)

        log_module_info(logger, 'RND Pred Network', self.pred_net)
        log_module_info(logger, 'RND Targ Network', self.targ_net)

    def update(self, batch):
        """Update the RND predictor network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        # Transfer to device
        state = torch.Tensor(batch['obs0']).to(self.device)
        action = torch.Tensor(batch['acs']).to(self.device)

        # Compute loss
        _loss = F.mse_loss(
            self.pred_net(state, action),
            self.targ_net(state, action),
            reduction='none',
        )
        loss = _loss.mean()
        metrics['loss'].append(loss)

        # Update running moments for intrinsic rewards
        int_rews = F.mse_loss(
            self.pred_net(state, action),
            self.targ_net(state, action),
            reduction='none',
        ).mean(dim=-1, keepdim=True).detach()
        self.rms_int_rews.update(int_rews)

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics

    def get_int_rew(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.Tensor(action)
        # Compute intrinsic reward
        int_rew = F.mse_loss(
            self.pred_net(state, action),
            self.targ_net(state, action),
            reduction='none',
        ).mean(dim=-1, keepdim=True).detach()
        # Normalize intrinsic reward
        int_rew = self.rms_int_rews.divide_by_std(int_rew)
        return int_rew
