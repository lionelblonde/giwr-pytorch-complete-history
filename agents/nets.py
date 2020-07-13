import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.distributed_util import RunMoms


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def init(weight_scale=1., constant_bias=0.):
    """Perform orthogonal initialization"""

    def _init(m):

        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=weight_scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, nn.BatchNorm2d) or
              isinstance(m, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Models.

class Actor(nn.Module):

    def __init__(self, env, hps):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 300)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(300)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.a_fc_stack = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(300, 200)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(200)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.a_head = nn.Linear(200, ac_dim)
        if self.hps.kye_p:
            self.r_fc_stack = nn.Sequential(OrderedDict([
                ('fc_block_1', nn.Sequential(OrderedDict([
                    ('fc', nn.Linear(300, 200)),
                    ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(200)),
                    ('nl', nn.ReLU()),
                ]))),
            ]))
            self.r_head = nn.Linear(200, 1)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.a_fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.a_head.apply(init(weight_scale=0.01))
        if self.hps.kye_p:
            self.r_fc_stack.apply(init(weight_scale=math.sqrt(2)))
            self.r_head.apply(init(weight_scale=0.01))

    def act(self, ob):
        out = self.forward(ob)
        return out[0]  # ac

    def auxo(self, ob):
        if self.hps.kye_p:
            out = self.forward(ob)
            return out[1]  # aux
        else:
            raise ValueError("should not be called")

    def forward(self, ob):
        ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = self.fc_stack(ob)
        ac = float(self.ac_max) * torch.tanh(self.a_head(self.a_fc_stack(x)))
        out = [ac]
        if self.hps.kye_p:
            aux = self.r_head(self.r_fc_stack(x))
            out.append(aux)
        return out

    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' not in n]

    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' in n]


class Critic(nn.Module):

    def __init__(self, env, hps):
        super(Critic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.use_c51:
            num_heads = hps.c51_num_atoms
        elif hps.use_qr:
            num_heads = hps.num_tau
        else:
            num_heads = 1
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 400)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(400)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(400, 300)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(300)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(300, num_heads)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def QZ(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = torch.cat([ob, ac], dim=-1)
        x = self.fc_stack(x)
        x = self.head(x)
        if self.hps.use_c51:
            # Return a categorical distribution
            x = F.log_softmax(x, dim=1).exp()
        return x

    @property
    def out_params(self):
        return [p for p in self.head.parameters()]
