import math
from collections import OrderedDict

import numpy as np
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
        self.max_ac = max(np.abs(np.amax(env.action_space.high.astype('float32'))),
                          np.abs(np.amin(env.action_space.low.astype('float32'))))
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 300)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(300)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(300, 200)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(200)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(200, ac_dim)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def act(self, ob):
        return self.forward(ob)

    def forward(self, ob):
        ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = self.fc_stack(ob)
        ac = float(self.max_ac) * torch.tanh(self.head(x))
        return ac

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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BCQ-specific models.

class ActorPhi(nn.Module):

    def __init__(self, env, hps, phi=0.05):
        super(ActorPhi, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.max_ac = max(np.abs(np.amax(env.action_space.high.astype('float32'))),
                          np.abs(np.amin(env.action_space.low.astype('float32'))))
        self.hps = hps
        self.phi = phi
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
        self.head = nn.Linear(300, ac_dim)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def act(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = torch.cat([ob, ac], dim=-1)
        x = self.fc_stack(x)
        a = self.phi * float(self.max_ac) * torch.tanh(self.head(x))
        return ac + a

    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' not in n]

    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' in n]


class ActorVAE(nn.Module):

    def __init__(self, env, hps):
        super(ActorVAE, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.latent_dim = 2 * ac_dim
        self.max_ac = max(np.abs(np.amax(env.action_space.high.astype('float32'))),
                          np.abs(np.amin(env.action_space.low.astype('float32'))))
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.encoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 750)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(750)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(750, 750)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(750)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.mean_head = nn.Linear(750, self.latent_dim)
        self.log_std_head = nn.Linear(750, self.latent_dim)
        self.decoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + self.latent_dim, 750)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(750)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(750, 750)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(750)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.ac_head = nn.Linear(750, ac_dim)
        # Perform initialization
        self.encoder.apply(init(weight_scale=math.sqrt(2)))
        self.mean_head.apply(init(weight_scale=0.01))
        self.log_std_head.apply(init(weight_scale=0.01))
        self.decoder.apply(init(weight_scale=math.sqrt(2)))
        self.ac_head.apply(init(weight_scale=0.01))

    def decode(self, ob, z=None):
        """Used in BCQ"""
        if z is None:
            # When z is not provided as input (or is literally None), sample it (and clip it)
            # z effectively sampled from a truncated (0, 1) isotropic normal distribution
            size = (ob.size(0), self.latent_dim)
            z = torch.randn(*size).to(ob).clamp(-0.5, 0.5)
        # Pass through the decoder and output head
        x = torch.cat([ob, z], dim=-1)
        x = self.decoder(x)
        ac = self.ac_head(x)
        ac = float(self.max_ac) * torch.tanh(ac)
        return ac

    def decodex(self, ob, z=None, expansion=4):
        """Used in BEAR"""
        if z is None:
            # When z is not provided as input (or is literally None), sample it (and clip it)
            # z effectively sampled from a truncated (0, 1) isotropic normal distribution
            size = (ob.size(0), expansion, self.latent_dim)
            z = torch.randn(*size).to(ob).clamp(-0.5, 0.5)
        # Pass through the decoder and output head
        ob = ob.unsqueeze(0).repeat(expansion, 1, 1).permute(1, 0, 2)
        x = torch.cat([ob, z], dim=-1)
        x = self.decoder(x)
        ac = self.ac_head(x)
        return torch.tanh(ac), ac

    def forward(self, ob, ac):
        x = torch.cat([ob, ac], dim=-1)
        # Encode
        x = self.encoder(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(-4, 15)  # clipped for numerical stability (BCQ)
        std = log_std.exp()
        z = mean + (std * torch.randn_like(std))
        # Decode
        ac = self.decode(ob, z)
        return ac, mean, std


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> SAM/BEAR-specific models.

def arctanh(x):
    """Implementation of the arctanh function.
    Can be very numerically unstable, hence the clamping.
    """
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x / one_minus_x)
    # return 0.5 * (x.log1p() - (-x).log1p())


class NormalToolkit(object):

    @staticmethod
    def logp(x, mean, std):
        neglogp = (0.5 * ((x - mean) / std).pow(2).sum(dim=-1, keepdim=True) +
                   0.5 * math.log(2 * math.pi) +
                   std.log().sum(dim=-1, keepdim=True))
        return -neglogp

    @staticmethod
    def sample(mean, std):
        # Reparametrization trick
        eps = torch.empty(mean.size()).normal_().to(mean.device)
        eps.requires_grad = False
        return mean + std * eps

    @staticmethod
    def mode(mean):
        return mean


class TanhNormalToolkit(object):

    @staticmethod
    def logp(x, mean, std):
        # We need to assemble the logp of a sample which comes from a Gaussian sample
        # after being mapped through a tanh. This needs a change of variable.
        # See appendix C of the SAC paper for an explanation of this change of variable.
        logp = NormalToolkit.logp(arctanh(x), mean, std) - torch.log(1 - x.pow(2) + 1e-6)
        return logp.sum(-1, keepdim=True)

    @staticmethod
    def nonsquashed_sample(mean, std):
        sample = NormalToolkit.sample(mean, std)
        return sample

    @staticmethod
    def sample(mean, std):
        sample = NormalToolkit.sample(mean, std)
        return torch.tanh(sample)

    @staticmethod
    def mode(mean):
        return torch.tanh(mean)


class TanhGaussActor(nn.Module):

    def __init__(self, env, hps, hidden_size):
        super(TanhGaussActor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Define perception stack
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, hidden_size)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_size)),
                ('nl', nn.Tanh()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_size, hidden_size)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_size)),
                ('nl', nn.Tanh()),
            ]))),
        ]))
        self.head = nn.Linear(hidden_size, 2 * ac_dim)
        # self.head = nn.Linear(hidden_size, ac_dim)
        # self.ac_logstd_head = nn.Parameter(torch.full((ac_dim,), math.log(0.6)))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=5./3.))
        self.head.apply(init(weight_scale=0.01))

    def logp(self, ob, ac):
        out = self.forward(ob)
        return TanhNormalToolkit.logp(ac, *out[0:2])  # mean, std

    def entropy(self, ob):
        out = self.forward(ob)
        return TanhNormalToolkit.entropy(out[1])  # std

    def act(self, ob):
        # Special for BEAR
        out = self.forward(ob)
        ac = TanhNormalToolkit.sample(*out[0:2])  # mean, std
        nonsquashed_ac = TanhNormalToolkit.nonsquashed_sample(*out[0:2])  # mean, std
        return ac, nonsquashed_ac

    def sample(self, ob, sg=True):
        if sg:
            with torch.no_grad():
                out = self.forward(ob)
                ac = TanhNormalToolkit.sample(*out[0:2])  # mean, std
        else:
            out = self.forward(ob)
            ac = TanhNormalToolkit.sample(*out[0:2])  # mean, std
        return ac

    def mode(self, ob, sg=True):
        if sg:
            with torch.no_grad():
                out = self.forward(ob)
                ac = TanhNormalToolkit.mode(out[0])  # mean
        else:
            out = self.forward(ob)
            ac = TanhNormalToolkit.mode(out[0])  # mean
        return ac

    def kl(self, ob, other):
        assert isinstance(other, TanhGaussActor)
        with torch.no_grad():
            out_a = self.forward(ob)
            out_b = other.forward(ob)
            kl = TanhNormalToolkit.kl(*out_a[0:2],
                                      *out_b[0:2])  # mean, std
        return kl

    def forward(self, ob):
        ob = torch.clamp(self.rms_obs.standardize(ob), -5., 5.)
        x = self.fc_stack(ob)

        ac_mean, ac_log_std = self.head(x).chunk(2, dim=-1)
        ac_mean = ac_mean.clamp(-9.0, 9.0)
        ac_std = ac_log_std.clamp(-5.0, 2.0).exp()

        # ac_mean = self.head(x).clamp(-9.0, 9.0)
        # ac_std = self.ac_logstd_head.expand_as(ac_mean).clamp(-5.0, 2.0).exp()

        # Note, clipping values were taken from the SAC/BEAR codebases
        return ac_mean, ac_std


class Critic2(nn.Module):

    def __init__(self, env, hps, hidden_size):
        super(Critic2, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Define observation whitening
        self.rms_obs = RunMoms(shape=env.observation_space.shape, use_mpi=True)
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, hidden_size)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_size)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_size, hidden_size)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_size)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hidden_size, 1)
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
        return x

    @property
    def out_params(self):
        return [p for p in self.head.parameters()]
