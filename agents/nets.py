import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STANDARDIZED_OB_CLAMPS = [-5., 5.]
BCQ_Z_CLAMPS = [-0.5, 0.5]
BCQ_LOG_STD_CLAMPS = [-4., 15.]
SAC_MEAN_CLAMPS = [-9., 9.]
SAC_LOG_STD_CLAMPS = [-5., 2.]
GAUSS_MIXTURE_COMPS = 5  # like in CRR


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


def perception_stack_parser(stack_string):
    """Parse the perception stack description given in string format.
    Note, only works with shallow networks of depth of 2 layers.

    Example of stack string: '300 200, 400 300, 750 750'
    """
    out = []
    for piece in stack_string.split(','):
        pieces = piece.strip().split(' ')
        assert len(pieces) == 2, "must be of length 2"
        pieces = [int(p) for p in pieces]
        out.append(pieces)  # careful, not extend
    return out


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Models.

class Actor(nn.Module):

    def __init__(self, env, hps, rms_obs, hidden_dims):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.max_ac = max(np.abs(np.amax(env.action_space.high.astype('float32'))),
                          np.abs(np.amin(env.action_space.low.astype('float32'))))
        self.hps = hps
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, hidden_dims[0])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[0])),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dims[0], hidden_dims[1])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[1])),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hidden_dims[1], ac_dim)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def act(self, ob):
        return self.forward(ob)

    def forward(self, ob):
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
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

    def __init__(self, env, hps, rms_obs, hidden_dims):
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
        self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, hidden_dims[0])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[0])),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dims[0], hidden_dims[1])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[1])),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hidden_dims[1], num_heads)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def QZ(self, ob, ac):
        x = self.forward(ob, ac)
        x = self.head(x)
        if self.hps.use_c51:
            # Return a categorical distribution
            x = F.log_softmax(x, dim=1).exp()
        return x

    def forward(self, ob, ac):
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = torch.cat([ob, ac], dim=-1)
        x = self.fc_stack(x)
        return x

    @property
    def out_params(self):
        return [p for p in self.head.parameters()]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BCQ-specific models.

class ActorPhi(nn.Module):

    def __init__(self, env, hps, rms_obs, hidden_dims):
        super(ActorPhi, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.max_ac = max(np.abs(np.amax(env.action_space.high.astype('float32'))),
                          np.abs(np.amin(env.action_space.low.astype('float32'))))
        self.hps = hps
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, hidden_dims[0])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[0])),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dims[0], hidden_dims[1])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[1])),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hidden_dims[1], ac_dim)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def act(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = torch.cat([ob, ac], dim=-1)
        x = self.fc_stack(x)
        a = self.hps.bcq_phi * float(self.max_ac) * torch.tanh(self.head(x))
        return ac + a


class ActorVAE(nn.Module):

    def __init__(self, env, hps, rms_obs, hidden_dims):
        super(ActorVAE, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.latent_dim = 2 * ac_dim
        self.max_ac = max(np.abs(np.amax(env.action_space.high.astype('float32'))),
                          np.abs(np.amin(env.action_space.low.astype('float32'))))
        self.hps = hps
        # Exceptionally here, we force the hidden dims to be identical
        assert hidden_dims[0] == hidden_dims[1], "must be identical"
        hidden_dim = hidden_dims[0]  # arbitrarily chose index 0
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        self.encoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, hidden_dim)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dim)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dim, hidden_dim)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dim)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.mean_head = nn.Linear(hidden_dim, self.latent_dim)
        self.log_std_head = nn.Linear(hidden_dim, self.latent_dim)
        self.decoder = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + self.latent_dim, hidden_dim)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dim)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dim, hidden_dim)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dim)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.ac_head = nn.Linear(hidden_dim, ac_dim)
        # Perform initialization
        self.encoder.apply(init(weight_scale=math.sqrt(2)))
        self.mean_head.apply(init(weight_scale=0.01))
        self.log_std_head.apply(init(weight_scale=0.01))
        self.decoder.apply(init(weight_scale=math.sqrt(2)))
        self.ac_head.apply(init(weight_scale=0.01))

        # Create generator object which manages the random state of the actor
        # All the sampling is done by giving the generator as input
        self.gen = torch.Generator(device=hps.device).manual_seed(self.hps.seed)
        # Note, the device needs to be specified here, not carried over by the model

    def decode(self, ob, z=None):
        """Used in BCQ"""
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        if z is None:
            # When z is not provided as input (or is literally None), sample it (and clip it)
            # z effectively sampled from a truncated (0, 1) isotropic normal distribution
            size = (ob.size(0), self.latent_dim)
            z = torch.randn(*size, device=self.hps.device, generator=self.gen).to(ob).clamp(*BCQ_Z_CLAMPS)
        # Pass through the decoder and output head
        x = torch.cat([ob, z], dim=-1)
        x = self.decoder(x)
        ac = self.ac_head(x)
        ac = float(self.max_ac) * torch.tanh(ac)
        return ac

    def decodex(self, ob, z=None, expansion=4):
        """Used in BEAR"""
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        if z is None:
            # When z is not provided as input (or is literally None), sample it (and clip it)
            # z effectively sampled from a truncated (0, 1) isotropic normal distribution
            size = (ob.size(0), expansion, self.latent_dim)
            z = torch.randn(*size, device=self.hps.device, generator=self.gen).to(ob).clamp(*BCQ_Z_CLAMPS)
        # Pass through the decoder and output head
        ob = ob.unsqueeze(0).repeat(expansion, 1, 1).permute(1, 0, 2)
        x = torch.cat([ob, z], dim=-1)
        x = self.decoder(x)
        ac = self.ac_head(x)
        return torch.tanh(ac), ac

    def forward(self, ob, ac):
        unnormalized_ob = ob.clone()
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = torch.cat([ob, ac], dim=-1)
        # Encode
        x = self.encoder(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(*BCQ_LOG_STD_CLAMPS)  # clipped for numerical stability (BCQ)
        std = log_std.exp()
        z = mean + (std * torch.randn(*std.shape, device=self.hps.device, generator=self.gen))
        # Decode
        ac = self.decode(unnormalized_ob, z)  # the ob will be normalized in `decode`
        return ac, mean, std


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Tanh-wrapped models.

def arctanh(x):
    """Implementation of the arctanh function.
    Can be very numerically unstable, hence the clamping.
    """
    one_plus_x = (1 + x).clamp(min=1e-6)
    one_minus_x = (1 - x).clamp(min=1e-6)
    return 0.5 * torch.log(one_plus_x / one_minus_x)
    # return 0.5 * (x.log1p() - (-x).log1p())


class NormalToolkit(object):
    """Technically, multivariate normal with diagonal covariance"""

    @staticmethod
    def logp(x, mean, std):
        neglogp = (0.5 * ((x - mean) / std).pow(2).sum(dim=-1, keepdim=True) +
                   0.5 * math.log(2 * math.pi) +
                   std.log().sum(dim=-1, keepdim=True))
        return -neglogp

    @staticmethod
    def sample(mean, std, generator):
        # Re-parametrization trick
        eps = torch.empty(mean.size()).to(mean.device).normal_(generator=generator)
        eps.requires_grad = False
        return mean + std * eps

    @staticmethod
    def mode(mean):
        return mean


class TanhNormalToolkit(object):
    """Technically, multivariate normal with diagonal covariance"""

    @staticmethod
    def logp(x, mean, std):
        # We need to assemble the logp of a sample which comes from a Gaussian sample
        # after being mapped through a tanh. This needs a change of variable.
        # See appendix C of the SAC paper for an explanation of this change of variable.
        logp = NormalToolkit.logp(arctanh(x), mean, std) - torch.log(1 - x.pow(2) + 1e-6)
        return logp.sum(-1, keepdim=True)

    @staticmethod
    def nonsquashed_sample(mean, std, generator):
        # Special for BEAR
        sample = NormalToolkit.sample(mean, std, generator)
        return sample

    @staticmethod
    def sample(mean, std, generator):
        sample = NormalToolkit.sample(mean, std, generator)
        return torch.tanh(sample)

    @staticmethod
    def mode(mean):
        return torch.tanh(mean)


class MixtureTanhNormalToolkit(object):
    """Technically, mixture of multivariate normal with diagonal covariances"""

    @staticmethod
    def logp(x, log_mixture, mean, std):
        # `x` has shape[batch_size, ac_dim], while mean and std have shape [batch_size, K, ac_dim]
        logp_per_comp = TanhNormalToolkit.logp(x.unsqueeze(1).repeat(1, GAUSS_MIXTURE_COMPS, 1), mean, std)
        # `logp_per_comp` should have shape [batch_size, K], and now we compute the mixture weighted average
        logp = (torch.logsumexp(logp_per_comp + log_mixture.unsqueeze(-1), dim=1) -
                torch.logsumexp(log_mixture.unsqueeze(-1), dim=1))
        # The resulting logp should have shape [batch_size, 1]
        return logp

    @staticmethod
    def sample(log_mixture, mean, std, generator):
        # `mixture` has shape [batch_size, K]
        # Sample a component of the mixture, `component` has shape [batch_size, 1]
        comp_index = torch.multinomial(log_mixture.exp(), num_samples=1, generator=generator)
        # Note, exopnential is here because PyTorch's `multinomial` only takes positive logits
        # Create an index vector that indicate which component in the mixture has been selected
        comp_index = comp_index.unsqueeze(-1).expand(-1, -1, mean.shape[-1])
        # Gather the mean and std of the components of the mixture corresponding to the indices
        comp_mean = torch.gather(mean, 1, comp_index).squeeze(dim=1)
        comp_std = torch.gather(std, 1, comp_index).squeeze(dim=1)
        sample = TanhNormalToolkit.sample(comp_mean, comp_std, generator)
        return sample

    @staticmethod
    def mode(mean):
        mode = mean.max(dim=1).values
        return mode


class TanhGaussActor(nn.Module):

    def __init__(self, env, hps, rms_obs, hidden_dims):
        super(TanhGaussActor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Define observation whitening
        self.rms_obs = rms_obs
        # Define perception stack
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, hidden_dims[0])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[0])),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dims[0], hidden_dims[1])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[1])),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        if self.hps.state_dependent_std:
            self.head = nn.Linear(hidden_dims[1], 2 * ac_dim)
        else:
            self.head = nn.Linear(hidden_dims[1], ac_dim)
            self.ac_logstd_head = nn.Parameter(torch.full((ac_dim,), math.log(0.6)))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=5. / 3.))
        self.head.apply(init(weight_scale=0.01))

        # Create generator object which manages the random state of the actor
        # All the sampling is done by giving the generator as input
        self.gen = torch.Generator(device=hps.device).manual_seed(self.hps.seed)
        # Note, the device needs to be specified here, not carried over by the model

    def logp(self, ob, ac):
        out = self.forward(ob)
        return TanhNormalToolkit.logp(ac, *out[0:2])  # mean, std

    def entropy(self, ob):
        raise NotImplementedError

    def act(self, ob):
        # Special for BEAR
        # Note: it lets gradients flow through
        out = self.forward(ob)
        ac = TanhNormalToolkit.sample(*out[0:2], generator=self.gen)  # mean, std
        nonsquashed_ac = TanhNormalToolkit.nonsquashed_sample(*out[0:2], generator=self.gen)  # mean, std
        return ac, nonsquashed_ac

    def sample(self, ob, sg=True):
        if sg:
            with torch.no_grad():
                out = self.forward(ob)
                ac = TanhNormalToolkit.sample(*out[0:2], generator=self.gen)  # mean, std
        else:
            out = self.forward(ob)
            ac = TanhNormalToolkit.sample(*out[0:2], generator=self.gen)  # mean, std
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
        raise NotImplementedError

    def forward(self, ob):
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        # Note, the following clipping values were taken from the SAC/BEAR codebases
        if self.hps.state_dependent_std:
            ac_mean, ac_log_std = self.head(x).chunk(2, dim=-1)
            ac_mean = ac_mean.clamp(*SAC_MEAN_CLAMPS)
            ac_std = ac_log_std.clamp(*SAC_LOG_STD_CLAMPS).exp()
        else:
            ac_mean = self.head(x).clamp(*SAC_MEAN_CLAMPS)
            ac_std = self.ac_logstd_head.expand_as(ac_mean).clamp(*SAC_LOG_STD_CLAMPS).exp()
        return ac_mean, ac_std


class MixtureTanhGaussActor(nn.Module):

    def __init__(self, env, hps, rms_obs, hidden_dims):
        super(MixtureTanhGaussActor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_dim = ac_dim  # defined here for convenience later
        self.hps = hps
        # Define observation whitening
        self.rms_obs = rms_obs
        # Define perception stack
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, hidden_dims[0])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[0])),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dims[0], hidden_dims[1])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[1])),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        if self.hps.state_dependent_std:
            self.head = nn.Linear(hidden_dims[1], GAUSS_MIXTURE_COMPS * ((2 * ac_dim) + 1))
        else:
            self.head = nn.Linear(hidden_dims[1], GAUSS_MIXTURE_COMPS * (ac_dim + 1))
            self.ac_logstd_head = nn.Parameter(torch.full((GAUSS_MIXTURE_COMPS, ac_dim,), math.log(0.6)))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=5. / 3.))
        self.head.apply(init(weight_scale=0.01))

        # Create generator object which manages the random state of the actor
        # All the sampling is done by giving the generator as input
        self.gen = torch.Generator(device=hps.device).manual_seed(self.hps.seed)
        # Note, the device needs to be specified here, not carried over by the model

    def logp(self, ob, ac):
        out = self.forward(ob)
        return MixtureTanhNormalToolkit.logp(ac, *out[0:3])  # mixture, mean, std

    def entropy(self, ob):
        raise NotImplementedError

    def sample(self, ob, sg=True):
        if sg:
            with torch.no_grad():
                out = self.forward(ob)
                ac = MixtureTanhNormalToolkit.sample(*out[0:3], generator=self.gen)  # mixture, mean, std
        else:
            out = self.forward(ob)
            ac = MixtureTanhNormalToolkit.sample(*out[0:3], generator=self.gen)  # mixture, mean, std
        return ac

    def mode(self, ob, sg=True):
        if sg:
            with torch.no_grad():
                out = self.forward(ob)
                ac = MixtureTanhNormalToolkit.mode(out[1])  # mean
        else:
            out = self.forward(ob)
            ac = MixtureTanhNormalToolkit.mode(out[1])  # mean
        return ac

    def kl(self, ob, other):
        assert isinstance(other, MixtureTanhGaussActor)
        raise NotImplementedError

    def forward(self, ob):
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        # Note, the following clipping values were taken from the SAC/BEAR codebases
        if self.hps.state_dependent_std:
            ac_suff_stats_packed = self.head(x).view(-1, GAUSS_MIXTURE_COMPS, (2 * self.ac_dim) + 1)
            ac_log_mixture_weights = ac_suff_stats_packed[..., 0]
            ac_mean = ac_suff_stats_packed[..., 1:1 + self.ac_dim]
            ac_log_std = ac_suff_stats_packed[..., 1 + self.ac_dim:]
            ac_mean = ac_mean.clamp(*SAC_MEAN_CLAMPS)
            ac_std = ac_log_std.clamp(*SAC_LOG_STD_CLAMPS).exp()
        else:
            ac_suff_stats_packed = self.head(x).view(-1, GAUSS_MIXTURE_COMPS, self.ac_dim + 1)
            ac_log_mixture_weights = ac_suff_stats_packed[..., 0]
            ac_mean = ac_suff_stats_packed[..., 1:]
            ac_mean = ac_mean.clamp(*SAC_MEAN_CLAMPS)
            ac_std = self.ac_logstd_head.expand_as(ac_mean).clamp(*SAC_LOG_STD_CLAMPS).exp()
        return ac_log_mixture_weights, ac_mean, ac_std


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Reward averager.

class RewardAverager(nn.Module):

    def __init__(self, env, hps, rms_obs, hidden_dims):
        super(RewardAverager, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.hps = hps
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim + ob_dim, hidden_dims[0])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[0])),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(hidden_dims[0], hidden_dims[1])),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(hidden_dims[1])),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(hidden_dims[1], 1)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def forward(self, ob, ac, next_ob):
        if self.rms_obs is not None:
            ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
            next_ob = self.rms_obs.standardize(next_ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = torch.cat([ob, ac, next_ob], dim=-1)
        x = self.fc_stack(x)
        x = self.head(x)
        return x
