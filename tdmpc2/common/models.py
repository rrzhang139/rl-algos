import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimNorm(nn.Module):
	"""
	Simplicial normalization. Same shape, don't care batch.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""
	
	def __init__(self, simnorm_dim):
		super().__init__()
		self.dim = simnorm_dim
	
	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)
		
	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout. Don't care batch
	"""

	def __init__(self, *args, dropout=0., act=nn.Mish(inplace=True), **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=True) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))
	
	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


class Encoder(nn.Module):
    """Encodes observations into latent features."""

    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        # Two layer MLP producing a compact latent representation.
        self.net = nn.Sequential(
            NormedLinear(obs_dim, 256, act=nn.Mish()),
            NormedLinear(256, 512, act=SimNorm(8)),
        )

    def forward(self, obs):
        # obs: [B, obs_dim]
        return self.net(obs)

class Dynamics(nn.Module):
    """Predicts next latent state given current latent and action."""

    def __init__(self, latent_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            NormedLinear(latent_dim + act_dim, 512, act=nn.Mish()),
            NormedLinear(512, 512, act=nn.Mish()),
            NormedLinear(512, latent_dim, act=SimNorm(8)),
        )

    def forward(self, latent, act):
        # Concatenate along the feature dimension: [B, latent+act]
        x = torch.cat([latent, act], dim=-1)
        return self.net(x)


class Actor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        act_dim: int,
        act_limit: float,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.act_limit   = act_limit
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mu_net = nn.Sequential(
            NormedLinear(latent_dim, 512, act=nn.Mish()),
            NormedLinear(512, 512, act=nn.Mish()),
            nn.Linear(512, 2*act_dim),
        )
        
        nn.init.zeros_(self.mu_net[-1].weight)
        nn.init.zeros_(self.mu_net[-1].bias)

        # self.log_std = nn.Parameter(torch.zeros(act_dim))

    # @torch.jit.script
    # def _squash(self, pi):
    #     return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)

    def squash(self, mu, pi, log_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
        # print(f"Det Jacob: {torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)}")
        return mu, pi, log_pi

    def forward(self, latent):
        mu, log_std = self.mu_net(latent).chunk(2, dim=-1)

        # Clamp log-σ
        log_std = self.log_std_min + 0.5 * (
                    torch.tanh(log_std) + 1.0
                ) * (self.log_std_max - self.log_std_min)
        print(f"Actor log std: {log_std[0][0]}")

        # Reparameterised sampling
        eps = torch.randn_like(mu)
        pi  = mu + eps * log_std.exp()

        log_pi = (-0.5 * ((eps ** 2) + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)

        mu, pi, log_pi = self.squash(mu, pi, log_pi)
        mu = mu * self.act_limit
        pi = pi * self.act_limit

        return mu, pi, log_pi, log_std

class Critic(nn.Module):
    """Twin Q networks for TD learning."""

    def __init__(self, latent_dim, act_dim):
        super().__init__()

        def build():
            return nn.Sequential(
                NormedLinear(latent_dim + act_dim, 512, act=nn.Mish(), dropout=0.01),
                NormedLinear(512, 512, act=nn.Mish()),
                nn.Linear(512, 1),
            )

        self.q1 = build()
        self.q2 = build()

    def forward(self, latent, act):
        # Input is concatenation of latent state and action.
        x = torch.cat([latent, act], dim=-1)
        return self.q1(x), self.q2(x)

class RewardModel(nn.Module):
    """Reward prediction model."""

    def __init__(self, latent_dim, act_dim, mlp_dim=128, num_bins=1):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            NormedLinear(latent_dim + act_dim, 512, act=nn.Mish()),
            NormedLinear(512, 512, act=nn.Mish()),
            nn.Linear(512, max(num_bins, 1)),
        )

    def forward(self, latent, act):
        # Concatenate latent state and action
        x = torch.cat([latent, act], dim=-1)
        return self.net(x)
