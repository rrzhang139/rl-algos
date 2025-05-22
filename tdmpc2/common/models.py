import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Encoder(nn.Module):
    """Encodes observations into latent features."""

    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        # Two layer MLP producing a compact latent representation.
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, obs):
        # obs: [B, obs_dim]
        return self.net(obs)

class Dynamics(nn.Module):
    """Predicts next latent state given current latent and action."""

    def __init__(self, latent_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, latent, act):
        # Concatenate along the feature dimension: [B, latent+act]
        x = torch.cat([latent, act], dim=-1)
        return self.net(x)


class Actor(nn.Module):
    """
    Stochastic squashed-Gaussian policy used by TD-MPC2.
    * Minimal: no task-conditioning, no action-masking.*
    """

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
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    @torch.jit.script
    def _squash(self, pi):
        return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)

    def squash(self, mu, pi, log_pi):
        """Apply squashing function."""
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        log_pi -= self._squash(pi).sum(-1, keepdim=True)
        return mu, pi, log_pi

    def forward(self, latent):
        mu = self.mu_net(latent)

        # Clamp log-σ
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mu)

        # Reparameterised sampling
        eps = torch.randn_like(mu)
        pi  = mu + eps * log_std.exp()

        # Log-prob under the unsquashed Gaussian
        log_pi = (-0.5 * ((eps ** 2) + 2 * log_std + math.log(2 * math.pi))).sum(dim=-1, keepdim=True)

        # Squash to (-act_limit, act_limit)
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
                nn.Linear(latent_dim + act_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
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
            nn.Linear(latent_dim + act_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, max(num_bins, 1)),
        )

    def forward(self, latent, act):
        # Concatenate latent state and action
        x = torch.cat([latent, act], dim=-1)
        return self.net(x)
