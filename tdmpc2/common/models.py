import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Deterministic policy."""

    def __init__(self, latent_dim, act_dim, act_limit):
        super().__init__()
        self.act_limit = act_limit
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )

    def forward(self, latent):
        # Output is scaled to environment action bounds.
        return self.act_limit * self.net(latent)

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
