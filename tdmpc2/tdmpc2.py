import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .common.models import Encoder, Dynamics, Actor, Critic
from .common.buffer import ReplayBuffer


class TD_MPC2:
    """Minimal TD-MPC2 implementation for continuous control."""

    def __init__(self, obs_dim, act_dim, act_limit, device="cpu"):
        self.device = device

        # --- Model definitions ---
        latent_dim = 32  # size of encoded observation
        self.encoder = Encoder(obs_dim, latent_dim).to(device)
        self.dynamics = Dynamics(latent_dim, act_dim).to(device)
        self.actor = Actor(latent_dim, act_dim, act_limit).to(device)
        self.critic = Critic(latent_dim, act_dim).to(device)

        # Optimisers for policy, critic and dynamics/encoder.
        self.actor_opt = Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = Adam(self.critic.parameters(), lr=3e-4)
        self.dynamics_opt = Adam(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()),
            lr=3e-4,
        )

        # Experience replay buffer used for off-policy updates.
        self.replay = ReplayBuffer(obs_dim, act_dim, size=100000)
        self.gamma = 0.99
        self.act_dim = act_dim
        self.act_limit = act_limit

    @torch.no_grad()
    def act(self, obs, noise_scale=0.1):
        # Convert to tensor and add batch dimension.
        obs = torch.tensor(obs, dtype=torch.float32,
                           device=self.device).unsqueeze(0)
        latent = self.encoder(obs)
        # Deterministic action from actor.
        action = self.actor(latent).squeeze(0)
        # Add exploration noise when interacting with the env.
        action += noise_scale * torch.randn_like(action)
        # Clamp to action bounds and return as numpy array.
        return torch.clamp(action, -self.act_limit, self.act_limit).cpu().numpy()

    def update(self, batch_size=256):
        batch = self.replay.sample_batch(batch_size)
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        act = batch["act"].to(self.device)
        rew = batch["rew"].to(self.device)
        done = batch["done"].to(self.device)

        # --- Dynamics loss ---
        latent = self.encoder(obs)
        next_latent = self.encoder(next_obs).detach()
        pred_next = self.dynamics(latent, act)
        # Predict next latent state and regress to encoding of next_obs
        dyn_loss = F.mse_loss(pred_next, next_latent)
        self.dynamics_opt.zero_grad()
        dyn_loss.backward()
        self.dynamics_opt.step()

        with torch.no_grad():
            # Target policy value for next state.
            target_act = self.actor(next_latent)
            q1_next, q2_next = self.critic(next_latent, target_act)
            target_q = rew + self.gamma * \
                (1 - done) * torch.min(q1_next, q2_next).squeeze(-1)

        # --- Critic update ---
        q1, q2 = self.critic(latent, act)
        critic_loss = F.mse_loss(q1.squeeze(-1), target_q) + \
            F.mse_loss(q2.squeeze(-1), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- Actor update ---
        actor_loss = - \
            self.critic.q1(
                torch.cat([latent, self.actor(latent)], dim=-1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return dict(actor_loss=actor_loss.item(), critic_loss=critic_loss.item(), dyn_loss=dyn_loss.item())
