import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .common.models import Encoder, Dynamics, Actor, Critic, RewardModel
from .common.buffer import ReplayBuffer


class TD_MPC2:
    """Minimal TD-MPC2 implementation for continuous control."""

    def __init__(self, obs_dim, act_dim, act_limit, mpc=False, horizon=5, gamma=0.99, device="cpu"):
        self.device = device
        self.mpc = mpc

        # --- Model definitions ---
        latent_dim = 32  # size of encoded observation
        self.encoder = Encoder(obs_dim, latent_dim).to(device)
        self.dynamics = Dynamics(latent_dim, act_dim).to(device)
        self.actor = Actor(latent_dim, act_dim, act_limit).to(device)
        self.critic = Critic(latent_dim, act_dim).to(device)
        self.reward = RewardModel(latent_dim, act_dim).to(device)

        # Optimisers for policy, critic and dynamics/encoder.
        self.optimizer = Adam(
            list(self.encoder.parameters()) + list(self.dynamics.parameters()) + \
                list(self.critic.parameters()) + list(self.actor.parameters()),
            lr=3e-4,
        )

        # Experience replay buffer used for off-policy updates.
        self.horizon = horizon
        self.replay = ReplayBuffer(obs_dim, act_dim, size=100000, horizon=self.horizon)
        self.gamma = gamma
        self.act_dim = act_dim
        self.act_limit = act_limit
        
    def plan(self, latent):
        pass

    @torch.no_grad()
    def act(self, obs, noise_scale=0.1):
        obs = torch.as_tensor(obs, dtype=torch.float32,
                           device=self.device)
        # breakpoint()
        latent = self.encoder(obs)
        if self.mpc:
            action = self.plan(latent)
        else:
            action = self.actor(latent).squeeze(0)
        # Add exploration noise when interacting with the env.
        action += noise_scale * torch.randn_like(action)
        # Clamp to action bounds
        return torch.clamp(action, -self.act_limit, self.act_limit)

    def update(self, batch_size=256):
        batch = self.replay.sample_batch(batch_size)
        obs      = batch["obs"].permute(1, 0, 2).to(self.device)   # (K+1, B, obs_dim)
        act      = batch["act"].permute(1, 0, 2).to(self.device)   # (K,   B, act_dim)
        rew      = batch["rew"].permute(1, 0).to(self.device)      # (K,   B)
        done     = batch["done"].permute(1, 0).to(self.device)     # (K,   B)
        
        with torch.no_grad():
            next_latent = self.encoder(obs[1:])
            _, pi_act, _, _ = self.actor(next_latent)
            td_targets = rew + self.gamma * min(self.critic(next_latent, pi_act))
            
        # --- Dynamics loss ---
        latents = torch.empty_like(obs, device=self.device)
        latent = self.encoder(obs[0])
        dyn_loss = 0
        for t in range(self.horizon):
            latent = self.dynamics(latent, act[t])
            dyn_loss += F.mse_loss(latent, next_latent[t]) * self.gamma**t
            latents[t+1] = latent
            
            
        _latents = latents[:-1]
        q1, q2 = self.critic(_latents, act)
        reward_preds = self.reward(_latents, act)
        
        # Compute losses
        reward_loss, value_loss = 0, 0
        for t in range(self.horizon):
            reward_loss += F.mse_loss(reward_preds[t], rew[t]).mean() * self.gamma**t
            value_loss += F.mse_loss(q1[t], td_targets[t]).mean() * self.gamma**t
            value_loss += F.mse_loss(q2[t], td_targets[t]).mean() * self.gamma**t
        
        dyn_loss *= (1 / self.horizon)
        reward_loss*=(1 / self.horizon)
        value_loss*=(1 / self.horizon * 2) # 2 critic networks
        
        total_loss = (
            # TODO: add coeff for each loss
            dyn_loss + 
            reward_loss + 
            value_loss
        )
            
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        pi_loss = self.actor()

        return dict(dynamics_loss=dyn_loss.item(), reward_loss=reward_loss.item(), value_loss=value_loss.item(),
                    total_loss=total_loss.item(), )
