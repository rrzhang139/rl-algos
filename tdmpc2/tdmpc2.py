import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import numpy as np

from .common.models import Encoder, Dynamics, Actor, Critic, RewardModel
from .common.buffer import ReplayBuffer


class TD_MPC2:
    """Minimal TD-MPC2 implementation for continuous control."""

    def __init__(self, obs_dim, act_dim, latent_dim, act_limit, mpc=True, horizon=5, gamma=0.99, entropy_coef=0.2, device="cpu"):
        self.device = device
        self.mpc = mpc

        # --- Model definitions ---
        self.latent_dim = latent_dim
        self.encoder = Encoder(obs_dim, latent_dim).to(device)
        self.dynamics = Dynamics(latent_dim, act_dim).to(device)
        self.actor = Actor(latent_dim, act_dim, act_limit).to(device)
        self.critic = Critic(latent_dim, act_dim).to(device)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.reward = RewardModel(latent_dim, act_dim).to(device)

        # Optimisers for policy, critic and dynamics/encoder.
        self.model_parameters = list(self.encoder.parameters()) + list(self.dynamics.parameters()) + \
                list(self.critic.parameters()) + list(self.reward.parameters())
        self.optimizer = Adam(
            self.model_parameters,
            lr=3e-4,
        )
        self.act_optimizer = Adam(self.actor.parameters(), lr=3e-4)

        # Experience replay buffer used for off-policy updates.
        self.horizon = horizon
        self.replay = ReplayBuffer(obs_dim, act_dim, size=100000, horizon=self.horizon)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip_norm = 20.0
        self.tau = 0.01
        self.act_dim = act_dim
        self.act_limit = act_limit
    
        
    @torch.no_grad()
    def plan(self, z, eval_mode=False):
        """
        MPPI action-sequence optimiser   (single-task, no masking).

        Parameters
        ----------
        z :  (B, latent_dim)  – current latent state(s)
        eval_mode : bool      – set True during evaluation (no added noise)

        Returns
        -------
        a0 : (B, act_dim) – first action in the optimised sequence, scaled
                            to [-act_limit, act_limit] already.
        """

        # -------------  short names from cfg  ---------------------------------
        H          = self.horizon
        N          = 256                    # samples per env
        N_elite    = 16
        T          = 5                      # optimisation iterations
        max_std    = 0.5
        min_std    = 0.05
        temperature= 1.0
        num_pi = 24

        B          = z.shape[0]             # number of parallel envs
        device     = z.device
        
        if num_pi > 0:
            breakpoint()
            actions_pi = torch.empty(B, H, num_pi, self.act_dim, device=device)
            z_pi = z.unsqueeze(1).repeat(1, num_pi, 1)            # (B,num_pi,D)
            for t in range(H - 1):
                _, a_t, _, _ = self.actor(z_pi)                   # (B,num_pi,A)
                actions_pi[:, t] = a_t
                z_pi = self.dynamics(z_pi, a_t)                   # next latent
            actions_pi[:, -1] = self.actor(z_pi)[1]               # last step

        # -------------  hold mean / std over the horizon  ---------------------
        mean = torch.zeros(B, H, self.act_dim, device=device)
        std  = torch.ones_like(mean) * max_std

        z_repeat = z.unsqueeze(1).repeat(1, N, 1)          # (B,N,D)

        # -------------  main optimisation loop  -------------------------------
        for _ in range(T):

            # Sample action sequences  (B, H, N, A)
            eps = torch.randn(B, H, N, self.act_dim, device=device)
            actions = (mean.unsqueeze(2) + std.unsqueeze(2) * eps).clamp(-1, 1)

            # ----- rollout value ------------------------------------------------
            value = torch.zeros(B, N, 1, device=device)
            z_roll = z_repeat                                                # (B,N,D)
            for t in range(H):
                a_t = actions[:, t]                                          # (B,N,A)
                z_roll = self.dynamics(z_roll, a_t)                          # next z
                q1, q2 = self.target_critic(z_roll, a_t)
                value += torch.min(q1, q2)                                   # accumulate

            # ----- select elites -----------------------------------------------
            elite_idx = torch.topk(value.squeeze(-1), N_elite, dim=1).indices     # (B,E)
            elite_actions = torch.gather(actions, 2,
                                        elite_idx.unsqueeze(1)
                                                .unsqueeze(3)
                                                .expand(-1, H, -1, self.act_dim))   # (B,H,E,A)
            elite_val = torch.gather(value, 1, elite_idx.unsqueeze(-1))            # (B,E,1)

            # ----- update mean / std  ------------------------------------------
            max_val = elite_val.max(1, keepdim=True).values                        # (B,1,1)
            score   = torch.exp(temperature * (elite_val - max_val))               # (B,E,1)
            weight  = score / (score.sum(1, keepdim=True) + 1e-9)

            mean = (weight.unsqueeze(1) * elite_actions).sum(2)
            std  = torch.sqrt(
                    (weight.unsqueeze(1) *
                    (elite_actions - mean.unsqueeze(2)).pow(2)
                    ).sum(2)
                ).clamp_(min_std, max_std)

        # -------------  pick one elite trajectory per env  ---------------------
        weight = weight.squeeze(-1).cpu().numpy()                       # (B,E)
        a_seq  = torch.zeros(B, H, self.act_dim, device=device)
        for i in range(B):
            j = np.random.choice(N_elite, p=weight[i] / weight[i].sum())
            a_seq[i] = elite_actions[i, :, j]

        a0 = a_seq[:, 0]                                                # first action

        if not eval_mode:                                               # exploration
            a0 += std[:, 0] * torch.randn_like(a0)

        return (a0 * self.act_limit).clamp(-self.act_limit, self.act_limit)

    @torch.no_grad()
    def act(self, obs, noise_scale=0.1):
        obs = torch.as_tensor(obs, dtype=torch.float32,
                           device=self.device)
        breakpoint()
        latent = self.encoder(obs)
        
        if self.mpc:
            action = self.plan(latent)
        else:
            _, action, _, _ = self.actor(latent)
        # Add exploration noise when interacting with the env.
        action += noise_scale * torch.randn_like(action)
        # Clamp to action bounds
        return torch.clamp(action, -self.act_limit, self.act_limit)


    def update_pi(self, latents):
        rho_vec = torch.pow(self.gamma, torch.arange(self.horizon, device=self.device))

        _, pi_act, log_pi, _ = self.actor(latents)

        q1, q2 = self.critic(latents, pi_act)
        q_min  = torch.min(q1, q2).squeeze(-1)
        
        log_pi = log_pi.squeeze(-1)
        term   = self.entropy_coef * log_pi - q_min

        pi_loss = (term.mean(dim=1) * rho_vec).mean()

        self.act_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.act_optimizer.step()

        return pi_loss
    
    @torch.no_grad()
    def soft_update_target_critic(self):
        for p, p_tgt in zip(self.critic.parameters(),
                            self.target_critic.parameters()):
            p_tgt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch_size=256):
        breakpoint()
        batch = self.replay.sample_batch(batch_size)
        obs      = batch["obs"].permute(1, 0, 2).to(self.device)   # (K+1, B, obs_dim)
        act      = batch["act"].permute(1, 0, 2).to(self.device)   # (K,   B, act_dim)
        rew      = batch["rew"].permute(1, 0).to(self.device)      # (K,   B)
        done     = batch["done"].permute(1, 0).to(self.device)     # (K,   B)
        
        with torch.no_grad():
            next_latent = self.encoder(obs[1:])
            print(f"Next latent: {next_latent[0][0]}")
            _, pi_act, _, _ = self.actor(next_latent)
            q1_t, q2_t = self.target_critic(next_latent, pi_act)
            q_next = torch.min(q1_t, q2_t).squeeze(-1)
            td_targets = rew + self.gamma * (1 - done) * q_next

        latents = torch.empty(self.horizon + 1, batch_size, self.latent_dim, device=self.device)
        latent = self.encoder(obs[0])
        dyn_loss = 0
        for t in range(self.horizon):
            latent = self.dynamics(latent, act[t])
            dyn_loss += F.mse_loss(latent, next_latent[t]) * self.gamma**t
            latents[t+1] = latent
        breakpoint()
        _latents = latents[:-1]
        q1, q2 = self.critic(_latents, act)
        q1, q2 = q1.squeeze(-1), q2.squeeze(-1)
        reward_preds = self.reward(_latents, act).squeeze(-1)
        
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
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip_norm)
        self.optimizer.step()
        
        latents_detached = _latents.detach()
        pi_loss = self.update_pi(latents_detached)
        
        self.soft_update_target_critic()
        
        print("finished update")

        return dict(dynamics_loss=dyn_loss.item(), reward_loss=reward_loss.item(), value_loss=value_loss.item(),
                    total_loss=total_loss.item(), pi_loss=pi_loss.item(), grad_norm=grad_norm)
