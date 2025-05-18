# A **bare‑bones PlaNet‑style RSSM demo** in PyTorch
#
# – 1‑D point‑mass environment  (state x ∈ ℝ,   action a ∈ {‑1,0,1})
# – Learn encoder, stochastic latent dynamics (RSSM), reward model, decoder
# – Train on collected experience via a simple ELBO (recon + KL + reward)
# – Plan with CEM in latent space (after warm‑up) to improve returns
#
# The model + training are intentionally tiny so they run in <30 s on CPU.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from collections import deque, namedtuple

# ----------- ENVIRONMENT -----------


class PointEnv1D:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
        self.action_space = torch.tensor([-1.0, 0.0, 1.0])
        self.reset()

    def reset(self):
        self.x = float(np.random.uniform(-5, 5))
        return torch.tensor([self.x])

    def step(self, a):
        a = float(a)
        x_next = self.x + a + np.random.randn() * self.noise_std
        r = -x_next**2
        self.x = x_next
        done = False
        return torch.tensor([x_next]), torch.tensor([r]), done


# ----------- MODEL -----------
LatentDim = 3
HiddenDim = 16


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, HiddenDim)
        self.mu = nn.Linear(HiddenDim, LatentDim)
        self.logvar = nn.Linear(HiddenDim, LatentDim)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LatentDim, 1)

    def forward(self, z):
        return self.fc(z)


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        # latent + action -> hidden
        self.rnn = nn.GRUCell(LatentDim + 1, HiddenDim)
        self.prior_mu = nn.Linear(HiddenDim, LatentDim)
        self.prior_logvar = nn.Linear(HiddenDim, LatentDim)

    def forward(self, prev_z, action, prev_h):
        inp = torch.cat([prev_z, action], dim=-1)
        h = self.rnn(inp, prev_h)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h).clamp(-5, 5)
        return mu, logvar, h


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LatentDim + HiddenDim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))

# ----------- UTILITIES -----------


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


Transition = namedtuple("Transition", "obs act rew next_obs")

# ----------- TRAINING SETUP -----------
device = torch.device("cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
rssm = RSSM().to(device)
reward_m = RewardModel().to(device)

params = list(encoder.parameters())+list(decoder.parameters()) + \
    list(rssm.parameters())+list(reward_m.parameters())
opt = torch.optim.Adam(params, lr=5e-3)

env = PointEnv1D()
replay = deque(maxlen=10000)

episodes = 120
steps_per_ep = 30
warmup_steps = 200   # collect random data before planning
CEM_samples = 120
CEM_iters = 3
CEM_elite_frac = 0.15
planning_horizon = 12

returns = []


def plan_action(obs, h, z):
    # CEM in latent space using current model
    with torch.no_grad():
        mu = torch.zeros(planning_horizon)
        std = torch.ones(planning_horizon)
        for _ in range(CEM_iters):
            seq = torch.randn(CEM_samples, planning_horizon) * std + mu
            seq = seq.round().clamp(-1, 1)  # discrete {-1,0,1}
            rewards = torch.zeros(CEM_samples)
            for i in range(CEM_samples):
                z_roll = z.clone()
                h_roll = h.clone()
                total = 0.0
                for t in range(planning_horizon):
                    a = seq[i, t:t+1]
                    prior_mu, prior_logvar, h_roll = rssm(
                        z_roll, a.unsqueeze(0), h_roll)
                    z_roll = prior_mu  # use mean
                    r_pred = reward_m(z_roll, h_roll)
                    total += r_pred.item()
                rewards[i] = total
            elite_idx = rewards.topk(
                max(1, int(CEM_samples*CEM_elite_frac))).indices
            elite = seq[elite_idx]
            mu, std = elite.mean(0), elite.std(0)+1e-3
        return float(mu[0].round().clamp(-1, 1))


# ------------- MAIN LOOP -------------
global_step = 0
for ep in range(episodes):
    obs = env.reset()
    # initial latent & hidden
    mu, logvar = encoder(obs)
    z = reparameterize(mu, logvar)
    h = torch.zeros(1, HiddenDim)
    ep_ret = 0.0

    for t in range(steps_per_ep):
        global_step += 1
        if global_step < warmup_steps:
            action = random.choice([-1.0, 0.0, 1.0])
        else:
            action = plan_action(obs, h, z)
        next_obs, reward, done = env.step(action)

        replay.append(Transition(obs, action, reward, next_obs))
        ep_ret += reward.item()

        # update latent & hidden with encoder on *next* obs (posterior)
        mu_post, logvar_post = encoder(next_obs)
        z_post = reparameterize(mu_post, logvar_post)
        # update hidden using posterior z (per RSSM paper)
        _, _, h = rssm(z_post.detach(), torch.tensor([[action]]), h)

        obs = next_obs
        z = z_post

        # ----- MODEL TRAINING -----
        if len(replay) >= 512:
            batch = random.sample(replay, 64)
            obs_b = torch.stack([b.obs for b in batch]).to(device)
            act_b = torch.tensor([[b.act] for b in batch]).to(device)
            next_obs_b = torch.stack([b.next_obs for b in batch]).to(device)
            rew_b = torch.stack([b.rew for b in batch]).to(device)

            # Encode current & next obs
            mu_enc, logvar_enc = encoder(obs_b)
            z_enc = reparameterize(mu_enc, logvar_enc)

            mu_next, logvar_next = encoder(next_obs_b)
            z_next = reparameterize(mu_next, logvar_next)

            # RSSM prior from z_enc + a
            h0 = torch.zeros(len(batch), HiddenDim)
            prior_mu, prior_logvar, _ = rssm(z_enc.detach(), act_b, h0)

            # Losses
            kl_loss = 0.5 * torch.mean(
                prior_logvar - logvar_next +
                (torch.exp(logvar_next) + (mu_next - prior_mu)**2) /
                torch.exp(prior_logvar) - 1
            )

            recon_obs = decoder(z_enc)
            recon_loss = F.mse_loss(recon_obs, obs_b)

            pred_rew = reward_m(z_enc, h0)
            rew_loss = F.mse_loss(pred_rew, rew_b)

            loss = recon_loss + rew_loss + 0.1*kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

    returns.append(ep_ret)

    if (ep+1) % 20 == 0:
        print(
            f"Ep {ep+1:03d} | AvgReturn last20: {np.mean(returns[-20:]):6.2f} | Loss {loss.item():.4f}")

# --------- VISUALIZE EPISODE RETURNS ----------
plt.figure()
plt.plot(np.convolve(returns, np.ones(10)/10, mode='valid'))
plt.xlabel("Episode")
plt.ylabel("Return (10‑ep MA)")
plt.title("PlaNet‑lite: episode return")
plt.show()
