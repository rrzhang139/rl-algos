# Minimal PlaNet-style RSSM on image-based grid world
# ---------------------------------------------------
# This implementation is inspired by the PlaNet paper but simplified
# for clarity and to run quickly on CPU. It does not depend on gym.
# The environment is a small 2D grid that returns 8x8 grayscale images.
# The agent learns an encoder, RSSM dynamics, decoder and reward model
# and plans in latent space via CEM.

import math
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------- Environment -------------------------------------------------
class GridGoalEnv:
    """Simple grid world with image observations."""

    def __init__(self, size: int = 8):
        self.size = size
        self.action_space = list(range(4))  # 0:up,1:down,2:left,3:right
        self.reset()

    def reset(self):
        self.agent = np.random.randint(0, self.size, size=2)
        self.goal = np.random.randint(0, self.size, size=2)
        while np.all(self.goal == self.agent):
            self.goal = np.random.randint(0, self.size, size=2)
        return self._get_obs()

    def step(self, action: int):
        if action == 0:  # up
            self.agent[0] = max(0, self.agent[0] - 1)
        elif action == 1:  # down
            self.agent[0] = min(self.size - 1, self.agent[0] + 1)
        elif action == 2:  # left
            self.agent[1] = max(0, self.agent[1] - 1)
        elif action == 3:  # right
            self.agent[1] = min(self.size - 1, self.agent[1] + 1)

        done = np.all(self.agent == self.goal)
        reward = 1.0 if done else -0.1
        return self._get_obs(), reward, done

    def _get_obs(self):
        obs = np.zeros((1, self.size, self.size), dtype=np.float32)
        obs[0, self.goal[0], self.goal[1]] = 0.5
        obs[0, self.agent[0], self.agent[1]] = 1.0
        return torch.tensor(obs)

# -------- Models ------------------------------------------------------
LatentDim = 16
HiddenDim = 32


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.fc = nn.Linear(16 * 2 * 2, HiddenDim)
        self.mu = nn.Linear(HiddenDim, LatentDim)
        self.logvar = nn.Linear(HiddenDim, LatentDim)

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.mu(x), self.logvar(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LatentDim, 16 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 16, 2, 2)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRUCell(LatentDim + 4, HiddenDim)
        self.prior_mu = nn.Linear(HiddenDim, LatentDim)
        self.prior_logvar = nn.Linear(HiddenDim, LatentDim)

    def forward(self, prev_z, action_onehot, prev_h):
        inp = torch.cat([prev_z, action_onehot], dim=-1)
        h = self.rnn(inp, prev_h)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h).clamp(-5, 5)
        return mu, logvar, h


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(LatentDim + HiddenDim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z, h):
        x = torch.cat([z, h], dim=-1)
        return self.fc2(F.relu(self.fc1(x)))


# -------- Utilities ---------------------------------------------------
Transition = namedtuple("Transition", "obs act rew next_obs done")

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def one_hot(actions, num_actions=4):
    return F.one_hot(actions.long(), num_actions).float()

# -------- Training setup ----------------------------------------------
device = torch.device("cpu")
encoder = Encoder().to(device)
decoder = Decoder().to(device)
rssm = RSSM().to(device)
reward_m = RewardModel().to(device)
params = list(encoder.parameters()) + list(decoder.parameters()) + \
    list(rssm.parameters()) + list(reward_m.parameters())
opt = torch.optim.Adam(params, lr=1e-3)

env = GridGoalEnv()
replay = deque(maxlen=5000)

episodes = 150
steps_per_ep = 30
warmup_steps = 400
CEM_samples = 50
CEM_iters = 2
CEM_elite_frac = 0.2
planning_horizon = 6

returns = []


def plan_action(z, h):
    with torch.no_grad():
        mu = torch.zeros(planning_horizon, dtype=torch.float)
        std = torch.ones(planning_horizon, dtype=torch.float)
        for _ in range(CEM_iters):
            seq = torch.randint(0, 4, (CEM_samples, planning_horizon))
            rewards = torch.zeros(CEM_samples)
            for i in range(CEM_samples):
                z_roll = z.clone()
                h_roll = h.clone()
                total = 0.0
                for t in range(planning_horizon):
                    a = seq[i, t]
                    a_oh = one_hot(a.unsqueeze(0))
                    prior_mu, _, h_roll = rssm(z_roll, a_oh, h_roll)
                    z_roll = prior_mu
                    r_pred = reward_m(z_roll, h_roll)
                    total += r_pred.item()
                rewards[i] = total
            elite_idx = rewards.topk(max(1, int(CEM_samples * CEM_elite_frac))).indices
            elite = seq[elite_idx]
            mu = elite.float().mean(0)
            std = elite.float().std(0) + 1e-3
        return int(mu[0].round().clamp(0, 3).item())


# ---------------- Main Loop -------------------------------------------
loss = torch.tensor(0.0)
global_step = 0
for ep in range(episodes):
    obs = env.reset()
    mu, logvar = encoder(obs.unsqueeze(0))
    z = reparameterize(mu, logvar)
    h = torch.zeros(1, HiddenDim)
    ep_ret = 0.0

    for t in range(steps_per_ep):
        global_step += 1
        if global_step < warmup_steps:
            action = random.choice(env.action_space)
        else:
            action = plan_action(z, h)
        next_obs, reward, done = env.step(action)

        replay.append(Transition(obs, action, reward, next_obs, done))
        ep_ret += reward

        mu_post, logvar_post = encoder(next_obs.unsqueeze(0))
        z_post = reparameterize(mu_post, logvar_post)
        a_onehot = one_hot(torch.tensor([action]))
        _, _, h = rssm(z_post.detach(), a_onehot, h)

        obs = next_obs
        z = z_post

        if len(replay) >= 512:
            batch = random.sample(replay, 32)
            obs_b = torch.stack([b.obs for b in batch]).to(device)
            act_b = torch.tensor([b.act for b in batch]).unsqueeze(-1).to(device)
            next_obs_b = torch.stack([b.next_obs for b in batch]).to(device)
            rew_b = torch.tensor([b.rew for b in batch]).unsqueeze(-1).to(device)

            mu_enc, logvar_enc = encoder(obs_b)
            z_enc = reparameterize(mu_enc, logvar_enc)
            mu_next, logvar_next = encoder(next_obs_b)
            z_next = reparameterize(mu_next, logvar_next)

            act_oh = one_hot(act_b.squeeze(-1))
            h0 = torch.zeros(len(batch), HiddenDim)
            prior_mu, prior_logvar, _ = rssm(z_enc.detach(), act_oh, h0)

            kl_loss = 0.5 * torch.mean(
                prior_logvar - logvar_next +
                (torch.exp(logvar_next) + (mu_next - prior_mu) ** 2) /
                torch.exp(prior_logvar) - 1
            )
            recon_obs = decoder(z_enc)
            recon_loss = F.mse_loss(recon_obs, obs_b)
            pred_rew = reward_m(z_enc, h0)
            rew_loss = F.mse_loss(pred_rew, rew_b)

            loss = recon_loss + rew_loss + 0.1 * kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

        if done:
            break

    returns.append(ep_ret)
    if (ep + 1) % 20 == 0:
        avg_ret = np.mean(returns[-20:])
        print(f"Ep {ep+1:03d} | AvgReturn last20: {avg_ret:5.2f} | Loss {loss.item():.4f}")

# -------- Visualize learning curve -----------------------------------
plt.figure()
ma = np.convolve(returns, np.ones(10) / 10, mode='valid')
plt.plot(ma)
plt.xlabel('Episode')
plt.ylabel('Return (10-ep MA)')
plt.title('PlaNet-pixel: episode return')
plt.savefig('planet_pixel_returns.png')
print('Saved training curve to planet_pixel_returns.png')
