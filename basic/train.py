import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---- Hyperparams ----
epochs = 300
steps_per_epoch = 2048
gamma = 0.99
lam = 0.95
clip_ratio = 0.2
policy_lr = 3e-4
value_lr = 1e-3

# ---- PPO Actor-Critic ----


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)


# ---- Env Setup ----
env = gym.make("Ant-v5")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

ac = ActorCritic(obs_dim, act_dim)
pi_opt = optim.Adam(ac.actor.parameters(), lr=policy_lr)
vf_opt = optim.Adam(ac.critic.parameters(), lr=value_lr)

# ---- PPO Training Loop ----
for epoch in range(epochs):
    obs_buf, act_buf, adv_buf, ret_buf, logp_buf = [], [], [], [], []
    obs, _ = env.reset()
    done = False
    ep_rews, values, logps = [], [], []
    t = 0
    while t < steps_per_epoch:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        mean_action, value = ac(obs_t)
        dist = torch.distributions.Normal(mean_action, 0.2)
        action = dist.sample()
        logp = dist.log_prob(action).sum()

        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        ep_rews.append(reward)
        values.append(value.item())
        logps.append(logp.item())

        obs_buf.append(obs)
        act_buf.append(action.detach().numpy())
        obs = next_obs
        t += 1

        if done:
            obs, _ = env.reset()

    # ---- Compute GAE-Lambda Advantages ----
    rewards = np.array(ep_rews)
    values = np.array(values + [0])
    deltas = rewards + gamma * values[1:] - values[:-1]

    adv = []
    adv_t = 0
    for d in reversed(deltas):
        adv_t = d + gamma * lam * adv_t
        adv.insert(0, adv_t)
    adv = torch.tensor(adv, dtype=torch.float32)

    obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(act_buf), dtype=torch.float32)
    logp_old = torch.tensor(logps, dtype=torch.float32)
    ret = adv + torch.tensor(values[:-1], dtype=torch.float32)

    # ---- Update Policy ----
    for _ in range(10):
        mean_action, _ = ac(obs_tensor)
        dist = torch.distributions.Normal(mean_action, 0.2)
        logp = dist.log_prob(act_tensor).sum(axis=-1)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -torch.min(ratio * adv, clip_adv).mean()
        pi_opt.zero_grad()
        loss_pi.backward()
        pi_opt.step()

    # ---- Update Value Function ----
    for _ in range(10):
        _, val = ac(obs_tensor)
        loss_v = ((val.squeeze() - ret) ** 2).mean()
        vf_opt.zero_grad()
        loss_v.backward()
        vf_opt.step()

    print(f"Epoch {epoch} complete.")
torch.save(ac.state_dict(), "ppo_ant_policy.pt")
env.close()
