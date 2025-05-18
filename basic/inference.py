import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

# Define the same ActorCritic class


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        self.critic = nn.Sequential(  # must match training
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)


# Set up environment with GUI
env = gym.make("Ant-v5", render_mode="human")  # v5 is the updated version
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

# Initialize model and load weights
ac = ActorCritic(obs_dim, act_dim)
ac.load_state_dict(torch.load("ppo_ant_policy.pt"))
ac.eval()

obs, _ = env.reset()
for _ in range(1000):
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    mean_action = ac(obs_tensor)[0]
    action = mean_action.detach().numpy()
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
