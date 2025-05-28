"""Minimal policy sequencing example on PegInsertionSide-v1."""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import numpy as np
from dataclasses import dataclass
import tyro


def make_env(env_id: str):
    env = gym.make(env_id, obs_mode="state", control_mode="pd_joint_delta_pos")
    return FlattenActionSpaceWrapper(env)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        mean = self.net(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def act(self, obs: np.ndarray):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        mean, std = self.forward(obs_t)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action.detach().numpy(), log_prob


@dataclass
class Args:
    env_id: str = "PegInsertionSide-v1"
    episodes_stage1: int = 10
    episodes_stage2: int = 10
    steps_per_episode: int = 200
    lr: float = 3e-4
    gamma: float = 0.99


def train_policy(env, policy, optimizer, episodes, steps, gamma):
    for ep in range(episodes):
        obs, _ = env.reset()
        log_probs = []
        rewards = []
        for _ in range(steps):
            action, log_prob = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            obs = next_obs
            if terminated or truncated:
                break
        # compute return
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ep_ret = sum(rewards)
        print(f"Episode {ep}: return={ep_ret:.2f}")


def run(args: Args):
    env = make_env(args.env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Stage 1
    policy1 = GaussianPolicy(obs_dim, act_dim)
    optim1 = optim.Adam(policy1.parameters(), lr=args.lr)
    print("Training stage 1 policy...")
    train_policy(env, policy1, optim1, args.episodes_stage1, args.steps_per_episode, args.gamma)
    torch.save(policy1.state_dict(), "stage1_policy.pth")

    # Stage 2 - start from stage 1 weights
    policy2 = GaussianPolicy(obs_dim, act_dim)
    policy2.load_state_dict(policy1.state_dict())
    optim2 = optim.Adam(policy2.parameters(), lr=args.lr)
    print("Training stage 2 policy starting from stage 1...")
    train_policy(env, policy2, optim2, args.episodes_stage2, args.steps_per_episode, args.gamma)
    torch.save(policy2.state_dict(), "stage2_policy.pth")
    env.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    run(args)
