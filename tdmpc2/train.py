"""Train TD-MPC2 on a ManiSkill environment."""
import gymnasium as gym
from collections import deque
import mani_skill
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import tyro
import torch
import numpy as np
from dataclasses import dataclass

from tdmpc2 import TD_MPC2


@dataclass
class TrainArgs:
    """Command line arguments for training TD-MPC2."""
    env_id: str = "PushCube-v1"
    episodes: int = 50
    latent_dim: int = 512
    steps_per_episode: int = 200
    num_envs: int = 256
    horizon: int = 3
    gamma: float = 0.99
    mpc: bool = True


def run(args: TrainArgs):
    """Run a short training loop on the specified ManiSkill task."""
    env = gym.make(args.env_id, num_envs=args.num_envs)
    env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
    print("Training configuration:")
    for field in args.__dataclass_fields__:
        print(f"  {field}: {getattr(args, field)}")
    agent = TD_MPC2(
        env.observation_space.shape[-1],
        env.action_space.shape[-1],
        args.latent_dim,
        env.action_space.high[0][0], # action upper bound
        mpc=args.mpc,
        horizon=args.horizon,
        gamma=args.gamma,
    )

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_return = 0
        update_stats = None
        for t in range(args.steps_per_episode):
            act = agent.act(obs)
            next_obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated | truncated
            for i in range(args.num_envs):
                agent.replay.store(
                    obs[i].cpu().numpy(),
                    act[i].cpu().numpy(),
                    rew[i].cpu().numpy().item(),
                    next_obs[i].cpu().numpy(),
                    done[i].cpu().numpy().item()
                )
            obs = next_obs
            ep_return += rew.sum().item()
            if agent.replay.size > 1000:
                update_stats = agent.update(args.num_envs)
        returns.append(ep_return)
        avg_ret = sum(returns[-10:]) / len(returns[-10:])
        print(f"Episode {ep}: return={ep_return:.1f}, avg_return={avg_ret:.1f}")
        if update_stats is not None:
            print("  Update stats:", ", ".join(f"{k}={v:.4f}" for k, v in update_stats.items()))
    env.close()


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    run(args)
