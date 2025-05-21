"""Train TD-MPC2 on a ManiSkill environment."""
import gymnasium as gym
from collections import deque
import mani_skill
import tyro
from dataclasses import dataclass

from tdmpc2 import TD_MPC2


@dataclass
class TrainArgs:
    """Command line arguments for training TD-MPC2."""
    env_id: str = "PickCube-v1"
    episodes: int = 50
    steps_per_episode: int = 200
    num_envs: int = 32


def run(args: TrainArgs):
    """Run a short training loop on the specified ManiSkill task."""
    env = gym.make(args.env_id, num_envs=args.num_envs)
    # breakpoint()
    agent = TD_MPC2(
        env.observation_space.shape[-1],
        env.action_space.shape[-1],
        env.action_space.high[0][0], # action upper bound
    )

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_return = 0
        for t in range(args.steps_per_episode):
            # 1. Query policy for action.
            act = agent.act(obs)
            # 2. Step the environment.
            next_obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated | truncated
            # 3. Store transition in the replay buffer.
            agent.replay.store(obs, act, rew, next_obs, done)
            obs = next_obs
            ep_return += rew.item()
            # Start updating once enough data has been collected.
            if agent.replay.size > 1000:
                agent.update(args.num_envs)
            if done:
                break
        returns.append(ep_return)
        avg_ret = sum(returns[-10:]) / len(returns[-10:])
        # Print recent performance for monitoring progress.
        print(
            f"Episode {ep}: return={ep_return:.1f}, avg_return={avg_ret:.1f}")
    env.close()


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    run(args)
