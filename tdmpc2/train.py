"""Train TD-MPC2 on a ManiSkill environment."""
import gymnasium as gym
from collections import deque
import mani_skill

from tdmpc2 import TD_MPC2


def make_env(env_name):
    return gym.make(env_name)


def run(env_name="PickCube-v1", episodes=50, steps_per_episode=200):
    """Run a short training loop on the specified ManiSkill task."""
    env = make_env(env_name)
    agent = TD_MPC2(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.high[0],
    )

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_return = 0
        for t in range(steps_per_episode):
            # 1. Query policy for action.
            act = agent.act(obs)
            # 2. Step the environment.
            next_obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            # 3. Store transition in the replay buffer.
            agent.replay.store(obs, act, rew, next_obs, float(done))
            obs = next_obs
            ep_return += rew
            # Start updating once enough data has been collected.
            if agent.replay.size > 1000:
                agent.update()
            if done:
                break
        returns.append(ep_return)
        avg_ret = sum(returns[-10:]) / len(returns[-10:])
        # Print recent performance for monitoring progress.
        print(
            f"Episode {ep}: return={ep_return:.1f}, avg_return={avg_ret:.1f}")
    env.close()


if __name__ == "__main__":
    run()
