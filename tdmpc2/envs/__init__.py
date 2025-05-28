"""Utility functions for creating ManiSkill environments."""

try:
    import gymnasium as gym
except ImportError:
    import gym


def make_env(env_id="PickCube-v1", **kwargs):
    """Create a ManiSkill environment.

    Parameters
    ----------
    env_id : str
        Name of the ManiSkill environment, e.g. ``"PickCube-v1"``.
    kwargs : any
        Additional arguments passed to ``gym.make``.
    """
    return gym.make(env_id, **kwargs)
