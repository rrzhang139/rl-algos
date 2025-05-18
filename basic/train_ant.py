#!/usr/bin/env python3
"""
Train MuJoCo Ant to reach a fixed red-sphere target with PPO.
Dependencies
------------
pip install mujoco==3.1.3 gymnasium==0.29.1 stable-baselines3==2.3.3
"""
import pathlib
import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

ROOT = pathlib.Path(__file__).resolve().parent
XML_PATH = ROOT / "ant_target.xml"          # ↓ see Section 2


class AntTargetEnv(gym.Env):
    metadata = {"render_mode": "human", "render_fps": 60}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.nq, self.nv = self.model.nq, self.model.nv

        # gym spaces
        act_low = self.model.actuator_ctrlrange[:, 0]
        act_high = self.model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        obs_high = np.inf * np.ones(self.nq + self.nv + 3, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high,
                                            obs_high, dtype=np.float32)

        target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_xyz = self.model.body_pos[target_id].copy()
        self.t = 0.0

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel,
                               self.target_xyz - self.data.qpos[:3]])

    def reset(self, seed=None, **_):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.t += self.model.opt.timestep
        obs = self._get_obs()

        # ---------- custom reward ----------
        # 1. forward progress toward target
        torso_xyz = self.data.qpos[:3]
        dist_old = np.linalg.norm(
            self.target_xyz - (torso_xyz - self.data.qvel[:3]*self.model.opt.timestep))
        dist_new = np.linalg.norm(self.target_xyz - torso_xyz)
        progress = dist_old - dist_new                       # >0 if we moved closer

        # 2. survival bonus & energy penalty
        alive = 1.0 if torso_xyz[2] > 0.2 else -1.0
        ctrl_cost = 0.005 * np.square(action).sum()
        reward = 3.0*progress + alive - ctrl_cost
        terminated = bool(dist_new < 0.3)        # reached target
        truncated = self.t > 30.0               # 30 s horizon
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.sync()

    def close(self): pass


if __name__ == "__main__":
    env = AntTargetEnv(render_mode="human")
    model = PPO("MlpPolicy", env, verbose=1, batch_size=2048,
                n_steps=8192, gamma=0.99, clip_range=0.2,
                ent_coef=0.0, learning_rate=3e-4)
    model.learn(total_timesteps=2_000_000)
    model.save("ant_target_ppo")
