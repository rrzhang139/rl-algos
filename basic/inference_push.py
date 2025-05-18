import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import mujoco

# ---- Custom Ant Push Environment ----


class AntPushEnv(gym.Wrapper):
    def __init__(self, render_mode="human"):
        import os
        xml_path = os.path.join(os.getcwd(), "ant_push.xml")
        env = gym.make("Ant-v5",
                       xml_file=xml_path,
                       render_mode=render_mode)
        super().__init__(env)
        self.initial_cube_pos = np.array([10.0, 0.0, 0.5])
        self.target_pos = np.array([3.0, 0.0, 0.5])

        # Extend observation space to include cube and target info
        # +4 for cube_pos_xy and target_relative_xy
        self.obs_dim = self.observation_space.shape[0] + 4

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Get the joint ID associated with the cube
        joint_id = mujoco.mj_name2id(
            self.unwrapped.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            'cube'
        )

        # Get the address of the joint in qpos
        qpos_addr = self.unwrapped.model.jnt_qposadr[joint_id]

        # Set the desired position
        self.unwrapped.data.qpos[qpos_addr:qpos_addr+3] = self.initial_cube_pos

        return self._augment_observation(obs), info

    def _get_cube_pos(self):
        cube_id = mujoco.mj_name2id(
            self.unwrapped.model,
            mujoco.mjtObj.mjOBJ_BODY,
            'cube'
        )
        return self.unwrapped.data.xpos[cube_id].copy()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Get actual cube position from physics
        cube_pos = self._get_cube_pos()

        # Calculate distance to target
        dist_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        push_reward = -dist_to_target

        # Success detection
        success = dist_to_target < 0.3
        if success:
            print("Success! Cube reached target.")
            terminated = True

        return self._augment_observation(obs), reward + push_reward, terminated, truncated, info

    def _augment_observation(self, obs):
        # Get actual cube position
        cube_pos = self._get_cube_pos()

        # Calculate vector from cube to target
        cube_to_target = self.target_pos - cube_pos

        # Add cube position and relative target position to observation
        extra_obs = np.concatenate([
            cube_pos[:2],          # x,y position of cube
            cube_to_target[:2],    # relative x,y vector to target
        ])

        return np.concatenate([obs, extra_obs])

    def _get_ant_position(self):
        # Get ant position from the underlying environment
        return self.unwrapped.get_body_com("torso").copy()

    def _get_ant_orientation(self):
        import mujoco
        # Get ant orientation from the underlying environment
        torso_idx = mujoco.mj_name2id(
            self.unwrapped.model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            'torso'
        )
        quat = self.unwrapped.data.xquat[torso_idx]
        # Calculate yaw from quaternion
        return np.arctan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                          1.0 - 2.0 * (quat[2]**2 + quat[3]**2))

    def render(self):
        # No need for custom rendering as cube and target are now part of the environment
        super().render()

        # Print status
        ant_pos = self._get_ant_position()
        cube_pos = self._get_cube_pos()
        print(f"Ant position: {ant_pos[:2]}")
        print(f"Cube position: {cube_pos[:2]}")
        print(
            f"Distance to target: {np.linalg.norm(cube_pos[:2] - self.target_pos[:2]):.2f}")


# ---- Actor-Critic Architecture ----
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


# ---- Setup ----
env = AntPushEnv(render_mode="human")
obs_dim = env.obs_dim
act_dim = env.action_space.shape[0]

# Initialize model and load weights
ac = ActorCritic(obs_dim, act_dim)
ac.load_state_dict(torch.load("ppo_ant_push_policy.pt"))
ac.eval()

# ---- Run inference ----
obs, _ = env.reset()
done = False
episode_reward = 0
step = 0
max_steps = 1000

# Store trajectory data for visualization
cube_positions = [env._get_cube_pos()[:2].copy()]
ant_positions = [env._get_ant_position()[:2].copy()]

print("Starting inference...")
print(f"Target position: {env.target_pos[:2]}")

while not done and step < max_steps:
    # Convert observation to tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    # Get action from model
    with torch.no_grad():
        mean_action, _ = ac(obs_tensor)

    # Take action in environment
    action = mean_action.numpy()
    next_obs, reward, terminated, truncated, _ = env.step(action)

    # Check if episode is done
    done = terminated or truncated
    episode_reward += reward

    # Record positions
    # Record positions
    cube_positions.append(env._get_cube_pos()[:2].copy())
    ant_positions.append(env._get_ant_position()[:2].copy())
    # Update observation
    obs = next_obs
    step += 1

    # Optional: print every 50 steps
    if step % 50 == 0:
        cube_pos = env._get_cube_pos()
        print(
            f"Step {step}, Distance to target: {np.linalg.norm(cube_pos[:2] - env.target_pos[:2]):.2f}")

print(f"Episode finished after {step} steps with reward {episode_reward:.2f}")
env.close()

# Plot trajectory
fig, ax = plt.subplots(figsize=(10, 8))
cube_positions = np.array(cube_positions)
ant_positions = np.array(ant_positions)

ax.plot(ant_positions[:, 0], ant_positions[:, 1], 'b-', label='Ant')
ax.plot(cube_positions[:, 0], cube_positions[:, 1], 'r-', label='Cube')
ax.scatter([env.target_pos[0]], [env.target_pos[1]],
           color='g', marker='*', s=200, label='Target')
ax.scatter([1.0], [0.0], color='r', marker='o', s=100, label='Cube Start')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_title('Ant Push Trajectory')
ax.legend()
ax.grid(True)
ax.axis('equal')
plt.savefig('ant_push_trajectory.png')
plt.show()
