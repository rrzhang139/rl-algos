import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco

# ---- Custom Ant Push Environment ----


class AntPushEnv(gym.Wrapper):
    def __init__(self):
        import os
        xml_path = os.path.join(os.getcwd(), "ant_push.xml")
        env = gym.make("Ant-v5",
                       xml_file=xml_path,
                       render_mode="human")
        super().__init__(env)
        self.initial_cube_pos = np.array([2.0, 0.0, 0.5])
        self.target_pos = np.array([5.0, 0.0, 0.5])

        # Extend observation space to include cube and target info
        # +4 for cube_pos_xy and target_relative_xy
        self.obs_dim = self.observation_space.shape[0] + 4

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Get the cube body ID

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

        # Set the target position - create a visual marker at the target location
        target_site_id = mujoco.mj_name2id(
            self.unwrapped.model,
            mujoco.mjtObj.mjOBJ_SITE,
            'target'
        )

        # Update the target site position to match our target position
        if target_site_id >= 0:  # Make sure the site exists
            self.unwrapped.model.site_pos[target_site_id] = self.target_pos

        # Call mj_forward to update the simulation with our changes
        mujoco.mj_forward(self.unwrapped.model, self.unwrapped.data)

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

        ant_pos = self._get_ant_position()
        cube_pos = self._get_cube_pos()

        # Approach cube
        ant_to_cube_dist = np.linalg.norm(ant_pos[:2] - cube_pos[:2])
        approach_reward = -ant_to_cube_dist  # Negative distance

        # # Phase 2: Maintain contact
        in_contact = ant_to_cube_dist < 0.7  # Threshold for "contact"
        contact_reward = 2.0 if in_contact else 0.0

        # Phase 3: Push to target
        cube_to_target_dist = np.linalg.norm(
            cube_pos[:2] - self.target_pos[:2])
        push_reward = -cube_to_target_dist  # Negative distance to target

        # Only reward pushing when in contact
        if in_contact:
            push_reward *= 2.0  # Amplify push reward when in contact
        else:
            push_reward = 0.0  # No push reward when not in contact

        # Success bonus
        success = cube_to_target_dist < 0.3
        success_reward = 10.0 if success else 0.0

        # Total reward with phase weighting
        total_reward = (
            approach_reward +   # Phase 1 weight
            1.5 * contact_reward +   # Phase 2 weight
            2.0 * push_reward +      # Phase 3 weight
            success_reward           # Success bonus
        )

        if success:
            print("Success! Cube reached target.")
            terminated = True

        return self._augment_observation(obs), total_reward, terminated, truncated, info

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
        return self.unwrapped.get_body_com("torso").copy()


# ---- Hyperparams ----
epochs = 1000
steps_per_epoch = 2048
gamma = 0.99
lam = 0.95
clip_ratio = 0.2
policy_lr = 3e-4
value_lr = 1e-3

# ---- PPO Actor-Critic with extended observation ----


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
env = AntPushEnv()
obs_dim = env.obs_dim
act_dim = env.action_space.shape[0]

ac = ActorCritic(obs_dim, act_dim)
pi_opt = optim.Adam(ac.actor.parameters(), lr=policy_lr)
vf_opt = optim.Adam(ac.critic.parameters(), lr=value_lr)

# ---- PPO Training Loop ----
for epoch in range(epochs):
    obs_buf, act_buf, adv_buf, ret_buf, logp_buf, dones_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    done = False
    ep_rews, values, logps = [], [], []
    t = 0
    # ---- Data Collection ----
    while t < steps_per_epoch:
        obs_t = torch.tensor(obs, dtype=torch.float32)
        mean_action, value = ac(obs_t)
        dist = torch.distributions.Normal(mean_action, 0.2)
        pre_squash_action = dist.rsample()
        action = torch.tanh(pre_squash_action)
        action = action * \
            torch.tensor(env.action_space.high, dtype=torch.float32)
        # print(type(action))
        logp = dist.log_prob(action).sum()

        next_obs, reward, terminated, truncated, _ = env.step(
            action.detach().numpy())
        done = terminated or truncated
        ep_rews.append(reward)
        values.append(value.item())
        logps.append(logp.item())
        dones_buf.append(done)

        obs_buf.append(obs)
        act_buf.append(action.detach().numpy())
        obs = next_obs
        t += 1

        if done:
            obs, _ = env.reset()

    # ---- Compute GAE-Lambda Advantages ----
    rewards = np.array(ep_rews)
    values = np.array(values + [0])
    dones = np.array(dones_buf)
    deltas = rewards + gamma * values[1:] - values[:-1]

    adv = []
    adv_t = 0
    for d, done in zip(reversed(deltas), reversed(dones)):
        if done:
            adv_t = 0
        adv_t = d + gamma * lam * adv_t
        adv.insert(0, adv_t)
    adv = torch.tensor(adv, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32)
    act_tensor = torch.tensor(np.array(act_buf), dtype=torch.float32)
    logp_old = torch.tensor(logps, dtype=torch.float32)
    ret = adv + torch.tensor(values[:-1], dtype=torch.float32)

    # ---- Update Policy ----
    for _ in range(10):
        breakpoint()
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

    # Calculate mean episode reward for this epoch
    mean_reward = np.mean(rewards)
    print(f"Epoch {epoch}, Mean Reward: {mean_reward:.3f}")

# Save the model
torch.save(ac.state_dict(), "ppo_ant_push_policy.pt")
env.close()
