"""
Minimal DQN Implementation using Stable Baselines 3
Environment: CartPole-v1
Author: Richard Zhang
Date: 2025-04-29

This script implements a basic DQN (Deep Q-Network) using Stable Baselines 3.
It demonstrates core concepts of DQN on the CartPole environment.
"""

import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import numpy as np

# =====================================
# CONSTANTS AND HYPERPARAMETERS
# =====================================
# Standard parameters from DQN literature
TOTAL_TIMESTEPS = 5000  # Total timesteps for training
LEARNING_RATE = 1e-4     # Original DQN paper uses 1e-4
BUFFER_SIZE = 10000      # Size of replay buffer
LEARNING_STARTS = 1000   # Timesteps before starting training
BATCH_SIZE = 32         # Size of batches for training
GAMMA = 0.99            # Discount factor
TRAIN_FREQ = 4          # Update the model every 4 steps
GRADIENT_STEPS = 1      # How many gradient steps after each rollout
TARGET_UPDATE_INTERVAL = 1000  # Update target network every 1000 steps

# Paths for saving
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# =====================================
# ENVIRONMENT SETUP
# =====================================


def make_env():
    """Create and wrap the CartPole environment"""
    env = gym.make("CartPole-v1", render_mode="human")
    env = Monitor(env, LOGS_DIR)  # Wrapper for logging
    return env

# =====================================
# MODEL SETUP
# =====================================


def create_dqn_model(env):
    """
    Create DQN model with standard parameters from literature
    Returns: DQN model instance
    """
    model = DQN(
        "MlpPolicy",  # Standard MLP policy for non-image inputs
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        verbose=1,
        tensorboard_log=LOGS_DIR
    )
    return model

# =====================================
# TRAINING AND EVALUATION
# =====================================


def train_and_evaluate():
    """Main training and evaluation loop"""
    # Create environment
    env = make_env()

    # Initialize model
    model = create_dqn_model(env)

    # Training
    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True
    )

    # Save the model
    model.save(f"{MODELS_DIR}/dqn_cartpole")

    # Evaluation
    print("Evaluating model...")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Plot learning curve
    plot_results()

    return model, env

# =====================================
# VISUALIZATION
# =====================================


def plot_results():
    """Plot the training curve"""
    # Load monitoring data
    with open(f"{LOGS_DIR}/monitor.csv", "r") as f:
        # Skip first two lines (headers)
        f.readline()
        f.readline()

        # Read rewards
        rewards = []
        for line in f:
            r = float(line.split(',')[0])  # First column is reward
            rewards.append(r)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"{LOGS_DIR}/learning_curve.png")
    plt.close()


# =====================================
# MAIN EXECUTION
# =====================================
if __name__ == "__main__":
    model, env = train_and_evaluate()

    # Optional: Watch trained agent
    print("\nWatching trained agent...")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        try:
            env.render()
        except:
            print("Rendering not supported in this environment")
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
