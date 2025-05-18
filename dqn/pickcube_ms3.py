"""
PPO implementation for PickCube environment using ManiSkill3 and Stable-Baselines3
"""

import os
import numpy as np
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import mani_skill
from mani_skill.utils.visualization import images_to_video
import gymnasium as gym

# Create directories for logs and models
log_dir = "logs"
model_dir = "models"
video_dir = "logs"
epochs = 1
total_timesteps = 50000
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


def make_env():
    """Create and wrap PickCube environment with rendering"""
    env = gym.make('PickCube-v1',
                   obs_mode='state',
                   #    reward_mode='sparse',
                   render_mode='rgb_array')
    # env = Monitor(env, log_dir)
    return env


def create_model(env):
    """Create model with appropriate hyperparameters"""
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        batch_size=32,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir
    )
    return model


def train_and_evaluate(model):
    """Main training and evaluation loop"""

    print("Starting training...")
    for epoch in range(epochs):
        print(f"=========Epoch {epoch}=========")
        try:
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving model...")

    # Save the model
    model.save(f"{model_dir}/pickcube")

    # Plot learning curve
    plot_results()


def plot_results():
    """Plot training metrics"""
    import pandas as pd

    # Load monitor file
    monitor_data = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot rewards
    ax1.plot(monitor_data['r'], label='Episode Reward')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()

    # Plot episode lengths
    ax2.plot(monitor_data['l'], label='Episode Length', color='orange')
    ax2.set_title('Episode Lengths')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{log_dir}/learning_curves.png")
    plt.close()


def evaluate_model(model, env, num_episodes=5):
    """Evaluate trained model with rendering"""
    rewards = []
    for episode in range(num_episodes):
        images = []
        images.append(env.render().cpu().numpy()[0])
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward.cpu().numpy()[0]
            done = terminated or truncated
            images.append(env.render().cpu().numpy()[0])

        rewards.append(episode_reward)
        images_to_video(images, output_dir=video_dir,
                        video_name="example", fps=20)
        print(f"Episode {episode + 1} reward: {episode_reward:.2f}")

    print(
        f"\nMean reward over {num_episodes} episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == "__main__":
    # Create environment
    env = make_env()

    # Initialize model
    model = create_model(env)

    # Train the model
    train_and_evaluate(model)

    # Evaluate the trained model
    print("\nEvaluating trained model...")
    evaluate_model(model, env)

    # Clean up
    env.close()
