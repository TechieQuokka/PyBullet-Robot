"""
Main training script for Kuka Pick-and-Place task
"""

import os
from typing import Callable
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from config import (
    ppo_config,
    network_config,
    training_config,
    logging_config
)
from utils import create_normalized_env


class SaveVecNormalizeCallback(EvalCallback):
    """
    Custom EvalCallback that also saves VecNormalize statistics with best model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        # Call parent's _on_step (handles evaluation and best model saving)
        continue_training = super()._on_step()

        # After parent saves best model, also save VecNormalize if it was updated
        if self.best_mean_reward == self.last_mean_reward:  # Best model was just saved
            if isinstance(self.training_env, VecNormalize):
                vec_normalize_path = os.path.join(self.best_model_save_path, 'vec_normalize.pkl')
                self.training_env.save(vec_normalize_path)
                if self.verbose > 0:
                    print(f"Saved VecNormalize to {vec_normalize_path}")

        return continue_training


def create_directories() -> None:
    """Create necessary directories for training"""
    directories = [
        logging_config.checkpoint_dir,
        logging_config.best_model_dir,
        logging_config.final_model_dir,
        "./logs/eval/",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")


def setup_callbacks(eval_env: VecNormalize) -> CallbackList:
    """
    Setup training callbacks

    Args:
        eval_env: Separate environment for evaluation

    Returns:
        CallbackList: Combined callback list
    """
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config.save_freq,
        save_path=logging_config.checkpoint_dir,
        name_prefix='ppo_pickplace',
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # Evaluation callback - evaluate and save best model with VecNormalize
    eval_callback = SaveVecNormalizeCallback(
        eval_env,
        best_model_save_path=logging_config.best_model_dir,
        log_path='./logs/eval/',
        eval_freq=training_config.eval_freq,
        n_eval_episodes=training_config.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    return callback_list


def main() -> None:
    """Main training function"""

    print("=" * 50)
    print("PyBullet Kuka Pick-and-Place RL Training")
    print("=" * 50)

    # Create directories
    print("\nCreating directories...")
    create_directories()

    # Check device
    device = training_config.device
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    # Environment Setup
    print("\nCreating training environment...")
    train_env = create_normalized_env(n_envs=training_config.n_envs, training=True)

    print("Creating evaluation environment...")
    eval_env = create_normalized_env(n_envs=1, training=False)

    # Agent Setup
    print("\nCreating PPO agent...")

    # Get activation function
    if network_config.activation_fn == "tanh":
        activation_fn = torch.nn.Tanh
    elif network_config.activation_fn == "relu":
        activation_fn = torch.nn.ReLU
    else:
        activation_fn = torch.nn.Tanh

    policy_kwargs = dict(
        net_arch=dict(
            pi=network_config.policy_net_arch,
            vf=network_config.value_net_arch
        ),
        activation_fn=activation_fn
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=ppo_config.learning_rate,
        n_steps=ppo_config.n_steps,
        batch_size=ppo_config.batch_size,
        n_epochs=ppo_config.n_epochs,
        gamma=ppo_config.gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        clip_range_vf=ppo_config.clip_range_vf,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        target_kl=ppo_config.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=logging_config.tensorboard_log,
        verbose=training_config.verbose,
        device=device
    )

    # Callbacks
    callbacks = setup_callbacks(eval_env)

    # Training
    print(f"\nStarting training for {training_config.total_timesteps:,} timesteps...")
    print(f"Expected time: ~1 hour on RTX 3060")
    print("Monitor progress with: tensorboard --logdir=./logs/tensorboard/")
    print("-" * 50)

    try:
        model.learn(
            total_timesteps=training_config.total_timesteps,
            callback=callbacks,
            log_interval=logging_config.log_interval,
            tb_log_name=logging_config.tb_log_name,
            reset_num_timesteps=True,
            progress_bar=True
        )

        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("Training interrupted by user")
        print("=" * 50)

    finally:
        # Save Final Model
        print("\nSaving final model...")
        final_model_path = os.path.join(
            logging_config.final_model_dir,
            "ppo_pickplace_final"
        )
        model.save(final_model_path)

        # Save normalization stats
        vec_normalize_path = os.path.join(
            logging_config.final_model_dir,
            "vec_normalize.pkl"
        )
        train_env.save(vec_normalize_path)

        print(f"Model saved to: {final_model_path}.zip")
        print(f"Normalization stats saved to: {vec_normalize_path}")

        # Cleanup
        train_env.close()
        eval_env.close()

        print("\nTraining pipeline completed!")


if __name__ == "__main__":
    main()
