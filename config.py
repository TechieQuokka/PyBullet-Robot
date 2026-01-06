"""
Central configuration file for all hyperparameters
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters - Optimized for parallel environments"""
    learning_rate: float = 3e-4  # Higher learning rate for faster learning
    n_steps: int = 2048  # Per environment (total buffer = 2048 * 8 = 16,384)
    batch_size: int = 1024  # Scaled with n_envs (128 * 8) for efficient updates
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01  # Reduced from 0.1 to prevent exploration explosion
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None


@dataclass
class NetworkConfig:
    """Neural network architecture - Larger network for better capacity"""
    policy_net_arch: List[int] = None
    value_net_arch: List[int] = None
    activation_fn: str = "relu"  # ReLU often works better for robotics

    def __post_init__(self):
        if self.policy_net_arch is None:
            self.policy_net_arch = [256, 128]  # Increased from [64, 64]
        if self.value_net_arch is None:
            self.value_net_arch = [256, 128]  # Increased from [64, 64]


@dataclass
class EnvConfig:
    """Environment configuration"""
    # Simulation
    timestep: float = 1/240
    gravity: float = -9.81
    max_episode_steps: int = 1000  # Very long episodes for learning

    # Robot
    max_joint_velocity: float = 2.0  # Much faster movement
    joint_force: float = 200  # Stronger joints

    # Objects
    cube_size: float = 0.05
    cube_mass: float = 0.05  # Lighter cube

    # Task
    cube_start_pos: List[float] = None
    target_pos: List[float] = None
    success_distance: float = 0.08  # More lenient success criteria
    gripper_attach_distance: float = 0.08  # Much easier to grasp

    def __post_init__(self):
        if self.cube_start_pos is None:
            self.cube_start_pos = [0.35, 0.0, 0.05]  # Much closer to robot
        if self.target_pos is None:
            self.target_pos = [0.35, 0.08, 0.05]  # Very short distance (8cm)


@dataclass
class RewardConfig:
    """Reward function parameters - Improved dense reward shaping"""
    # Dense rewards for continuous feedback
    distance_reward_scale: float = 50.0  # Very strong learning signal
    reach_reward_coef: float = 3.0  # High reward for approaching
    lift_reward_coef: float = 4.0  # High reward for lifting
    place_reward_coef: float = 3.0  # High reward for placing

    # Milestone bonuses for major achievements
    approach_bonus: float = 50.0  # Big bonus when getting close
    grasp_bonus: float = 100.0  # Big bonus for grasping
    lift_bonus: float = 80.0  # Big bonus for lifting
    success_bonus: float = 500.0  # Huge success bonus

    # Penalties to discourage bad behavior
    time_penalty: float = -0.002  # Minimal time penalty
    collision_penalty: float = -0.5  # Reduced collision penalty
    drop_penalty: float = -5.0  # Reduced drop penalty
    distance_penalty: float = 0.0  # No distance penalty

    # Thresholds for milestone detection
    approach_threshold: float = 0.15  # Easier to trigger approach bonus
    lift_height_threshold: float = 0.08  # Lower lift threshold


@dataclass
class TrainingConfig:
    """Training configuration - Optimized for i7-6700K (4C/8T with HT)"""
    total_timesteps: int = 500_000  # Increased for better convergence
    n_envs: int = 8  # Max out all 8 logical cores
    save_freq: int = 25_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    device: str = "cpu"  # CPU is faster for MlpPolicy (2-3x vs GPU)
    verbose: int = 1


@dataclass
class LoggingConfig:
    """Logging configuration"""
    tensorboard_log: str = "./logs/tensorboard/"
    tb_log_name: str = "PPO_PickPlace"
    log_interval: int = 10
    checkpoint_dir: str = "./models/checkpoints/"
    best_model_dir: str = "./models/best/"
    final_model_dir: str = "./models/final/"


# Create default configuration instances
ppo_config = PPOConfig()
network_config = NetworkConfig()
env_config = EnvConfig()
reward_config = RewardConfig()
training_config = TrainingConfig()
logging_config = LoggingConfig()
