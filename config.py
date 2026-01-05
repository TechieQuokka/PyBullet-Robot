"""
Central configuration file for all hyperparameters
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters - Optimized for parallel environments"""
    learning_rate: float = 1e-4  # Reduced for more stable learning
    n_steps: int = 2048  # Per environment (total buffer = 2048 * 8 = 16,384)
    batch_size: int = 1024  # Scaled with n_envs (128 * 8) for efficient updates
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.02  # Increased for more exploration
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
    max_episode_steps: int = 200

    # Robot
    max_joint_velocity: float = 0.5
    joint_force: float = 150

    # Objects
    cube_size: float = 0.05
    cube_mass: float = 0.1

    # Task
    cube_start_pos: List[float] = None
    target_pos: List[float] = None
    success_distance: float = 0.05
    gripper_attach_distance: float = 0.03

    def __post_init__(self):
        if self.cube_start_pos is None:
            self.cube_start_pos = [0.5, 0.0, 0.05]
        if self.target_pos is None:
            self.target_pos = [0.5, 0.2, 0.05]


@dataclass
class RewardConfig:
    """Reward function parameters - Improved dense reward shaping"""
    # Dense rewards for continuous feedback
    distance_reward_scale: float = 10.0  # Scale for distance-based rewards
    reach_reward_coef: float = 1.0  # Coefficient for reaching phase
    lift_reward_coef: float = 2.0  # Coefficient for lifting cube
    place_reward_coef: float = 1.5  # Coefficient for placing phase

    # Milestone bonuses for major achievements
    approach_bonus: float = 10.0  # When gripper gets close to cube
    grasp_bonus: float = 50.0  # When cube is grasped
    lift_bonus: float = 30.0  # When cube is lifted above table
    success_bonus: float = 200.0  # When cube reaches target

    # Penalties to discourage bad behavior
    time_penalty: float = -0.01  # Small penalty per step
    collision_penalty: float = -1.0  # Collision with environment
    drop_penalty: float = -10.0  # Dropping the cube
    distance_penalty: float = -0.001  # Very small penalty for being far

    # Thresholds for milestone detection
    approach_threshold: float = 0.1  # Distance to trigger approach bonus
    lift_height_threshold: float = 0.1  # Height to trigger lift bonus


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
