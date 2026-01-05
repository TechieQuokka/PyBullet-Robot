"""
Utility functions for training and evaluation
"""

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from env.pick_place_env import PickPlaceEnv
from config import training_config


def make_env(render_mode: str = None) -> gym.Env:
    """
    Create and wrap the environment

    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)

    Returns:
        gym.Env: Monitored environment
    """
    env = PickPlaceEnv(render_mode=render_mode)
    env = Monitor(env)
    return env


def create_normalized_env(
    n_envs: int = 1,
    render_mode: str = None,
    training: bool = True
) -> VecNormalize:
    """
    Create vectorized and normalized environment with multiprocessing

    Args:
        n_envs: Number of parallel environments (runs in separate processes)
        render_mode: Rendering mode for the environment
        training: Whether this is for training (affects reward normalization)

    Returns:
        VecNormalize: Normalized vectorized environment
    """
    # Use SubprocVecEnv for true parallelization across CPU cores
    env = SubprocVecEnv([lambda: make_env(render_mode) for _ in range(n_envs)])

    env = VecNormalize(
        env,
        norm_obs=training_config.normalize_obs,
        norm_reward=training_config.normalize_reward if training else False,
        clip_obs=training_config.clip_obs,
        clip_reward=training_config.clip_reward,
        training=training
    )

    return env
