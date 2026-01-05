"""
Evaluation script for trained models
"""

import numpy as np
import argparse
import os
from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from utils import create_normalized_env


def evaluate_model(model_path: str, n_episodes: int = 100, render: bool = False) -> Dict[str, Any]:
    """
    Evaluate trained model

    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes

    Returns:
        dict: Evaluation metrics
    """
    print(f"Loading model from: {model_path}")

    # Load model
    model = PPO.load(model_path)

    # Create evaluation environment
    render_mode = "human" if render else None
    env = create_normalized_env(n_envs=1, render_mode=render_mode, training=False)

    # Try to load normalization statistics
    vec_normalize_paths = [
        model_path.replace('.zip', '') + '_vec_normalize.pkl',  # CheckpointCallback format
        os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl'),  # Directory format
    ]

    vec_normalize_loaded = False
    for vec_normalize_path in vec_normalize_paths:
        try:
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            print(f"Loaded normalization stats from: {vec_normalize_path}")
            vec_normalize_loaded = True
            break
        except FileNotFoundError:
            continue

    if not vec_normalize_loaded:
        print("Warning: Normalization stats not found in any expected location")
        print(f"Searched paths: {vec_normalize_paths}")
        print("Continuing without normalization - results may be inaccurate")

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print(f"\nEvaluating for {n_episodes} episodes...")
    print("-" * 50)

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while True:
            # Predict action (deterministic)
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward[0]
            episode_length += 1

            if done:
                if info[0].get('is_success', False):
                    success_count += 1
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Reward: {episode_reward:.2f}, "
                  f"Length: {episode_length}, "
                  f"Success: {success_count}/{episode + 1}")

    # Calculate statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': (success_count / n_episodes) * 100,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'total_episodes': n_episodes,
        'successful_episodes': success_count
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Episodes:    {results['total_episodes']}")
    print(f"Successful:        {results['successful_episodes']}")
    print(f"Success Rate:      {results['success_rate']:.1f}%")
    print(f"Mean Reward:       {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length:       {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Reward Range:      [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("=" * 50)

    env.close()
    return results


def main() -> None:
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument(
        '--model',
        type=str,
        default='./models/final/ppo_pickplace_final.zip',
        help='Path to model'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes'
    )
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render
    )


if __name__ == "__main__":
    main()
