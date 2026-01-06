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

    # Load model - Force CPU device (optimal for MlpPolicy)
    print("Using CPU device for evaluation (optimal for this model)")
    model = PPO.load(model_path, device='cpu')

    # Create evaluation environment (single env like EvalCallback)
    render_mode = "human" if render else None
    env = create_normalized_env(n_envs=1, render_mode=render_mode, training=False)

    # Try to load normalization statistics
    # Support multiple naming conventions:
    # 1. EvalCallback: {model_dir}/vec_normalize.pkl
    # 2. CheckpointCallback old: {model_name}_vec_normalize.pkl
    # 3. CheckpointCallback new: {prefix}_vecnormalize_{steps}_steps.pkl

    import re
    import glob

    model_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path).replace('.zip', '')

    vec_normalize_paths = []

    # Format 1: Directory format (final models, eval callback)
    vec_normalize_paths.append(os.path.join(model_dir, 'vec_normalize.pkl'))

    # Format 2: Old checkpoint format
    vec_normalize_paths.append(os.path.join(model_dir, f'{model_filename}_vec_normalize.pkl'))

    # Format 3: CheckpointCallback format - extract steps and reconstruct name
    # Example: ppo_pickplace_400000_steps.zip -> ppo_pickplace_vecnormalize_400000_steps.pkl
    steps_match = re.search(r'_(\d+)_steps$', model_filename)
    if steps_match:
        steps = steps_match.group(1)
        prefix = model_filename.split(f'_{steps}_steps')[0]
        checkpoint_format = os.path.join(model_dir, f'{prefix}_vecnormalize_{steps}_steps.pkl')
        vec_normalize_paths.append(checkpoint_format)

    # Format 4: Fallback - search for any vecnormalize file in directory
    vecnorm_files = glob.glob(os.path.join(model_dir, '*vecnormalize*.pkl'))
    vec_normalize_paths.extend(vecnorm_files)

    # Remove duplicates while preserving order
    vec_normalize_paths = list(dict.fromkeys(vec_normalize_paths))

    vec_normalize_loaded = False
    for vec_normalize_path in vec_normalize_paths:
        try:
            if os.path.exists(vec_normalize_path):
                env = VecNormalize.load(vec_normalize_path, env)
                env.training = False
                env.norm_reward = False
                print(f"Loaded normalization stats from: {vec_normalize_path}")
                vec_normalize_loaded = True
                break
        except (FileNotFoundError, Exception) as e:
            continue

    if not vec_normalize_loaded:
        print("Warning: Normalization stats not found in any expected location")
        print(f"Searched paths:")
        for path in vec_normalize_paths[:5]:  # Show first 5 paths
            print(f"  - {path}")
        print("Continuing without normalization - results may be inaccurate")

    # Evaluation loop (matches EvalCallback behavior)
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    print(f"\nEvaluating for {n_episodes} episodes...")
    print("-" * 50)

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Predict action (deterministic)
            action, _states = model.predict(obs, deterministic=True)

            # Step environment - handle both old (4-value) and new (5-value) Gym API
            step_result = env.step(action)

            if len(step_result) == 5:
                # New Gymnasium API: (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = step_result
                done = terminated[0] or truncated[0]
            else:
                # Old Gym API: (obs, reward, done, info)
                obs, reward, done_array, info = step_result
                done = done_array[0]

            episode_reward += reward[0]
            episode_length += 1

        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if info[0].get('is_success', False):
            success_count += 1

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
