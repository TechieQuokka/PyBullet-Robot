"""
Test environment functionality
"""

import sys
import os
from typing import NoReturn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.pick_place_env import PickPlaceEnv
import numpy as np


def test_basic_functionality() -> None:
    """Test basic environment operations"""
    print("Testing environment basic functionality...")
    print("-" * 50)

    env = PickPlaceEnv(render_mode=None)

    # Test reset
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    print(f"✓ Reset works, observation shape: {obs.shape}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, (float, np.floating)), "Reward should be float"
    print(f"✓ Step works, reward: {reward:.3f}")

    # Test episode
    total_reward = 0
    steps = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    print(f"✓ Episode completed in {steps} steps, total reward: {total_reward:.3f}")

    env.close()
    print("\n✓ All basic tests passed!")


def test_reward_function() -> None:
    """Test reward function components"""
    print("\nTesting reward function...")
    print("-" * 50)

    env = PickPlaceEnv(render_mode=None)
    env.reset()

    # Test time penalty
    _, reward, _, _, _ = env.step(np.zeros(7))
    assert reward < 0, "Time penalty should make reward negative"
    print(f"✓ Time penalty works: {reward:.3f}")

    # Test distance-based reward
    for _ in range(10):
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        # Reward should be reasonable (not NaN or inf)
        assert not np.isnan(reward) and not np.isinf(reward), "Reward should be valid"

    print(f"✓ Distance-based rewards work")

    env.close()
    print("✓ Reward function tests passed!")


def test_observation_space() -> None:
    """Test observation space"""
    print("\nTesting observation space...")
    print("-" * 50)

    env = PickPlaceEnv(render_mode=None)
    obs, _ = env.reset()

    # Check observation shape
    assert obs.shape == (17,), f"Expected shape (17,), got {obs.shape}"
    print(f"✓ Observation shape correct: {obs.shape}")

    # Check observation values are valid
    assert not np.isnan(obs).any(), "Observation contains NaN"
    assert not np.isinf(obs).any(), "Observation contains Inf"
    print(f"✓ Observation values are valid")

    # Joint positions (first 7 values)
    joint_positions = obs[:7]
    print(f"  Joint positions range: [{joint_positions.min():.2f}, {joint_positions.max():.2f}]")

    # Joint velocities (next 7 values)
    joint_velocities = obs[7:14]
    print(f"  Joint velocities range: [{joint_velocities.min():.2f}, {joint_velocities.max():.2f}]")

    # Cube position (last 3 values)
    cube_position = obs[14:17]
    print(f"  Cube position: [{cube_position[0]:.3f}, {cube_position[1]:.3f}, {cube_position[2]:.3f}]")

    env.close()
    print("✓ Observation space tests passed!")


def test_action_space() -> None:
    """Test action space"""
    print("\nTesting action space...")
    print("-" * 50)

    env = PickPlaceEnv(render_mode=None)
    env.reset()

    # Check action space shape
    assert env.action_space.shape == (7,), f"Expected shape (7,), got {env.action_space.shape}"
    print(f"✓ Action space shape correct: {env.action_space.shape}")

    # Test action bounds
    assert env.action_space.low.min() == -1.0, "Action space low should be -1.0"
    assert env.action_space.high.max() == 1.0, "Action space high should be 1.0"
    print(f"✓ Action space bounds correct: [-1.0, 1.0]")

    # Test applying actions
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (17,), "Observation shape changed after step"

    print(f"✓ Action application works correctly")

    env.close()
    print("✓ Action space tests passed!")


def run_all_tests() -> None:
    """Run all tests"""
    print("=" * 50)
    print("PyBullet Pick-and-Place Environment Tests")
    print("=" * 50)
    print()

    try:
        test_basic_functionality()
        test_reward_function()
        test_observation_space()
        test_action_space()

        print("\n" + "=" * 50)
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)

    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ TEST FAILED: {str(e)}")
        print("=" * 50)
        raise


if __name__ == "__main__":
    run_all_tests()
