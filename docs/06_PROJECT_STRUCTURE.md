# Project Structure and Implementation Guide

## Complete Directory Structure

```
robot/
├── docs/                           # Documentation (this directory)
│   ├── 00_ARCHITECTURE_OVERVIEW.md
│   ├── 01_ENVIRONMENT_DESIGN.md
│   ├── 02_RL_ALGORITHM_DESIGN.md
│   ├── 03_REWARD_DESIGN.md
│   ├── 04_TRAINING_PIPELINE.md
│   ├── 05_CONFIG_HYPERPARAMETERS.md
│   └── 06_PROJECT_STRUCTURE.md
│
├── env/                            # Custom Gym environment
│   ├── __init__.py
│   └── pick_place_env.py          # Main environment implementation
│
├── models/                         # Trained models storage
│   ├── checkpoints/               # Periodic training checkpoints
│   │   ├── ppo_pickplace_10000_steps.zip
│   │   ├── ppo_pickplace_20000_steps.zip
│   │   └── ...
│   ├── best/                      # Best performing model
│   │   └── best_model.zip
│   └── final/                     # Final trained model
│       ├── ppo_pickplace_final.zip
│       └── vec_normalize.pkl
│
├── logs/                          # Training logs
│   ├── tensorboard/               # TensorBoard logs
│   │   └── PPO_PickPlace_1/
│   │       └── events.out.tfevents...
│   └── eval/                      # Evaluation logs
│       ├── evaluations.npz
│       └── monitor.csv
│
├── urdf/                          # Robot model files
│   └── kuka_iiwa/                 # Kuka robot URDF
│       ├── model.urdf
│       ├── meshes/
│       └── ...
│
├── scripts/                       # Utility scripts
│   ├── download_urdf.py          # Download robot models
│   ├── test_env.py               # Environment testing
│   └── visualize_policy.py       # Policy visualization
│
├── config.py                      # Central configuration file
├── train.py                       # Main training script
├── test.py                        # Evaluation script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project overview
```

## File-by-File Implementation Guide

### 1. requirements.txt
```txt
# Core RL dependencies
gymnasium>=0.29.0
stable-baselines3>=2.0.0
torch>=2.0.0
tensorboard>=2.14.0

# Physics simulation
pybullet>=3.2.5

# Utilities
numpy>=1.24.0
matplotlib>=3.7.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

### 2. config.py
See [05_CONFIG_HYPERPARAMETERS.md](./05_CONFIG_HYPERPARAMETERS.md) for complete implementation.

**Key exports:**
```python
from config import (
    ppo_config,
    network_config,
    env_config,
    reward_config,
    training_config,
    logging_config
)
```

### 3. env/__init__.py
```python
"""
Custom Gym environments
"""

from env.pick_place_env import PickPlaceEnv

__all__ = ['PickPlaceEnv']
```

### 4. env/pick_place_env.py
See [01_ENVIRONMENT_DESIGN.md](./01_ENVIRONMENT_DESIGN.md) for complete implementation.

**Class structure:**
```python
class PickPlaceEnv(gymnasium.Env):
    """Kuka iiwa pick-and-place environment"""

    def __init__(self, render_mode=None, **kwargs):
        """Initialize environment"""

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""

    def step(self, action):
        """Execute action and return new state"""

    def render(self):
        """Render environment (if applicable)"""

    def close(self):
        """Clean up resources"""

    # Private helper methods
    def _setup_simulation(self): ...
    def _get_obs(self): ...
    def _apply_action(self, action): ...
    def _compute_reward(self): ...
    def _check_success(self): ...
```

### 5. train.py
See [04_TRAINING_PIPELINE.md](./04_TRAINING_PIPELINE.md) for complete implementation.

**Main components:**
```python
def make_env():
    """Create environment instance"""

def create_vectorized_env(n_envs=1):
    """Create vectorized environment"""

def setup_callbacks(eval_env):
    """Setup training callbacks"""

def main():
    """Main training function"""
    # 1. Create directories
    # 2. Setup environments
    # 3. Create PPO agent
    # 4. Setup callbacks
    # 5. Train
    # 6. Save final model
```

**Usage:**
```bash
python train.py
```

### 6. test.py
```python
"""
Evaluation script for trained models
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.pick_place_env import PickPlaceEnv


def evaluate_model(model_path, n_episodes=100, render=False):
    """
    Evaluate trained model

    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes

    Returns:
        dict: Evaluation metrics
    """
    # Load model
    model = PPO.load(model_path)

    # Create evaluation environment
    render_mode = "human" if render else None
    env = PickPlaceEnv(render_mode=render_mode)
    env = DummyVecEnv([lambda: env])

    # Load normalization statistics
    try:
        env = VecNormalize.load(
            model_path.replace('.zip', '_vec_normalize.pkl'),
            env
        )
        env.training = False  # Don't update stats during eval
        env.norm_reward = False  # Don't normalize rewards during eval
    except FileNotFoundError:
        print("Warning: Normalization stats not found")

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # Predict action (deterministic)
            action, _states = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            episode_length += 1

            if done:
                if info[0].get('is_success', False):
                    success_count += 1
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")

    # Calculate statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': (success_count / n_episodes) * 100,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes:      {n_episodes}")
    print(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Length:   {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate:  {results['success_rate']:.1f}%")
    print(f"Reward Range:  [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
    print("=" * 50)

    env.close()
    return results


def main():
    """Main evaluation function"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, default='./models/final/ppo_pickplace_final.zip',
                        help='Path to model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render episodes')
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render
    )


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Evaluate with rendering
python test.py --model ./models/best/best_model.zip --episodes 10 --render

# Evaluate without rendering (faster)
python test.py --model ./models/final/ppo_pickplace_final.zip --episodes 100
```

### 7. scripts/download_urdf.py
```python
"""
Download Kuka iiwa URDF models
"""

import os
import urllib.request
import zipfile


def download_kuka_urdf():
    """Download Kuka iiwa URDF from PyBullet data"""

    # PyBullet already includes URDF files
    # Just copy from pybullet_data package
    import pybullet_data

    data_path = pybullet_data.getDataPath()
    kuka_path = os.path.join(data_path, "kuka_iiwa")

    print(f"PyBullet data path: {data_path}")
    print(f"Kuka URDF path: {kuka_path}")

    # Create symlink or copy
    target_path = "./urdf/kuka_iiwa"
    os.makedirs("./urdf", exist_ok=True)

    if not os.path.exists(target_path):
        import shutil
        shutil.copytree(kuka_path, target_path)
        print(f"Copied Kuka URDF to {target_path}")
    else:
        print(f"Kuka URDF already exists at {target_path}")


if __name__ == "__main__":
    download_kuka_urdf()
```

**Usage:**
```bash
python scripts/download_urdf.py
```

### 8. scripts/test_env.py
```python
"""
Test environment functionality
"""

from env.pick_place_env import PickPlaceEnv
import numpy as np


def test_basic_functionality():
    """Test basic environment operations"""
    print("Testing environment basic functionality...")

    env = PickPlaceEnv(render_mode=None)

    # Test reset
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch"
    print(f"✓ Reset works, observation shape: {obs.shape}")

    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float), "Reward should be float"
    print(f"✓ Step works, reward: {reward:.3f}")

    # Test episode
    total_reward = 0
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"✓ Episode completed, total reward: {total_reward:.3f}")

    env.close()
    print("\n✓ All basic tests passed!")


def test_reward_function():
    """Test reward function components"""
    print("\nTesting reward function...")

    env = PickPlaceEnv(render_mode=None)
    env.reset()

    # Test time penalty
    _, reward, _, _, _ = env.step(np.zeros(7))
    assert reward < 0, "Time penalty should make reward negative"
    print(f"✓ Time penalty works: {reward:.3f}")

    # Test grasp bonus
    env.cube_grasped = True
    env.grasp_bonus_given = False
    reward_with_bonus = env._compute_reward()
    assert reward_with_bonus > 40, "Grasp bonus should be significant"
    print(f"✓ Grasp bonus works: {reward_with_bonus:.3f}")

    env.close()
    print("✓ Reward function tests passed!")


def test_rendering():
    """Test rendering modes"""
    print("\nTesting rendering...")

    # Test DIRECT mode
    env = PickPlaceEnv(render_mode=None)
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    env.close()
    print("✓ DIRECT mode works")

    # Test GUI mode
    env = PickPlaceEnv(render_mode="human")
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
        env.render()
    env.close()
    print("✓ GUI mode works")


if __name__ == "__main__":
    test_basic_functionality()
    test_reward_function()
    # test_rendering()  # Uncomment to test GUI
```

**Usage:**
```bash
python scripts/test_env.py
```

### 9. .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# Training artifacts
models/
logs/
*.zip
*.pkl

# PyBullet
*.bullet

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data
data/
*.mp4
*.avi
```

### 10. README.md
```markdown
# PyBullet Kuka Pick-and-Place RL

Reinforcement learning project for training a Kuka iiwa robot arm to perform pick-and-place tasks using PyBullet and PPO.

## Features
- Custom Gymnasium environment with Kuka iiwa robot
- PPO algorithm (Stable-Baselines3)
- TensorBoard logging
- GPU acceleration (CUDA)
- Shaped reward function for efficient learning

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Download Robot Models
```bash
python scripts/download_urdf.py
```

### Test Environment
```bash
python scripts/test_env.py
```

### Train
```bash
python train.py
```

### Evaluate
```bash
python test.py --model ./models/best/best_model.zip --render
```

### Monitor Training
```bash
tensorboard --logdir=./logs/tensorboard/
```

## Project Structure
See [docs/06_PROJECT_STRUCTURE.md](docs/06_PROJECT_STRUCTURE.md)

## Documentation
- [Architecture Overview](docs/00_ARCHITECTURE_OVERVIEW.md)
- [Environment Design](docs/01_ENVIRONMENT_DESIGN.md)
- [RL Algorithm](docs/02_RL_ALGORITHM_DESIGN.md)
- [Reward Function](docs/03_REWARD_DESIGN.md)
- [Training Pipeline](docs/04_TRAINING_PIPELINE.md)
- [Configuration](docs/05_CONFIG_HYPERPARAMETERS.md)

## Expected Results
- Training time: ~1 hour (RTX 3060)
- Success rate: >80% (fixed positions)
- GPU memory: ~2-3 GB

## Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- WSL2 Ubuntu (or native Linux/macOS)

## License
MIT
```

## Implementation Sequence

### Step-by-Step Setup

#### 1. Project Initialization
```bash
# Create project directory
cd /home/beethoven/workspace/deeplearning/reinforcement/pyBullet/robot

# Create all directories
mkdir -p env models/{checkpoints,best,final} logs/{tensorboard,eval} urdf scripts docs
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Download Robot Models
```bash
python scripts/download_urdf.py
```

#### 4. Implement Core Files
Order of implementation:
1. `config.py` - Configuration first
2. `env/pick_place_env.py` - Environment implementation
3. `scripts/test_env.py` - Test environment
4. `train.py` - Training script
5. `test.py` - Evaluation script

#### 5. Test Environment
```bash
python scripts/test_env.py
```

#### 6. Start Training
```bash
python train.py
```

#### 7. Monitor Progress
```bash
# In separate terminal
tensorboard --logdir=./logs/tensorboard/
```

## Common Setup Issues

### Issue 1: PyBullet Import Error
```python
ImportError: libpython3.x.so: cannot open shared object file
```
**Solution:**
```bash
sudo apt-get install python3-dev
pip install --upgrade pybullet
```

### Issue 2: CUDA Not Available
```python
torch.cuda.is_available() # Returns False
```
**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: Missing URDF Files
```python
pybullet.error: Cannot load URDF file
```
**Solution:**
```bash
python scripts/download_urdf.py
```

## Development Workflow

### Typical Development Cycle
```
1. Modify environment/config
2. Test with scripts/test_env.py
3. Train for short period (50K steps)
4. Evaluate results
5. Adjust hyperparameters
6. Repeat
```

### Debugging Tips
```python
# Enable verbose logging
VERBOSE = 2

# Reduce training time for testing
TOTAL_TIMESTEPS = 10_000

# Enable GUI for visual debugging
env = PickPlaceEnv(render_mode="human")

# Add breakpoints
import pdb; pdb.set_trace()
```

## Next Steps After Implementation
1. ✅ Complete basic training (fixed positions)
2. 📈 Add position randomization (medium difficulty)
3. 🎯 Try different reward functions
4. 🤖 Implement real gripper control
5. 📷 Add visual observations
6. 🔄 Try other algorithms (SAC, TD3)
7. 🎮 Multi-object tasks
