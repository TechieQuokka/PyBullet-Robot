# Configuration and Hyperparameters

## Complete Hyperparameter Reference

### PPO Algorithm Parameters

#### Core Learning Parameters
```python
LEARNING_RATE = 3e-4          # Adam optimizer learning rate
N_STEPS = 2048                # Steps to collect before policy update
BATCH_SIZE = 64               # Minibatch size for SGD
N_EPOCHS = 10                 # Number of epochs per update
GAMMA = 0.99                  # Discount factor
GAE_LAMBDA = 0.95             # GAE parameter for advantage estimation
```

**Tuning Guidelines:**

| Parameter | Default | If too slow | If unstable |
|-----------|---------|-------------|-------------|
| LEARNING_RATE | 3e-4 | 5e-4 to 1e-3 | 1e-4 to 3e-5 |
| N_STEPS | 2048 | Keep same | 4096 to 8192 |
| BATCH_SIZE | 64 | 128 to 256 | 32 to 64 |
| N_EPOCHS | 10 | 15 to 20 | 5 to 10 |

#### PPO-Specific Parameters
```python
CLIP_RANGE = 0.2              # PPO clipping parameter
CLIP_RANGE_VF = None          # Value function clipping (None = no clip)
ENT_COEF = 0.01               # Entropy coefficient
VF_COEF = 0.5                 # Value function coefficient
MAX_GRAD_NORM = 0.5           # Gradient clipping
TARGET_KL = None              # Target KL divergence (None = no limit)
```

**Detailed Explanations:**

**CLIP_RANGE (ε):**
- Controls how much policy can change per update
- Higher = more aggressive updates, potentially unstable
- Lower = conservative updates, slower learning
- Typical range: 0.1 to 0.3

**ENT_COEF:**
- Encourages exploration by penalizing deterministic policies
- Higher = more exploration, slower convergence
- Lower = less exploration, faster convergence (risk of local optima)
- Task-specific tuning: simple tasks use lower values

**VF_COEF:**
- Weight of value function loss in total loss
- Higher = prioritize value learning
- Lower = prioritize policy learning
- Usually keep at 0.5

**MAX_GRAD_NORM:**
- Clips gradient norm to prevent exploding gradients
- Higher = less clipping, potentially unstable
- Lower = more clipping, slower learning
- Rarely needs tuning

### Network Architecture

#### Policy Network (Actor)
```python
POLICY_NET_ARCH = [64, 64]    # Hidden layer sizes
ACTIVATION_FN = "tanh"        # Activation function
```

**Options:**
```python
# Small network (faster, less capacity)
POLICY_NET_ARCH = [32, 32]

# Medium network (default, balanced)
POLICY_NET_ARCH = [64, 64]

# Large network (slower, more capacity)
POLICY_NET_ARCH = [128, 128]

# Deep network (for complex tasks)
POLICY_NET_ARCH = [64, 64, 64]
```

**Activation Functions:**
- `tanh`: Default, smooth gradients, bounded output [-1, 1]
- `relu`: Faster computation, risk of dead neurons
- `elu`: Smooth, allows negative values

#### Value Network (Critic)
```python
VALUE_NET_ARCH = [64, 64]     # Usually same as policy network
```

**Note:** Keeping policy and value networks the same size is common practice.

### Environment Configuration

#### Physical Parameters
```python
# Simulation
TIMESTEP = 1/240              # Physics timestep (240 Hz)
GRAVITY = -9.81               # Gravity (m/s²)
MAX_EPISODE_STEPS = 200       # Episode timeout

# Robot
MAX_JOINT_VELOCITY = 0.5      # Max joint velocity (rad/s)
JOINT_FORCE = 150             # Max joint force (N⋅m)

# Objects
CUBE_SIZE = 0.05              # Cube edge length (m)
CUBE_MASS = 0.1               # Cube mass (kg)

# Workspace
WORKSPACE_X = [0.3, 0.7]      # X bounds (m)
WORKSPACE_Y = [-0.3, 0.3]     # Y bounds (m)
WORKSPACE_Z = [0.0, 0.5]      # Z bounds (m)
```

#### Task Configuration
```python
# Fixed positions (easy difficulty)
CUBE_START_POS = [0.5, 0.0, 0.05]
TARGET_POS = [0.5, 0.2, 0.05]

# Success criteria
SUCCESS_DISTANCE = 0.05       # Distance threshold (m)
GRIPPER_ATTACH_DISTANCE = 0.03  # Auto-grasp distance (m)
```

#### Difficulty Levels
```python
# Level 1: Fixed positions
DIFFICULTY = "easy"
CUBE_START_POS = [0.5, 0.0, 0.05]
TARGET_POS = [0.5, 0.2, 0.05]

# Level 2: Random start, fixed target
DIFFICULTY = "medium"
CUBE_START_POS = "random"     # Random within workspace
TARGET_POS = [0.5, 0.2, 0.05]

# Level 3: Both random
DIFFICULTY = "hard"
CUBE_START_POS = "random"
TARGET_POS = "random"

# Level 4: Multiple objects
DIFFICULTY = "expert"
NUM_CUBES = 3
```

### Reward Function Parameters

```python
# Distance reward coefficients
REACH_REWARD_COEF = -0.1      # Reward for approaching cube
PLACE_REWARD_COEF = -0.1      # Reward for moving to target

# Milestone bonuses
GRASP_BONUS = 50.0            # Bonus for successful grasp
SUCCESS_BONUS = 100.0         # Bonus for task completion

# Penalties
TIME_PENALTY = -0.01          # Per-step time penalty
COLLISION_PENALTY = -1.0      # Self-collision penalty
DROP_PENALTY = -5.0           # Cube dropped penalty
```

**Tuning Guide:**

**If agent not approaching cube:**
```python
REACH_REWARD_COEF = -0.2      # Increase magnitude
```

**If agent not grasping:**
```python
GRASP_BONUS = 100.0           # Increase bonus
# Or add proximity bonus
PROXIMITY_BONUS = 10.0        # When very close to cube
```

**If episodes too long:**
```python
TIME_PENALTY = -0.02          # Increase penalty
```

**If agent too cautious:**
```python
TIME_PENALTY = -0.005         # Decrease penalty
```

### Training Configuration

#### Training Duration
```python
TOTAL_TIMESTEPS = 200_000     # Total training steps
SAVE_FREQ = 10_000            # Checkpoint save frequency
EVAL_FREQ = 5_000             # Evaluation frequency
N_EVAL_EPISODES = 10          # Episodes per evaluation
```

**Recommended Timesteps by Difficulty:**
- Easy (fixed): 100K - 200K
- Medium (random start): 300K - 500K
- Hard (both random): 500K - 1M
- Expert (multi-object): 1M+

#### Vectorization
```python
N_ENVS = 1                    # Number of parallel environments
```

**Options:**
- `N_ENVS = 1`: Simplest, easier debugging
- `N_ENVS = 4`: 4x faster data collection
- `N_ENVS = 8`: 8x faster, more GPU memory

**Note:** For beginners, start with 1. Increase for faster training later.

#### Normalization
```python
NORMALIZE_OBS = True          # Normalize observations
NORMALIZE_REWARD = True       # Normalize rewards
CLIP_OBS = 10.0               # Clip normalized obs
CLIP_REWARD = 10.0            # Clip normalized reward
```

**Effects:**
- Normalization improves learning stability
- Recommended to keep enabled
- Only disable for debugging

### Hardware Configuration

#### Device Selection
```python
import torch

# Automatic device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Or force specific device
DEVICE = "cuda"               # Use GPU
DEVICE = "cpu"                # Use CPU
```

#### PyBullet Rendering
```python
# Training mode (fast)
RENDER_MODE = None            # DIRECT mode, no visualization

# Evaluation mode (slow)
RENDER_MODE = "human"         # GUI mode, visualization
```

#### Simulation Speed
```python
# Real-time playback (60 FPS)
TIME_SLEEP = 1/60

# Fast playback
TIME_SLEEP = 0

# Slow-motion
TIME_SLEEP = 1/30
```

### Logging Configuration

#### TensorBoard
```python
TENSORBOARD_LOG = "./logs/tensorboard/"
TB_LOG_NAME = "PPO_PickPlace"
LOG_INTERVAL = 10             # Log every N policy updates
```

#### Console Logging
```python
VERBOSE = 1                   # 0: no output, 1: info, 2: debug
```

#### Custom Logging
```python
# Log additional metrics
LOG_DISTANCE_TO_CUBE = True
LOG_DISTANCE_TO_TARGET = True
LOG_GRASP_SUCCESS_RATE = True
LOG_COLLISION_RATE = True
```

## Configuration File Structure

### config.py
```python
"""
Central configuration file for all hyperparameters
"""

import torch
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters"""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None


@dataclass
class NetworkConfig:
    """Neural network architecture"""
    policy_net_arch: list = None
    value_net_arch: list = None
    activation_fn: str = "tanh"

    def __post_init__(self):
        if self.policy_net_arch is None:
            self.policy_net_arch = [64, 64]
        if self.value_net_arch is None:
            self.value_net_arch = [64, 64]


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
    cube_start_pos: list = None
    target_pos: list = None
    success_distance: float = 0.05
    gripper_attach_distance: float = 0.03

    def __post_init__(self):
        if self.cube_start_pos is None:
            self.cube_start_pos = [0.5, 0.0, 0.05]
        if self.target_pos is None:
            self.target_pos = [0.5, 0.2, 0.05]


@dataclass
class RewardConfig:
    """Reward function parameters"""
    reach_reward_coef: float = -0.1
    place_reward_coef: float = -0.1
    grasp_bonus: float = 50.0
    success_bonus: float = 100.0
    time_penalty: float = -0.01
    collision_penalty: float = -1.0
    drop_penalty: float = -5.0


@dataclass
class TrainingConfig:
    """Training configuration"""
    total_timesteps: int = 200_000
    n_envs: int = 1
    save_freq: int = 10_000
    eval_freq: int = 5_000
    n_eval_episodes: int = 10
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
```

### Usage in Scripts
```python
from config import (
    ppo_config,
    network_config,
    env_config,
    training_config
)

# Use in training script
model = PPO(
    learning_rate=ppo_config.learning_rate,
    n_steps=ppo_config.n_steps,
    batch_size=ppo_config.batch_size,
    ...
)

# Use in environment
env = PickPlaceEnv(
    max_episode_steps=env_config.max_episode_steps,
    success_distance=env_config.success_distance,
    ...
)
```

## Hyperparameter Tuning Strategies

### Grid Search (Systematic)
```python
learning_rates = [1e-4, 3e-4, 1e-3]
n_steps_options = [1024, 2048, 4096]

for lr in learning_rates:
    for n_steps in n_steps_options:
        # Train and evaluate
        results = train_model(lr=lr, n_steps=n_steps)
```

### Random Search (Efficient)
```python
import random

for trial in range(10):
    lr = random.uniform(1e-4, 1e-3)
    batch_size = random.choice([32, 64, 128])
    ent_coef = random.uniform(0.0, 0.1)

    results = train_model(lr=lr, batch_size=batch_size, ent_coef=ent_coef)
```

### Manual Tuning (Beginner-Friendly)
1. Start with defaults
2. Train for 50K steps
3. Check if learning is happening
4. Adjust one parameter at a time
5. Repeat

## Performance Benchmarks

### Expected Performance by Configuration

#### Default Configuration (Recommended)
```python
# As specified above
TOTAL_TIMESTEPS = 200_000
LEARNING_RATE = 3e-4
```
**Expected Results:**
- Training time: ~60 minutes
- Final success rate: 80-90%
- GPU memory: 2-3 GB

#### Fast Configuration (Quick Test)
```python
TOTAL_TIMESTEPS = 50_000
LEARNING_RATE = 5e-4
N_STEPS = 1024
```
**Expected Results:**
- Training time: ~15 minutes
- Final success rate: 40-60%
- Good for debugging

#### High-Performance Configuration
```python
TOTAL_TIMESTEPS = 500_000
LEARNING_RATE = 3e-4
N_ENVS = 4
POLICY_NET_ARCH = [128, 128]
```
**Expected Results:**
- Training time: ~2 hours
- Final success rate: 90-95%
- GPU memory: 6-8 GB

## Common Configuration Mistakes

### ❌ Mistake 1: Learning Rate Too High
```python
LEARNING_RATE = 1e-2  # Too high!
```
**Symptom:** Unstable learning, performance oscillates
**Fix:** Use 3e-4 or lower

### ❌ Mistake 2: Batch Size Too Small
```python
BATCH_SIZE = 8  # Too small!
```
**Symptom:** Noisy updates, slow convergence
**Fix:** Use at least 64

### ❌ Mistake 3: Too Few Steps
```python
N_STEPS = 256  # Too few!
```
**Symptom:** Poor advantage estimates
**Fix:** Use at least 1024, ideally 2048

### ❌ Mistake 4: Wrong Device
```python
DEVICE = "cpu"  # When GPU is available!
```
**Symptom:** Very slow training
**Fix:** Use "cuda" when available

## Saving and Loading Configurations

### Save Configuration
```python
import json

config_dict = {
    "ppo": ppo_config.__dict__,
    "network": network_config.__dict__,
    "env": env_config.__dict__,
    "training": training_config.__dict__
}

with open("./config.json", "w") as f:
    json.dump(config_dict, f, indent=2)
```

### Load Configuration
```python
with open("./config.json", "r") as f:
    config_dict = json.load(f)

ppo_config = PPOConfig(**config_dict["ppo"])
network_config = NetworkConfig(**config_dict["network"])
```
