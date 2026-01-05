# Training Pipeline Design

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
│                                                              │
│  Environment Setup → Agent Creation → Training Loop →       │
│  Monitoring → Evaluation → Model Saving                     │
└─────────────────────────────────────────────────────────────┘
```

## Training Script Structure

### Main Training Script (`train.py`)

```python
"""
Main training script for Kuka Pick-and-Place task
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from env.pick_place_env import PickPlaceEnv


def make_env():
    """Create and wrap the environment"""
    env = PickPlaceEnv(render_mode=None)  # DIRECT mode for speed
    env = Monitor(env)  # Add monitoring wrapper
    return env


def create_vectorized_env(n_envs=1):
    """
    Create vectorized environment for parallel training

    Args:
        n_envs: Number of parallel environments (1 for simplicity)
    """
    env = DummyVecEnv([make_env for _ in range(n_envs)])

    # Normalize observations and rewards (optional but recommended)
    env = VecNormalize(
        env,
        norm_obs=True,      # Normalize observations
        norm_reward=True,   # Normalize rewards
        clip_obs=10.0,      # Clip normalized obs
        clip_reward=10.0,   # Clip normalized reward
    )

    return env


def setup_callbacks(eval_env):
    """
    Setup training callbacks

    Args:
        eval_env: Separate environment for evaluation
    """
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/checkpoints/',
        name_prefix='ppo_pickplace',
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best/',
        log_path='./logs/eval/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Combine callbacks
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    return callback_list


def main():
    """Main training function"""

    # ============ Configuration ============
    TOTAL_TIMESTEPS = 200_000
    LEARNING_RATE = 3e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ============ Environment Setup ============
    print("Creating training environment...")
    train_env = create_vectorized_env(n_envs=1)

    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # ============ Agent Setup ============
    print("Creating PPO agent...")

    policy_kwargs = dict(
        net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        activation_fn=torch.nn.Tanh
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tensorboard/",
        verbose=1,
        device=DEVICE
    )

    # ============ Callbacks ============
    callbacks = setup_callbacks(eval_env)

    # ============ Training ============
    print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Expected time: ~1 hour on RTX 3060")
    print("-" * 50)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,  # Log every 10 updates
            tb_log_name="PPO_PickPlace",
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
        # ============ Save Final Model ============
        print("\nSaving final model...")
        model.save("./models/final/ppo_pickplace_final")
        train_env.save("./models/final/vec_normalize.pkl")

        print("Model saved to: ./models/final/")

        # ============ Cleanup ============
        train_env.close()
        eval_env.close()

        print("\nTraining pipeline completed!")


if __name__ == "__main__":
    main()
```

## Directory Structure Creation

### Automatic Directory Setup
```python
import os

def create_directories():
    """Create necessary directories for training"""
    directories = [
        "./models/checkpoints/",
        "./models/best/",
        "./models/final/",
        "./logs/tensorboard/",
        "./logs/eval/",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

# Call at start of main()
create_directories()
```

## Training Phases

### Phase 1: Initialization (Timesteps 0-10K)
**What happens:**
- Random exploration dominates
- Policy and value networks initialize
- Observation normalization statistics collected
- Baseline performance established

**Expected metrics:**
- Episode reward: -20 to -10
- Success rate: 0-5%
- Episode length: 180-200 (timeout)

**Monitoring:**
- Check that environment runs without errors
- Verify GPU utilization
- Confirm TensorBoard logging works

### Phase 2: Discovery (Timesteps 10K-50K)
**What happens:**
- Agent discovers approaching cube increases reward
- Occasional grasps occur by chance
- Policy begins to form structure

**Expected metrics:**
- Episode reward: -10 to +5
- Success rate: 5-20%
- Episode length: 150-200

**Monitoring:**
- Policy loss should decrease
- Entropy should gradually decrease
- Grasp bonuses start appearing in logs

### Phase 3: Skill Development (Timesteps 50K-100K)
**What happens:**
- Consistent grasping behavior learned
- Agent attempts to move cube after grasping
- Some complete successes

**Expected metrics:**
- Episode reward: +5 to +30
- Success rate: 20-50%
- Episode length: 100-150

**Monitoring:**
- Success rate curve should show upward trend
- Value function predictions become more accurate
- Clip fraction should stabilize

### Phase 4: Optimization (Timesteps 100K-150K)
**What happens:**
- Refinement of pick-and-place sequence
- More direct trajectories
- Consistent success on fixed position

**Expected metrics:**
- Episode reward: +30 to +50
- Success rate: 50-75%
- Episode length: 80-120

**Monitoring:**
- Explained variance should be high (>0.7)
- Episode length decreases
- Success rate continues improving

### Phase 5: Convergence (Timesteps 150K-200K)
**What happens:**
- Near-optimal policy for fixed position
- Efficient, consistent behavior
- Minimal further improvement

**Expected metrics:**
- Episode reward: +50 to +70
- Success rate: 75-90%
- Episode length: 60-100

**Monitoring:**
- Metrics should plateau
- Policy updates become smaller
- Ready for evaluation

## Monitoring and Logging

### TensorBoard Visualization

#### Launch TensorBoard
```bash
tensorboard --logdir=./logs/tensorboard/
```

#### Key Plots to Monitor

**Performance Plots:**
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `eval/mean_reward` - Evaluation performance
- `eval/success_rate` - Success percentage

**Training Plots:**
- `train/learning_rate` - Current LR
- `train/policy_loss` - Policy gradient loss
- `train/value_loss` - Value function loss
- `train/entropy_loss` - Policy entropy

**Diagnostic Plots:**
- `train/approx_kl` - KL divergence (stability indicator)
- `train/clip_fraction` - Fraction of clipped ratios
- `train/explained_variance` - Value function quality
- `time/fps` - Simulation speed

### Console Output

#### Training Progress
```
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 156         |
|    ep_rew_mean          | -8.42       |
| time/                   |             |
|    fps                  | 1024        |
|    iterations           | 10          |
|    time_elapsed         | 19          |
|    total_timesteps      | 20480       |
| train/                  |             |
|    approx_kl            | 0.0089      |
|    clip_fraction        | 0.098       |
|    entropy_loss         | -0.84       |
|    explained_variance   | 0.23        |
|    learning_rate        | 0.0003      |
|    policy_gradient_loss | -0.012      |
|    value_loss           | 45.2        |
-----------------------------------------
```

### Custom Logging

#### Episode Statistics Logger
```python
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggerCallback(BaseCallback):
    """
    Custom callback for additional logging
    """
    def __init__(self, verbose=0):
        super(CustomLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]

            # Log episode metrics
            ep_reward = info.get('episode', {}).get('r', 0)
            ep_length = info.get('episode', {}).get('l', 0)
            is_success = info.get('is_success', False)

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_length)

            if is_success:
                self.success_count += 1

            # Log to TensorBoard
            self.logger.record('custom/success_count', self.success_count)
            self.logger.record('custom/avg_reward_100',
                             np.mean(self.episode_rewards[-100:]))

        return True
```

## Evaluation During Training

### Periodic Evaluation
```python
# Handled by EvalCallback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/best/',
    eval_freq=5000,        # Every 5K timesteps
    n_eval_episodes=10,    # 10 episodes per eval
    deterministic=True,    # Use deterministic policy
    render=False
)
```

### Evaluation Metrics
- **Mean reward**: Average over 10 episodes
- **Success rate**: Percentage of successful episodes
- **Mean episode length**: Average steps to completion
- **Best model**: Saved when mean reward improves

## Checkpointing Strategy

### Checkpoint Frequency
```python
checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Every 10K timesteps
    save_path='./models/checkpoints/',
    name_prefix='ppo_pickplace'
)
```

### Checkpoint Files
```
models/checkpoints/
├── ppo_pickplace_10000_steps.zip
├── ppo_pickplace_20000_steps.zip
├── ppo_pickplace_30000_steps.zip
...
└── ppo_pickplace_200000_steps.zip
```

### Loading Checkpoint
```python
# Resume training from checkpoint
model = PPO.load(
    "./models/checkpoints/ppo_pickplace_100000_steps.zip",
    env=train_env,
    device="cuda"
)

# Continue training
model.learn(
    total_timesteps=100000,  # Train for 100K more steps
    reset_num_timesteps=False  # Don't reset step counter
)
```

## Resource Management

### GPU Memory Monitoring
```python
import torch

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, "
              f"Reserved: {reserved:.2f} GB")

# Call periodically during training
print_gpu_memory()
```

### Expected Resource Usage
- **GPU Memory**: 2-3 GB
- **RAM**: 4-6 GB
- **Disk**: ~500 MB (all checkpoints)
- **Training Time**: ~60 minutes on RTX 3060

## Error Handling

### Common Errors and Solutions

#### CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 32  # Instead of 64

# Or reduce network size
policy_kwargs = dict(
    net_arch=[dict(pi=[32, 32], vf=[32, 32])]  # Smaller network
)
```

#### NaN in Observations
```python
# Check observation bounds
assert not np.isnan(obs).any(), "NaN in observation!"

# Add observation clipping in environment
obs = np.clip(obs, -10, 10)
```

#### Environment Crashes
```python
# Wrap in exception handler
try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
except Exception as e:
    print(f"Training error: {e}")
    model.save("./models/emergency_save.zip")
```

## Training Completion

### Final Steps
```python
# 1. Save final model
model.save("./models/final/ppo_pickplace_final")

# 2. Save normalization stats
train_env.save("./models/final/vec_normalize.pkl")

# 3. Run final evaluation
from eval import evaluate_model
results = evaluate_model(
    model_path="./models/final/ppo_pickplace_final.zip",
    n_episodes=100,
    render=True
)

# 4. Print summary
print(f"\nFinal Results:")
print(f"  Success Rate: {results['success_rate']:.1f}%")
print(f"  Mean Reward: {results['mean_reward']:.2f}")
print(f"  Mean Length: {results['mean_length']:.1f}")
```

### Success Criteria
- ✅ Success rate > 80% on evaluation
- ✅ Training completes without crashes
- ✅ Metrics show clear improvement
- ✅ Model saved successfully

## Next Steps After Training
1. Run comprehensive evaluation (`test.py`)
2. Visualize successful episodes
3. Analyze failure cases
4. Consider curriculum learning (randomize positions)
5. Try different algorithms (SAC, TD3)
