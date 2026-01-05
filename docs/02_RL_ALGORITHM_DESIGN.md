# Reinforcement Learning Algorithm Design

## Algorithm Selection: PPO (Proximal Policy Optimization)

### Why PPO?
- **Beginner-friendly**: Well-documented, stable, widely used
- **Stable learning**: Clipped objective prevents large policy updates
- **Sample efficient**: Better than vanilla policy gradient methods
- **Proven success**: Works well for continuous control tasks
- **Great defaults**: Stable-Baselines3 provides well-tuned hyperparameters

### Alternatives Considered
| Algorithm | Pros | Cons | Decision |
|-----------|------|------|----------|
| **SAC** | Very sample efficient, off-policy | More complex, harder to tune | Not chosen (overkill for simple task) |
| **TD3** | Good for continuous control | Requires careful tuning | Not chosen (learning project) |
| **DDPG** | Simpler than TD3 | Unstable, sensitive | Not chosen (stability concerns) |
| **A2C** | Simpler than PPO | Less sample efficient | Not chosen (PPO is better) |

## PPO Algorithm Overview

### Core Concepts

#### Policy Network (Actor)
```
Input: State (17-dim observation)
  ↓
Hidden Layer 1: 64 units, tanh activation
  ↓
Hidden Layer 2: 64 units, tanh activation
  ↓
Output: Action mean (7-dim) + log_std
  ↓
Sample action from Gaussian distribution
```

#### Value Network (Critic)
```
Input: State (17-dim observation)
  ↓
Hidden Layer 1: 64 units, tanh activation
  ↓
Hidden Layer 2: 64 units, tanh activation
  ↓
Output: State value (scalar)
```

### PPO Objective Function

#### Clipped Surrogate Objective
```
L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  (probability ratio)
- A_t = advantage estimate at timestep t
- ε = clip range (typically 0.2)
```

#### Value Function Loss
```
L^VF(θ) = E[(V_θ(s_t) - V_target)^2]

where:
- V_target = R_t + γ * V(s_{t+1})  (TD target)
```

#### Entropy Bonus
```
L^ENT(θ) = -E[entropy(π_θ)]

Total Loss = L^CLIP - c_1 * L^VF + c_2 * L^ENT
```

## Stable-Baselines3 Integration

### PPO Configuration
```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

model = PPO(
    policy="MlpPolicy",              # Multi-Layer Perceptron policy
    env=env,                         # Custom gym environment
    learning_rate=3e-4,              # Adam optimizer learning rate
    n_steps=2048,                    # Steps to collect before update
    batch_size=64,                   # Minibatch size for SGD
    n_epochs=10,                     # Number of epochs per update
    gamma=0.99,                      # Discount factor
    gae_lambda=0.95,                 # GAE parameter
    clip_range=0.2,                  # PPO clip parameter
    clip_range_vf=None,              # No value function clipping
    ent_coef=0.01,                   # Entropy coefficient
    vf_coef=0.5,                     # Value function coefficient
    max_grad_norm=0.5,               # Gradient clipping
    use_sde=False,                   # No state-dependent exploration
    sde_sample_freq=-1,
    target_kl=None,                  # No KL divergence limit
    tensorboard_log="./logs/",       # TensorBoard logging
    verbose=1,                       # Print training info
    device="cuda"                    # Use GPU
)
```

### Policy Network Architecture (MlpPolicy)
```python
policy_kwargs = dict(
    net_arch=[
        dict(pi=[64, 64], vf=[64, 64])  # Actor and Critic networks
    ],
    activation_fn=torch.nn.Tanh,        # Activation function
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    ...
)
```

## Training Process

### Data Collection Phase
```
1. Collect n_steps (2048) transitions using current policy
2. Store (state, action, reward, next_state, done) in rollout buffer
3. Calculate advantages using GAE (Generalized Advantage Estimation)
```

### Policy Update Phase
```
1. Compute advantages: A = Q(s,a) - V(s)
2. Normalize advantages (mean=0, std=1)
3. Split data into minibatches (batch_size=64)
4. For each epoch (10 epochs):
   a. For each minibatch:
      - Compute policy loss (clipped objective)
      - Compute value loss (MSE)
      - Compute entropy bonus
      - Total loss = policy_loss - vf_coef*value_loss + ent_coef*entropy
      - Backpropagation and gradient descent
      - Clip gradients (max_grad_norm=0.5)
```

### Update Cycle
```
Timesteps: 0 → 2048 → 4096 → ... → 200,000

At each 2048 steps:
  - Collect rollout buffer (2048 transitions)
  - Update policy (10 epochs × 32 minibatches)
  - Log metrics to TensorBoard
  - Save checkpoint (every 10K steps)
```

## Advantage Estimation (GAE)

### Generalized Advantage Estimation
```
A_t = δ_t + (γλ)δ_{t+1} + (γλ)^2δ_{t+2} + ...

where:
- δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
- γ = discount factor (0.99)
- λ = GAE parameter (0.95)
```

### Why GAE?
- **Bias-variance trade-off**: λ controls balance
- **λ=0**: Low variance, high bias (TD learning)
- **λ=1**: High variance, low bias (Monte Carlo)
- **λ=0.95**: Good balance for most tasks

## Exploration Strategy

### Action Sampling
```python
# During training: stochastic policy
action, _states = model.predict(observation, deterministic=False)

# During evaluation: deterministic policy
action, _states = model.predict(observation, deterministic=True)
```

### Gaussian Exploration
```
action ~ N(μ_θ(s), σ²)

where:
- μ_θ(s) = mean predicted by policy network
- σ² = learned or fixed standard deviation
```

### Entropy Regularization
```
Encourages exploration by:
- Adding entropy bonus to objective
- Prevents premature convergence to deterministic policy
- ent_coef = 0.01 (relatively small, task is simple)
```

## Training Monitoring

### Key Metrics (TensorBoard)

#### Performance Metrics
- **ep_rew_mean**: Average episode reward (should increase)
- **ep_len_mean**: Average episode length (should decrease when learning)
- **success_rate**: Percentage of successful episodes (goal: >80%)

#### Learning Metrics
- **policy_loss**: PPO clipped objective (should decrease)
- **value_loss**: Value function MSE (should decrease)
- **entropy_loss**: Policy entropy (should gradually decrease)
- **approx_kl**: Approximate KL divergence (monitor stability)
- **clip_fraction**: Fraction of clipped policy ratios

#### Technical Metrics
- **learning_rate**: Current learning rate
- **explained_variance**: How well value function predicts returns
- **fps**: Frames per second (simulation speed)

### Expected Training Curves

#### Episode Reward
```
Timesteps:    0     50K   100K  150K  200K
Reward:      -50    -20    +10   +30   +50
Trend: Gradual increase, may plateau
```

#### Episode Length
```
Timesteps:    0     50K   100K  150K  200K
Length:      200   150    100    80    60
Trend: Decrease as policy becomes more efficient
```

#### Success Rate
```
Timesteps:    0     50K   100K  150K  200K
Success:      0%    20%    50%   70%   85%
Trend: Gradual increase, target >80%
```

## Callbacks

### Checkpoint Callback
```python
checkpoint_callback = CheckpointCallback(
    save_freq=10000,              # Save every 10K steps
    save_path='./models/',        # Save directory
    name_prefix='ppo_pickplace'   # Model filename prefix
)
```

### Evaluation Callback
```python
eval_callback = EvalCallback(
    eval_env,                     # Separate evaluation environment
    best_model_save_path='./models/best/',
    log_path='./logs/eval/',
    eval_freq=5000,               # Evaluate every 5K steps
    n_eval_episodes=10,           # 10 episodes for evaluation
    deterministic=True,           # Use deterministic policy
    render=False                  # No rendering during training
)
```

## Training Workflow

### Phase 1: Initial Training (0-50K steps)
- **Goal**: Learn basic motion patterns
- **Expected**: Random exploration, low success rate
- **Monitor**: Policy loss convergence, entropy decrease

### Phase 2: Skill Acquisition (50K-100K steps)
- **Goal**: Learn to reach cube and grasp
- **Expected**: Increasing success in grasping
- **Monitor**: Distance to cube decreasing

### Phase 3: Task Optimization (100K-150K steps)
- **Goal**: Learn complete pick-and-place sequence
- **Expected**: Occasional successes, improving consistency
- **Monitor**: Success rate increasing

### Phase 4: Policy Refinement (150K-200K steps)
- **Goal**: Optimize trajectory efficiency
- **Expected**: Consistent success, shorter episodes
- **Monitor**: Episode length decreasing, success rate >80%

## Hyperparameter Tuning Guidelines

### If Learning Too Slow
- Increase `learning_rate` (3e-4 → 5e-4)
- Increase `batch_size` (64 → 128)
- Adjust reward shaping (more dense rewards)

### If Learning Unstable
- Decrease `learning_rate` (3e-4 → 1e-4)
- Decrease `clip_range` (0.2 → 0.1)
- Increase `n_steps` (2048 → 4096)

### If Not Exploring Enough
- Increase `ent_coef` (0.01 → 0.05)
- Add action noise

### If Overfitting to Fixed Position
- Add position randomization
- Increase environment diversity

## GPU Utilization

### PyTorch CUDA Settings
```python
import torch

# Verify GPU availability
assert torch.cuda.is_available(), "CUDA not available"
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set device
device = "cuda"

# Create model with GPU
model = PPO(..., device=device)
```

### Expected GPU Usage
- **Memory**: ~2-3 GB (RTX 3060 has 12GB, plenty)
- **Utilization**: 30-50% (simulation is bottleneck, not training)
- **Speed**: ~1000 FPS in DIRECT mode

## Troubleshooting

### Common Issues

#### Policy Not Learning
- Check reward function (are rewards received?)
- Verify observation space (normalized?)
- Check action scaling (too fast/slow?)

#### Unstable Learning
- Reduce learning rate
- Increase n_steps for more data
- Check for NaN values in observations

#### Slow Training
- Verify GPU is being used (`device="cuda"`)
- Use DIRECT mode (not GUI)
- Reduce logging frequency

#### Gripper Not Attaching
- Check attachment distance threshold
- Verify end-effector link index
- Debug with visualization

## Next Steps After Basic Training
1. Add position randomization (medium difficulty)
2. Try different algorithms (SAC, TD3)
3. Add visual observations (camera input)
4. Multi-object tasks
5. Real gripper control (not simplified)
