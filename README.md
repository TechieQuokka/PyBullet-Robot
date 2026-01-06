# PyBullet Kuka Pick-and-Place RL

Reinforcement learning project for training a Kuka iiwa robot arm to perform pick-and-place tasks using PyBullet and PPO.

## Features
- ✅ **Custom Gymnasium environment** with Kuka iiwa robot (7-DOF arm)
- ✅ **PPO algorithm** from Stable-Baselines3 with optimized hyperparameters
- ✅ **97% success rate** achieved on pick-and-place task
- ✅ **Fast training**: 5 minutes on CPU (500K timesteps)
- ✅ **Dense reward shaping** with milestone bonuses for efficient learning
- ✅ **TensorBoard integration** for real-time training monitoring
- ✅ **Parallel environments** (8 envs) for sample-efficient learning
- ✅ **Comprehensive documentation** with implementation guides

## System Requirements
- **OS**: Windows 11 with WSL2 Ubuntu (or any Linux distribution)
- **CPU**: Modern multi-core processor (8 cores recommended)
- **RAM**: 8GB minimum
- **Python**: 3.8+
- **GPU**: Optional (CPU training is fast for this task)

## Quick Start

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify CUDA is available (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Download Robot Models
```bash
python scripts/download_urdf.py
```

### 3. Test Environment
```bash
python scripts/test_env.py
```

### 4. Train
```bash
python train.py
```

Training takes approximately 5 minutes on CPU for 500K timesteps with the optimized configuration.

### 5. Monitor Training
Open a new terminal and run:
```bash
tensorboard --logdir=./logs/tensorboard/
```

Then open http://localhost:6006 in your browser.

### 6. Evaluate
```bash
# Evaluate best model (no rendering)
python test.py --model ./models/best/best_model.zip --episodes 100

# Evaluate with visualization (slower)
python test.py --model ./models/best/best_model.zip --episodes 10 --render

# Evaluate final model
python test.py --model ./models/final/ppo_pickplace_final.zip --episodes 100
```

## Project Structure
```
robot/
├── docs/                      # Documentation
├── env/                       # Custom Gym environment
│   ├── __init__.py
│   └── pick_place_env.py
├── models/                    # Trained models (created during training)
│   ├── checkpoints/
│   ├── best/
│   └── final/
├── logs/                      # Training logs (created during training)
│   ├── tensorboard/
│   └── eval/
├── urdf/                      # Robot model files
│   └── kuka_iiwa/
├── scripts/                   # Utility scripts
│   ├── download_urdf.py
│   └── test_env.py
├── config.py                  # Central configuration
├── train.py                   # Training script
├── test.py                    # Evaluation script
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Configuration

All hyperparameters can be modified in `config.py`:

- **PPO parameters**: learning rate, batch size, epochs, etc.
- **Network architecture**: layer sizes, activation functions
- **Environment parameters**: robot speed, cube size, distances
- **Reward function**: distance coefficients, bonuses, penalties
- **Training settings**: total timesteps, save frequency, device

## Training Results

### Performance Metrics
Our optimized configuration achieved excellent performance:

| Metric | Value |
|--------|-------|
| **Final Success Rate** | 97% |
| **Training Time** | ~5 minutes (500K timesteps on CPU) |
| **First Success** | 160K timesteps |
| **Episode Efficiency** | 159-238 steps (successful episodes) |
| **Peak Reward** | 843 |

### Learning Progression

```
Timesteps    Success Rate    Avg Reward    Episode Length
------------ --------------- ------------- ---------------
80K          0%              234           1001 (timeout)
160K         100%            741           342
240K         100%            733           238
320K         100%            738           197
400K         100%            749           159
480K         100%            843           463
500K (final) 97%             723           225
```

### Key Achievements
- ✅ **Rapid Learning**: Achieved 100% success rate by 160K timesteps
- ✅ **Stable Performance**: Maintained 100% evaluation success rate across 240K-480K timesteps
- ✅ **Efficiency Gains**: Episode length improved from 1001 → 159 steps (6.3x faster)
- ✅ **High Final Performance**: 97% success rate in training rollouts

### Training Configuration
The successful training used optimized hyperparameters in `config.py`:
- **Task**: 8cm cube displacement (simplified for initial learning)
- **Episode Length**: 1000 steps maximum
- **Learning Rate**: 3e-4 (3x higher for faster convergence)
- **Exploration**: ent_coef=0.1 (5x higher for better exploration)
- **Reward Scale**: 50.0 (strong learning signals)
- **Success Bonus**: 500 (significant achievement reward)

### Learning Curve Analysis

**Phase 1 (0-80K): Exploration**
- Agent explores the environment randomly
- No successful episodes yet
- Episode length maxed out (timeout)

**Phase 2 (80K-160K): Breakthrough** 🎯
- Agent discovers successful strategy
- Success rate jumps from 0% → 100%
- Reward increases from 234 → 741

**Phase 3 (160K-500K): Optimization**
- Maintains 100% eval success rate
- Improves efficiency: 342 → 159 steps
- Final training performance: 97% success rate

## Documentation
Detailed documentation is available in the `docs/` directory:

- [Architecture Overview](docs/00_ARCHITECTURE_OVERVIEW.md)
- [Environment Design](docs/01_ENVIRONMENT_DESIGN.md)
- [RL Algorithm](docs/02_RL_ALGORITHM_DESIGN.md)
- [Reward Function](docs/03_REWARD_DESIGN.md)
- [Training Pipeline](docs/04_TRAINING_PIPELINE.md)
- [Configuration](docs/05_CONFIG_HYPERPARAMETERS.md)
- [Project Structure](docs/06_PROJECT_STRUCTURE.md)

## Troubleshooting

### CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### PyBullet Import Error
```bash
# Install missing dependencies
sudo apt-get install python3-dev
pip install --upgrade pybullet
```

### Missing URDF Files
```bash
python scripts/download_urdf.py
```

### Slow Training
- For this task, CPU is actually faster than GPU for MlpPolicy
- Use DIRECT mode (not GUI) during training
- Ensure `n_envs=8` to utilize all CPU cores

### Zero Success Rate
If training shows 0% success rate throughout:
1. **Task too difficult**: Reduce target distance in `config.py`
2. **Time too short**: Increase `max_episode_steps`
3. **Exploration insufficient**: Increase `ent_coef` in PPO config
4. **Reward signal weak**: Increase `distance_reward_scale`

Our optimized configuration addresses all these issues and achieves 97% success rate.

## Next Steps

### Progressive Difficulty Scaling
Now that basic pick-and-place works (97% success), gradually increase task difficulty:

#### Phase 2: Moderate Challenge
```python
# config.py modifications
target_pos = [0.35, 0.12, 0.05]  # Increase distance: 8cm → 12cm
max_episode_steps = 800           # Reduce time allowance
success_distance = 0.06           # Tighter success criteria
```

#### Phase 3: Original Difficulty
```python
cube_start_pos = [0.5, 0.0, 0.05]   # Move cube further from robot
target_pos = [0.5, 0.2, 0.05]        # Full 20cm displacement
max_episode_steps = 500               # More efficient episodes required
gripper_attach_distance = 0.05        # Precise grasping needed
```

#### Phase 4: Advanced Challenges
1. **Position Randomization**: Random cube and target positions
2. **Different Algorithms**: Experiment with SAC or TD3 for continuous control
3. **Visual Observations**: Camera-based perception instead of state vectors
4. **Real Gripper Control**: Implement actual gripper finger control
5. **Multi-Object Tasks**: Pick and place multiple cubes
6. **Obstacle Avoidance**: Add obstacles between start and target
7. **Dynamic Targets**: Moving target positions during episodes

## License
Educational/Research Project

## Acknowledgments
- PyBullet for physics simulation
- Stable-Baselines3 for RL algorithms
- Gymnasium for environment interface
