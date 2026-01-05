# PyBullet Robot Arm RL - Architecture Documentation

Complete architectural design documentation for PyBullet-based Kuka iiwa robot arm reinforcement learning project.

## Project Overview

**Goal:** Train a Kuka iiwa robot arm to perform pick-and-place tasks using reinforcement learning (PPO algorithm).

**Key Features:**
- PyBullet physics simulation
- Custom Gymnasium environment
- PPO algorithm (Stable-Baselines3)
- Shaped reward function
- GPU acceleration (CUDA)
- TensorBoard monitoring

## Documentation Index

### Core Architecture
1. **[00_ARCHITECTURE_OVERVIEW.md](./00_ARCHITECTURE_OVERVIEW.md)**
   - System architecture and component interaction
   - Technology stack and design principles
   - Data flow and key design decisions
   - Performance targets

### Technical Design
2. **[01_ENVIRONMENT_DESIGN.md](./01_ENVIRONMENT_DESIGN.md)**
   - Gymnasium environment interface
   - Observation and action spaces
   - PyBullet simulation setup
   - Episode management and gripper logic

3. **[02_RL_ALGORITHM_DESIGN.md](./02_RL_ALGORITHM_DESIGN.md)**
   - PPO algorithm selection and configuration
   - Network architecture (Actor-Critic)
   - Training process and phases
   - GPU utilization and troubleshooting

4. **[03_REWARD_DESIGN.md](./03_REWARD_DESIGN.md)**
   - Shaped reward function components
   - Distance-based rewards and milestone bonuses
   - Reward tuning guidelines
   - Alternative reward strategies

### Implementation Guides
5. **[04_TRAINING_PIPELINE.md](./04_TRAINING_PIPELINE.md)**
   - Complete training script structure
   - Training phases and monitoring
   - TensorBoard visualization
   - Checkpointing and evaluation

6. **[05_CONFIG_HYPERPARAMETERS.md](./05_CONFIG_HYPERPARAMETERS.md)**
   - Complete hyperparameter reference
   - PPO-specific parameters
   - Environment and reward configuration
   - Tuning strategies and benchmarks

7. **[06_PROJECT_STRUCTURE.md](./06_PROJECT_STRUCTURE.md)**
   - Complete directory structure
   - File-by-file implementation guide
   - Setup instructions and workflow
   - Troubleshooting common issues

## Quick Navigation

### For First-Time Setup
1. Start with [06_PROJECT_STRUCTURE.md](./06_PROJECT_STRUCTURE.md) for setup
2. Read [00_ARCHITECTURE_OVERVIEW.md](./00_ARCHITECTURE_OVERVIEW.md) for big picture
3. Follow implementation sequence in project structure doc

### For Understanding the System
1. [00_ARCHITECTURE_OVERVIEW.md](./00_ARCHITECTURE_OVERVIEW.md) - System overview
2. [01_ENVIRONMENT_DESIGN.md](./01_ENVIRONMENT_DESIGN.md) - How environment works
3. [02_RL_ALGORITHM_DESIGN.md](./02_RL_ALGORITHM_DESIGN.md) - How learning works
4. [03_REWARD_DESIGN.md](./03_REWARD_DESIGN.md) - How rewards guide learning

### For Training and Tuning
1. [04_TRAINING_PIPELINE.md](./04_TRAINING_PIPELINE.md) - Training workflow
2. [05_CONFIG_HYPERPARAMETERS.md](./05_CONFIG_HYPERPARAMETERS.md) - Parameter tuning
3. [03_REWARD_DESIGN.md](./03_REWARD_DESIGN.md) - Reward tuning

### For Development
1. [06_PROJECT_STRUCTURE.md](./06_PROJECT_STRUCTURE.md) - Code organization
2. [01_ENVIRONMENT_DESIGN.md](./01_ENVIRONMENT_DESIGN.md) - Environment API
3. [05_CONFIG_HYPERPARAMETERS.md](./05_CONFIG_HYPERPARAMETERS.md) - Configuration system

## Key Specifications Summary

### Hardware
- **Platform:** WSL2 Ubuntu
- **GPU:** NVIDIA GeForce RTX 3060 (12GB)
- **IDE:** VS Code

### Software Stack
- **Python:** 3.8+
- **PyBullet:** Physics simulation
- **Gymnasium:** RL environment interface
- **Stable-Baselines3:** PPO implementation
- **PyTorch:** Neural network backend
- **TensorBoard:** Training visualization

### Task Configuration
- **Robot:** Kuka iiwa (7-DOF)
- **Task:** Pick cube and place at target
- **Difficulty:** Fixed positions (beginner-friendly)
- **Gripper:** Simplified (auto-attach)
- **Success:** 5cm distance threshold

### Learning Configuration
- **Algorithm:** PPO (Proximal Policy Optimization)
- **Observation:** Joint angles + velocities + cube position (17-dim)
- **Action:** Joint velocity control (7-dim)
- **Reward:** Shaped (distance-based + bonuses)
- **Training:** ~200K timesteps (~1 hour)

### Expected Performance
- **Success Rate:** 80-90%
- **Episode Length:** 60-100 steps
- **Training Time:** ~60 minutes
- **GPU Memory:** 2-3 GB

## Implementation Checklist

### Phase 1: Setup
- [ ] Install dependencies (`requirements.txt`)
- [ ] Download robot models (`scripts/download_urdf.py`)
- [ ] Create directory structure
- [ ] Verify CUDA availability

### Phase 2: Environment
- [ ] Implement `env/pick_place_env.py`
- [ ] Create `config.py`
- [ ] Test environment (`scripts/test_env.py`)
- [ ] Verify observation/action spaces

### Phase 3: Training
- [ ] Implement `train.py`
- [ ] Setup callbacks and logging
- [ ] Run test training (10K steps)
- [ ] Verify TensorBoard logging

### Phase 4: Full Training
- [ ] Train for 200K timesteps
- [ ] Monitor learning curves
- [ ] Save checkpoints
- [ ] Verify success rate improvement

### Phase 5: Evaluation
- [ ] Implement `test.py`
- [ ] Evaluate best model
- [ ] Visualize successful episodes
- [ ] Analyze failure cases

## Design Decisions Summary

### Why PPO?
- Stable and beginner-friendly
- Well-tuned defaults in SB3
- Good for continuous control
- Balance of sample efficiency and simplicity

### Why Shaped Rewards?
- Faster learning than sparse rewards
- Clear guidance for exploration
- Avoids pure trial-and-error
- Still robust (not over-shaped)

### Why Simplified Gripper?
- Focus on motion planning
- Reduce complexity for learning
- Faster training
- Can add real gripper later

### Why Fixed Positions?
- Easier to debug
- Faster initial learning
- Progressive difficulty increase
- Can randomize later for generalization

## Next Steps After Basic Implementation

### Immediate
1. Complete basic training
2. Achieve >80% success rate
3. Understand learning curves
4. Analyze behavior patterns

### Short-term
1. Add position randomization
2. Try different reward weights
3. Experiment with network sizes
4. Compare different algorithms

### Long-term
1. Implement real gripper control
2. Add visual observations (camera)
3. Multi-object manipulation
4. Transfer to real robot

## Troubleshooting Guide

### Environment Issues
- **PyBullet errors:** Check URDF paths and installation
- **Observation NaN:** Check joint limits and physics
- **Slow simulation:** Verify DIRECT mode (not GUI)

### Training Issues
- **Not learning:** Check reward function, increase learning rate
- **Unstable:** Reduce learning rate, increase n_steps
- **Slow convergence:** Adjust reward shaping, check exploration

### Technical Issues
- **CUDA not available:** Reinstall PyTorch with CUDA
- **Out of memory:** Reduce batch size or network size
- **Import errors:** Check virtual environment and dependencies

## Resources

### Internal Documentation
- All docs in this directory
- Code comments in implementation files
- TensorBoard logs in `logs/`

### External Resources
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PyBullet Quickstart](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## Contributors
Project documentation created as architectural design reference.

## License
Educational/Research Project
