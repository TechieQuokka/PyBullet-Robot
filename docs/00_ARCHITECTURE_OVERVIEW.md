# Architecture Overview

## Project Goal
PyBullet-based Kuka iiwa robot arm reinforcement learning project for learning simple cube pick-and-place tasks

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Pipeline                        │
│                                                              │
│  ┌────────────┐      ┌──────────────┐      ┌─────────────┐ │
│  │   Train    │─────▶│  PPO Agent   │─────▶│   Model     │ │
│  │   Script   │      │ (SB3 + PT)   │      │   Storage   │ │
│  └────────────┘      └──────────────┘      └─────────────┘ │
│         │                    │                              │
│         │                    │                              │
│         ▼                    ▼                              │
│  ┌────────────────────────────────────────┐                │
│  │       Custom Gym Environment           │                │
│  │  ┌──────────────┐  ┌──────────────┐   │                │
│  │  │ Observation  │  │   Reward     │   │                │
│  │  │   Space      │  │   Function   │   │                │
│  │  └──────────────┘  └──────────────┘   │                │
│  │  ┌──────────────┐  ┌──────────────┐   │                │
│  │  │   Action     │  │   Episode    │   │                │
│  │  │   Space      │  │   Manager    │   │                │
│  │  └──────────────┘  └──────────────┘   │                │
│  └────────────────────────────────────────┘                │
│                        │                                    │
│                        ▼                                    │
│  ┌────────────────────────────────────────┐                │
│  │         PyBullet Simulation            │                │
│  │  ┌──────────────┐  ┌──────────────┐   │                │
│  │  │  Kuka iiwa   │  │    Cube      │   │                │
│  │  │   Robot      │  │   Object     │   │                │
│  │  └──────────────┘  └──────────────┘   │                │
│  │  ┌──────────────┐  ┌──────────────┐   │                │
│  │  │   Physics    │  │  Collision   │   │                │
│  │  │   Engine     │  │   Detection  │   │                │
│  │  └──────────────┘  └──────────────┘   │                │
│  └────────────────────────────────────────┘                │
│                        │                                    │
│                        ▼                                    │
│  ┌────────────────────────────────────────┐                │
│  │         TensorBoard Logging            │                │
│  └────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Evaluation Pipeline                        │
│                                                              │
│  ┌────────────┐      ┌──────────────┐      ┌─────────────┐ │
│  │   Test     │─────▶│ Trained PPO  │─────▶│Visualization│ │
│  │   Script   │      │    Model     │      │   & Stats   │ │
│  └────────────┘      └──────────────┘      └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. PyBullet Simulation Layer
- **Purpose**: Provide physics simulation and robot environment
- **Responsibilities**:
  - Load and control Kuka iiwa robot model
  - Create cube objects and handle physics interactions
  - Collision detection and physics engine updates
  - Visualization (during evaluation)

### 2. Custom Gym Environment
- **Purpose**: Provide standard reinforcement learning interface
- **Responsibilities**:
  - Define observation space and collect state information
  - Define action space and control robot
  - Calculate rewards and manage episodes
  - Implement reset and step logic

### 3. PPO Agent (Stable-Baselines3)
- **Purpose**: Execute reinforcement learning algorithm
- **Responsibilities**:
  - Train policy network
  - Train value network
  - Collect experience and perform updates
  - Save and load models

### 4. Training Pipeline
- **Purpose**: Manage overall training process
- **Responsibilities**:
  - Initialize and configure environment
  - Create PPO agent and execute training
  - Log to TensorBoard
  - Save model checkpoints

### 5. Evaluation Pipeline
- **Purpose**: Evaluate and visualize trained models
- **Responsibilities**:
  - Load trained models
  - Execute evaluation episodes
  - Collect success rate and performance metrics
  - Render visualization

## Technology Stack

### Core Framework
- **Python**: 3.8+
- **PyBullet**: Physics simulation
- **Gymnasium**: RL environment standard interface
- **PyTorch**: Deep learning backend
- **Stable-Baselines3**: RL algorithm library

### Supporting Tools
- **TensorBoard**: Training monitoring
- **NumPy**: Numerical computation
- **VS Code**: Development environment

### Infrastructure
- **OS**: WSL2 Ubuntu
- **GPU**: NVIDIA GeForce RTX 3060 (12GB)
- **CUDA**: PyTorch GPU acceleration

## Design Principles

### 1. Simplicity First
- Minimize complexity for beginner learning
- Simplified gripper (auto-attach)
- Lower difficulty with fixed starting positions

### 2. Modularity
- Independent component design
- Separate environment, agent, and evaluation
- Reusable structure

### 3. Extensibility
- Structure allowing future complexity increase
- Configurable environment parameters
- Enable experimentation with various reward functions

### 4. Observability
- Visualize learning process through TensorBoard
- Clear logging and metrics
- Debugging-friendly structure

## Data Flow

### Training Loop
```
1. Environment.reset()
   ↓
2. Agent.predict(observation)
   ↓
3. Environment.step(action)
   ↓
4. Compute reward
   ↓
5. Store transition
   ↓
6. Update policy (PPO)
   ↓
7. Log to TensorBoard
   ↓
8. Repeat until done
```

### Evaluation Loop
```
1. Load trained model
   ↓
2. Environment.reset()
   ↓
3. Agent.predict(observation, deterministic=True)
   ↓
4. Environment.step(action)
   ↓
5. Render visualization
   ↓
6. Collect metrics
   ↓
7. Repeat for N episodes
   ↓
8. Report statistics
```

## Key Design Decisions

### 1. PPO Algorithm Choice
- **Reason**: Beginner-friendly, stable learning, well-tuned hyperparameters
- **Alternatives**: SAC, TD3 (more complex, powerful for continuous control)

### 2. Joint Velocity Control
- **Reason**: Smooth control, physically realistic
- **Alternatives**: Joint position (less smooth), End-effector control (more complex)

### 3. Shaped Reward
- **Reason**: Faster learning, clear learning signals
- **Alternatives**: Sparse reward (slow learning), Dense reward (overfitting risk)

### 4. Simplified Gripper
- **Reason**: Reduced complexity, focus on learning objectives
- **Alternatives**: Real gripper control (more realistic, much more complex)

### 5. Fixed Starting Position
- **Reason**: Progressive difficulty increase, easier debugging
- **Future**: Generalize to random positions

## Performance Targets

### Training
- **Time**: ~1 hour (100K-200K timesteps)
- **GPU Usage**: RTX 3060 utilization
- **Memory**: <8GB

### Inference
- **Success Rate**: >80% (fixed positions)
- **Episode Length**: Average 50-100 steps
- **Inference Speed**: Real-time capable

## Related Documents
- [01_ENVIRONMENT_DESIGN.md](./01_ENVIRONMENT_DESIGN.md) - Detailed Gym environment design
- [02_RL_ALGORITHM_DESIGN.md](./02_RL_ALGORITHM_DESIGN.md) - PPO algorithm configuration
- [03_REWARD_DESIGN.md](./03_REWARD_DESIGN.md) - Detailed reward function design
- [04_TRAINING_PIPELINE.md](./04_TRAINING_PIPELINE.md) - Training pipeline
- [05_CONFIG_HYPERPARAMETERS.md](./05_CONFIG_HYPERPARAMETERS.md) - Configuration and hyperparameters
- [06_PROJECT_STRUCTURE.md](./06_PROJECT_STRUCTURE.md) - Project directory structure
