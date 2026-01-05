# PyBullet Kuka Pick-and-Place RL

Reinforcement learning project for training a Kuka iiwa robot arm to perform pick-and-place tasks using PyBullet and PPO.

## Features
- Custom Gymnasium environment with Kuka iiwa robot
- PPO algorithm (Stable-Baselines3)
- TensorBoard logging
- GPU acceleration (CUDA)
- Shaped reward function for efficient learning

## System Requirements
- **OS**: Windows 11 with WSL2 Ubuntu
- **GPU**: NVIDIA GeForce RTX 3060 (12GB) or similar
- **Python**: 3.8+
- **CUDA**: Compatible with PyTorch

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

Training will take approximately 1 hour on RTX 3060 for 200K timesteps.

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

## Expected Results
- **Training time**: ~1 hour (200K timesteps on RTX 3060)
- **Success rate**: >80% (fixed positions)
- **GPU memory**: ~2-3 GB
- **Episode length**: 60-100 steps (successful episodes)

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
- Make sure you're using GPU (check `config.py` device setting)
- Verify CUDA is available
- Use DIRECT mode (not GUI) during training

## Next Steps

After completing basic training:

1. **Increase difficulty**: Randomize cube positions in `config.py`
2. **Try different algorithms**: Experiment with SAC or TD3
3. **Add visual observations**: Implement camera-based observations
4. **Real gripper control**: Replace simplified gripper with actual control
5. **Multi-object tasks**: Extend to multiple cubes

## License
Educational/Research Project

## Acknowledgments
- PyBullet for physics simulation
- Stable-Baselines3 for RL algorithms
- Gymnasium for environment interface
