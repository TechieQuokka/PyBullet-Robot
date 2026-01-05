# Setup Guide

Step-by-step guide to set up the PyBullet Kuka Pick-and-Place RL environment.

## Prerequisites

- Windows 11 with WSL2 Ubuntu
- NVIDIA GeForce RTX 3060 (12GB) or similar
- Python 3.8+
- VS Code (recommended)

## Step 1: Verify System

### Check WSL2 Ubuntu
```bash
lsb_release -a
# Should show Ubuntu
```

### Check GPU
```bash
nvidia-smi
# Should show your GPU (e.g., RTX 3060)
```

### Check Python
```bash
python3 --version
# Should be Python 3.8 or higher
```

## Step 2: Create Virtual Environment (Recommended)

```bash
# Navigate to project directory
cd /home/beethoven/workspace/deeplearning/reinforcement/pyBullet/robot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# This will install:
# - gymnasium
# - stable-baselines3
# - torch (with CUDA support)
# - tensorboard
# - pybullet
# - numpy
# - matplotlib
```

**Note**: If torch doesn't detect CUDA, install it explicitly:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Step 4: Verify CUDA

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA GeForce RTX 3060
```

## Step 5: Download Robot Models

```bash
python scripts/download_urdf.py
```

Expected output:
```
PyBullet data path: /path/to/pybullet_data
Kuka URDF path: /path/to/pybullet_data/kuka_iiwa
✓ Copied Kuka URDF to ./urdf/kuka_iiwa
```

## Step 6: Test Environment

```bash
python scripts/test_env.py
```

Expected output:
```
==================================================
PyBullet Pick-and-Place Environment Tests
==================================================

Testing environment basic functionality...
--------------------------------------------------
✓ Reset works, observation shape: (17,)
✓ Step works, reward: -0.011
✓ Episode completed in 100 steps, total reward: -1.100

✓ All basic tests passed!

... (more tests)

==================================================
✓ ALL TESTS PASSED!
==================================================
```

## Step 7: Quick Training Test (Optional)

Run a quick training test with reduced timesteps:

```bash
# Modify config.py temporarily
# Change: total_timesteps = 10_000

python train.py
```

This should run for about 5 minutes and verify everything works.

## Step 8: Full Training

Once everything is verified:

```bash
# Restore config.py
# Change back: total_timesteps = 200_000

# Start training
python train.py
```

## Step 9: Monitor Training

In a new terminal:

```bash
# Activate virtual environment if using one
source venv/bin/activate

# Start TensorBoard
tensorboard --logdir=./logs/tensorboard/

# Open browser to http://localhost:6006
```

## Troubleshooting

### Issue: CUDA Not Available

**Symptoms**:
```
CUDA available: False
```

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify again
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: PyBullet Import Error

**Symptoms**:
```
ImportError: libpython3.x.so: cannot open shared object file
```

**Solution**:
```bash
sudo apt-get update
sudo apt-get install python3-dev
pip install --upgrade pybullet
```

### Issue: Permission Denied

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
```

**Solution**:
```bash
# Make sure you're in the project directory
cd /home/beethoven/workspace/deeplearning/reinforcement/pyBullet/robot

# Check directory permissions
ls -la

# If needed, fix permissions
chmod +x scripts/*.py
```

### Issue: Module Not Found

**Symptoms**:
```
ModuleNotFoundError: No module named 'env'
```

**Solution**:
```bash
# Make sure you're running from project root
cd /home/beethoven/workspace/deeplearning/reinforcement/pyBullet/robot

# Run the script
python train.py
```

### Issue: Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
Edit `config.py`:
```python
# Reduce batch size
batch_size = 32  # instead of 64

# Or use smaller network
policy_net_arch = [32, 32]  # instead of [64, 64]
```

## VS Code Setup (Optional)

### Install Python Extension

1. Open VS Code
2. Install "Python" extension by Microsoft
3. Select Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose your venv

### Recommended Extensions

- Python (Microsoft)
- Pylance
- Python Debugger

### Launch Configuration

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/train.py",
            "console": "integratedTerminal"
        },
        {
            "name": "Test Environment",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/test_env.py",
            "console": "integratedTerminal"
        }
    ]
}
```

## Next Steps

After successful setup:

1. **Read Documentation**: Check `docs/` directory for detailed design docs
2. **Start Training**: Run `python train.py`
3. **Monitor Progress**: Use TensorBoard
4. **Evaluate Results**: Run `python test.py` after training
5. **Experiment**: Modify `config.py` to try different settings

## Quick Reference Commands

```bash
# Activate environment (if using venv)
source venv/bin/activate

# Download models
python scripts/download_urdf.py

# Test environment
python scripts/test_env.py

# Train
python train.py

# Monitor (in new terminal)
tensorboard --logdir=./logs/tensorboard/

# Evaluate
python test.py --model ./models/best/best_model.zip --episodes 100

# Evaluate with rendering
python test.py --model ./models/best/best_model.zip --episodes 10 --render
```

## Expected Timeline

- **Setup**: 10-15 minutes
- **Testing**: 2-3 minutes
- **Training**: ~1 hour (200K timesteps on RTX 3060)
- **Evaluation**: 5-10 minutes

## Support

If you encounter issues not covered here, check:
1. Documentation in `docs/` directory
2. PyBullet documentation
3. Stable-Baselines3 documentation
4. Gymnasium documentation
