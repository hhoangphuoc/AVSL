# Whisper-Flamingo Setup and Fine-tuning Guide

This guide provides step-by-step instructions to set up and fine-tune Whisper-Flamingo on your HPC cluster.

## Issue Analysis

The main issues you encountered were:

1. **Missing fairseq module**:
```
ModuleNotFoundError: No module named 'fairseq'
```

2. **NumPy compatibility issues**:
```
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`.
```

These happen because:
- The AV-HuBERT component requires `fairseq`, but it wasn't properly installed
- Fairseq uses deprecated NumPy aliases that were removed in newer NumPy versions

## Solution Overview

1. **Install fairseq from the av_hubert submodule**
2. **Fix NumPy compatibility issues in fairseq**
3. **Fix fairseq import pattern in Whisper-Flamingo**
4. **Download required model weights**
5. **Run the training**

## Step-by-Step Setup

### Step 1: Install fairseq and fix NumPy compatibility

Run the manual setup script to install fairseq:

```bash
cd /home/s2587130/AVSL/avsl
bash scripts/fix/manual_setup_fairseq.sh
```

This script will:
- Activate your conda environment
- Navigate to the av_hubert/fairseq directory
- Downgrade pip to version 24.0 (as recommended)
- Install fairseq in editable mode
- Fix NumPy compatibility issues (deprecated `np.float`, `np.int`, etc.)
- Test the installation

**Note**: If you encounter NumPy compatibility errors, you can also run the quick fix:
```bash
bash scripts/quick_numpy_fix.sh
```

### Step 2: Download Model Weights

Download the required pre-trained models:

```bash
cd /home/s2587130/AVSL
bash scripts/download_models.sh
```

This will download:
- `large_noise_pt_noise_ft_433h_only_weights.pt` (AV-HuBERT weights)
- `whisper_en_large.pt` (Pre-trained Whisper model)
- `whisper-flamingo_en_large.pt` (Optional reference model)

### Step 3: Verify Installation

Run the comprehensive test script:

```bash
cd /home/s2587130/AVSL
bash scripts/test_installation.sh
```

This will test:
- Basic fairseq import
- Fairseq checkpoint_utils and utils access
- Whisper-Flamingo import
- Training script import
- Model file availability
- Data path existence

If all tests pass, you'll see:
```
✓ Fairseq: Working
✓ Whisper-Flamingo: Working
✓ Training script: Working
```

### Step 4: Submit Training Job

Now you can submit your SLURM job:

```bash
cd /home/s2587130/AVSL/avsl
sbatch scripts/whisper_flamingo_ft.sh
```

## Key Changes Made

### 1. Fixed Import Paths

Updated `avsl/whisper_flamingo_ft_ami.py` to properly add the whisper_flamingo and av_hubert paths:

```python
# Add paths for whisper_flamingo and av_hubert
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo')
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert')

# Add to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, whisper_flamingo_path)
sys.path.insert(0, av_hubert_path)
```

### 2. Added Fairseq Installation Check

Updated `scripts/whisper_flamingo_ft.sh` to check for fairseq installation and run setup if needed.

### 3. Created Setup Scripts

- `scripts/manual_setup_fairseq.sh`: Manual fairseq installation with NumPy fixes
- `scripts/download_models.sh`: Download required model weights
- `scripts/setup_whisper_flamingo_env.sh`: Complete environment setup
- `scripts/fix_numpy_compatibility.sh`: Comprehensive NumPy compatibility fix
- `scripts/quick_numpy_fix.sh`: Quick fix for immediate NumPy issues
- `scripts/test_installation.sh`: Comprehensive installation verification

## Configuration Details

Your configuration file `avsl/config/ami_whisper_flamingo_large.yaml` is properly set up with:

- **Model**: Whisper Large-V2 with AV-HuBERT encoder
- **Batch size**: 2 (suitable for large model on single GPU)
- **Gradient accumulation**: 4 steps
- **Learning rate**: 1e-5
- **Training steps**: 8000
- **Video model**: AV-HuBERT with gated cross-attention

## Troubleshooting

### If fairseq import still fails:

1. Check that the av_hubert submodule is properly initialized:
```bash
cd /home/s2587130/AVSL/whisper_flamingo/av_hubert
git submodule status
```

2. If submodules are not initialized:
```bash
cd /home/s2587130/AVSL/whisper_flamingo/av_hubert
git submodule init
git submodule update
```

3. Reinstall fairseq:
```bash
cd /home/s2587130/AVSL/whisper_flamingo/av_hubert/fairseq
pip install --editable ./
```

### If model loading fails:

1. Verify model files exist:
```bash
ls -la /home/s2587130/AVSL/avsl/models/
```

2. Check file permissions and sizes
3. Re-download if necessary

### If CUDA/GPU issues:

1. Check GPU availability:
```bash
nvidia-smi
```

2. Verify CUDA version compatibility with PyTorch
3. Adjust precision settings in config if needed

## Expected Training Behavior

1. **Initialization**: Model loads with AV-HuBERT encoder
2. **Token embedding resize**: Adds custom `<|laugh|>` token
3. **Freezing**: Video model weights are frozen (only cross-attention trains)
4. **Training**: Gated cross-attention layers learn to fuse audio-visual features
5. **Validation**: Evaluates on both test and validation sets

## Monitoring Training

1. **Logs**: Check SLURM output in `logs/whisper_flamingo_ft_*.log`
2. **TensorBoard**: Monitor training progress
3. **Checkpoints**: Saved in `/home/s2587130/AVSL/avsl/checkpoints/whisper_flamingo_large_ft`

## Next Steps After Training

1. **Evaluate**: Test the trained model on your AMI dataset
2. **Compare**: Compare with audio-only Whisper baseline
3. **Fine-tune**: Adjust hyperparameters if needed
4. **Deploy**: Use the trained model for inference

## References

- [Original Whisper-Flamingo Repository](https://github.com/roudimit/whisper-flamingo)
- [AV-HuBERT Repository](https://github.com/facebookresearch/av_hubert)
- [Whisper-Flamingo Paper](https://arxiv.org/abs/2406.10082) 