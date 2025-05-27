# AV-HuBERT Registration in Whisper-Flamingo Fix

This document explains the AV-HuBERT task registration issue in your whisper-flamingo implementation and provides the complete solution.

## The Problem

When using AV-HuBERT with fairseq, you encounter the error:
```
ValueError: Cannot register duplicate model (av_hubert)
```

This happens because:
1. Fairseq automatically registers model tasks when modules are imported
2. AV-HuBERT models get registered multiple times during the import process
3. The fairseq registration system doesn't handle duplicate registrations gracefully

## The Solution

The fix involves two main components:

### 1. Correct Path Setup
- Proper `sys.path` configuration to ensure all modules are found
- Correct `av_hubert_path` pointing to the `avhubert` directory within the `av_hubert` repository
- Passing the correct path to `whisper.load_model()`

### 2. Dummy Argument Workaround
- Adding a `'dummy'` argument to `sys.argv` before importing fairseq modules
- This prevents the duplicate registration error (known workaround from AV-HuBERT GitHub issues)

## Files Modified

1. **`whisper_flamingo_ft_ami.py`** - Main training script with the fix
2. **`test_av_hubert_fix.py`** - Test script to verify the fix works
3. **`config/ami_whisper_flamingo_large.yaml`** - Configuration file for AMI dataset

## How to Use

### Step 1: Verify Your Setup

Ensure you have the following directory structure:
```
AVSL/
├── whisper_flamingo/           # Cloned from roudimit/whisper-flamingo
│   ├── av_hubert/              # Submodule
│   │   ├── avhubert/           # This is the user_dir for fairseq
│   │   └── fairseq/            # Fairseq installation
│   └── whisper/
├── avsl/
│   ├── whisper_flamingo_ft_ami.py
│   ├── test_av_hubert_fix.py
│   └── config/
└── data/ami/av_hubert/         # Your AMI HF datasets
```

### Step 2: Download Required Models

Ensure you have the AV-HuBERT checkpoint:
```bash
# Create models directory if it doesn't exist
mkdir -p /home/s2587130/AVSL/avsl/models

# Download the AV-HuBERT weights (example - adjust URL as needed)
wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt \
     -O /home/s2587130/AVSL/avsl/models/large_noise_pt_noise_ft_433h_only_weights.pt
```

### Step 3: Test the Fix

Run the test script to verify everything works:
```bash
cd /home/s2587130/AVSL/avsl
python test_av_hubert_fix.py config/ami_whisper_flamingo_large.yaml
```

Expected output:
```
✓ Added fairseq path to sys.path: ...
✓ AV-HuBERT user dir set to: ...
✓ Fairseq imported successfully from: ...
✓ Adding dummy argument to prevent AV-HuBERT task registration conflicts...
============================================================
Testing AV-HuBERT Task Registration Fix
============================================================
...
🎉 ALL TESTS PASSED!
✓ AV-HuBERT task registration fix is working correctly
✓ Your whisper_flamingo_ft_ami.py should now work properly
============================================================
```

### Step 4: Run Training

If the test passes, run the training script:
```bash
cd /home/s2587130/AVSL/avsl
python whisper_flamingo_ft_ami.py config/ami_whisper_flamingo_large.yaml
```

## Configuration Notes

### Key Config Parameters

In your `ami_whisper_flamingo_large.yaml`:

```yaml
# Video model settings
use_av_hubert_encoder: True           # Enable AV-HuBERT encoder
add_gated_x_attn: 1                   # Enable Flamingo-style cross-attention
av_fusion: separate                   # AV fusion strategy
video_model_ckpt: '/path/to/large_noise_pt_noise_ft_433h_only_weights.pt'

# Training settings
model_name: large-v2                  # Whisper model size
batch_size: 2                         # Reduced for large model
gradient_accumulation_steps: 4        # To maintain effective batch size
```

### Path Requirements

The fix automatically sets up these paths:
- `avhubert_user_dir`: Points to `whisper_flamingo/av_hubert/avhubert/`
- `fairseq_path`: Points to `whisper_flamingo/av_hubert/fairseq/`
- All paths are added to `sys.path` for proper imports

## Troubleshooting

### Common Issues

1. **"Cannot register duplicate model" still appears**
   - Ensure the dummy argument fix is applied before any fairseq imports
   - Check that you're using the modified `whisper_flamingo_ft_ami.py`

2. **"Could not infer task type"**
   - Verify the AV-HuBERT checkpoint file exists and is accessible
   - Check that the `av_hubert` submodule is properly initialized

3. **Import errors**
   - Ensure the whisper-flamingo repository is properly cloned
   - Verify all submodules are initialized: `git submodule update --init --recursive`

4. **FileNotFoundError for video_model_ckpt**
   - Download the AV-HuBERT checkpoint file to the specified path
   - Update the path in your config file

### Debug Commands

```bash
# Check git submodules
cd /home/s2587130/AVSL/whisper_flamingo
git submodule status

# Verify fairseq installation
cd /home/s2587130/AVSL/whisper_flamingo/av_hubert/fairseq
pip list | grep fairseq

# Test basic imports
python -c "import sys; sys.path.insert(0, '/path/to/whisper_flamingo'); import fairseq; print('OK')"
```

## Technical Details

### What the Fix Does

1. **Path Management**: Ensures all required directories are in `sys.path`
2. **Import Order**: Pre-imports fairseq before other operations
3. **Dummy Argument**: Adds `'dummy'` to `sys.argv` to prevent registration conflicts
4. **User Module Loading**: Properly calls `utils.import_user_module()` with correct path

### Why It Works

The dummy argument workaround works because:
- AV-HuBERT's model registration logic checks for command-line arguments
- When no arguments are present, it triggers a special debug mode that causes registration conflicts
- Adding a dummy argument prevents this debug mode from activating

## References

- [AV-HuBERT GitHub Issue #36](https://github.com/facebookresearch/av_hubert/issues/36)
- [Original whisper-flamingo repository](https://github.com/roudimit/whisper-flamingo)
- [AV-HuBERT repository](https://github.com/facebookresearch/av_hubert)

## Contact

If you encounter issues with this fix, please:
1. Run the test script and share the output
2. Check the troubleshooting section above
3. Verify your directory structure matches the expected layout 