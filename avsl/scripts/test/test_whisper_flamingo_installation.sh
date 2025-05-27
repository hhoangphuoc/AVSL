#!/bin/bash

# Test script to verify Whisper-Flamingo installation

echo "=== Testing Whisper-Flamingo Installation ==="

# Activate conda environment
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo "Testing in conda environment: $PYTHON_VIRTUAL_ENVIRONMENT"

cd /home/s2587130/AVSL

# Test 1: Basic fairseq import
echo "Test 1: Basic fairseq import..."
python -c "
import fairseq
print('✓ Basic fairseq import successful')
print('  Available modules:', [x for x in dir(fairseq) if not x.startswith('_')][:5], '...')
" || { echo "✗ Basic fairseq import failed"; exit 1; }

# Test 2: Fairseq checkpoint_utils and utils
echo "Test 2: Fairseq checkpoint_utils and utils..."
python -c "
import fairseq
checkpoint_utils = fairseq.checkpoint_utils
utils = fairseq.utils
print('✓ checkpoint_utils and utils accessible')
print('  load_model_ensemble_and_task available:', hasattr(checkpoint_utils, 'load_model_ensemble_and_task'))
" || { echo "✗ Fairseq utils access failed"; exit 1; }

# Test 3: Whisper-Flamingo import
echo "Test 3: Whisper-Flamingo import..."
python -c "
import sys
sys.path.insert(0, '/home/s2587130/AVSL')
import whisper_flamingo.whisper as whisper
print('✓ Whisper-Flamingo import successful')
print('  Available functions:', [x for x in dir(whisper) if not x.startswith('_')][:5], '...')
" || { echo "✗ Whisper-Flamingo import failed"; exit 1; }

# Test 4: Training script import
echo "Test 4: Training script import..."
cd /home/s2587130/AVSL/avsl
python -c "
import whisper_flamingo_ft_ami
print('✓ Training script import successful')
" || { echo "✗ Training script import failed"; exit 1; }

# Test 5: Check required model files
echo "Test 5: Checking model files..."
MODELS_DIR="/home/s2587130/AVSL/avsl/models"
if [ -f "$MODELS_DIR/large_noise_pt_noise_ft_433h_only_weights.pt" ]; then
    echo "✓ AV-HuBERT weights found"
else
    echo "⚠ AV-HuBERT weights not found at $MODELS_DIR/large_noise_pt_noise_ft_433h_only_weights.pt"
    echo "  Run: bash scripts/download_models.sh"
fi

if [ -f "$MODELS_DIR/whisper_en_large.pt" ]; then
    echo "✓ Whisper model weights found"
else
    echo "⚠ Whisper model weights not found at $MODELS_DIR/whisper_en_large.pt"
    echo "  Run: bash scripts/download_models.sh"
fi

# Test 6: Check data paths
echo "Test 6: Checking data paths..."
DATA_PATHS=(
    "/home/s2587130/AVSL/data/ami/av_hubert/train"
    "/home/s2587130/AVSL/data/ami/av_hubert/validation"
    "/home/s2587130/AVSL/data/ami/av_hubert/test"
)

for path in "${DATA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "✓ Data path exists: $path"
    else
        echo "⚠ Data path not found: $path"
    fi
done

echo ""
echo "=== Installation Test Summary ==="
echo "✓ Fairseq: Working"
echo "✓ Whisper-Flamingo: Working"
echo "✓ Training script: Working"
echo ""
echo "If all tests passed, you can now run:"
echo "  sbatch scripts/whisper_flamingo_ft.sh"
echo ""
echo "If model files are missing, run:"
echo "  bash scripts/download_models.sh" 