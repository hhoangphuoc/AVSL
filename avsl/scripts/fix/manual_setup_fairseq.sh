#!/bin/bash

# Manual setup script for fairseq installation
# Run this script manually before submitting the SLURM job

set -e  # Exit on any error

echo "=== Manual Fairseq Setup for Whisper-Flamingo ==="

# Activate the conda environment
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo "Activated conda environment: $PYTHON_VIRTUAL_ENVIRONMENT"

# Navigate to the whisper_flamingo directory
cd /home/s2587130/AVSL/whisper_flamingo

# Check if av_hubert directory exists and has fairseq
if [ ! -d "av_hubert/fairseq" ]; then
    echo "Error: av_hubert/fairseq directory not found."
    echo "Please ensure the av_hubert submodule is properly initialized."
    echo "Run: cd av_hubert && git submodule init && git submodule update"
    exit 1
fi

# Navigate to fairseq directory and install
echo "Installing fairseq from av_hubert/fairseq..."
cd av_hubert/fairseq

# Downgrade pip as recommended in the original instructions
echo "Downgrading pip to version 24.0..."
python -m pip install pip==24.0

# Install fairseq in editable mode
echo "Installing fairseq in editable mode..."
pip install --editable ./

echo "Fairseq installation completed!"

# Fix NumPy compatibility issues
echo "Fixing NumPy compatibility issues..."
cd /home/s2587130/AVSL/avsl/scripts/fix
bash fix_numpy_compatibility.sh

# Test the installation
echo "Testing fairseq import..."
cd /home/s2587130/AVSL
python -c "
import sys
sys.path.insert(0, '/home/s2587130/AVSL/whisper_flamingo/av_hubert')
from fairseq import checkpoint_utils, utils
print('✓ Fairseq import successful!')
"

echo "=== Setup completed successfully! ==="
echo ""
echo "You can now submit your SLURM job with:"
echo "sbatch scripts/whisper_flamingo_ft.sh" 