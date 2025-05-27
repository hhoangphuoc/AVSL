#!/bin/bash

# Setup script for Whisper-Flamingo environment
# Based on the original repository instructions

set -e  # Exit on any error

echo "Setting up Whisper-Flamingo environment..."

# Check if we're in the correct directory
if [ ! -d "whisper_flamingo" ]; then
    echo "Error: whisper_flamingo directory not found. Please run this script from the AVSL root directory."
    exit 1
fi

cd whisper_flamingo

# Check if av_hubert directory exists
if [ ! -d "av_hubert" ]; then
    echo "Cloning av_hubert repository..."
    # Clone the "muavic" branch of av_hubert's repo
    git clone -b muavic https://github.com/facebookresearch/av_hubert.git
fi

cd av_hubert

# Initialize and update submodules (this includes fairseq)
echo "Initializing and updating submodules..."
git submodule init
git submodule update

# Install av-hubert's requirements
echo "Installing av-hubert requirements..."
pip install -r requirements.txt

# Install fairseq
echo "Installing fairseq..."
cd fairseq
pip install --editable ./

echo "Fairseq installation completed!"

# Go back to av_hubert directory
cd ..

# Apply the fix for mWhisper-Flamingo (comment out line 624 and add the fix)
echo "Applying mWhisper-Flamingo fix to avhubert/hubert.py..."
HUBERT_FILE="avhubert/hubert.py"
if [ -f "$HUBERT_FILE" ]; then
    # Check if the fix is already applied
    if ! grep -q "features_audio = torch.zeros_like(features_video)" "$HUBERT_FILE"; then
        # Create a backup
        cp "$HUBERT_FILE" "${HUBERT_FILE}.backup"
        
        # Apply the fix using sed
        # Comment out line 624 and add the fix after line 625
        sed -i '624s/^/#/' "$HUBERT_FILE"
        sed -i '625a\                features_audio = torch.zeros_like(features_video)' "$HUBERT_FILE"
        
        echo "Applied mWhisper-Flamingo fix to $HUBERT_FILE"
    else
        echo "mWhisper-Flamingo fix already applied to $HUBERT_FILE"
    fi
else
    echo "Warning: $HUBERT_FILE not found. You may need to apply the fix manually."
fi

# Go back to whisper_flamingo directory
cd ..

echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Download the required model weights:"
echo "   mkdir -p ../avsl/models"
echo "   wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt -P ../avsl/models/"
echo "   wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en_large.pt -P ../avsl/models/"
echo ""
echo "2. Test the installation by running a simple import test:"
echo "   python -c \"from av_hubert.fairseq.fairseq import checkpoint_utils, utils; print('Fairseq import successful!')\"" 