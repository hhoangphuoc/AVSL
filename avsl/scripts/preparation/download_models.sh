#!/bin/bash

# Script to download required model weights for Whisper-Flamingo

set -e  # Exit on any error

echo "=== Downloading Whisper-Flamingo Model Weights ==="

# Create models directory
MODELS_DIR="/home/s2587130/AVSL/avsl/models"
mkdir -p "$MODELS_DIR"

echo "Models will be downloaded to: $MODELS_DIR"

# Download AV-HuBERT weights (required for video processing)
echo "Downloading AV-HuBERT weights..."
if [ ! -f "$MODELS_DIR/large_noise_pt_noise_ft_433h_only_weights.pt" ]; then
    wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/large_noise_pt_noise_ft_433h_only_weights.pt -P "$MODELS_DIR"
    echo "✓ Downloaded AV-HuBERT weights"
else
    echo "✓ AV-HuBERT weights already exist"
fi

# Download pre-trained Whisper model (audio-only, fine-tuned on English)
echo "Downloading pre-trained Whisper model..."
if [ ! -f "$MODELS_DIR/whisper_en_large.pt" ]; then
    wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper_en_large.pt -P "$MODELS_DIR"
    echo "✓ Downloaded Whisper English Large model"
else
    echo "✓ Whisper English Large model already exists"
fi

# Optional: Download a pre-trained Whisper-Flamingo model for reference
echo "Downloading pre-trained Whisper-Flamingo model (optional)..."
if [ ! -f "$MODELS_DIR/whisper-flamingo_en_large.pt" ]; then
    wget https://data.csail.mit.edu/public-release-sls/whisper-flamingo/models/whisper-flamingo_en_large.pt -P "$MODELS_DIR"
    echo "✓ Downloaded Whisper-Flamingo English Large model"
else
    echo "✓ Whisper-Flamingo English Large model already exists"
fi

echo ""
echo "=== Download completed! ==="
echo "Downloaded models:"
ls -lh "$MODELS_DIR"/*.pt

echo ""
echo "Models are ready for training!" 