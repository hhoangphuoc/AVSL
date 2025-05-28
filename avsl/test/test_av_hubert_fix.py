#!/usr/bin/env python3
"""
Test script to verify AV-HuBERT task registration fix.

This script tests if the AV-HuBERT models can be loaded properly without 
the "Cannot register duplicate model" error.

Usage:
    python test_av_hubert_fix.py [config_file]
"""

import os
import sys
import yaml
import types
import torch

# Add paths for whisper_flamingo and av_hubert (same as in whisper_flamingo_ft_ami.py)
current_dir = os.path.dirname(os.path.abspath(__file__)) # avsl/test
parent_dir = os.path.dirname(current_dir) # avsl
project_root = os.path.dirname(parent_dir) # AVSL
print(f"Project root: {project_root}")
print(f"Parent dir: {parent_dir}")
print(f"Current dir: {current_dir}")

utils_path = os.path.join(project_root, 'utils') # AVSL/utils
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo') # AVSL/whisper_flamingo
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert') # AVSL/whisper_flamingo/av_hubert

# Add to Python path
sys.path.insert(0, project_root) # AVSL
sys.path.insert(0, utils_path) # AVSL/utils
sys.path.insert(0, whisper_flamingo_path) # AVSL/whisper_flamingo
sys.path.insert(0, av_hubert_path) # AVSL/whisper_flamingo/av_hubert


fairseq_path = os.path.join(av_hubert_path, 'fairseq')
if os.path.exists(fairseq_path) and fairseq_path not in sys.path:
    sys.path.insert(0, fairseq_path)
    print(f"✓ Added fairseq path to sys.path: {fairseq_path}")

# Set the correct av_hubert path for user module import
avhubert_user_dir = os.path.join(av_hubert_path, 'avhubert')
print(f"✓ AV-HuBERT user dir set to: {avhubert_user_dir}")

#===============================================================================================================
# Pre-import fairseq to ensure it's loaded correctly
try:
    import fairseq
    print(f"✓ Fairseq imported successfully from: {fairseq.__file__}")
    # Verify that the required modules are accessible
    _ = fairseq.checkpoint_utils
    _ = fairseq.utils
    print("✓ Fairseq checkpoint_utils and utils are accessible")
except Exception as e:
    print(f"⚠ Warning: Fairseq import issue: {e}")
    print("Continuing with the assumption that the original repository fix is in place...")


#===============================================================================================================
# FIXME: IS THIS NEEDED?
# CRITICAL FIX: Add dummy argument to prevent AV-HuBERT duplicate model registration
# This is a known issue with AV-HuBERT: https://github.com/facebookresearch/av_hubert/issues/36
# The workaround is to add a dummy command line argument to prevent the registration conflict
if 'dummy' not in sys.argv:
    print("✓ Adding dummy argument to prevent AV-HuBERT task registration conflicts...")
    sys.argv.append('dummy')
    print(f"✓ sys.argv is now: {sys.argv}")
else:
    print("✓ Dummy argument already present in sys.argv")
#===============================================================================================================

# Import the Whisper modules
import whisper_flamingo.whisper as whisper


def test_av_hubert_loading(config_path=None):
    """Test AV-HuBERT model loading with different configurations."""
    
    print("=" * 60)
    print("Testing AV-HuBERT Task Registration Fix")
    print("=" * 60)
    
    # Default test configuration
    default_config = {
        'model_name': 'large-v2',
        'video_model_ckpt': '/home/s2587130/AVSL/avsl/models/large_noise_pt_noise_ft_433h_only_weights.pt',
        'use_av_hubert_encoder': True,
        'av_fusion': 'separate',
        'add_gated_x_attn': 1,
        'dropout_rate': 0.0,
        'prob_use_av': 1.0,
        'prob_use_a': 0.0,
    }
    
    # Load config from file if provided
    if config_path and os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as file:
            file_config = yaml.safe_load(file)
            default_config.update(file_config)
    else:
        print("Using default test configuration")
    
    cfg = types.SimpleNamespace(**default_config)
    
    # Check if video model checkpoint exists
    if not os.path.exists(cfg.video_model_ckpt):
        print(f"⚠ Warning: Video model checkpoint not found at: {cfg.video_model_ckpt}")
        print("The model loading test will fail, but we can still test the path setup")
        return False
    
    print(f"Model configuration:")
    print(f"  Model name: {cfg.model_name}")
    print(f"  Video model checkpoint: {cfg.video_model_ckpt}")
    print(f"  AV-HuBERT encoder: {cfg.use_av_hubert_encoder}")
    print(f"  AV fusion: {cfg.av_fusion}")
    print(f"  Gated cross-attention: {cfg.add_gated_x_attn}")
    print(f"  AV-HuBERT user dir: {avhubert_user_dir}")
    
    try:
        print("\n" + "-" * 40)
        print("Step 1: Testing Whisper model loading without AV-HuBERT...")
        
        # First test: Load model without AV-HuBERT
        model_audio_only = whisper.load_model(
            cfg.model_name,
            device='cpu',
            video=False,
        )
        print("✓ Audio-only Whisper model loaded successfully")
        del model_audio_only  # Free memory
        
        print("\n" + "-" * 40)
        print("Step 2: Testing Whisper model with AV-HuBERT encoder...")
        
        # Second test: Load model with AV-HuBERT (this is where the registration issue occurs)
        model_av = whisper.load_model(
            cfg.model_name,
            device='cpu',
            video=True,
            video_model_path=cfg.video_model_ckpt,
            av_hubert_path=avhubert_user_dir,
            av_hubert_encoder=cfg.use_av_hubert_encoder,
            av_fusion=cfg.av_fusion,
            add_gated_x_attn=cfg.add_gated_x_attn,
            dropout_rate=cfg.dropout_rate,
            prob_av=cfg.prob_use_av,
            prob_a=cfg.prob_use_a,
        )
        print("✓ Whisper model with AV-HuBERT loaded successfully!")
        print(f"  Encoder type: {type(model_av.encoder)}")
        if hasattr(model_av.encoder, 'video_model'):
            print(f"  Video model type: {type(model_av.encoder.video_model)}")
        del model_av  # Free memory
        
        print("\n" + "-" * 40)
        print("Step 3: Testing multiple model loads (duplicate registration test)...")
        
        # Third test: Load the same model again to test for duplicate registration error
        model_av2 = whisper.load_model(
            cfg.model_name,
            device='cpu',
            video=True,
            video_model_path=cfg.video_model_ckpt,
            av_hubert_path=avhubert_user_dir,
            av_hubert_encoder=cfg.use_av_hubert_encoder,
            av_fusion=cfg.av_fusion,
            add_gated_x_attn=cfg.add_gated_x_attn,
            dropout_rate=cfg.dropout_rate,
            prob_av=cfg.prob_use_av,
            prob_a=cfg.prob_use_a,
        )
        print("✓ Second AV-HuBERT model loaded successfully!")
        print("✓ No duplicate task registration error detected!")
        del model_av2  # Free memory
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("✓ AV-HuBERT task registration fix is working correctly")
        print("✓ Your whisper_flamingo_ft_ami.py should now work properly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during model loading: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Check for specific errors and provide helpful messages
        if "Cannot register duplicate model" in str(e):
            print("\n🔍 DIAGNOSIS: AV-HuBERT task registration issue detected!")
            print("This means the fix in whisper_flamingo_ft_ami.py may need further adjustment.")
        elif "Could not infer task type" in str(e):
            print("\n🔍 DIAGNOSIS: AV-HuBERT task type inference issue!")
            print("This is related to how AV-HuBERT tasks are registered in fairseq.")
        elif "FileNotFoundError" in str(e) or "No such file" in str(e):
            print("\n🔍 DIAGNOSIS: Missing file or path issue!")
            print("Check that all model files and paths are correct.")
        else:
            print(f"\n🔍 DIAGNOSIS: Unexpected error: {e}")
        
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("1. Ensure the AV-HuBERT checkpoint file exists and is accessible")
        print("2. Verify that the whisper-flamingo repository is properly cloned")
        print("3. Check that the av_hubert submodule is properly initialized")
        print("4. Ensure fairseq is properly installed in the av_hubert/fairseq directory")
        
        return False


def main():
    """Main function for the test script."""
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    success = test_av_hubert_loading(config_path)
    
    if success:
        print(f"\n✅ Test completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Test failed! Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 