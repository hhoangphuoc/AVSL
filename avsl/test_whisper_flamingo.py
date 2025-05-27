#!/usr/bin/env python3
"""
Simple test script to verify whisper-flamingo functionality
"""
import os
import sys
import torch
import numpy as np
from datasets import load_from_disk

# Add the whisper_flamingo directory to path
sys.path.append('/home/s2587130/AVSL/whisper_flamingo')

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import whisper_flamingo.whisper as whisper
        print("✓ whisper_flamingo.whisper imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import whisper_flamingo.whisper: {e}")
        return False
    
    try:
        from whisper_flamingo.utils import WhisperVideoCollatorWithPadding
        print("✓ WhisperVideoCollatorWithPadding imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import WhisperVideoCollatorWithPadding: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that whisper model can be loaded"""
    print("\nTesting model loading...")
    try:
        import whisper_flamingo.whisper as whisper
        
        # Test basic whisper model loading
        model = whisper.load_model("small", device='cpu', video=False)
        print("✓ Basic Whisper model loaded successfully")
        
        # Test video-enabled model loading
        model_video = whisper.load_model("small", 
                                        device='cpu', 
                                        video=True,
                                        av_hubert_encoder=False,
                                        av_fusion='early')
        print("✓ Video-enabled Whisper model loaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

def test_dataset_loading():
    """Test that the AMI dataset can be loaded"""
    print("\nTesting dataset loading...")
    try:
        train_path = "/home/s2587130/AVSL/data/ami/av_hubert/train"
        if os.path.exists(train_path):
            dataset = load_from_disk(train_path)
            print(f"✓ Train dataset loaded: {len(dataset)} samples")
            
            # Check first sample
            sample = dataset[0]
            print(f"✓ Sample keys: {list(sample.keys())}")
            
            if 'audio' in sample:
                print(f"✓ Audio shape: {sample['audio']['array'].shape}")
            if 'transcript' in sample:
                print(f"✓ Transcript: {sample['transcript'][:50]}...")
            if 'lip_video' in sample:
                print(f"✓ Video path: {sample['lip_video']}")
                
            return True
        else:
            print(f"✗ Dataset path does not exist: {train_path}")
            return False
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

def test_video_loading():
    """Test video loading functionality"""
    print("\nTesting video loading...")
    try:
        # Import the video loading function from our script
        sys.path.append('/home/s2587130/AVSL/avsl')
        from whisper_flamingo_ft_ami import load_video_feats
        
        # Test with a dummy video path (this will fail but we can check the function exists)
        print("✓ load_video_feats function imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import video loading function: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Whisper-Flamingo Functionality Test ===\n")
    
    tests = [
        test_imports,
        test_model_loading,
        test_dataset_loading,
        test_video_loading
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Ready to run training.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 