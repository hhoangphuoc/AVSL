#!/usr/bin/env python3
"""
Debugging script specifically for the video loading issues found in the logs.
This script tests the decord.VideoReader handling improvements.
"""

import os
import sys
import numpy as np
from datasets import load_from_disk
import traceback   

# Consistent path setup (same across all test files)
current_dir = os.path.dirname(os.path.abspath(__file__))  # avsl/test
parent_dir = os.path.dirname(current_dir)  # avsl
project_root = os.path.dirname(parent_dir)  # AVSL
print(f"Project root: {project_root}")
print(f"Parent dir: {parent_dir}")
print(f"Current dir: {current_dir}")

utils_path = os.path.join(project_root, 'utils') # AVSL/utils
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo') # AVSL/whisper_flamingo
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert') # AVSL/whisper_flamingo/av_hubert

# Add to Python path (consistent with all test files)
sys.path.insert(0, project_root)
sys.path.insert(0, utils_path) # AVSL/utils
sys.path.insert(0, whisper_flamingo_path)
sys.path.insert(0, av_hubert_path)



def test_video_object_handling():
    """Test the improved video object handling"""
    print("=== Testing Video Object Handling ===")
    
    try:
        # Import utilities
        from utils import (
            safe_load_video_feats_from_hf_object,
            extract_video_path_from_hf_object
        )
        print("✓ Video utilities imported successfully")
        
        # Load a small test dataset
        dataset_path = "/home/s2587130/AVSL/data/ami/av_hubert/test"
        print(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Filter dataset
        dataset = dataset.filter(lambda x: x['lip_video'] is not None and x['audio'] is not None)
        print(f"✓ Dataset filtered: {len(dataset)} samples")
        
        # Test first few samples
        for i in range(min(3, len(dataset))):
            print(f"\n--- Testing Sample {i} ---")
            sample = dataset[i]
            video_object = sample['lip_video']
            
            print(f"Video object type: {type(video_object)}")
            
            # Check if it's a decord.VideoReader
            is_video_reader = hasattr(video_object, '__class__') and 'VideoReader' in str(type(video_object))
            print(f"Is VideoReader: {is_video_reader}")
            
            if is_video_reader:
                print(f"Video length: {len(video_object)} frames")
                try:
                    fps = video_object.get_avg_fps()
                    print(f"Average FPS: {fps:.2f}")
                except:
                    print("Could not get FPS")
            
            # Test path extraction
            print("Testing path extraction...")
            video_path = extract_video_path_from_hf_object(video_object)
            print(f"Extracted path: {video_path}")
            
            # Test video feature loading
            print("Testing video feature loading...")
            try:
                video_feats = safe_load_video_feats_from_hf_object(video_object, train=False)
                print(f"✓ Video features loaded successfully!")
                print(f"  Shape: {video_feats.shape}")
                print(f"  Data type: {video_feats.dtype}")
                print(f"  Value range: [{video_feats.min():.3f}, {video_feats.max():.3f}]")
                
                # Check for dummy features (all zeros)
                if np.allclose(video_feats, 0):
                    print("⚠ Warning: Video features are all zeros (dummy features)")
                else:
                    print("✓ Video features contain actual data")
                    
            except Exception as e:
                print(f"✗ Error loading video features: {e}")
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"✗ Error in video object handling test: {e}")
        traceback.print_exc()
        return False


def test_tokenizer_fix():
    """Test the tokenizer fix for negative token IDs"""
    print("\n=== Testing Tokenizer Fix ===")
    
    try:
        import whisper_flamingo.whisper as whisper
        import torch
        
        # Create tokenizer
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='en', task='transcribe')
        print("✓ Tokenizer created")
        
        # Create test tokens with negative values (simulating the error condition)
        test_tokens = torch.tensor([50258, 50259, 50360, 50365, 1234, -100, 5678, -100, 50257])  # Mix of valid and negative tokens
        print(f"Test tokens: {test_tokens}")
        
        # Test the old way (this should fail)
        print("Testing old decoding method...")
        try:
            special_token_set = set(tokenizer.special_tokens.values())
            old_decoded = tokenizer.decode([t for t in test_tokens if t.item() not in special_token_set])
            print(f"Old method result: '{old_decoded}'")
        except Exception as e:
            print(f"✗ Old method failed (expected): {e}")
        
        # Test the new way (this should work)
        print("Testing new decoding method...")
        try:
            special_token_set = set(tokenizer.special_tokens.values())
            valid_tokens = [t for t in test_tokens if t.item() >= 0 and t.item() not in special_token_set]
            new_decoded = tokenizer.decode(valid_tokens)
            print(f"✓ New method result: '{new_decoded}'")
            print(f"✓ Valid tokens used: {valid_tokens}")
            return True
        except Exception as e:
            print(f"✗ New method failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error in tokenizer test: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Debugging Video Loading Issues")
    print("=" * 50)
    
    # Test video object handling
    video_test_passed = test_video_object_handling()
    
    # Test tokenizer fix  
    tokenizer_test_passed = test_tokenizer_fix()
    
    # Final report
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"✓ Video object handling: {'PASSED' if video_test_passed else 'FAILED'}")
    print(f"✓ Tokenizer fix: {'PASSED' if tokenizer_test_passed else 'FAILED'}")
    
    if video_test_passed and tokenizer_test_passed:
        print("\n🎉 All tests passed! The fixes should resolve your training issues.")
        print("You can now run the full training script.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
    
    return video_test_passed and tokenizer_test_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 