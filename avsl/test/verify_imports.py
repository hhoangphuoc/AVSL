#!/usr/bin/env python3
"""Verify all imports work correctly after fixes."""

import os
import sys

# Same path setup as other test files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
utils_path = os.path.join(project_root, 'utils')
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo')
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert')

sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)  # This was the missing piece
sys.path.insert(0, utils_path)
sys.path.insert(0, whisper_flamingo_path)
sys.path.insert(0, av_hubert_path)

def test_imports():
    """Test all the imports that were failing."""
    print("Testing imports...")
    
    # Test 1: HF video utilities from AVSL/utils
    try:
        original_path = sys.path.copy()
        temp_path = [p for p in sys.path if 'whisper_flamingo' not in p]
        sys.path = temp_path
        
        from hf_video_utils import (
            validate_hf_video_object,
            safe_load_video_feats_from_hf_object,
            create_robust_video_filter,
            debug_hf_video_object,
            extract_video_path_from_hf_object   
        )
        
        sys.path = original_path
        print("✅ HF video utilities from AVSL/utils imported successfully")
    except ImportError as e:
        if 'original_path' in locals():
            sys.path = original_path
        print(f"❌ HF video utilities import failed: {e}")
        return False
    
    # Test 2: whisper_flamingo_ft_ami module
    try:
        from whisper_flamingo_ft_ami import AmiVideoHFDataset, WhisperFlamingoModule
        print("✅ whisper_flamingo_ft_ami imports successful")
    except ImportError as e:
        print(f"❌ whisper_flamingo_ft_ami import failed: {e}")
        return False
    
    # Test 3: whisper_flamingo.whisper
    try:
        import whisper_flamingo.whisper as whisper
        print("✅ whisper_flamingo.whisper imported successfully")
    except ImportError as e:
        print(f"❌ whisper_flamingo.whisper import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    print(f"\n{'='*50}")
    if success:
        print("🎉 All imports are working correctly!")
        print("✅ The test files should now run without import errors.")
    else:
        print("❌ Some imports are still failing.")
    print(f"{'='*50}")
    sys.exit(0 if success else 1)
