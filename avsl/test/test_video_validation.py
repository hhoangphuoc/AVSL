#!/usr/bin/env python3
"""
Robust video validation test script.
This script validates all videos in the dataset and identifies corrupted files.
"""

import os
import sys
import json
from datasets import load_from_disk

#===============================================================================================================
#                           PATH SETUP
#===============================================================================================================
# Consistent path setup (same across all test files)
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # avsl
project_root = os.path.dirname(parent_dir)  # AVSL
print(f"Project root: {project_root}")
print(f"Parent dir: {parent_dir}")

utils_path = os.path.join(project_root, 'utils') # AVSL/utils
whisper_flamingo_path = os.path.join(project_root, 'whisper_flamingo') # AVSL/whisper_flamingo
av_hubert_path = os.path.join(whisper_flamingo_path, 'av_hubert') # AVSL/whisper_flamingo/av_hubert

# Add to Python path (consistent with all test files)
sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)  # avsl
sys.path.insert(0, utils_path) # AVSL/utils
sys.path.insert(0, whisper_flamingo_path) # AVSL/whisper_flamingo
sys.path.insert(0, av_hubert_path) # AVSL/whisper_flamingo/av_hubert

# Import utilities
try:
    # Import from AVSL/utils using sys.path setup
    import sys
    # Temporarily modify import path to prioritize AVSL/utils
    original_path = sys.path.copy()
    # Remove whisper_flamingo from path temporarily to avoid conflicts
    temp_path = [p for p in sys.path if 'whisper_flamingo' not in p]
    sys.path = temp_path
    
    from utils import (
        safe_load_video_feats_from_hf_object,
        create_robust_video_filter,
    )
    
    # Restore original path
    sys.path = original_path
    
    print("✅ HuggingFace Video utilities imported successfully")
except ImportError as e:
    # Restore original path if error occurred
    if 'original_path' in locals():
        sys.path = original_path
    print(f"❌ Could not import HF Video utilities: {e}")
    print("💡 Note: Use test_hf_dataset_comprehensive.py for more robust testing")
    sys.exit(1)

#===============================================================================================================
#                           TEST 1: Robust Video Validation
#===============================================================================================================

def save_corrupted_files_report(corrupted_files, output_path):
    """Save report of corrupted files."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(corrupted_files, f, indent=2)
    
    print(f"📄 Corrupted files report saved to: {output_path}")


def test_video_validation():
    """Test the robust video validation on AMI dataset."""
    print("=" * 80)
    print("ROBUST VIDEO VALIDATION TEST")
    print("=" * 80)
    
    # Test on a small subset first
    # dataset_path = "/home/s2587130/AVSL/data/ami/av_hubert/test"
    dataset_path = "/home/s2587130/AVSL/data/ami/av_hubert/validation"
    
    try:
        print(f"📂 Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Test on first 100 samples for quick validation
        test_size = min(100, len(dataset))
        test_dataset = dataset.select(range(test_size))
        print(f"🧪 Testing on first {test_size} samples")
        
        # Run robust video validation
        valid_indices, corrupted_files = create_robust_video_filter(test_dataset)
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"   Total samples tested: {test_size}")
        print(f"   Valid videos: {len(valid_indices)}")
        print(f"   Corrupted videos: {len(corrupted_files)}")
        print(f"   Success rate: {len(valid_indices)/test_size*100:.1f}%")
        
        # Show details of corrupted files
        if corrupted_files:
            print(f"\n🚨 CORRUPTED FILES DETAILS:")
            for i, corrupted in enumerate(corrupted_files[:5]):  # Show first 5
                print(f"   {i+1}. Index {corrupted['index']}: {corrupted['reason']}")
                print(f"      File: {corrupted['file']}")
            
            if len(corrupted_files) > 5:
                print(f"   ... and {len(corrupted_files) - 5} more")
        
        # Save report
        report_path = "output/test_dataset/validation/corrupted_videos_report.json"
        save_corrupted_files_report(corrupted_files, report_path)
        
        # Test video loading on valid samples
        print(f"\n🎬 TESTING VIDEO LOADING:")
        successful_loads = 0
        
        for i, valid_idx in enumerate(valid_indices[:5]):  # Test first 5 valid videos
            try:
                sample = test_dataset[valid_idx]
                video_object = sample['lip_video']
                
                print(f"   Testing sample {valid_idx}...")
                video_feats = safe_load_video_feats_from_hf_object(video_object, train=False)
                
                if video_feats is not None:
                    print(f"   ✅ Loaded: {video_feats.shape}, dtype: {video_feats.dtype}")
                    successful_loads += 1
                else:
                    print(f"   ❌ Failed to load video features")
                    
            except Exception as e:
                print(f"   ❌ Error loading sample {valid_idx}: {e}")
        
        print(f"\n📈 Video loading success rate: {successful_loads}/5")
        
        return len(corrupted_files) == 0  # Return True if no corrupted files
        
    except Exception as e:
        print(f"❌ Error during video validation test: {e}")
        import traceback
        traceback.print_exc()
        return False

#===============================================================================================================
#                           TEST 2: Full Dataset Validation
#===============================================================================================================
def test_full_dataset_validation():
    """Test validation on full dataset (may take longer)."""
    print("=" * 80)
    print("FULL DATASET VIDEO VALIDATION")
    print("=" * 80)
    
    # dataset_path = "/home/s2587130/AVSL/data/ami/av_hubert/test"
    dataset_path = "/home/s2587130/AVSL/data/ami/av_hubert/validation"
    
    try:
        print(f"📂 Loading full dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Run validation on full dataset
        def progress_callback(current, total, valid_count, corrupted_count):
            if current % 1000 == 0:
                print(f"   Progress: {current}/{total} ({current/total*100:.1f}%), "
                      f"Valid: {valid_count}, Corrupted: {corrupted_count}")
        
        valid_indices, corrupted_files = create_robust_video_filter(
            dataset, 
            progress_callback=progress_callback
        )
        
        print(f"\n📊 FULL DATASET VALIDATION RESULTS:")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Valid videos: {len(valid_indices)}")
        print(f"   Corrupted videos: {len(corrupted_files)}")
        print(f"   Success rate: {len(valid_indices)/len(dataset)*100:.1f}%")
        
        # Save comprehensive report
        report_path = "output/test_dataset/validation/full_dataset_corrupted_videos_report.json"
        save_corrupted_files_report(corrupted_files, report_path)
        
        # Create a clean dataset with only valid samples
        if valid_indices:
            clean_dataset = dataset.select(valid_indices)
            clean_dataset_path = "/home/s2587130/AVSL/data/ami/av_hubert/validation_clean"
            
            print(f"💾 Saving clean dataset to: {clean_dataset_path}")
            clean_dataset.save_to_disk(clean_dataset_path)
            print(f"✅ Clean dataset saved: {len(clean_dataset)} samples")
        
        return len(corrupted_files) < len(dataset) * 0.1  # Pass if less than 10% corrupted
        
    except Exception as e:
        print(f"❌ Error during full dataset validation: {e}")
        import traceback
        traceback.print_exc()
        return False

#===============================================================================================================
#                           MAIN FUNCTION
#===============================================================================================================
if __name__ == "__main__":
    print("🧪 Starting robust video validation tests...")
    
    # Create output directory
    os.makedirs("output/test_dataset/validation", exist_ok=True)
    
    # Run quick validation test
    quick_test_passed = test_video_validation()
    
    # Ask if user wants to run full validation
    print(f"\n{'='*80}")
    print("QUICK TEST RESULTS:")
    if quick_test_passed:
        print("✅ Quick validation test PASSED")
    else:
        print("⚠️  Quick validation test found issues (expected)")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATION:")
    print("Running full dataset validation to identify all corrupted files...")
    
    # Run full validation
    full_test_passed = test_full_dataset_validation()
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS:")
    print(f"Quick test: {'PASSED' if quick_test_passed else 'FOUND ISSUES'}")
    print(f"Full test: {'PASSED' if full_test_passed else 'FOUND ISSUES'}")
    
    if full_test_passed:
        print("✅ Dataset validation completed successfully!")
        print("   Your dataset is ready for training.")
    else:
        print("⚠️  Dataset contains corrupted videos (this is normal).")
        print("   Clean dataset has been created for training.")
    
    print(f"{'='*80}")
    
    sys.exit(0 if full_test_passed else 1) 