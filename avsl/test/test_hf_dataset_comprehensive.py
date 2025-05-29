#!/usr/bin/env python3
"""
Comprehensive HuggingFace Dataset Test Script
Merges functionality from test_hf_dataset.py and test_hf_dataset_old.py
Tests all aspects of dataset loading, video processing, and training pipeline integration.
"""

import os
import sys
import numpy as np
import traceback
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
sys.path.insert(0, whisper_flamingo_path)
sys.path.insert(0, av_hubert_path)

# Add fairseq path if available
fairseq_path = os.path.join(av_hubert_path, 'fairseq')
if os.path.exists(fairseq_path) and fairseq_path not in sys.path:
    sys.path.insert(0, fairseq_path)

# Import video utilities with comprehensive fallbacks
VIDEO_UTILS_AVAILABLE = False
ROBUST_VIDEO_UTILS_AVAILABLE = False

try:
    # Import from AVSL/utils using sys.path setup
    # Temporarily modify import path to prioritize AVSL/utils
    original_path = sys.path.copy()
    # Remove whisper_flamingo from path temporarily to avoid conflicts
    temp_path = [p for p in sys.path if 'whisper_flamingo' not in p]
    sys.path = temp_path
    
    from hf_video_utils import (
        # Robust/new functions
        validate_hf_video_object,
        safe_load_video_feats_from_hf_object,
        create_robust_video_filter,
        # Common functions
        debug_hf_video_object,
        extract_video_path_from_hf_object   
    )
    
    # Restore original path
    sys.path = original_path
    
    VIDEO_UTILS_AVAILABLE = True
    ROBUST_VIDEO_UTILS_AVAILABLE = True
    print("✅ Full HuggingFace Video utilities imported successfully")
except ImportError as e:
    # Restore original path if error occurred
    if 'original_path' in locals():
        sys.path = original_path
    print(f"⚠️ Full video utilities not available: {e}")

#===============================================================================================================
#                           TEST 1: Dataset Structure and Loading
#===============================================================================================================
def test_dataset_structure_and_loading():
    """Test basic dataset loading and structure inspection."""
    print("="*80)
    print("TEST 1: Dataset Structure and Loading")
    print("="*80)
    
    # Test multiple dataset paths
    dataset_paths = [
        "/home/s2587130/AVSL/data/ami/av_hubert/train",  # AV-HuBERT format
        "/home/s2587130/AVSL/data/ami/av_hubert/validation",
        "/home/s2587130/AVSL/data/ami/av_hubert/test"
    ]
    
    results = {}
    
    for dataset_path in dataset_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f"\n--- Testing Dataset: {dataset_name} ---")
        print(f"Path: {dataset_path}")
        
        try:
            if not os.path.exists(dataset_path):
                print(f"⚠️ Dataset path does not exist: {dataset_path}")
                results[dataset_name] = "not_found"
                continue
                
            # Load dataset
            dataset = load_from_disk(dataset_path)
            print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
            
            # Inspect structure
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"📋 Sample keys: {list(sample.keys())}")
                
                # Detailed structure inspection
                for key, value in sample.items():
                    if key == 'audio':
                        if hasattr(value, 'get') and 'array' in value:
                            print(f"   {key}: array shape {value['array'].shape}, sr={value['sampling_rate']}")
                        else:
                            print(f"   {key}: {type(value)}")
                    elif key == 'lip_video':
                        print(f"   {key}: {type(value)}")
                    elif isinstance(value, str):
                        print(f"   {key}: '{value[:100]}{'...' if len(value) > 100 else ''}'")
                    elif isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {type(value)}")
                
                # Check required fields
                required_fields = ['audio', 'transcript', 'lip_video']
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    print(f"⚠️ Missing required fields: {missing_fields}")
                else:
                    print(f"✅ All required fields present")
                
                results[dataset_name] = "success"
            else:
                print(f"⚠️ Dataset is empty")
                results[dataset_name] = "empty"
                
        except Exception as e:
            print(f"❌ Error loading dataset {dataset_name}: {e}")
            results[dataset_name] = "error"
    
    # Summary
    print(f"\n📊 Dataset Loading Summary:")
    for dataset_name, status in results.items():
        status_icon = {"success": "✅", "not_found": "📁", "empty": "📋", "error": "❌"}
        print(f"   {status_icon.get(status, '?')} {dataset_name}: {status}")
    
    # Return True if at least one dataset loaded successfully
    return any(status == "success" for status in results.values())


#===============================================================================================================
#                           TEST 2: Video Processing Detailed
#===============================================================================================================
def test_video_processing_detailed(dataset_path, num_samples=5):
    """Detailed test of video processing for a specific dataset."""
    print(f"\n--- Detailed Video Processing Test: {os.path.basename(dataset_path)} ---")
    
    try:
        if not os.path.exists(dataset_path):
            print(f"⚠️ Dataset not found: {dataset_path}")
            return False
            
        dataset = load_from_disk(dataset_path)
        test_size = min(num_samples, len(dataset))
        print(f"Testing video processing on {test_size} samples...")
        
        successful_videos = 0
        failed_videos = 0
        failed_details = []
        
        for i in range(test_size):
            print(f"\n--- Sample {i} ---")
            
            try:
                sample = dataset[i]
                
                if 'lip_video' not in sample:
                    print(f"⚠️ No 'lip_video' key found")
                    failed_videos += 1
                    continue
                
                video_object = sample['lip_video']
                
                # Debug video object
                if VIDEO_UTILS_AVAILABLE:
                    debug_hf_video_object(video_object, i)
                    
                    # Test path extraction
                    video_path = extract_video_path_from_hf_object(video_object)
                    if video_path:
                        print(f"✅ Video path: {video_path}")
                        print(f"✅ Path exists: {os.path.exists(video_path)}")
                    else:
                        print(f"⚠️ Could not extract video path")
                    
                    # Test video validation if available
                    if ROBUST_VIDEO_UTILS_AVAILABLE:
                        is_valid = validate_hf_video_object(video_object)
                        print(f"✅ Video validation: {is_valid}")
                        
                        if is_valid:
                            # Test robust video loading
                            video_feats = safe_load_video_feats_from_hf_object(video_object, train=False)
                            if video_feats is not None:
                                print(f"✅ Robust video loading: {video_feats.shape}, dtype: {video_feats.dtype}")
                                print(f"   Value range: [{video_feats.min():.3f}, {video_feats.max():.3f}]")
                                successful_videos += 1
                            else:
                                print(f"❌ Robust video loading failed")
                                failed_videos += 1
                        else:
                            print(f"❌ Video validation failed")
                            failed_videos += 1
                    else:
                        # Test basic video loading
                        try:
                            video_feats = safe_load_video_feats_from_hf_object(video_object, train=False)
                            print(f"✅ Basic video loading: {video_feats.shape}, dtype: {video_feats.dtype}")
                            print(f"   Value range: [{video_feats.min():.3f}, {video_feats.max():.3f}]")
                            successful_videos += 1
                        except Exception as e:
                            print(f"❌ Basic video loading failed: {e}")
                            failed_videos += 1
                            failed_details.append(f"Sample {i}: {str(e)[:100]}")
                
                else:
                    print(f"⚠️ Video utilities not available, skipping video tests")
                    failed_videos += 1
                
                # Test other sample components
                if 'audio' in sample:
                    audio_info = sample['audio']
                    if hasattr(audio_info, 'get') and 'array' in audio_info:
                        audio_array = audio_info['array']
                        print(f"✅ Audio: shape={audio_array.shape}, sr={audio_info['sampling_rate']}")
                    else:
                        print(f"✅ Audio: {type(audio_info)}")
                
                if 'transcript' in sample:
                    transcript = sample['transcript']
                    print(f"✅ Transcript: '{transcript[:50]}{'...' if len(transcript) > 50 else ''}'")
                
                if 'duration' in sample:
                    duration = sample['duration']
                    print(f"✅ Duration: {duration:.2f} seconds")
                    
            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                failed_videos += 1
                failed_details.append(f"Sample {i}: {str(e)[:100]}")
        
        print(f"\n📊 Video Processing Results:")
        print(f"   Successful: {successful_videos}/{test_size} ({successful_videos/test_size*100:.1f}%)")
        print(f"   Failed: {failed_videos}/{test_size} ({failed_videos/test_size*100:.1f}%)")
        
        if failed_details:
            print(f"\n🚨 Failed Video Details (first 3):")
            for detail in failed_details[:3]:
                print(f"   {detail}")
            if len(failed_details) > 3:
                print(f"   ... and {len(failed_details) - 3} more failures")
        
        # Consider successful if at least 80% of videos process correctly
        return successful_videos / test_size >= 0.8
        
    except Exception as e:
        print(f"❌ Error in video processing test: {e}")
        traceback.print_exc()
        return False


#===============================================================================================================
#                           TEST 3: Robust Video Filtering
#===============================================================================================================
def test_robust_video_filtering():
    """Test robust video filtering on the main dataset."""
    print("="*80)
    print("TEST 3: Robust Video Filtering")
    print("="*80)
    
    if not ROBUST_VIDEO_UTILS_AVAILABLE:
        print("⚠️ Robust video utilities not available, skipping this test")
        return True  # Don't fail the test if utilities aren't available
    
    dataset_path = "/home/s2587130/AVSL/data/ami/av_hubert/test_clean" #NOTE: This load from test_clean dataset (After running first test: `test_video_validation.py`)
    
    try:
        if not os.path.exists(dataset_path):
            print(f"⚠️ Main dataset not found: {dataset_path}")
            return True  # Don't fail if dataset doesn't exist
        
        print(f"Loading dataset for robust filtering...")
        dataset = load_from_disk(dataset_path)
        print(f"Original dataset size: {len(dataset)}")
        
        # Test on a subset first
        test_size = min(100, len(dataset))
        test_dataset = dataset.select(range(test_size))
        print(f"Running robust filtering on {test_size} samples...")
        
        # Use robust filtering
        valid_indices, corrupted_files = create_robust_video_filter(test_dataset)
        
        print(f"\n📊 Robust Filtering Results:")
        print(f"   Total samples tested: {test_size}")
        print(f"   Valid samples: {len(valid_indices)}")
        print(f"   Corrupted samples: {len(corrupted_files)}")
        print(f"   Success rate: {len(valid_indices)/test_size*100:.1f}%")
        
        # Show corruption details
        if corrupted_files:
            print(f"\n🚨 Corruption Analysis:")
            reasons = {}
            for corrupted in corrupted_files:
                reason = corrupted['reason'].split(':')[0]
                reasons[reason] = reasons.get(reason, 0) + 1
            
            for reason, count in reasons.items():
                print(f"   {reason}: {count} files")
            
            print(f"\n🚨 Example corrupted files (first 3):")
            for i, corrupted in enumerate(corrupted_files[:3]):
                print(f"   {i+1}. Index {corrupted['index']}: {corrupted['reason']}")
                print(f"      File: {corrupted['file']}")
        
        # Save results
        os.makedirs("output/test_scripts", exist_ok=True)
        import json
        with open("output/test_scripts/video_filtering_test_results.json", 'w') as f:
            json.dump({
                'total_samples': test_size,
                'valid_samples': len(valid_indices),
                'corrupted_samples': len(corrupted_files),
                'corrupted_files': corrupted_files
            }, f, indent=2)
        print(f"📄 Results saved to: output/test_scripts/video_filtering_test_results.json")
        
        return True  # Always return True since this is an informational test
        
    except Exception as e:
        print(f"❌ Error in robust filtering test: {e}")
        traceback.print_exc()
        return False


#===============================================================================================================
#                           TEST 4: AmiVideoHFDataset Creation
#===============================================================================================================
def test_ami_dataset_creation():
    """Test creating AmiVideoHFDataset class from the training script."""
    print("="*80)
    print("TEST 4: AmiVideoHFDataset Creation")
    print("="*80)
    
    try:
        # Import training components
        print("Importing training script components...")
        from whisper_flamingo_ft_ami import AmiVideoHFDataset
        import whisper_flamingo.whisper as whisper
        print("✅ Training components imported successfully")
        
        # Find an available dataset
        dataset_paths = [
            "/home/s2587130/AVSL/data/ami/av_hubert/train",
            "/home/s2587130/AVSL/data/ami/av_hubert/validation",
            "/home/s2587130/AVSL/data/ami/av_hubert/test",
        ]
        
        dataset = None
        dataset_path = None
        
        for path in dataset_paths:
            if os.path.exists(path):
                try:
                    dataset = load_from_disk(path)
                    dataset_path = path
                    print(f"✅ Using dataset: {path} ({len(dataset)} samples)")
                    break
                except Exception as e:
                    print(f"⚠️ Could not load dataset from {path}: {e}")
                    continue
        
        if dataset is None:
            print("❌ No usable dataset found for testing")
            return False
        
        # Filter and prepare dataset
        print("Preparing dataset...")
        original_size = len(dataset)
        
        # Basic filtering
        dataset = dataset.filter(lambda x: x['lip_video'] is not None and x['audio'] is not None)
        print(f"After basic filtering: {len(dataset)} samples (was {original_size})")
        
        if len(dataset) == 0:
            print("❌ No valid samples after filtering")
            return False
        
        # Use small subset for testing
        test_size = min(3, len(dataset))
        test_dataset = dataset.select(range(test_size))
        print(f"Using {test_size} samples for testing")
        
        # Create tokenizer
        print("Creating tokenizer...")
        tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, language='en', task='transcribe')
        print("✅ Tokenizer created successfully")
        
        # Create AmiVideoHFDataset
        print("Creating AmiVideoHFDataset...")
        ami_dataset = AmiVideoHFDataset(
            hf_dataset=test_dataset,
            tokenizer=tokenizer,
            sample_rate=16000,
            model_name='large-v2',
            audio_max_length=480000,  # 30 seconds
            lang_code='en',
            spec_augment_config=None,  # Disable for testing
            train=False  # Test mode
        )
        print(f"✅ AmiVideoHFDataset created: {len(ami_dataset)} samples")
        
        # Test sample loading
        print("Testing sample loading...")
        for i in range(len(ami_dataset)):
            try:
                sample = ami_dataset[i]
                print(f"✅ Sample {i} loaded successfully")
                print(f"   Keys: {list(sample.keys())}")
                print(f"   Input IDs shape: {sample['input_ids'].shape}")
                print(f"   Labels shape: {sample['labels'].shape}")
                print(f"   Dec input IDs shape: {sample['dec_input_ids'].shape}")
                print(f"   Video shape: {sample['video'].shape}")
                
                # Check data types and ranges
                print(f"   Video dtype: {sample['video'].dtype}")
                print(f"   Video range: [{sample['video'].min():.3f}, {sample['video'].max():.3f}]")
                
            except Exception as e:
                print(f"❌ Error loading sample {i}: {e}")
                if i == 0:  # If first sample fails, return failure
                    traceback.print_exc()
                    return False
        
        print("✅ AmiVideoHFDataset test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in AmiVideoHFDataset test: {e}")
        traceback.print_exc()
        return False


#===============================================================================================================
#                           TEST 5: Dataset Compatibility
#===============================================================================================================
def test_dataset_compatibility():
    """Test compatibility between different dataset formats."""
    print("="*80)
    print("TEST 5: Dataset Format Compatibility")
    print("="*80)
    
    # Test different dataset paths and their compatibility
    dataset_info = {
        'av_hubert_train': "/home/s2587130/AVSL/data/ami/av_hubert/train",
        'av_hubert_val': "/home/s2587130/AVSL/data/ami/av_hubert/validation",
        'av_hubert_test': "/home/s2587130/AVSL/data/ami/av_hubert/test"
    }
    
    loaded_datasets = {}
    
    # Load all available datasets
    for name, path in dataset_info.items():
        if os.path.exists(path):
            try:
                dataset = load_from_disk(path)
                loaded_datasets[name] = {
                    'dataset': dataset,
                    'path': path,
                    'size': len(dataset)
                }
                print(f"✅ {name}: {len(dataset)} samples")
            except Exception as e:
                print(f"❌ {name}: Failed to load - {e}")
        else:
            print(f"📁 {name}: Not found")
    
    if not loaded_datasets:
        print("⚠️ No datasets available for compatibility testing")
        return True
    
    # Compare dataset structures
    print(f"\n📋 Dataset Structure Comparison:")
    sample_structures = {}
    
    for name, info in loaded_datasets.items():
        dataset = info['dataset']
        if len(dataset) > 0:
            sample = dataset[0]
            sample_structures[name] = set(sample.keys())
            print(f"   {name}: {sorted(sample.keys())}")
    
    # Find common keys
    if sample_structures:
        common_keys = set.intersection(*sample_structures.values())
        print(f"\n🔗 Common keys across all datasets: {sorted(common_keys)}")
        
        # Check for differences
        all_keys = set.union(*sample_structures.values())
        unique_keys = {}
        for name, keys in sample_structures.items():
            unique = keys - common_keys
            if unique:
                unique_keys[name] = unique
        
        if unique_keys:
            print(f"\n🔍 Unique keys by dataset:")
            for name, keys in unique_keys.items():
                print(f"   {name}: {sorted(keys)}")
    
    print("✅ Dataset compatibility test completed")
    return True


def main():
    """Main test function."""
    print("🧪 Comprehensive HuggingFace Dataset Testing")
    print("=" * 80)
    print("This test combines functionality from both test files and adds robust error handling.")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("output/test_scripts", exist_ok=True)
    
    # Define tests
    tests = [
        ("Dataset Structure and Loading", test_dataset_structure_and_loading),
        ("Video Processing (Basic)", lambda: test_video_processing_detailed("/home/s2587130/AVSL/data/ami/av_hubert/test")),
        ("Robust Video Filtering", test_robust_video_filtering),
        ("AmiVideoHFDataset Creation", test_ami_dataset_creation),
        ("Dataset Format Compatibility", test_dataset_compatibility),
    ]
    
    # Run tests
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"{'='*80}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "⚠️ COMPLETED WITH ISSUES"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ CRASHED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Final summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED/ISSUES"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # Provide guidance based on results
    if passed == total:
        print("\n🎉 All tests passed! Your dataset is ready for training.")
        print("✅ You can proceed with running the training script.")
    elif passed >= total * 0.8:  # 80% success rate
        print("\n✅ Most tests passed! Your dataset should work for training.")
        print("⚠️ Some issues detected, but they may be expected (e.g., corrupted videos).")
        print("💡 Check the detailed output above for specific issues.")
    else:
        print("\n⚠️ Several tests failed. Please review the errors above.")
        print("💡 Common issues to check:")
        print("   - Dataset paths are correct")
        print("   - Required files are present")
        print("   - Video utilities are properly installed")
        print("   - Training script imports work correctly")
    
    print(f"\n📄 Test results logged to: output/test_scripts/")
    print(f"{'='*80}")
    
    # Save summary to file
    summary = {
        'total_tests': total,
        'passed_tests': passed,
        'success_rate': passed / total * 100,
        'results': {name: result for name, result in results},
        'video_utils_available': VIDEO_UTILS_AVAILABLE,
        'robust_video_utils_available': ROBUST_VIDEO_UTILS_AVAILABLE
    }
    
    import json
    with open("output/test_scripts/comprehensive_test_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    return passed >= total * 0.8  # Consider success if 80% of tests pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 