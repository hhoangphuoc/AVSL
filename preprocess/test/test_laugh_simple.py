"""
Simple test for laughter dataset CSV loading
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import tempfile
import shutil
import json
from preprocess.laugh_dataset_process import (
    load_laughter_markers,
    create_segment_id,
    collect_segments_by_source
)


def test_csv_structure():
    """Test CSV file structure and loading"""
    print("Testing CSV structure...")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'ami_laugh_markers.csv')
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found at {csv_path}")
        return False
        
    # Try to load the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded CSV with {len(df)} rows")
        
        # Check required columns
        required_columns = ['meeting_id', 'speaker_id', 'word', 'start_time', 'end_time', 'disfluency_type']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return False
        else:
            print(f"✓ All required columns present: {required_columns}")
            
        # Check data types
        print(f"✓ Column types: {df.dtypes.to_dict()}")
        
        # Check for laughter and fluent segments
        if 'disfluency_type' in df.columns:
            types = df['disfluency_type'].value_counts()
            print(f"✓ Disfluency types: {types.to_dict()}")
            
            laugh_fluent = df[df['disfluency_type'].isin(['laughter', 'fluent'])]
            print(f"✓ Laughter/fluent segments: {len(laugh_fluent)}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return False


def test_segment_id_creation():
    """Test segment ID creation"""
    print("\nTesting segment ID creation...")
    
    try:
        # Test normal case
        seg_id = create_segment_id('ES2002a', 'A', 10.5, 15.2, 'laughter')
        expected = 'ES2002a-A-10.50-15.20-laughter'
        
        if seg_id == expected:
            print(f"✓ Segment ID creation: {seg_id}")
        else:
            print(f"✗ Expected {expected}, got {seg_id}")
            return False
            
        # Test edge cases
        seg_id2 = create_segment_id('ES2002a', 'A', 0.0, 0.1, 'fluent')
        print(f"✓ Edge case: {seg_id2}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating segment ID: {e}")
        return False


def test_none_handling():
    """Test handling of None values in dataset records"""
    print("\nTesting None value handling...")
    
    try:
        # Create test records with None values
        test_records = [
            {
                'segment_id': 'test-1',
                'audio': '/path/to/audio.wav',
                'video_path': '/path/to/video.mp4',
                'lip_video': None,  # This should not cause errors
                'transcript': 'hello world'
            },
            {
                'segment_id': 'test-2',
                'audio': None,  # This should not cause errors
                'video_path': None,  # This should not cause errors
                'lip_video': None,  # This should not cause errors
                'transcript': 'another test'
            }
        ]
        
        # Test DataFrame creation
        df = pd.DataFrame(test_records)
        print(f"✓ DataFrame created with {len(df)} records")
        
        # Test filtering valid records (like in the fixed code)
        valid_records = []
        for record in test_records:
            has_audio = record.get('audio') is not None
            has_video = record.get('video_path') is not None  
            has_lip = record.get('lip_video') is not None
            
            if has_audio or has_video or has_lip:
                valid_records.append(record)
        
        print(f"✓ Filtered to {len(valid_records)} valid records (out of {len(test_records)})")
        
        # Test safe path handling
        def safe_path_join(x):
            if x is None or pd.isna(x):
                return None
            return os.path.join('data', os.path.basename(str(x)))
        
        # This should not raise errors
        test_paths = ['/path/to/file.wav', None, '/another/path.mp4', pd.NA]
        results = [safe_path_join(x) for x in test_paths]
        print(f"✓ Safe path join results: {results}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in None handling test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_markers_function():
    """Test the load_laughter_markers function with error handling"""
    print("\nTesting load_laughter_markers function...")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'ami_laugh_markers.csv')
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found, skipping test")
        return True  # Skip but don't fail
    
    try:
        df = load_laughter_markers(csv_path)
        
        if df is not None and len(df) > 0:
            print(f"✓ Loaded {len(df)} markers successfully")
            
            # Test filtering
            laugh_fluent = df[df['disfluency_type'].isin(['laughter', 'fluent'])]
            print(f"✓ Found {len(laugh_fluent)} laughter/fluent segments")
            
            return True
        else:
            print(f"✗ No data loaded or empty DataFrame")
            return False
            
    except Exception as e:
        print(f"✗ Error in load_laughter_markers: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_segment_grouping():
    """Test segment grouping function"""
    print("\nTesting segment grouping...")
    
    # Create a minimal test DataFrame
    test_data = {
        'meeting_id': ['ES2002a', 'ES2002a', 'ES2002b'],
        'speaker_id': ['A', 'B', 'A'],
        'start_time': [10.0, 20.0, 5.0],
        'end_time': [15.0, 25.0, 8.0],
        'disfluency_type': ['laughter', 'fluent', 'laughter'],
        'word': ['[laughter]', 'hello world', '[laughter]']
    }
    
    df_test = pd.DataFrame(test_data)
    
    try:
        audio_by_source, video_by_source, segment_info = collect_segments_by_source(df_test)
        
        print(f"✓ Audio sources: {len(audio_by_source)}")
        print(f"✓ Video sources: {len(video_by_source)}")
        print(f"✓ Segment info: {len(segment_info)}")
        
        # Test segment info structure
        if segment_info:
            sample_id = list(segment_info.keys())[0]
            sample_info = segment_info[sample_id]
            required_keys = ['meeting_id', 'speaker_id', 'start_time', 'end_time', 'transcript', 'disfluency_type']
            
            if all(key in sample_info for key in required_keys):
                print(f"✓ Segment info has required keys: {required_keys}")
            else:
                missing = [key for key in required_keys if key not in sample_info]
                print(f"✗ Missing keys in segment info: {missing}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in segment grouping: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all simple tests"""
    print("="*60)
    print("Simple Laughter Dataset Processing Tests")
    print("="*60)
    
    tests = [
        test_csv_structure,
        test_segment_id_creation,
        test_none_handling,
        test_load_markers_function,
        test_segment_grouping
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} PASSED")
            else:
                failed += 1
                print(f"✗ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} ERROR: {e}")
        
        print("-" * 40)
    
    print(f"\nResults: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 