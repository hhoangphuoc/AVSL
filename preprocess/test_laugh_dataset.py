"""
Test script for verifying the laughter dataset processing
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import tempfile
import shutil
from preprocess.laugh_dataset_process import (
    load_laughter_markers,
    create_segment_id,
    collect_segments_by_source,
    process_laughter_dataset
)
from preprocess.constants import DATA_PATH


def test_load_markers():
    """Test loading the CSV file"""
    print("Testing load_laughter_markers...")
    
    csv_path = os.path.join(os.path.dirname(__file__), 'ami_laugh_markers.csv')
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return None
        
    df = load_laughter_markers(csv_path)
    
    print(f"✓ Loaded {len(df)} segments")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Disfluency types: {df['disfluency_type'].unique()}")
    
    # Show sample data
    print("\nSample data:")
    print(df.head())
    
    return df


def test_segment_grouping(df):
    """Test segment grouping by source"""
    print("\nTesting segment grouping...")
    
    # Filter to a small subset for testing
    test_meetings = df['meeting_id'].unique()[:2]  # First 2 meetings
    df_test = df[df['meeting_id'].isin(test_meetings)]
    df_test = df_test[df_test['disfluency_type'].isin(['laughter', 'fluent'])]
    
    print(f"Using {len(df_test)} segments from meetings: {test_meetings}")
    
    audio_by_source, video_by_source, segment_info = collect_segments_by_source(df_test)
    
    print(f"✓ Audio sources: {len(audio_by_source)}")
    print(f"✓ Video sources: {len(video_by_source)}")
    print(f"✓ Total segments: {len(segment_info)}")
    
    # Show sample segment info
    if segment_info:
        sample_id = list(segment_info.keys())[0]
        print(f"\nSample segment info for {sample_id}:")
        print(segment_info[sample_id])
    
    return df_test


def test_mini_dataset(df):
    """Test creating a mini dataset with a few segments"""
    print("\nTesting mini dataset creation...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a mini CSV with just a few segments
        mini_df = df[df['disfluency_type'].isin(['laughter', 'fluent'])].head(10)
        mini_csv = os.path.join(temp_dir, 'mini_markers.csv')
        mini_df.to_csv(mini_csv, index=False)
        
        output_dir = os.path.join(temp_dir, 'output')
        dataset_path = os.path.join(temp_dir, 'dataset')
        
        print(f"Processing {len(mini_df)} segments as a test...")
        print(f"Output directory: {output_dir}")
        print(f"Dataset path: {dataset_path}")
        
        try:
            # Process without lip videos for speed
            records = process_laughter_dataset(
                csv_path=mini_csv,
                output_dir=output_dir,
                dataset_path=dataset_path,
                extract_lip_videos=False,
                use_shards=False
            )
            
            print(f"✓ Created {len(records)} dataset records")
            
            # Check output structure
            if os.path.exists(output_dir):
                print("\nOutput directory structure:")
                for root, dirs, files in os.walk(output_dir):
                    level = root.replace(output_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files[:5]:  # Show first 5 files
                        print(f"{subindent}{file}")
                    if len(files) > 5:
                        print(f"{subindent}... and {len(files)-5} more files")
            
            # Verify dataset structure
            if os.path.exists(dataset_path):
                print("\nDataset structure:")
                for item in os.listdir(dataset_path):
                    print(f"  {item}")
                    
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Run all tests"""
    print("="*60)
    print("Testing Laughter Dataset Processing")
    print("="*60)
    
    # Test 1: Load markers
    df = test_load_markers()
    if df is None:
        print("\nCannot proceed without CSV file")
        return
    
    # Test 2: Segment grouping
    df_test = test_segment_grouping(df)
    
    # Test 3: Mini dataset
    test_mini_dataset(df_test)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
    
    print("\nTo run the full dataset processing, use:")
    print("python preprocess/laugh_dataset_process.py")


if __name__ == "__main__":
    main() 