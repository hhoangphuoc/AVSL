"""
Script to create a HuggingFace dataset from AMI meeting segments based on ami-segments-info.csv
This function processes existing AMI segment files and creates a validated HuggingFace dataset
where each record contains segment-id, meeting-id, speaker-id, transcript, audio as HuggingFace Audio() object, and lip_video as HuggingFace Video() object.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import argparse

from preprocess.constants import SOURCE_PATH
from utils import av_to_hf_dataset

def load_ami_segments_csv(csv_path):
    """
    Load and validate the AMI segments CSV file.
    
    Args:
        csv_path: Path to the ami-segments-info.csv file
        
    Returns:
        DataFrame with AMI segments information
    """
    print(f"Loading AMI segments from {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_columns = ['id', 'meeting_id', 'speaker_id', 'start_time', 'end_time', 'duration', 'transcript', 
                       'has_audio', 'has_video', 'has_lip_video', 'audio', 'video', 'lip_video', '_audio_abs', '_video_abs', '_lip_video_abs']
    
    print("Columns in CSV:", df.columns)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    # Use absolute paths as the path to the audio, video, and lip video files
    # replace relative paths in df['audio'], df['video'], and df['lip_video'] 
    # with absolute paths in df['_audio_abs'], df['_video_abs'], and df['_lip_video_abs']
    if '_audio_abs' in df.columns:
        df['audio_path'] = df['_audio_abs']
        print("Using absolute audio paths from '_audio_abs' column")
    else:
        df['audio_path'] = None
        print("No audio path column found")
    
    if '_video_abs' in df.columns:
        df['video_path'] = df['_video_abs']
        print("Using absolute video paths from '_video_abs' column")
    else:
        df['video_path'] = None
        print("No video path column found")
    
    if '_lip_video_abs' in df.columns:
        df['lip_video_path'] = df['_lip_video_abs']
        print("Using absolute lip video paths from '_lip_video_abs' column")
    else:
        df['lip_video_path'] = None
        print("No lip video path column found")
    
    # Convert boolean columns
    for col in ['has_audio', 'has_video', 'has_lip_video']:
        df[col] = df[col].astype(bool)
    
    # Convert time columns to float
    for col in ['start_time', 'end_time', 'duration']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with invalid time data
    initial_count = len(df)
    df = df.dropna(subset=['start_time', 'end_time', 'duration'])
    df = df[df['end_time'] > df['start_time']]
    df = df[df['duration'] > 0]
    
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} rows with invalid time data")
    
    print(f"Loaded {len(df)} valid segments")
    print(f"Segments with audio: {df['has_audio'].sum()}")
    print(f"Segments with video: {df['has_video'].sum()}")
    print(f"Segments with lip video: {df['has_lip_video'].sum()}")
    
    return df


def create_dataset_records(df):
    """
    Create dataset records from validated segment DataFrame.
    Args:
        df: DataFrame with validated segments
        
    Returns:
        List of dataset records suitable for HuggingFace dataset creation
    """
    print("Creating dataset records...")
    
    dataset_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating records"):
        record = {
            'segment_id': row['id'],
            'meeting_id': row['meeting_id'],
            'speaker_id': row['speaker_id'],
            'start_time': row['start_time'],
            'end_time': row['end_time'],
            'duration': row['duration'],
            'transcript': row['transcript']
        }
        
        # Add audio path if valid
        if row.get('has_audio', False) and pd.notna(row['audio_path']):
            audio_path = row['audio_path']
            record['audio'] = audio_path
        
        # Add video path if valid
        if row.get('has_video', False) and pd.notna(row['video_path']):
            video_path = row['video_path']
            record['video'] = video_path
        
        # Add lip video path if valid (only if HF Video compatible)
        if row.get('has_lip_video', False) and pd.notna(row['lip_video_path']):
            lip_video_path = row['lip_video_path']
            record['lip_video'] = lip_video_path
        
        dataset_records.append(record)
    
    print(f"Created {len(dataset_records)} dataset records")
    
    return dataset_records


def process_ami_segments_dataset(
    csv_path,
    output_dir,
    dataset_path=None,
):
    """
    MAIN FUNCTION to process AMI segments CSV into a validated HuggingFace dataset.
    
    Args:
        csv_path: Path to ami-segments-info.csv file
        output_dir: Directory to save processing results
        dataset_path: Path to save HuggingFace dataset
        
    Returns:
        List of dataset records
    """
    print("="*80)
    print("AMI SEGMENTS DATASET PROCESSING")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV
    df = load_ami_segments_csv(csv_path)
    
    # Create dataset records ------------------------------------------------------------------------------
    print("\n" + "-"*40)
    print("CREATING DATASET RECORDS")
    print("-"*40)
    dataset_records = create_dataset_records(df)
    
    # Save dataset records to JSON
    records_path = os.path.join(output_dir, 'dataset_records.json')
    with open(records_path, 'w') as f:
        json.dump(dataset_records, f, indent=2)
    print(f"Saved dataset records to {records_path}")
    
    # Create HuggingFace dataset
    if dataset_path:
        print(f"Creating HuggingFace dataset at {dataset_path}")
        os.makedirs(dataset_path, exist_ok=True)
        
        try:
            dataset = av_to_hf_dataset(
                dataset_records,
                dataset_path=dataset_path,
                prefix="ami_clean"
            )
            
            print(f"Dataset successfully saved to {dataset_path}")
        except Exception as e:
            print(f"Error creating HuggingFace dataset: {e}")
    
    return dataset, dataset_records


if __name__ == "__main__":
    # Add project root and utils to path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    utils_path = os.path.join(project_root, 'utils')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    
    parser = argparse.ArgumentParser(description='Process AMI segments CSV into HuggingFace dataset')
    
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to ami-segments-info.csv file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processing results')
    parser.add_argument('--dataset_path', type=str,
                       help='Path to save HuggingFace dataset')
    
    args = parser.parse_args()
    
    # Process the dataset
    dataset, dataset_records = process_ami_segments_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
    )
    
    print(f"Processing complete. {len(dataset_records)} records created.")
    print(f"Dataset (HuggingFace Dataset): {dataset}")