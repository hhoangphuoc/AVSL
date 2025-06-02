"""
Script to create a HuggingFace dataset for laughter events and fluent speech segments
from the AMI Meeting Corpus based on ami_laugh_markers.csv
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
import argparse
import time

from preprocess.constants import (
    DATA_PATH, 
    SOURCE_PATH, 
    AMI_SPEAKERS,
    FACE_PREDICTOR_PATH,
    CNN_DETECTOR_PATH,
    MEAN_FACE_PATH
)
from preprocess.audio_process import batch_segment_audio
from preprocess.video_process import batch_segment_video, extract_and_save_lip_video, batch_process_lip_videos
from utils import av_to_hf_dataset, av_to_hf_dataset_with_shards


def load_laughter_markers(csv_path):
    """
    Load the laughter markers CSV file containing disfluency and laughter annotations.
    
    Args:
        csv_path: Path to the ami_laugh_markers.csv file
        
    Returns:
        DataFrame with columns: meeting_id, speaker_id, word, start_time, end_time, disfluency_type
    """
    print(f"Loading laughter markers from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Convert time columns to float
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')
    
    # Filter out invalid rows
    df = df.dropna(subset=['start_time', 'end_time'])
    df = df[df['end_time'] > df['start_time']]
    
    print(f"Loaded {len(df)} valid segments")
    print(f"Disfluency types: {df['disfluency_type'].value_counts().to_dict()}")
    
    return df


def create_segment_id(meeting_id, speaker_id, start_time, end_time, disfluency_type):
    """
    Create a unique segment ID for a laughter/fluent segment.
    
    Args:
        meeting_id: Meeting ID (e.g. 'ES2002a')
        speaker_id: Speaker ID (e.g. 'A')
        start_time: Start time in seconds
        end_time: End time in seconds
        disfluency_type: Type of segment ('laughter' or 'fluent')
        
    Returns:
        String segment ID
    """
    # Format times with 2 decimal places
    start_str = f"{start_time:.2f}"
    end_str = f"{end_time:.2f}"
    # Include disfluency type in ID to handle overlapping segments
    return f"{meeting_id}-{speaker_id}-{start_str}-{end_str}-{disfluency_type}"


def collect_segments_by_source(df_markers):
    """
    Group segments by their source audio/video files for efficient batch processing.
    
    Args:
        df_markers: DataFrame with laughter markers
        
    Returns:
        Tuple of (audio_segments_by_source, video_segments_by_source, segment_info)
    """
    audio_segments_by_source = defaultdict(list)
    video_segments_by_source = defaultdict(list)
    segment_info = {}  # Store segment metadata
    
    print("Grouping segments by source files...")
    
    for _, row in tqdm(df_markers.iterrows(), total=len(df_markers), desc="Processing segments"):
        meeting_id = row['meeting_id']
        speaker_id = row['speaker_id']
        start_time = row['start_time']
        end_time = row['end_time']
        disfluency_type = row['disfluency_type']
        transcript = row['word']
        
        # Skip if speaker not in mapping
        if speaker_id not in AMI_SPEAKERS:
            continue
            
        # Get source files
        audio_source = AMI_SPEAKERS[speaker_id]['audio']
        video_source = AMI_SPEAKERS[speaker_id]['video']
        
        source_audio_file = os.path.join(SOURCE_PATH, meeting_id, 'audio', f"{meeting_id}.{audio_source}.wav")
        source_video_file = os.path.join(SOURCE_PATH, meeting_id, 'video', f"{meeting_id}.{video_source}.avi")
        
        # Create segment ID
        segment_id = create_segment_id(meeting_id, speaker_id, start_time, end_time, disfluency_type)
        
        # Store segment info
        segment_info[segment_id] = {
            'meeting_id': meeting_id,
            'speaker_id': speaker_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'transcript': transcript,
            'disfluency_type': disfluency_type
        }
        
        # Add to source lists if files exist
        if os.path.exists(source_audio_file):
            audio_segments_by_source[source_audio_file].append((start_time, end_time, segment_id))
            
        if os.path.exists(source_video_file):
            video_segments_by_source[source_video_file].append((start_time, end_time, segment_id))
    
    print(f"Found {len(segment_info)} unique segments")
    print(f"Audio sources: {len(audio_segments_by_source)}, Video sources: {len(video_segments_by_source)}")
    
    return audio_segments_by_source, video_segments_by_source, segment_info


def process_laughter_dataset(
    csv_path,
    output_dir,
    dataset_path=None,
    extract_lip_videos=True,
    to_grayscale=True,
    batch_size=8,
    use_shards=True,
    files_per_shard=2000,
    use_parallel=None,
    num_workers=None,
    max_tasks_per_child=10
):
    """
    Main function to process laughter and fluent speech segments into a HuggingFace dataset.
    
    Args:
        csv_path: Path to ami_laugh_markers.csv
        output_dir: Directory to save processed audio/video segments
        dataset_path: Path to save HuggingFace dataset
        extract_lip_videos: Whether to extract lip regions from videos
        to_grayscale: Whether to save lip videos in grayscale
        batch_size: Batch size for lip extraction
        use_shards: Whether to use sharded dataset format
        files_per_shard: Number of files per shard if using sharded format
        use_parallel: Whether to use parallel processing for lip extraction (None=auto)
        num_workers: Number of worker processes (None=auto)
        max_tasks_per_child: Maximum tasks per worker before respawning
    """
    # Create output directories
    audio_segment_dir = os.path.join(output_dir, 'audio_segments')
    video_segment_dir = os.path.join(output_dir, 'video_segments')
    lip_video_dir = os.path.join(output_dir, 'lip_segments') if extract_lip_videos else None
    
    os.makedirs(audio_segment_dir, exist_ok=True)
    os.makedirs(video_segment_dir, exist_ok=True)
    if lip_video_dir:
        os.makedirs(lip_video_dir, exist_ok=True)
    
    # Load laughter markers
    df_markers = load_laughter_markers(csv_path)
    
    # Filter to only laughter and fluent segments
    df_markers = df_markers[df_markers['disfluency_type'].isin(['laughter', 'fluent'])]
    print(f"Processing {len(df_markers)} laughter/fluent segments")
    
    # Group segments by source
    audio_segments_by_source, video_segments_by_source, segment_info = collect_segments_by_source(df_markers)
    
    # Process audio segments
    print("\n=========================================== Processing Audio Segments ============================================\n")
    audio_results = {}
    successful_audio = 0
    
    for audio_file, segments in tqdm(audio_segments_by_source.items(), desc="Processing audio files"):
        # Prepare batch segments
        batch_segments = []
        segment_ids = []
        
        for start, end, seg_id in segments:
            output_path = os.path.join(audio_segment_dir, f"{seg_id}.wav")
            batch_segments.append((start, end, output_path))
            segment_ids.append(seg_id)
        
        # Process batch
        results = batch_segment_audio(audio_file, batch_segments)
        
        # Record results
        for i, (success, output_file) in enumerate(results):
            seg_id = segment_ids[i]
            audio_results[seg_id] = (success, output_file)
            if success:
                successful_audio += 1
    
    print(f"Successfully processed {successful_audio}/{len(segment_info)} audio segments (Success rate: {successful_audio/len(segment_info)*100:.1f}%)")
    
    # Process video segments
    print("\n=========================================== Processing Video Segments ============================================\n")
    video_results = {}
    successful_video = 0
    
    for video_file, segments in tqdm(video_segments_by_source.items(), desc="Processing video files"):
        # Prepare batch segments
        batch_segments = []
        segment_ids = []
        
        for start, end, seg_id in segments:
            output_path = os.path.join(video_segment_dir, f"{seg_id}.mp4")
            batch_segments.append((start, end, output_path))
            segment_ids.append(seg_id)
        
        # Process batch
        results = batch_segment_video(video_file, batch_segments)
        
        # Record results
        for i, (success, output_file) in enumerate(results):
            seg_id = segment_ids[i]
            video_results[seg_id] = (success, output_file)
            if success:
                successful_video += 1
    
    print(f"Successfully processed {successful_video}/{len(segment_info)} video segments (Success rate: {successful_video/len(segment_info)*100:.1f}%)")
    
    # Process lip videos
    lip_results = {}
    successful_lip = 0
    
    if extract_lip_videos and successful_video > 0:
        print("\n=========================================== Processing Lip Videos ============================================\n")
        
        # Check if we should use parallel processing
        if use_parallel is None:
            # Auto-decide based on dataset size
            use_parallel = batch_size > 1 and len(video_results) > 10
        
        if use_parallel:
            # Import batch processing function
            from preprocess.video_process import batch_process_lip_videos
            
            print(f"Using parallel processing for {len(video_results)} lip videos...")
            
            # Prepare batch inputs
            video_paths = []
            lip_output_paths = []
            segment_ids_ordered = []
            
            for seg_id, (video_success, video_path) in video_results.items():
                if video_success and video_path:
                    lip_output_path = os.path.join(lip_video_dir, f"{seg_id}_lip.mp4")
                    video_paths.append(video_path)
                    lip_output_paths.append(lip_output_path)
                    segment_ids_ordered.append(seg_id)
            
            if video_paths:
                # Configure batch processing parameters
                if num_workers is None:
                    num_workers = min(8, os.cpu_count() - 1) if hasattr(os, 'cpu_count') else 4
                
                batch_kwargs = {
                    'to_grayscale': to_grayscale,
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                    'max_tasks_per_child': max_tasks_per_child,
                    'use_multiprocessing': True,
                    'adaptive_memory': True
                }
                
                print(f"Processing with {num_workers} workers, batch size {batch_size}")
                
                # Process in batches
                results, successful_count = batch_process_lip_videos(
                    video_paths,
                    lip_output_paths,
                    **batch_kwargs
                )
                
                # Map results back to segment IDs
                for i, (success, lip_path) in enumerate(results):
                    if i < len(segment_ids_ordered):
                        seg_id = segment_ids_ordered[i]
                        lip_results[seg_id] = (success, lip_path)
                        
                successful_lip = successful_count
                
                # Handle any videos that weren't processed
                for seg_id in video_results:
                    if seg_id not in lip_results:
                        lip_results[seg_id] = (False, None)
            
        else:
            # Use sequential processing for small datasets
            print(f"Using sequential processing for {len(video_results)} lip videos...")
            
            for seg_id, (video_success, video_path) in tqdm(video_results.items(), desc="Extracting lip regions"):
                if video_success and video_path:
                    lip_output_path = os.path.join(lip_video_dir, f"{seg_id}_lip.mp4")
                    
                    try:
                        lip_success, lip_path = extract_and_save_lip_video(
                            video_path,
                            lip_output_path,
                            face_predictor_path=FACE_PREDICTOR_PATH,
                            cnn_detector_path=CNN_DETECTOR_PATH,
                            mean_face_path=MEAN_FACE_PATH,
                            to_grayscale=to_grayscale,
                            batch_size=batch_size
                        )
                        
                        lip_results[seg_id] = (lip_success, lip_path)
                        if lip_success:
                            successful_lip += 1
                            
                    except Exception as e:
                        print(f"Error processing lip video for {seg_id}: {e}")
                        lip_results[seg_id] = (False, None)
                else:
                    lip_results[seg_id] = (False, None)
        
        print(f"Successfully processed {successful_lip}/{len(video_results)} lip videos (Success rate: {successful_lip/len(video_results)*100:.1f}%)")
    
    # Create dataset records
    print("\n=========================================== Creating Dataset Records ============================================\n")
    dataset_records = []
    
    for seg_id, info in segment_info.items():
        record = {
            'segment_id': seg_id,
            'meeting_id': info['meeting_id'],
            'speaker_id': info['speaker_id'],
            'transcript': info['transcript'],
            'disfluency_type': info['disfluency_type'],
            'start_time': info['start_time'],
            'end_time': info['end_time'],
            'duration': info['duration']
        }
        
        # Add audio information
        if seg_id in audio_results:
            success, path = audio_results[seg_id]
            if success and path and os.path.exists(path):
                record['audio'] = path
                record['audio_path'] = path
            else:
                record['audio'] = None
                record['audio_path'] = None
        else:
            record['audio'] = None
            record['audio_path'] = None
        
        # Add video information
        if seg_id in video_results:
            success, path = video_results[seg_id]
            if success and path and os.path.exists(path):
                record['video_path'] = path
            else:
                record['video_path'] = None
        else:
            record['video_path'] = None
        
        # Add lip video information
        if seg_id in lip_results:
            success, path = lip_results[seg_id]
            if success and path and os.path.exists(path):
                record['lip_video'] = path
            else:
                record['lip_video'] = None
        else:
            record['lip_video'] = None
        
        dataset_records.append(record)
    
    print(f"Created {len(dataset_records)} dataset records")
    
    # Save dataset records to JSON
    records_path = os.path.join(output_dir, 'dataset_records.json')
    with open(records_path, 'w') as f:
        json.dump(dataset_records, f, indent=2)
    print(f"Saved dataset records to {records_path}")
    
    # Create HuggingFace dataset
    if dataset_path:
        print(f"\n=========================================== Creating HuggingFace Dataset ============================================\n")
        os.makedirs(dataset_path, exist_ok=True)
        
        # Filter out records with all None paths to avoid HuggingFace errors
        valid_records = []
        for record in dataset_records:
            # Check if record has at least one valid media file
            has_audio = record.get('audio') is not None
            has_video = record.get('video_path') is not None  
            has_lip = record.get('lip_video') is not None
            
            if has_audio or has_video or has_lip:
                valid_records.append(record)
        
        print(f"Using {len(valid_records)} valid records (out of {len(dataset_records)}) for HuggingFace dataset")
        
        if valid_records:
            try:
                if use_shards:
                    av_to_hf_dataset_with_shards(
                        valid_records,
                        dataset_path=dataset_path,
                        prefix="ami_laughter",
                        files_per_shard=files_per_shard
                    )
                else:
                    av_to_hf_dataset(
                        valid_records,
                        dataset_path=dataset_path,
                        prefix="ami_laughter"
                    )
                
                print(f"Dataset saved to {dataset_path}")
            except Exception as e:
                print(f"Error creating HuggingFace dataset: {e}")
                print("Saving dataset records only...")
        else:
            print("No valid records found with media files. Skipping HuggingFace dataset creation.")
    
    # Print summary statistics
    print("\n=========================================== Summary Statistics ============================================")
    df_records = pd.DataFrame(dataset_records)
    
    print(f"Total segments: {len(df_records)}")
    print(f"Disfluency type distribution:")
    print(df_records['disfluency_type'].value_counts())
    
    print(f"\nMedia availability:")
    print(f"Has audio: {df_records['audio'].notna().sum()} ({df_records['audio'].notna().sum()/len(df_records)*100:.1f}%)")
    print(f"Has video: {df_records['video_path'].notna().sum()} ({df_records['video_path'].notna().sum()/len(df_records)*100:.1f}%)")
    print(f"Has lip video: {df_records['lip_video'].notna().sum()} ({df_records['lip_video'].notna().sum()/len(df_records)*100:.1f}%)")
    
    print(f"\nDuration statistics:")
    print(f"Mean duration: {df_records['duration'].mean():.2f}s")
    print(f"Min duration: {df_records['duration'].min():.2f}s")
    print(f"Max duration: {df_records['duration'].max():.2f}s")
    
    return dataset_records


def find_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint to resume processing.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Tuple of (last_chunk_processed, checkpoint_data)
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} not found, starting from beginning")
        return -1, None
    
    # Look for checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('checkpoint_chunk_') and f.endswith('.json')]
    
    if not checkpoint_files:
        print("No checkpoints found, starting from beginning")
        return -1, None
    
    # Extract chunk numbers
    chunk_nums = [int(f.split('_')[2].split('.')[0]) for f in checkpoint_files]
    if not chunk_nums:
        return -1, None
    
    # Find the latest checkpoint
    last_chunk = max(chunk_nums)
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_chunk_{last_chunk}.json')
    
    # Load the checkpoint
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        completion_status = checkpoint.get('chunk_completed', False)
        
        if completion_status:
            # If chunk is marked complete, start next chunk from beginning
            print(f"Chunk {last_chunk} was completed, resuming from chunk {last_chunk + 1}")
            return last_chunk, checkpoint
        else:
            # If chunk is not complete, resume from this chunk
            print(f"Resuming from chunk {last_chunk}")
            return last_chunk, checkpoint
            
    except Exception as e:
        print(f"Error reading checkpoint file {checkpoint_file}: {str(e)}")
        print("Starting from beginning")
        return -1, None


def save_checkpoint(checkpoint_dir, chunk_id, completed=False, results=None, metadata=None):
    """
    Save a checkpoint for the current processing state.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        chunk_id: Current chunk ID
        completed: Whether the chunk was completed
        results: Processing results for the chunk
        metadata: Additional metadata to save
    """
    import datetime
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'chunk_id': chunk_id,
        'chunk_completed': completed,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    if results:
        checkpoint['results'] = results
    
    if metadata:
        checkpoint.update(metadata)
    
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_chunk_{chunk_id}.json')
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"Saved checkpoint for chunk {chunk_id}, completed: {completed}")


def process_laughter_dataset_in_chunks(
    csv_path,
    output_dir,
    dataset_path=None,
    chunk_size=1000,
    extract_lip_videos=True,
    to_grayscale=True,
    batch_size=8,
    use_shards=True,
    files_per_shard=2000,
    use_parallel=None,
    num_workers=None,
    max_tasks_per_child=10
):
    """
    Process the laughter dataset in chunks with checkpointing for resumption.
    
    Args:
        csv_path: Path to ami_laugh_markers.csv
        output_dir: Directory to save processed segments
        dataset_path: Path to save HuggingFace dataset
        chunk_size: Number of segments per chunk
        extract_lip_videos: Whether to extract lip regions from videos
        to_grayscale: Whether to save lip videos in grayscale
        batch_size: Batch size for lip extraction
        use_shards: Whether to use sharded dataset format
        files_per_shard: Number of files per shard if using sharded format
        use_parallel: Whether to use parallel processing for lip extraction (None=auto)
        num_workers: Number of worker processes (None=auto)
        max_tasks_per_child: Maximum tasks per worker before respawning
    """
    import math
    import datetime
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load laughter markers
    df_markers = load_laughter_markers(csv_path)
    
    # Filter to only laughter and fluent segments
    df_markers = df_markers[df_markers['disfluency_type'].isin(['laughter', 'fluent'])]
    print(f"Processing {len(df_markers)} laughter/fluent segments in chunks of {chunk_size}")
    
    # Calculate number of chunks
    total_segments = len(df_markers)
    num_chunks = math.ceil(total_segments / chunk_size)
    
    print(f"Total segments: {total_segments}")
    print(f"Number of chunks: {num_chunks}")
    
    # Find checkpoint
    resume_chunk, checkpoint_data = find_checkpoint(checkpoint_dir)
    
    # Track overall statistics
    overall_stats = {
        'total_processed': 0,
        'total_successful_audio': 0,
        'total_successful_video': 0,
        'total_successful_lip': 0,
        'start_time': datetime.datetime.now().isoformat(),
        'chunk_stats': []
    }
    
    all_dataset_records = []
    
    # Process each chunk
    for chunk_idx in range(num_chunks):
        # Skip chunks before resume point
        if chunk_idx <= resume_chunk and checkpoint_data and checkpoint_data.get('chunk_completed', False):
            print(f"Skipping chunk {chunk_idx} (already completed)")
            # Load existing results if available
            if 'results' in checkpoint_data:
                chunk_records = checkpoint_data['results']
                if isinstance(chunk_records, list):
                    all_dataset_records.extend(chunk_records)
            continue
        
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_segments)
        
        chunk_df = df_markers.iloc[start_idx:end_idx].copy()
        
        print(f"\nProcessing chunk {chunk_idx+1}/{num_chunks} ({len(chunk_df)} segments)...")
        chunk_start_time = time.time()
        
        # Save chunk to temporary CSV
        chunk_csv = os.path.join(checkpoint_dir, f"chunk_{chunk_idx}.csv")
        chunk_df.to_csv(chunk_csv, index=False)
        
        # Create chunk-specific output directories
        chunk_output = os.path.join(output_dir, f"chunk_{chunk_idx:04d}")
        
        try:
            # Process the chunk
            chunk_records = process_laughter_dataset(
                csv_path=chunk_csv,
                output_dir=chunk_output,
                dataset_path=None,  # Don't create HF dataset for individual chunks
                extract_lip_videos=extract_lip_videos,
                to_grayscale=to_grayscale,
                batch_size=batch_size,
                use_shards=False,  # Individual chunks don't use shards
                files_per_shard=files_per_shard,
                use_parallel=use_parallel,
                num_workers=num_workers,
                max_tasks_per_child=max_tasks_per_child
            )
            
            # Calculate chunk statistics
            chunk_time = time.time() - chunk_start_time
            successful_audio = sum(1 for r in chunk_records if r.get('audio') is not None)
            successful_video = sum(1 for r in chunk_records if r.get('video_path') is not None)
            successful_lip = sum(1 for r in chunk_records if r.get('lip_video') is not None)
            
            # Update overall statistics
            overall_stats['total_processed'] += len(chunk_records)
            overall_stats['total_successful_audio'] += successful_audio
            overall_stats['total_successful_video'] += successful_video
            overall_stats['total_successful_lip'] += successful_lip
            
            chunk_stats = {
                'chunk_idx': chunk_idx,
                'processed': len(chunk_records),
                'successful_audio': successful_audio,
                'successful_video': successful_video,
                'successful_lip': successful_lip,
                'time_seconds': chunk_time
            }
            overall_stats['chunk_stats'].append(chunk_stats)
            
            # Save chunk completion status
            save_checkpoint(
                checkpoint_dir, 
                chunk_idx, 
                completed=True, 
                results=chunk_records,
                metadata=chunk_stats
            )
            
            # Add to all records
            all_dataset_records.extend(chunk_records)
            
            print(f"Chunk {chunk_idx} completed in {chunk_time:.2f} seconds")
            print(f"  Audio: {successful_audio}/{len(chunk_records)}")
            print(f"  Video: {successful_video}/{len(chunk_records)}")
            print(f"  Lip: {successful_lip}/{len(chunk_records)}")
            
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Save error checkpoint
            save_checkpoint(
                checkpoint_dir, 
                chunk_idx, 
                completed=False, 
                results=None,
                metadata={'error': str(e)}
            )
            
        finally:
            # Clean up temporary CSV
            if os.path.exists(chunk_csv):
                os.remove(chunk_csv)
    
    # Save overall statistics
    overall_stats['end_time'] = datetime.datetime.now().isoformat()
    overall_stats['total_time_minutes'] = (
        datetime.datetime.now() - 
        datetime.datetime.fromisoformat(overall_stats['start_time'])
    ).total_seconds() / 60
    
    with open(os.path.join(checkpoint_dir, 'overall_stats.json'), 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    # Create final HuggingFace dataset from all chunks
    if dataset_path and all_dataset_records:
        print(f"\nCreating final HuggingFace dataset with {len(all_dataset_records)} records...")
        
        # Filter out records with all None paths to avoid HuggingFace errors
        valid_records = []
        for record in all_dataset_records:
            has_audio = record.get('audio') is not None
            has_video = record.get('video_path') is not None  
            has_lip = record.get('lip_video') is not None
            
            if has_audio or has_video or has_lip:
                valid_records.append(record)
        
        print(f"Using {len(valid_records)} valid records for final dataset")
        
        if valid_records:
            try:
                os.makedirs(dataset_path, exist_ok=True)
                
                if use_shards:
                    av_to_hf_dataset_with_shards(
                        valid_records,
                        dataset_path=dataset_path,
                        prefix="ami_laughter",
                        files_per_shard=files_per_shard
                    )
                else:
                    av_to_hf_dataset(
                        valid_records,
                        dataset_path=dataset_path,
                        prefix="ami_laughter"
                    )
                
                print(f"Final dataset saved to {dataset_path}")
            except Exception as e:
                print(f"Error creating final HuggingFace dataset: {e}")
                import traceback
                traceback.print_exc()
    
    # Print final summary
    print("\n" + "="*80)
    print("CHUNKED PROCESSING SUMMARY")
    print("="*80)
    print(f"Total segments processed: {overall_stats['total_processed']}")
    print(f"Successful audio segments: {overall_stats['total_successful_audio']}")
    print(f"Successful video segments: {overall_stats['total_successful_video']}")
    print(f"Successful lip segments: {overall_stats['total_successful_lip']}")
    print(f"Total processing time: {overall_stats['total_time_minutes']:.2f} minutes")
    print("="*80)
    
    return all_dataset_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create HuggingFace dataset for laughter and fluent speech segments')
    
    parser.add_argument('--csv_path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'ami_laugh_markers.csv'),
                       help='Path to ami_laugh_markers.csv file')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(DATA_PATH, 'laughter_dataset', 'segments'),
                       help='Directory to save processed segments')
    parser.add_argument('--dataset_path', type=str,
                       default="/home/s2587130/AVSL/data/ami_laugh/dataset",
                       help='Path to save HuggingFace dataset')
    parser.add_argument('--extract_lip_videos', action='store_true', default=True,
                       help='Extract lip videos from video segments')
    parser.add_argument('--no_lip_videos', dest='extract_lip_videos', action='store_false',
                       help='Skip lip video extraction')
    parser.add_argument('--to_grayscale', action='store_true', default=True,
                       help='Convert lip videos to grayscale')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for lip extraction')
    parser.add_argument('--use_shards', action='store_true', default=True,
                       help='Use sharded dataset format')
    parser.add_argument('--no_shards', dest='use_shards', action='store_false',
                       help='Use standard dataset format')
    parser.add_argument('--files_per_shard', type=int, default=2000,
                       help='Number of files per shard')
    
    # Chunked processing arguments
    parser.add_argument('--chunked', action='store_true', default=False,
                       help='Process dataset in chunks with checkpointing')
    parser.add_argument('--chunk_size', type=int, default=1000,
                       help='Number of segments per chunk (for chunked processing)')
    
    # Add parallel processing arguments
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes for parallel lip extraction (default: auto)')
    parser.add_argument('--use_parallel', action='store_true', default=None,
                       help='Force use of parallel processing for lip extraction')
    parser.add_argument('--no_parallel', dest='use_parallel', action='store_false',
                       help='Disable parallel processing for lip extraction')
    parser.add_argument('--max_tasks_per_child', type=int, default=10,
                       help='Maximum tasks per worker process before respawning')
    
    args = parser.parse_args()
    
    print("="*50)
    print("AMI Laughter Dataset Processing")
    print("="*50)
    print(f"CSV path: {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Extract lip videos: {args.extract_lip_videos}")
    print(f"Grayscale lip videos: {args.to_grayscale}")
    print(f"Use shards: {args.use_shards}")
    print(f"Chunked processing: {args.chunked}")
    if args.chunked:
        print(f"Chunk size: {args.chunk_size}")
    if args.extract_lip_videos:
        print(f"Parallel processing: {args.use_parallel if args.use_parallel is not None else 'auto'}")
        if args.num_workers:
            print(f"Number of workers: {args.num_workers}")
    print("="*50)
    
    if args.chunked:
        # Use chunked processing with checkpointing
        print("Using chunked processing with checkpointing...")
        records = process_laughter_dataset_in_chunks(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            dataset_path=args.dataset_path,
            chunk_size=args.chunk_size,
            extract_lip_videos=args.extract_lip_videos,
            to_grayscale=args.to_grayscale,
            batch_size=args.batch_size,
            use_shards=args.use_shards,
            files_per_shard=args.files_per_shard,
            use_parallel=args.use_parallel,
            num_workers=args.num_workers,
            max_tasks_per_child=args.max_tasks_per_child
        )
    else:
        # Use standard processing
        print("Using standard processing...")
        records = process_laughter_dataset(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            dataset_path=args.dataset_path,
            extract_lip_videos=args.extract_lip_videos,
            to_grayscale=args.to_grayscale,
            batch_size=args.batch_size,
            use_shards=args.use_shards,
            files_per_shard=args.files_per_shard,
            use_parallel=args.use_parallel,
            num_workers=args.num_workers,
            max_tasks_per_child=args.max_tasks_per_child
        ) 