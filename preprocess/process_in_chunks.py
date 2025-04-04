#!/usr/bin/env python3
"""
Script to process lip videos in smaller chunks with intelligent multiprocessing,
with checkpoint resumption capabilities. This script processes the main dataset CSV 
in smaller batches, saving progress after each chunk to allow for resuming if the 
job crashes or times out.
"""

import os
import sys
import pandas as pd
import argparse
import math
from tqdm import tqdm
import json
import time
import datetime
import gc
import traceback
import multiprocessing as mp

def prepare_chunks(
    csv_path, 
    output_dir, 
    chunk_size=1000, 
    filter_processed=True
):
    """
    Prepare dataset chunks for sequential processing.
    
    Args:
        csv_path: Path to the main dataset CSV
        output_dir: Directory to save chunk metadata
        chunk_size: Number of records per chunk
        filter_processed: Whether to filter out already processed videos
    
    Returns:
        List of chunks, each containing a list of video records
    """
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    # Filter for videos only
    video_df = df[df['has_video'] == True].copy()
    video_df = video_df.dropna(subset=['video'])
    print(f"Found {len(video_df)} records with videos")
    
    # Filter out already processed lip videos if requested
    if filter_processed and 'has_lip_video' in video_df.columns:
        # Keep only videos without lip processing or failed processing
        unprocessed_df = video_df[~video_df['has_lip_video']].copy()
        print(f"Found {len(unprocessed_df)} unprocessed videos")
        video_df = unprocessed_df
    
    # Check if we have videos to process
    if video_df.empty:
        print("No videos to process!")
        return []
    
    # Calculate number of chunks
    total_videos = len(video_df)
    num_chunks = math.ceil(total_videos / chunk_size)
    print(f"Dividing {total_videos} videos into {num_chunks} chunks of ~{chunk_size} videos each")
    
    # Reset index for proper chunking
    video_df = video_df.reset_index(drop=True)
    
    # Create chunks
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_videos)
        
        # Extract chunk
        chunk_df = video_df.iloc[start_idx:end_idx].copy()
        chunks.append({
            'chunk_id': i,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'num_videos': len(chunk_df),
            'records': chunk_df.to_dict('records')
        })
        
    # Save chunk metadata
    chunks_meta = {
        'total_chunks': num_chunks,
        'total_videos': total_videos,
        'chunk_size': chunk_size,
        'created_time': datetime.datetime.now().isoformat(),
        'chunks': [{'chunk_id': c['chunk_id'], 'num_videos': c['num_videos']} for c in chunks]
    }
    
    with open(os.path.join(output_dir, 'chunks_metadata.json'), 'w') as f:
        json.dump(chunks_meta, f, indent=2)
    
    return chunks

def find_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint to resume processing.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Tuple of (last_chunk_processed, last_video_processed)
    """
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} not found, starting from beginning")
        return -1, -1
    
    # Look for checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('checkpoint_chunk_') and f.endswith('.json')]
    
    if not checkpoint_files:
        print("No checkpoints found, starting from beginning")
        return -1, -1
    
    # Extract chunk numbers
    chunk_nums = [int(f.split('_')[2].split('.')[0]) for f in checkpoint_files]
    if not chunk_nums:
        return -1, -1
    
    # Find the latest checkpoint
    last_chunk = max(chunk_nums)
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_chunk_{last_chunk}.json')
    
    # Load the checkpoint
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        last_video = checkpoint.get('last_video_processed', -1)
        completion_status = checkpoint.get('chunk_completed', False)
        
        if completion_status:
            # If chunk is marked complete, start next chunk from beginning
            print(f"Chunk {last_chunk} was completed, resuming from chunk {last_chunk + 1}")
            return last_chunk, -1
        else:
            # If chunk is not complete, resume from last processed video
            print(f"Resuming from chunk {last_chunk}, video {last_video + 1}")
            return last_chunk, last_video
            
    except Exception as e:
        print(f"Error reading checkpoint file {checkpoint_file}: {str(e)}")
        print("Starting from beginning")
        return -1, -1

def save_checkpoint(checkpoint_dir, chunk_id, video_idx, completed=False, results=None):
    """
    Save a checkpoint for the current processing state.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        chunk_id: Current chunk ID
        video_idx: Index of last processed video in chunk
        completed: Whether the chunk was completed
        results: Processing results for the chunk
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'chunk_id': chunk_id,
        'last_video_processed': video_idx,
        'chunk_completed': completed,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    if results:
        checkpoint['results'] = results
    
    checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_chunk_{chunk_id}.json')
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"Saved checkpoint for chunk {chunk_id}, video {video_idx}, completed: {completed}")

def update_csv_with_results(csv_path, results):
    """
    Update the main CSV file with processing results.
    
    Args:
        csv_path: Path to the main CSV file
        results: Dictionary mapping segment IDs to (success, lip_video_path) tuples
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Function to map results back
        def get_lip_info(segment_id):
            if segment_id in results:
                success, path = results[segment_id]
                return pd.Series([success, path if success else None])
            return pd.Series([False, None]) # Default if not processed or failed
        
        # Ensure the columns exist before assigning
        if 'has_lip_video' not in df.columns:
            df['has_lip_video'] = False
        if 'lip_video' not in df.columns:
            df['lip_video'] = None
            
        # Update values that we processed (don't overwrite others)
        processed_ids = list(results.keys())
        mask = df['id'].isin(processed_ids)
        
        # Apply results only to rows with IDs we processed
        df.loc[mask, ['has_lip_video', 'lip_video']] = pd.DataFrame(
            [get_lip_info(id) for id in df.loc[mask, 'id']],
            index=df.loc[mask].index
        )
        
        # Convert lip_video to strings or NA if None
        df['lip_video'] = df['lip_video'].astype(str)
        df.loc[df['lip_video'] == 'None', 'lip_video'] = pd.NA # Use pandas NA for missing strings
        
        # Save the updated DataFrame back to the CSV
        df.to_csv(csv_path, index=False)
        print(f"Successfully updated {csv_path} with lip video information")
        
    except Exception as e:
        print(f"Error updating CSV with results: {str(e)}")
        traceback.print_exc()

def process_chunks_sequentially(
    chunks, 
    dataset_csv, 
    lip_video_dir, 
    checkpoint_dir,
    batch_size=8, 
    to_grayscale=True,
    max_videos_per_run=None
):
    """
    Process video chunks sequentially with checkpoint saving.
    
    Args:
        chunks: List of chunk dictionaries
        dataset_csv: Path to the dataset CSV file
        lip_video_dir: Directory to save lip videos
        checkpoint_dir: Directory to save checkpoints
        batch_size: Batch size for frame processing within a video
        to_grayscale: Whether to extract lip videos in grayscale
        max_videos_per_run: Maximum number of videos to process in a single run (for testing)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(lip_video_dir, exist_ok=True)
    
    # Import here to prevent circular dependencies
    from video_process import extract_and_save_lip_video
    
    # Find checkpoint
    resume_chunk, resume_video = find_checkpoint(checkpoint_dir)
    
    # Track overall statistics
    overall_stats = {
        'total_processed': 0,
        'total_successful': 0,
        'start_time': datetime.datetime.now().isoformat(),
        'chunk_stats': []
    }
    
    # Counter for max videos
    videos_processed = 0
    
    # Process each chunk
    for chunk in chunks:
        chunk_id = chunk['chunk_id']
        
        # Skip chunks before resume point
        if chunk_id < resume_chunk:
            print(f"Skipping chunk {chunk_id} (already processed)")
            continue
        
        chunk_records = chunk['records']
        chunk_results = {}
        chunk_start_time = time.time()
        successful_count = 0
        
        print(f"\n===== Processing chunk {chunk_id} ({len(chunk_records)} videos) =====")
        
        # Determine starting point within chunk
        start_idx = 0
        if chunk_id == resume_chunk and resume_video >= 0:
            start_idx = resume_video + 1
            print(f"Resuming chunk {chunk_id} from video {start_idx}")
        
        # Process each video in the chunk
        for video_idx, record in enumerate(chunk_records[start_idx:], start=start_idx):
            # Check if we've hit the maximum videos limit
            if max_videos_per_run is not None and videos_processed >= max_videos_per_run:
                print(f"Reached maximum videos limit ({max_videos_per_run})")
                save_checkpoint(checkpoint_dir, chunk_id, video_idx-1, completed=False, results=chunk_results)
                
                # Update the main CSV with results so far
                if chunk_results:
                    update_csv_with_results(dataset_csv, chunk_results)
                    
                return overall_stats
            
            video_id = record['id']
            video_path = record['video']
            
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}, skipping")
                chunk_results[video_id] = (False, None)
                continue
            
            # Setup output path
            lip_output_path = os.path.join(lip_video_dir, f"{video_id}-lip_video.mp4")
            
            # Process the video
            print(f"\nProcessing {video_id} ({video_idx+1}/{len(chunk_records)}) - {os.path.basename(video_path)}")
            try:
                # Force garbage collection
                gc.collect()
                
                success, lip_output_file = extract_and_save_lip_video(
                    video_path,
                    lip_output_path,
                    to_grayscale=to_grayscale,
                    batch_size=batch_size,
                    adaptive_memory=True,
                    max_frames=None # Disable max_frames limit: TODO: Change back to 300
                )
                
                # Record results
                chunk_results[video_id] = (success, lip_output_file)
                if success:
                    successful_count += 1
                    
                # Save checkpoint regularly (every 10 videos or at the end)
                if (video_idx + 1) % 10 == 0 or video_idx == len(chunk_records) - 1:
                    save_checkpoint(checkpoint_dir, chunk_id, video_idx, completed=False, results=chunk_results)
                
                # Update counters
                videos_processed += 1
                
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user")
                save_checkpoint(checkpoint_dir, chunk_id, video_idx, completed=False, results=chunk_results)
                
                # Update CSV with results so far
                if chunk_results:
                    update_csv_with_results(dataset_csv, chunk_results)
                    
                return overall_stats
                
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                chunk_results[video_id] = (False, None)
                traceback.print_exc()
        
        # Chunk completed
        chunk_time = time.time() - chunk_start_time
        videos_per_minute = len(chunk_records) / (chunk_time / 60) if chunk_time > 0 else 0
        
        print(f"\nCompleted chunk {chunk_id}")
        print(f"Time taken: {chunk_time:.2f} seconds ({videos_per_minute:.2f} videos/minute)")
        print(f"Success rate: {successful_count}/{len(chunk_records)} ({(successful_count/len(chunk_records))*100:.1f}%)")
        
        # Save final checkpoint for this chunk
        save_checkpoint(checkpoint_dir, chunk_id, len(chunk_records)-1, completed=True, results=chunk_results)
        
        # Update the main CSV with chunk results
        update_csv_with_results(dataset_csv, chunk_results)
        
        # Update overall stats
        overall_stats['total_processed'] += len(chunk_records)
        overall_stats['total_successful'] += successful_count
        overall_stats['chunk_stats'].append({
            'chunk_id': chunk_id,
            'processed': len(chunk_records),
            'successful': successful_count,
            'time_seconds': chunk_time,
            'videos_per_minute': videos_per_minute
        })
        
        # Save overall stats
        overall_stats['end_time'] = datetime.datetime.now().isoformat()
        overall_stats['total_time_minutes'] = (datetime.datetime.now() - 
                                           datetime.datetime.fromisoformat(overall_stats['start_time'])).total_seconds() / 60
        
        with open(os.path.join(checkpoint_dir, 'overall_stats.json'), 'w') as f:
            json.dump(overall_stats, f, indent=2)
    
    print("\n===== All chunks processed =====")
    print(f"Total videos: {overall_stats['total_processed']}")
    print(f"Successfully processed: {overall_stats['total_successful']}")
    success_rate = overall_stats['total_successful']/overall_stats['total_processed'] if overall_stats['total_processed'] > 0 else 0
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Total time: {overall_stats['total_time_minutes']:.2f} minutes")
    
    return overall_stats

def process_chunks_with_multiprocessing(
    chunks, 
    dataset_csv, 
    lip_video_dir, 
    checkpoint_dir,
    batch_size=8, 
    to_grayscale=True,
    max_videos_per_run=None,
    num_workers=None,
    max_tasks_per_child=10,
    use_multiprocessing=True
):
    """
    Process video chunks with multiprocessing and checkpoint saving.
    
    Args:
        chunks: List of chunk dictionaries
        dataset_csv: Path to the dataset CSV file
        lip_video_dir: Directory to save lip videos
        checkpoint_dir: Directory to save checkpoints
        batch_size: Batch size for frame processing within a video
        to_grayscale: Whether to extract lip videos in grayscale
        max_videos_per_run: Maximum number of videos to process in a single run (for testing)
        num_workers: Number of worker processes to use
        max_tasks_per_child: Maximum number of tasks per worker before respawning
        use_multiprocessing: Whether to use multiprocessing
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(lip_video_dir, exist_ok=True)
    
    # Import batch_process_lip_videos for chunk processing
    from video_process import batch_process_lip_videos
    
    # Find checkpoint
    resume_chunk, resume_video = find_checkpoint(checkpoint_dir)
    
    # Track overall statistics
    overall_stats = {
        'total_processed': 0,
        'total_successful': 0,
        'start_time': datetime.datetime.now().isoformat(),
        'chunk_stats': []
    }
    
    # Counter for max videos
    videos_processed = 0
    
    # Process each chunk
    for chunk in chunks:
        chunk_id = chunk['chunk_id']
        
        # Skip chunks before resume point
        if chunk_id < resume_chunk:
            print(f"Skipping chunk {chunk_id} (already processed)")
            continue
        
        chunk_records = chunk['records']
        chunk_start_time = time.time()
        
        print(f"\n===== Processing chunk {chunk_id} ({len(chunk_records)} videos) =====")
        
        # Determine starting point within chunk
        start_idx = 0
        if chunk_id == resume_chunk and resume_video >= 0:
            start_idx = resume_video + 1
            print(f"Resuming chunk {chunk_id} from video {start_idx}")
        
        # Check if we've hit the maximum videos limit
        remaining_quota = max_videos_per_run - videos_processed if max_videos_per_run is not None else None
        if remaining_quota is not None and remaining_quota <= 0:
            print(f"Reached maximum videos limit ({max_videos_per_run})")
            break
            
        # Determine how many videos to process in this chunk
        videos_to_process = chunk_records[start_idx:]
        if remaining_quota is not None:
            videos_to_process = videos_to_process[:remaining_quota]
            
        # Prepare batch processing inputs
        video_segment_results = {}
        for record in videos_to_process:
            video_id = record['id']
            video_path = record['video']
            if os.path.exists(video_path):
                video_segment_results[video_id] = (True, video_path)
        
        # Skip empty chunks
        if not video_segment_results:
            print(f"No valid videos to process in chunk {chunk_id}, skipping")
            continue
            
        print(f"Batch processing {len(video_segment_results)} videos from chunk {chunk_id}")
        
        # Process using batch_process_lip_videos
        try:
            # Configure processing parameters
            process_kwargs = {
                'to_grayscale': to_grayscale,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'max_tasks_per_child': max_tasks_per_child,
                'use_multiprocessing': use_multiprocessing,
                'adaptive_memory': True  # Enable adaptive memory handling instead of max_frames
            }
            
            # Process the chunk
            lip_segment_results, successful_lip_segments = batch_process_lip_videos(
                video_segment_results,
                lip_video_dir,
                **process_kwargs
            )
            
            # Update overall statistics
            videos_processed += len(video_segment_results)
            overall_stats['total_processed'] += len(video_segment_results)
            overall_stats['total_successful'] += successful_lip_segments
            
            # Calculate performance metrics
            chunk_time = time.time() - chunk_start_time
            videos_per_minute = len(video_segment_results) / (chunk_time / 60) if chunk_time > 0 else 0
            
            # Record chunk statistics
            chunk_stats = {
                'chunk_id': chunk_id,
                'processed': len(video_segment_results),
                'successful': successful_lip_segments,
                'time_seconds': chunk_time,
                'videos_per_minute': videos_per_minute,
                'success_rate': successful_lip_segments / len(video_segment_results) if video_segment_results else 0
            }
            overall_stats['chunk_stats'].append(chunk_stats)
            
            # Print chunk summary
            print(f"\nCompleted chunk {chunk_id}")
            print(f"Time taken: {chunk_time:.2f} seconds ({videos_per_minute:.2f} videos/minute)")
            print(f"Success rate: {successful_lip_segments}/{len(video_segment_results)} ({chunk_stats['success_rate']*100:.1f}%)")
            
            # Save checkpoint for this chunk
            save_checkpoint(checkpoint_dir, chunk_id, len(videos_to_process)-1, completed=True, results=lip_segment_results)
            
            # Update the main CSV with chunk results
            update_csv_with_results(dataset_csv, lip_segment_results)
            
            # Save overall stats after each chunk
            overall_stats['end_time'] = datetime.datetime.now().isoformat()
            overall_stats['total_time_minutes'] = (datetime.datetime.now() - 
                                               datetime.datetime.fromisoformat(overall_stats['start_time'])).total_seconds() / 60
            
            with open(os.path.join(checkpoint_dir, 'overall_stats.json'), 'w') as f:
                json.dump(overall_stats, f, indent=2)
                
        except KeyboardInterrupt:
            print(f"\nProcessing of chunk {chunk_id} interrupted by user")
            save_checkpoint(checkpoint_dir, chunk_id, start_idx + len(video_segment_results)//2, 
                           completed=False, results=lip_segment_results if 'lip_segment_results' in locals() else None)
            raise
            
        except Exception as e:
            print(f"\nError processing chunk {chunk_id}: {str(e)}")
            traceback.print_exc()
            save_checkpoint(checkpoint_dir, chunk_id, start_idx + len(video_segment_results)//2, 
                           completed=False, results=lip_segment_results if 'lip_segment_results' in locals() else None)
    
    print("\n===== All chunks processed =====")
    print(f"Total videos: {overall_stats['total_processed']}")
    print(f"Successfully processed: {overall_stats['total_successful']}")
    success_rate = overall_stats['total_successful']/overall_stats['total_processed'] if overall_stats['total_processed'] > 0 else 0
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Total time: {overall_stats['total_time_minutes']:.2f} minutes")
    
    return overall_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos in chunks with multiprocessing and checkpoint resumption")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the main dataset CSV")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs and checkpoints")
    parser.add_argument("--lip_video_dir", type=str, required=True, help="Directory to save lip videos")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of videos per chunk")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for frame processing within a video")
    parser.add_argument("--filter_processed", action="store_true", help="Filter out already processed videos")
    parser.add_argument("--to_grayscale", action="store_true", default=True, help="Extract lip videos in grayscale")
    parser.add_argument("--max_videos", type=int, help="Maximum number of videos to process (for testing)")
    
    #Sequential processing arguments
    parser.add_argument("--use_sequential", action="store_true", help="Use sequential processing instead of multiprocessing")
    
    # Multiprocessing arguments
    parser.add_argument("--adaptive_memory", action="store_true", help="Use adaptive memory management")
    parser.add_argument("--num_workers", type=int, help="Number of worker processes for multiprocessing (default: CPU count - 1)")
    parser.add_argument("--max_tasks_per_child", type=int, default=10, help="Maximum tasks per worker before respawning")
    parser.add_argument("--disable_multiprocessing", action="store_true", help="Disable multiprocessing and use sequential processing")
    
    
    args = parser.parse_args()
    
    # Create directories
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(args.lip_video_dir, exist_ok=True)
    
    # Prepare chunks
    chunks = prepare_chunks(
        args.csv_path,
        args.output_dir,
        args.chunk_size,
        args.filter_processed
    )
    
    if not chunks:
        print("No videos to process, exiting")
        sys.exit(0)
    
    # Process chunks
    try:
        if args.use_sequential:
            # Use sequential processing
            print("Using sequential processing")
            process_chunks_sequentially(
                chunks,
                args.csv_path,
                args.lip_video_dir,
                checkpoint_dir,
                args.batch_size,
                args.to_grayscale,
                args.max_videos
            )
        else:
            # Use multiprocessing
            print("Using multiprocessing")
            process_chunks_with_multiprocessing(
                chunks,
                args.csv_path,
                args.lip_video_dir,
                checkpoint_dir,
                args.batch_size,
                args.to_grayscale,
                args.max_videos,
                args.num_workers,
                args.max_tasks_per_child,
                not args.disable_multiprocessing
            )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 