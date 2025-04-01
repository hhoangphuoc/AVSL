import json
import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from tqdm import tqdm

from collections import defaultdict

from preprocess.constants import (
    SOURCE_PATH,
    DSFL_PATH,
    AMI_SPEAKERS,
    DATASET_PATH
)
from dataset_process import process_lip_videos
from audio_process import batch_segment_audio
from video_process import batch_segment_video
from utils import av_to_hf_dataset


def collect_segments_from_dsfl_csv(dsfl_laugh_dir, 
                                   audio_segment_dir, 
                                   video_segment_dir):
    """
    Collect segments from disfluency/laughter CSV file and group them by source file.
    
    Args:
        dsfl_laugh_dir: Directory containing disfluency/laughter CSV file
        audio_segment_dir: Directory to save audio segments
        video_segment_dir: Directory to save video segments
        
    Returns:
        Tuple of (total_segments, skipped_speaker_mapping, skipped_source_missing, 
                 skipped_short_duration, audio_segments_by_source, video_segments_by_source)
    """
    try:
        csv_file_path = os.path.join(dsfl_laugh_dir, 'disfluency_laughter_markers.csv')
        df_markers = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df_markers)} records from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return 0, 0, 0, 0, {}, {}
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return 0, 0, 0, 0, {}, {}

    # Data structures to track processing
    total_segments = 0
    skipped_speaker_mapping = 0
    skipped_source_missing = 0
    skipped_short_duration = 0
    
    # Data structures to group segments by their source
    audio_segments_by_source = defaultdict(list)  # {audio_file: [(start, end, output_file, metadata), ...]}
    video_segments_by_source = defaultdict(list)  # {video_file: [(start, end, output_file, metadata), ...]}
    
    # FIRST PASS: Group segments by source file
    print("Step 1: Collecting disfluency/laughter segments from CSV file...")
    
    for index, row in tqdm(df_markers.iterrows(), total=len(df_markers), desc="Processing CSV"):
        total_segments += 1
        meeting_id = row['meeting_id']
        ami_speaker_id = row['speaker_id'] # AMI uses A, B, C, D, E
        start_time = row['start_time']
        end_time = row['end_time']
        disfluency_type = row['disfluency_type']
        is_laugh = bool(row['is_laugh']) # Convert 0/1 to False/True
        word = row['word'] # Can be useful for context or filtering
        
        # Handle potential NaN in disfluency_type
        if pd.isna(disfluency_type):
            disfluency_type = None # Standardize to None
            
        # Skip if speaker not in mapping
        if ami_speaker_id not in AMI_SPEAKERS:
            skipped_speaker_mapping += 1
            continue
            
        # GET ORIGINAL SOURCE PATHS ------------------------------------------------------------------------------------------------    
        audio_source = AMI_SPEAKERS[ami_speaker_id]['audio'] # Headset-0, Headset-1, etc.
        video_source = AMI_SPEAKERS[ami_speaker_id]['video'] # Closeup1, Closeup2, etc.
        
        audio_file = os.path.join(SOURCE_PATH, meeting_id, 'audio', f"{meeting_id}.{audio_source}.wav")
        video_file = os.path.join(SOURCE_PATH, meeting_id, 'video', f"{meeting_id}.{video_source}.avi")
            
        # Check if original files exist
        process_audio = os.path.exists(audio_file)
        process_video = os.path.exists(video_file)
        
        # Skip if neither source exists
        if not process_audio and not process_video:
            skipped_source_missing += 1
            continue
        
        # Skip very short segments (less than 0.05 seconds) - adjust threshold if needed
        duration = end_time - start_time
        if duration < 0.05 or start_time >= end_time:
            skipped_short_duration += 1
            continue

        # Get start and end time segment
        start_time_str = f"{start_time:.2f}"
        end_time_str = f"{end_time:.2f}"
        
        # Determine event type for filename
        if is_laugh:
            event_label = "laugh"
        elif disfluency_type:
            # Ensure disfluency_type is treated as a string for the label
            event_label = str(disfluency_type)
        else:
            # If neither laugh nor disfluency, use the word if available, else 'speech'
            event_label = str(word) if pd.notna(word) else "speech"
            
        # Sanitize event_label for filename
        event_label_safe = re.sub(r'[\\/*?:"<>|]', '_', event_label) 
        segment_name = f"{meeting_id}-{ami_speaker_id}-{start_time_str}-{end_time_str}-{event_label_safe}"
        
        # Store metadata to associate with each segment
        metadata = {
            'segment_id': segment_name,
            'meeting_id': meeting_id,
            'speaker_id': ami_speaker_id,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'disfluency_type': disfluency_type,
            'is_laugh': is_laugh,
            'word': word
        }
        
        # Add to appropriate source lists
        if process_audio:
            audio_output_file = os.path.join(audio_segment_dir, f"{segment_name}-audio.wav")
            audio_segments_by_source[audio_file].append((start_time, end_time, audio_output_file, metadata))
            
        if process_video:
            video_output_file = os.path.join(video_segment_dir, "dsfl_original", f"{segment_name}-video.mp4") #this is the original video segment so it saved in: /dsfl_laugh/video_segments/dsfl_original
            video_segments_by_source[video_file].append((start_time, end_time, video_output_file, metadata))
    
    # Save collection stats for reporting
    return total_segments, skipped_speaker_mapping, skipped_source_missing, skipped_short_duration, audio_segments_by_source, video_segments_by_source

def process_dsfl_audio_segments(audio_segments_by_source):
    """
    Process audio segments in batches for disfluency/laughter data.
    This is a wrapper around batch_segment_audio that handles the specific format of disfluency segments.
    
    Args:
        audio_segments_by_source: Dictionary mapping audio source files to segments to extract
        
    Returns:
        Tuple of (audio_segment_results, successful_audio_segments)
    """
    print("\nStep 2: Processing disfluency audio files...")
    audio_segment_results = {}  # {segment_id: (success, output_file)}
    successful_audio_segments = 0
    
    for audio_file, segments in tqdm(audio_segments_by_source.items(), desc="Processing audio files"):
        # Extract the relevant parts for batch processing
        batch_segments = [(start, end, output) for start, end, output, _ in segments]
        batch_results = batch_segment_audio(audio_file, batch_segments)
        
        # Record results
        for i, (success, output_file) in enumerate(batch_results):
            metadata = segments[i][3]
            segment_id = metadata['segment_id']
            audio_segment_results[segment_id] = (success, output_file)
            if success:
                successful_audio_segments += 1
    
    return audio_segment_results, successful_audio_segments

def process_dsfl_video_segments(video_segments_by_source):
    """
    Process video segments in batches for disfluency/laughter data.
    This is a wrapper around batch_segment_video that handles the specific format of disfluency segments.
    
    Args:
        video_segments_by_source: Dictionary mapping video source files to segments to extract
        
    Returns:
        Tuple of (video_segment_results, successful_video_segments)
    """
    print("\nStep 3: Processing disfluency video files...")
    video_segment_results = {}  # {segment_id: (success, output_file)}
    successful_video_segments = 0
    
    for video_file, segments in tqdm(video_segments_by_source.items(), desc="Processing video files"):
        # Extract the relevant parts for batch processing
        batch_segments = [(start, end, output) for start, end, output, _ in segments]
        batch_results = batch_segment_video(video_file, batch_segments)
        
        # Record results
        for i, (success, output_file) in enumerate(batch_results):
            metadata = segments[i][3]
            segment_id = metadata['segment_id']
            video_segment_results[segment_id] = (success, output_file)
            if success:
                successful_video_segments += 1
    
    return video_segment_results, successful_video_segments

def create_dsfl_dataset_records(
        all_segment_metadata, 
        audio_segment_results, 
        video_segment_results, 
        lip_segment_results
    ):
    """
    Create dataset records for disfluency/laughter data.
    
    Args:
        all_segment_metadata: Dictionary of segment metadata keyed by segment_id
        audio_segment_results: Dictionary of audio processing results
        video_segment_results: Dictionary of video processing results
        lip_segment_results: Dictionary of lip video processing results
        
    Returns:
        List of dataset records
    """
    print("\nFinalizing disfluency/laughter dataset records...")
    dataset_records = []
    processed_count = 0
    
    # Create dataset records
    for segment_id, metadata in all_segment_metadata.items():
        audio_success, audio_file = audio_segment_results.get(segment_id, (False, None))
        video_success, video_file = video_segment_results.get(segment_id, (False, None))
        lip_success, lip_file = lip_segment_results.get(segment_id, (False, None))
        
        if audio_success or video_success:
            processed_count += 1
            record = {
                "id": segment_id,
                "meeting_id": metadata['meeting_id'],
                "speaker_id": metadata['speaker_id'],
                "word": metadata['word'],
                "start_time": metadata['start_time'],
                "end_time": metadata['end_time'],
                "duration": metadata['duration'],
                "disfluency_type": metadata['disfluency_type'],
                "is_laugh": metadata['is_laugh'],
                "has_audio": audio_success,
                "has_video": video_success,
                "has_lip_video": lip_success
            }
            
            if audio_success:
                record["audio"] = audio_file
            if video_success:
                record["video"] = video_file
            if lip_success:
                record["lip_video"] = lip_file
                
            dataset_records.append(record)
    
    return dataset_records, processed_count

def segment_disfluency_laughter(dsfl_laugh_dir, 
                               dataset_path, # Specific path for this dataset
                               to_dataset=True,
                               extract_lip_videos=True,
                               use_gpu=False,
                               use_parallel=True,
                               batch_size=16,
                               batch_process=True,
                               to_grayscale=True
                              ):
    """
    Segments audio and video based on timestamps in the disfluency/laughter CSV file,
    extracts lip videos, and saves information to a HuggingFace Dataset.

    The audio and video segments are saved in the subdirectories of `dsfl_laugh_dir`:
    - `/audio_segments` : Audio segments of disfluency/laughter events
    - `/video_segments`
        - `/dsfl_original` : Original video segments of disfluency/laughter events
        - `/dsfl_lip` : Lip video segments of disfluency/laughter events

    Args:
        dsfl_laugh_dir: Path to the disfluency/laughter directory.
        dataset_path: Path to save the output HuggingFace Dataset, defaults to `DSFL_PATH/dataset`
        to_dataset: Whether to create a HuggingFace dataset.
        extract_lip_videos: Whether to extract lip regions from video segments.
        use_gpu: Whether to use GPU acceleration if available
        use_parallel: Whether to use parallel processing for lip extraction
        batch_size: Batch size for processing frames
        batch_process: Whether to process multiple videos in parallel
        to_grayscale: Whether to extract lip videos in grayscale
    """
    # CREATE OUTPUT DIRECTORIES  ------------------------------------------------------------------------------------------------    
    os.makedirs(dsfl_laugh_dir, exist_ok=True)

    # Video and audio segment directories are inside the `dsfl_laugh_dir`
    video_segment_dir = os.path.join(dsfl_laugh_dir, "video_segments") # /dsfl_laugh/video_segments
    audio_segment_dir = os.path.join(dsfl_laugh_dir, "audio_segments") # /dsfl_laugh/audio_segments
    os.makedirs(audio_segment_dir, exist_ok=True)
    os.makedirs(video_segment_dir, exist_ok=True)

    # Specific subdirectories for original and lip videos for this mode
    original_video_dir = os.path.join(video_segment_dir, "dsfl_original") # /dsfl_laugh/video_segments/dsfl_original
    os.makedirs(original_video_dir, exist_ok=True)
    
    lip_video_dir = None
    if extract_lip_videos:
        lip_video_dir = os.path.join(video_segment_dir, "dsfl_lip") # /dsfl_laugh/video_segments/dsfl_lip
        os.makedirs(lip_video_dir, exist_ok=True)

    # Step 1: Collect segments from CSV file
    total_segments, skipped_speaker_mapping, skipped_source_missing, skipped_short_duration, audio_segments_by_source, video_segments_by_source = collect_segments_from_dsfl_csv(
        dsfl_laugh_dir, 
        audio_segment_dir, 
        video_segment_dir
    )
    
    if total_segments == 0:
        print("No segments found in disfluency/laughter CSV file")
        return
    
    # Step 2: Process audio segments
    audio_segment_results, successful_audio_segments = process_dsfl_audio_segments(audio_segments_by_source)
    # Save audio segment results to JSON
    audio_segment_results_path = os.path.join(audio_segment_dir, "audio_segment_results.json")
    with open(audio_segment_results_path, "w") as f:
        json.dump(audio_segment_results, f)
    
    # Step 3: Process video segments
    video_segment_results, successful_video_segments = process_dsfl_video_segments(video_segments_by_source)
    # Save video segment results to JSON
    video_segment_results_path = os.path.join(video_segment_dir, "video_segment_results.json")
    with open(video_segment_results_path, "w") as f:
        json.dump(video_segment_results, f)
    
    # Step 4: Process lip videos if requested
    lip_segment_results = {}
    successful_lip_segments = 0
    failed_lip_extraction = 0
    
    if extract_lip_videos and lip_video_dir and successful_video_segments > 0:
        lip_segment_results, successful_lip_segments = process_lip_videos(
            video_segment_results, 
            lip_video_dir,
            use_gpu=use_gpu,
            use_parallel=use_parallel,
            batch_size=batch_size,
            batch_process=batch_process,
            to_grayscale=to_grayscale
        )
        failed_lip_extraction = len([v for v in video_segment_results.values() if v[0]]) - successful_lip_segments
    
    # Step 5: Collect metadata from all segments
    all_segment_metadata = {}
    
    # Collect metadata from audio segments
    for audio_file, segments in audio_segments_by_source.items():
        for _, _, _, metadata in segments:
            segment_id = metadata['segment_id']
            all_segment_metadata[segment_id] = metadata
    
    # Add any missing metadata from video segments
    for video_file, segments in video_segments_by_source.items():
        for _, _, _, metadata in segments:
            segment_id = metadata['segment_id']
            if segment_id not in all_segment_metadata:
                all_segment_metadata[segment_id] = metadata
    
    # Save all segment metadata to JSON
    all_segment_metadata_path = os.path.join(dsfl_laugh_dir, "all_segment_metadata.json")
    with open(all_segment_metadata_path, "w") as f:
        json.dump(all_segment_metadata, f)
    

    # -------------------------- Step 6: Create dataset records -----------------------------------------------------------
    dataset_records, processed_count = create_dsfl_dataset_records(
        all_segment_metadata,
        audio_segment_results,
        video_segment_results,
        lip_segment_results
    )
    
    # Step 7: Create dataset if requested
    if to_dataset and dataset_records:
        # Use the specific dataset path
        av_to_hf_dataset(dataset_records, dataset_path=dataset_path)
        
    elif not dataset_records:
        print("No valid segments were processed, dataset creation skipped.")
    else:
        print("Dataset creation skipped as per request.")

    # Print statistics
    print("\nDisfluency/Laughter Processing Summary:")
    print(f"Total CSV rows: {total_segments}")
    print(f"Skipped (speaker mapping missing): {skipped_speaker_mapping}")
    print(f"Skipped (source audio/video missing): {skipped_source_missing}")
    print(f"Skipped (short/invalid duration): {skipped_short_duration}")
    print(f"Segments processed (audio or video created): {processed_count}")
    print(f"Audio segments created: {successful_audio_segments}")
    print(f"Video segments created: {successful_video_segments}")
    
    if extract_lip_videos:
        if successful_video_segments > 0:
            print(f"Lip video extraction successful: {successful_lip_segments} ({successful_lip_segments/successful_video_segments*100:.1f}% of videos)")
            print(f"Lip video extraction failed: {failed_lip_extraction}")
        else:
            print("Lip video extraction: 0 (no successful video segments)")
    
    print("Disfluency/Laughter segmentation completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment audio and video sources based on transcript timestamps')
    parser.add_argument('--dsfl_laugh_dir', type=str, default=DSFL_PATH, help='Directory to save disfluency/laughter segments')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, help='Path to save HuggingFace dataset')
    parser.add_argument('--to_dataset', action='store_true', default=True, help='Create HuggingFace dataset')
    parser.add_argument('--extract_lip_videos', action='store_true', default=True, help='Extract lip videos')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU for lip extraction')
    parser.add_argument('--use_parallel', action='store_true', default=True, help='Use parallel processing for lip extraction')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing frames')
    parser.add_argument('--batch_process', action='store_true', default=True, help='Process multiple videos in parallel')
    parser.add_argument('--to_grayscale', action='store_true', default=True, help='Extract lip videos in grayscale')

    args = parser.parse_args()
    
    segment_disfluency_laughter(
        dsfl_laugh_dir=args.dsfl_laugh_dir,
        dataset_path=args.dataset_path,
        to_dataset=args.to_dataset,
        extract_lip_videos=args.extract_lip_videos,
        use_gpu=args.use_gpu,
        use_parallel=args.use_parallel,
        batch_size=args.batch_size,
        batch_process=args.batch_process,
        to_grayscale=args.to_grayscale
    )
