"""
This script is used to generate the source segments (audio and video)
from the original audio and video files, based on the transcript_segments timestamps
corresponding to each [meeting_id]-[speaker_id].

The source segments are saved in the following format:
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].wav
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].mp4
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-lip_video.mp4 (lip regions)

- audio_segment_dir: the directory to save the audio segments
- video_segment_dir: the directory to save the video segments
- lip_video_dir: the directory to save the lip frame videos
- transcript_segments: the transcript segments dataframe

"""

import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, Audio, Video
import math # Added for isnan
from collections import defaultdict

from preprocess.constants import (
    DATA_PATH, 
    TRANS_SEG_PATH, 
    SOURCE_PATH, 
    AUDIO_PATH, 
    VIDEO_PATH, 
    DATASET_PATH, 
    DSFL_PATH,
)

from audio_process import segment_audio, batch_segment_audio
from video_process import segment_video, batch_segment_video, extract_and_save_lip_video


transcript_segments_dir = TRANS_SEG_PATH
source_original_dir = SOURCE_PATH
audio_segment_dir = AUDIO_PATH # data/audio_segments
video_segment_dir = VIDEO_PATH # data/video_segments
original_video_dir = os.path.join(VIDEO_PATH, "original_videos") # data/video_segments/original_videos
lip_video_dir = os.path.join(VIDEO_PATH, "lip_videos")  # data/video_segments/lip_videos


ami_speakers ={
    'A': {
        'audio': 'Headset-0',
        'video': 'Closeup1'
    },
    'B': {
        'audio': 'Headset-1',
        'video': 'Closeup2'
    },
    'C': {
        'audio': 'Headset-2',
        'video': 'Closeup3'
    },
    'D': {
        'audio': 'Headset-3',
        'video': 'Closeup4'
    },
    'E': {
        'audio': 'Headset-4',
        'video': 'Closeup5'
    },
}


def segment_sources(transcript_segments_dir,
                    audio_segment_dir,
                    video_segment_dir,
                    dataset_path=DATASET_PATH, # Added default dataset path
                    to_dataset=False,
                    extract_lip_videos=True,
                    ):
    """
    This function is used to segment the audio and video sources based on the `transcript_segments` timestamps and
    save the segmented audio and video resources.

    If `to_dataset` is True, the sources will be saved in a HuggingFace Dataset.
    If `extract_lip_videos` is True, lip regions will be extracted from successful video segments.

    The segmented audio and video sources will be saved in:
    - `audio_segment_dir` : Audio segments. Default: `AUDIO_PATH`   
    - `video_segment_dir` : Video segments. Default: `VIDEO_PATH`
        - `/original_videos` : Original video segments. Default: `VIDEO_PATH/original_videos`
        - `/lip_videos` : Lip video segments. Default: `VIDEO_PATH/lip_videos`
    
    with the following format:

    For audio:
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-audio.wav

    For video:
    - original video
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-video.mp4

    - lip video
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-lip_video.mp4

    Args:
        transcript_segments_dir: Directory containing transcript segment files
        audio_segment_dir: Directory to save audio segments
        video_segment_dir: Directory to save video segments
        dataset_path: Base path to save HuggingFace Dataset (default: `DATASET_PATH`)
        to_dataset: Whether to create a HuggingFace dataset
        extract_lip_videos: Whether to extract lip regions from video segments
    """

    # Create output directories if they don't exist
    os.makedirs(audio_segment_dir, exist_ok=True)
    os.makedirs(video_segment_dir, exist_ok=True)

    # Making original video directory if it doesn't exist
    original_video_dir = os.path.join(video_segment_dir, "original_videos")
    os.makedirs(original_video_dir, exist_ok=True)

    # Create lip video directory
    lip_video_dir = None
    if extract_lip_videos:
        lip_video_dir = os.path.join(video_segment_dir, "lip_videos")
        os.makedirs(lip_video_dir, exist_ok=True)

    # Keep track of segments with alignment issues
    alignment_issues = []

    # Extract timestamps and text
    time_pattern = re.compile(r'\[(\d+\.\d+)-(\d+\.\d+)\]\s+(.*)') #format: [start_time-end_time] text

    # Dataset records for HuggingFace dataset
    dataset_records = []

    # For tracking processed segments
    total_segments = 0
    successful_audio_segments = 0
    successful_video_segments = 0
    successful_lip_segments = 0

    # Data structures to group segments by their source
    audio_segments_by_source = defaultdict(list)  # {audio_file: [(start, end, output_file, text, segment_id), ...]}
    video_segments_by_source = defaultdict(list)  # {video_file: [(start, end, output_file, segment_id), ...]}
    
    # First pass: Collect all segments grouped by source file
    print("Pass 1: Collecting all segments from transcript files...")
    
    for file in tqdm(os.listdir(transcript_segments_dir), desc="Processing transcript files"):
        if file.endswith('.txt'):
            meeting_id, ami_speaker_id = file.split('.')[0].split('-')

            # Skip if speaker not in mapping
            if ami_speaker_id not in ami_speakers:
                print(f"Warning: Speaker {ami_speaker_id} not found in mapping. Skipping file {file}")
                continue
                
            # Get the corresponding audio and video sources for this speaker
            audio_source = ami_speakers[ami_speaker_id]['audio']
            video_source = ami_speakers[ami_speaker_id]['video']
                
            # Paths to the original audio and video files
            audio_file = os.path.join(source_original_dir, meeting_id, 'audio', f"{meeting_id}.{audio_source}.wav")
            video_file = os.path.join(source_original_dir, meeting_id, 'video', f"{meeting_id}.{video_source}.avi")
                
            # Check if original files exist
            process_audio = os.path.exists(audio_file)
            process_video = os.path.exists(video_file)
            
            if not process_audio and not process_video:
                print(f"Skipping file {file} as neither audio nor video source exists")
                continue

            # Process each line in the transcript file
            with open(os.path.join(transcript_segments_dir, file), 'r') as f:
                for line in f:
                    # Parse the line to get start_time, end_time and text
                    match = time_pattern.match(line.strip())
                    if match:
                        start_time = float(match.group(1))
                        end_time = float(match.group(2))
                        text = match.group(3)
                        
                        # Skip very short segments (less than 0.1 seconds)
                        if end_time - start_time < 0.1:
                            continue
                            
                        # Format times for filenames (2 decimal places)
                        start_time_str = f"{start_time:.2f}"
                        end_time_str = f"{end_time:.2f}"
                        
                        # Base segment identifier
                        segment_id = f"{meeting_id}-{ami_speaker_id}-{start_time_str}-{end_time_str}"
                        
                        total_segments += 1
                        
                        # Add to appropriate source lists
                        if process_audio:
                            audio_output_file = os.path.join(audio_segment_dir, f"{segment_id}-audio.wav")
                            audio_segments_by_source[audio_file].append((start_time, end_time, audio_output_file, text, segment_id))
                            
                        if process_video:
                            video_output_file = os.path.join(original_video_dir, f"{segment_id}-video.mp4")
                            video_segments_by_source[video_file].append((start_time, end_time, video_output_file, segment_id))
    
    # Second pass: Process each source file once and extract all segments
    print("\nPass 2: Processing audio files...")
    audio_segment_results = {}  # {segment_id: (success, output_file)}
    
    for audio_file, segments in tqdm(audio_segments_by_source.items(), desc="Processing audio files"):
        # Extract the relevant parts for batch processing
        batch_segments = [(start, end, output) for start, end, output, _, _ in segments]
        results = batch_segment_audio(audio_file, batch_segments)
        
        # Record results for each segment
        for i, (success, output_file) in enumerate(results):
            _, _, _, text, segment_id = segments[i]
            audio_segment_results[segment_id] = (success, output_file, text)
            if success:
                successful_audio_segments += 1
    
    print("\nPass 3: Processing video files...")
    video_segment_results = {}  # {segment_id: (success, output_file)}
    
    for video_file, segments in tqdm(video_segments_by_source.items(), desc="Processing video files"):
        # Extract the relevant parts for batch processing
        batch_segments = [(start, end, output) for start, end, output, _ in segments]
        results = batch_segment_video(video_file, batch_segments)
        
        # Record results for each segment
        for i, (success, output_file) in enumerate(results):
            _, _, _, segment_id = segments[i]
            video_segment_results[segment_id] = (success, output_file)
            if success:
                successful_video_segments += 1
    
    # Third pass: Extract lip videos if requested
    lip_segment_results = {}  # {segment_id: (success, output_file)}
    
    if extract_lip_videos and lip_video_dir:
        print("\nPass 4: Processing lip videos...")
        
        # Collect all segments that have successful video
        all_lip_segments = []
        for segment_id, (success, video_file) in video_segment_results.items():
            if success:
                lip_output_file = os.path.join(lip_video_dir, f"{segment_id}-lip_video.mp4")
                all_lip_segments.append((video_file, lip_output_file, segment_id))
        
        # Process lip videos
        for video_file, lip_output, segment_id in tqdm(all_lip_segments, desc="Extracting lip videos"):
            try:
                lip_success, lip_output_file = extract_and_save_lip_video(
                    video_file,
                    lip_output,
                    to_grayscale=True
                )
                lip_segment_results[segment_id] = (lip_success, lip_output_file)
                if lip_success:
                    successful_lip_segments += 1
            except Exception as e:
                print(f"Error extracting lip video for {segment_id}: {str(e)}")
                lip_segment_results[segment_id] = (False, None)
    
    # Final pass: Collect all records for the dataset
    print("\nFinalizing dataset...")
    
    for segment_id in set(list(audio_segment_results.keys()) + list(video_segment_results.keys())):
        audio_success, audio_file, text = audio_segment_results.get(segment_id, (False, None, None))
        video_success, video_file = video_segment_results.get(segment_id, (False, None))
        lip_success, lip_file = lip_segment_results.get(segment_id, (False, None))
        
        if audio_success or video_success:
            # Parse segment_id to extract metadata
            parts = segment_id.split('-')
            meeting_id = parts[0]
            speaker_id = parts[1]
            start_time = float(parts[2])
            end_time = float(parts[3])
            
            record = {
                "id": segment_id,
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "transcript": text,
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
            
            # Check for alignment issues
            if audio_success != video_success:
                msg = f"Alignment issue for segment {segment_id}: audio_success={audio_success}, video_success={video_success}"
                alignment_issues.append(msg)
    
    if alignment_issues:
        print(f"\nFound {len(alignment_issues)} segments with potential alignment issues")
        alignment_log_path = os.path.join(DATA_PATH, "transcript_alignment_issues.log")
        print(f"Saving alignment issues to {alignment_log_path}")
        with open(alignment_log_path, "w") as f:
            for issue in alignment_issues:
                f.write(f"{issue}\n")
    
    # Create dataset if requested
    if to_dataset and dataset_records:
        # Use a specific dataset name for transcript-based segments
        audio_video_to_dataset(dataset_records, dataset_path=dataset_path)
    
    # Report statistics
    print("\nTranscript Segmentation Statistics:")
    print(f"Total segments processed: {total_segments}")
    print(f"Audio segments created: {successful_audio_segments} ({successful_audio_segments/total_segments*100:.1f}%)")
    print(f"Video segments created: {successful_video_segments} ({successful_video_segments/total_segments*100:.1f}%)")
    
    if extract_lip_videos:
        if successful_video_segments > 0:
            print(f"Lip video segments created: {successful_lip_segments} ({successful_lip_segments/successful_video_segments*100:.1f}% of successful videos)")
        else:
            print("Lip video segments created: 0 (0% of successful videos)")

    print("Transcript-based source segmentation completed.")

def segment_disfluency_laughter(dsfl_laugh_dir, 
                               dataset_path, # Specific path for this dataset
                               to_dataset=True,
                               extract_lip_videos=True
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
    """
    # CREATE OUTPUT DIRECTORIES  ------------------------------------------------------------------------------------------------    
    os.makedirs(dsfl_laugh_dir, exist_ok=True)

    # Video and audio segment directories are inside the `dsfl_laugh_dir`
    video_segment_dir = os.path.join(dsfl_laugh_dir, "video_segments")
    audio_segment_dir = os.path.join(dsfl_laugh_dir, "audio_segments")
    os.makedirs(audio_segment_dir, exist_ok=True)
    os.makedirs(video_segment_dir, exist_ok=True)

    # Specific subdirectories for original and lip videos for this mode
    original_video_dir = os.path.join(video_segment_dir, "dsfl_original")
    os.makedirs(original_video_dir, exist_ok=True)
    
    lip_video_dir = None
    if extract_lip_videos:
        lip_video_dir = os.path.join(video_segment_dir, "dsfl_lip")
        os.makedirs(lip_video_dir, exist_ok=True)
    # ------------------------------------------------------------------------------------------------    

    # READ THE CSV FILE ------------------------------------------------------------------------------------------------    
    try:
        csv_file_path = os.path.join(dsfl_laugh_dir, 'disfluency_laughter_markers.csv')
        df_markers = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df_markers)} records from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_file_path}: {e}")
        return

    # Data structures to track processing
    total_segments = 0
    skipped_speaker_mapping = 0
    skipped_source_missing = 0
    skipped_short_duration = 0
    
    # Data structures to group segments by their source
    audio_segments_by_source = defaultdict(list)  # {audio_file: [(start, end, output_file, metadata), ...]}
    video_segments_by_source = defaultdict(list)  # {video_file: [(start, end, output_file, metadata), ...]}
    
    # FIRST PASS: Group segments by source file ------------------------------------------------------------------------------------------------
    print("Pass 1: Collecting disfluency/laughter segments from CSV file...")
    
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
        if ami_speaker_id not in ami_speakers:
            skipped_speaker_mapping += 1
            continue
            
        # GET ORIGINAL SOURCE PATHS ------------------------------------------------------------------------------------------------    
        audio_source = ami_speakers[ami_speaker_id]['audio'] # Headset-0, Headset-1, etc.
        video_source = ami_speakers[ami_speaker_id]['video'] # Closeup1, Closeup2, etc.
        
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
        # ------------------------------------------------------------------------------------------------    

        # PROCESSING AUDIO AND VIDEO ------------------------------------------------------------------------------------------------    
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
            video_output_file = os.path.join(original_video_dir, f"{segment_name}-video.mp4")
            video_segments_by_source[video_file].append((start_time, end_time, video_output_file, metadata))
    
    # SECOND PASS: Process audio files in batches ------------------------------------------------------------------------------------------------
    print("\nPass 2: Processing audio files...")
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
    
    # THIRD PASS: Process video files in batches ------------------------------------------------------------------------------------------------
    print("\nPass 3: Processing video files...")
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
    
    # FOURTH PASS: Extract lip videos if requested ------------------------------------------------------------------------------------------------
    lip_segment_results = {}  # {segment_id: (success, output_file)}
    successful_lip_segments = 0
    failed_lip_extraction = 0
    
    if extract_lip_videos and lip_video_dir:
        print("\nPass 4: Processing lip videos...")
        
        # Collect all segments that have successful video
        all_lip_segments = []
        for segment_id, (success, video_file) in video_segment_results.items():
            if success:
                lip_output_file = os.path.join(lip_video_dir, f"{segment_id}-lip_video.mp4")
                all_lip_segments.append((video_file, lip_output_file, segment_id))
        
        # Process lip videos (currently one at a time due to face detection complexity)
        for video_file, lip_output, segment_id in tqdm(all_lip_segments, desc="Extracting lip videos"):
            try:
                lip_success, lip_output_file = extract_and_save_lip_video(
                    video_file,
                    lip_output,
                    to_grayscale=True
                )
                lip_segment_results[segment_id] = (lip_success, lip_output_file)
                if lip_success:
                    successful_lip_segments += 1
                else:
                    failed_lip_extraction += 1
            except Exception as e:
                failed_lip_extraction += 1
                lip_segment_results[segment_id] = (False, None)
    
    # FINAL PASS: Collect dataset records ------------------------------------------------------------------------------------------------
    print("\nFinalizing dataset records...")
    dataset_records = []
    processed_count = 0
    
    # Combine all segment metadata from audio and video sources
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
    
    # Create dataset if requested
    if to_dataset and dataset_records:
        # Use the specific dataset path
        audio_video_to_dataset(dataset_records, dataset_path=dataset_path) 
    elif not dataset_records:
        print("No valid segments were processed, dataset creation skipped.")
    else:
        print("Dataset creation skipped as per request.")

    print("Disfluency/Laughter segmentation completed.")

def audio_video_to_dataset(recordings, dataset_path=None):
    """
    Create a HuggingFace dataset from the processed segments (audio, video, and lip videos), 
    along with the transcript text.
    
    Args:
        recordings: List of dictionaries containing segment information
        dataset_path: Path to HuggingFace Dataset. If None, defaults to `DATA_PATH/dataset`
    """
    print(f"Creating HuggingFace dataset with {len(recordings)} records")
    
    os.makedirs(dataset_path, exist_ok=True)
    
    # GENERATE THE DATASET FROM THE RECORDINGS
    df = pd.DataFrame(recordings)

    # save the dataframe to a csv file
    csv_path = os.path.join(dataset_path, 'ami-segmented-recordings.csv')
    print(f"Saving dataframe to csv file: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Create HuggingFace Dataset containing all recordings
    dataset = Dataset.from_pandas(df)
    
    # Add audio and video features
    if 'audio' in dataset.features:
        dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    
    if 'video' in dataset.features:
        dataset = dataset.cast_column('video', Video())
        
    if 'lip_video' in dataset.features:
        dataset = dataset.cast_column('lip_video', Video())
    
    # Save the dataset
    print(f"Saving dataset to {dataset_path}")
    dataset.save_to_disk(dataset_path)
    print(f"HuggingFace dataset saved: {dataset}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment sources based on transcript segments or disfluency/laughter markers.')
    
    # ARGUMENTS FOR SEGMENTATION ---------------------------------------------------------------------
    parser.add_argument('--mode', choices=['transcript', 'dsfl_laugh'], default='transcript', 
                        help='Mode of operation: segment based on transcript segments or disfluency/laughter CSV.')
    parser.add_argument('--to_dataset', default=True, 
                        help='Create HuggingFace dataset (True/False)')
    parser.add_argument('--extract_lip_videos', default=True, 
                        help='Extract lip videos from video segments (True/False)') 
    #-----------------------------------------------------------------------------------------------------
    
    args = parser.parse_args()
    
    if args.mode == 'transcript':
        print("Running in mode: TRANSCRIPT SEGMENTATION...")
        segment_sources(transcript_segments_dir=TRANS_SEG_PATH, 
                        audio_segment_dir=AUDIO_PATH, 
                        video_segment_dir=VIDEO_PATH, 
                    to_dataset=args.to_dataset,
                    extract_lip_videos=args.extract_lip_videos,
                       )

    elif args.mode == 'dsfl_laugh':
        print("Running in mode: DISFLUENCY/LAUGHTER SEGMENTATION...")
        dsfl_dataset_path = os.path.join(DSFL_PATH, 'dataset')

        print(f"DSFL_PATH: {DSFL_PATH}")
        print(f"DSFL_DATASET saved at: {dsfl_dataset_path}")

        segment_disfluency_laughter(
            dsfl_laugh_dir=DSFL_PATH,
            dataset_path=dsfl_dataset_path, # Use the specific dataset path (DSFL_PATH/dataset)
            to_dataset=args.to_dataset,
            extract_lip_videos=args.extract_lip_videos
                   )












