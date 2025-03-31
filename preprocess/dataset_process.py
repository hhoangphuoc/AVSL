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

from preprocess.constants import (
    DATA_PATH, 
    TRANS_SEG_PATH, 
    SOURCE_PATH, 
    AUDIO_PATH, 
    VIDEO_PATH, 
    DATASET_PATH, 
    DSFL_PATH,
)

from audio_process import segment_audio
from video_process import segment_video, extract_and_save_lip_video


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
    
    with the following format:\n
    
    For audio:
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-audio.wav\n
    
    For video:
    - original video
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-video.mp4\n

    - lip video
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-lip_video.mp4
    
    Args:
        transcript_segments_dir: Directory containing transcript segment files
        audio_segment_dir: Directory to save audio segments
        video_segment_dir: Directory to save video segments
        to_dataset: Whether to create a HuggingFace dataset
        extract_lip_videos: Whether to extract lip regions from video segments
        dataset_path: Path to HuggingFace Dataset (default: `DATA_PATH/dataset`)
    """
    
    # Create output directories if they don't exist
    os.makedirs(audio_segment_dir, exist_ok=True)
    os.makedirs(video_segment_dir, exist_ok=True)

    # Making original video directory if it doesn't exist
    original_video_dir = os.path.join(video_segment_dir, "original_videos")
    os.makedirs(original_video_dir, exist_ok=True)
    
    # Create lip video directory 
    if extract_lip_videos:
        lip_video_dir = os.path.join(video_segment_dir, "lip_videos")
        os.makedirs(lip_video_dir, exist_ok=True)
    
    # Keep track of segments with alignment issues
    alignment_issues = []
    
    # Extract timestamps and text
    time_pattern = re.compile(r'\[(\d+\.\d+)-(\d+\.\d+)\]\s+(.*)') #format: [start_time-end_time] text
    
    # Dataset records for HuggingFace dataset
    dataset_records = []
    
    # For train/test split tracking
    meeting_ids = set()

    # Process each transcript file
    for file in tqdm(os.listdir(transcript_segments_dir), desc="Processing transcript files"):
        if file.endswith('.txt'):
            meeting_id, ami_speaker_id = file.split('.')[0].split('-')
            # Track meeting IDs for later splitting
            meeting_ids.add(meeting_id)
            
            # Skip if speaker not in mapping
            if ami_speaker_id not in ami_speakers:
                print(f"Warning: Speaker {ami_speaker_id} not found in mapping. Skipping file {file}")
                continue

# --------------------------------------- Getting Original Audio and Video --------------------------------------- 
            # Get the corresponding audio and video sources for this speaker
            audio_source = ami_speakers[ami_speaker_id]['audio']
            video_source = ami_speakers[ami_speaker_id]['video']
                
            # Paths to the original audio and video files
            audio_file = os.path.join(source_original_dir, meeting_id, 'audio', f"{meeting_id}.{audio_source}.wav")
            video_file = os.path.join(source_original_dir, meeting_id, 'video', f"{meeting_id}.{video_source}.avi")
                
            # Check if original files exist
            if not os.path.exists(audio_file):
                print(f"Warning: Audio file {audio_file} not found. Skipping audio processing for {file}")
                process_audio = False
            else:
                process_audio = True
                
            if not os.path.exists(video_file):
                print(f"Warning: Video file {video_file} not found. Skipping video processing for {file}")
                process_video = False
            else:
                process_video = True
            
            # Skip if neither audio nor video can be processed
            if not process_audio and not process_video:
                print(f"Skipping file {file} as neither audio nor video source exists")
                continue

# --------------------------------------- Processing Transcript Segments ---------------------------------------     
            # Process each line in the transcript file
            with open(os.path.join(transcript_segments_dir, file), 'r') as f:
                for line in tqdm(f, desc=f"Processing {file}", leave=False):
                    # Parse the line to get start_time, end_time and text
                    match = time_pattern.match(line.strip())
                    if match:
                        start_time = float(match.group(1))
                        end_time = float(match.group(2))
                        text = match.group(3)
                        
                        # Skip very short segments (less than 0.1 seconds)
                        if end_time - start_time < 0.1:
                            print(f"Warning: Skipping very short segment {start_time}-{end_time} (duration: {end_time-start_time:.2f}s)")
                            continue
                        
                        # Format times for filenames (2 decimal places)
                        start_time_str = f"{start_time:.2f}"
                        end_time_str = f"{end_time:.2f}"
                        
                        # Base segment identifier for both audio and video
                        segment_id = f"{meeting_id}-{ami_speaker_id}-{start_time_str}-{end_time_str}"
                        
# --------------------------------------- Processing Audio and Video --------------------------------------- 
                        # Process audio if the file exists
                        audio_success = False
                        audio_output_file = None
                        if process_audio:
                            audio_output_file = os.path.join(
                                audio_segment_dir, 
                                f"{segment_id}-audio.wav"
                            )
                            audio_success, audio_output_file = segment_audio(audio_file, start_time, end_time, audio_output_file)
                            
                        # Process video if the file exists
                        video_success = False
                        video_output_file = None
                        if process_video:
                            video_output_file = os.path.join(
                                original_video_dir, 
                                f"{segment_id}-video.mp4"
                            )
                            video_success, video_output_file = segment_video(video_file, start_time, end_time, video_output_file)

# --------------------------------------- Extracting Lip Video --------------------------------------- 
                        lip_video_success = False
                        lip_video_output_file = None
                        if extract_lip_videos and video_success:
                            lip_video_output_file = os.path.join(
                                lip_video_dir, 
                                f"{segment_id}-lip_video.mp4"
                            )
                            try:
                                print(f"Extracting lip video for {segment_id}")
                                lip_video_success, lip_video_output_file = extract_and_save_lip_video(
                                    video_output_file,  # Use the segmented video as input
                                    lip_video_output_file,
                                    to_grayscale=True  # Use color for better visualization
                                )
                                if not lip_video_success:
                                    print(f"Warning: Failed to extract lip video for {segment_id}")
                            except Exception as e:
                                print(f"Error extracting lip video for {segment_id}: {str(e)}")
                                lip_video_success = False

# --------------------------------------- Checking for Alignment Issues --------------------------------------- 
                        # Check for alignment issues
                        if process_audio and process_video:
                            if audio_success != video_success:
                                msg = f"Alignment issue for segment {segment_id}: audio_success={audio_success}, video_success={video_success}"
                                print(msg)
                                alignment_issues.append(msg)
                                
# --------------------------------------- Adding to Dataset Records --------------------------------------- 

                        if (process_audio and audio_success) or (process_video and video_success):
                            record = {
                                "id": segment_id,
                                "meeting_id": meeting_id,
                                "speaker_id": ami_speaker_id,
                                "start_time": start_time,
                                "end_time": end_time,
                                "duration": end_time - start_time,
                                "transcript": text,
                                "has_audio": audio_success,
                                "has_video": video_success,
                                "has_lip_video": lip_video_success
                            }
                            
                            if audio_success:
                                record["audio"] = audio_output_file
                                
                            if video_success:
                                record["video"] = video_output_file
                                
                            if lip_video_success:
                                record["lip_video"] = lip_video_output_file

                            print("Record: ", record)
                                
                            dataset_records.append(record)
    
    if alignment_issues:
        print(f"\nFound {len(alignment_issues)} segments with potential alignment issues")
        with open(os.path.join(DATA_PATH, "alignment_issues.log"), "w") as f:
            for issue in alignment_issues:
                f.write(f"{issue}\n")
    
    # Create dataset if requested
    if to_dataset and dataset_records:
        audio_video_to_dataset(dataset_records, meeting_ids, DATASET_PATH)
    
# --------------------------------------- Reporting Statistics --------------------------------------- 
    total_segments = len(dataset_records)
    audio_segments = sum(1 for record in dataset_records if record["has_audio"])
    video_segments = sum(1 for record in dataset_records if record["has_video"])
    lip_video_segments = sum(1 for record in dataset_records if record.get("has_lip_video", False))
    
    print("\nSegmentation Statistics:")
    print(f"Total segments processed: {total_segments}")
    print(f"Audio segments: {audio_segments} ({audio_segments/total_segments*100:.1f}%)")
    print(f"Video segments: {video_segments} ({video_segments/total_segments*100:.1f}%)")
    print(f"Lip video segments: {lip_video_segments} ({lip_video_segments/total_segments*100:.1f}%)")

# -------------------------------------------------------------------------------------------
    print("Source segmentation completed.")



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

    # Specific subdirectories for original and lip videos
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

    # PROCESS EACH ROW IN THE CSV ------------------------------------------------------------------------------------------------    
    dataset_records = []
    processed_count = 0
    
    # Process each row in the CSV
    for index, row in tqdm(df_markers.iterrows(), total=len(df_markers), desc="Processing Disfluency/Laughter CSV"):
        meeting_id = row['meeting_id']
        ami_speaker_id = row['speaker_id'] # AMI uses A, B, C, D, E
        start_time = row['start_time']
        end_time = row['end_time']
        disfluency_type = row['disfluency_type']
        is_laugh = bool(row['is_laugh']) # Convert 0/1 to False/True
        word = row['word'] # Can be useful for context or filtering

        # Handle potential NaN in disfluency_type
        if isinstance(disfluency_type, float) and math.isnan(disfluency_type):
            disfluency_type = None # Represent NaN as None

        # Skip if speaker not in mapping
        if ami_speaker_id not in ami_speakers:
            # print(f"Warning: Speaker {ami_speaker_id} in meeting {meeting_id} not found in mapping. Skipping row {index}")
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
            print(f"Warning: Neither audio nor video source found for {meeting_id}-{ami_speaker_id}. Skipping row")
            continue
        
        # Skip very short segments (less than 0.05 seconds) - adjust threshold if needed
        if end_time - start_time < 0.05:
            print(f"Warning: Skipping very short segment {start_time}-{end_time} (duration: {end_time-start_time:.2f}s) for row {index}")
            continue
        # ------------------------------------------------------------------------------------------------    

        # PROCESSING AUDIO AND VIDEO ------------------------------------------------------------------------------------------------    
        # Get start and end time segment
        start_time_str = f"{start_time:.2f}"
        end_time_str = f"{end_time:.2f}"
        
        # Determine event type for filename
        event_label = "laugh" if is_laugh else (disfluency_type if disfluency_type else "speech") # Use disfluency type or 'speech' if neither
        event_label_safe = re.sub(r'[\\/*?:"<>|]', '_', event_label) 
        segment_name = f"{meeting_id}-{ami_speaker_id}-{start_time_str}-{end_time_str}-{event_label_safe}"
        
        # --- Process Audio ---
        audio_success = False
        audio_output_file = None
        if process_audio:
            audio_output_file = os.path.join(audio_segment_dir, f"{segment_name}-audio.wav")
            audio_success, audio_output_file = segment_audio(audio_file, start_time, end_time, audio_output_file)
            
        # --- Process Video ---
        video_success = False
        video_output_file = None
        if process_video:
            video_output_file = os.path.join(original_video_dir, f"{segment_name}-video.mp4")
            video_success, video_output_file = segment_video(video_file, start_time, end_time, video_output_file)

        # --- Extract Lip Video ---
        lip_video_success = False
        lip_video_output_file = None
        if extract_lip_videos and video_success and lip_video_dir:
            lip_video_output_file = os.path.join(lip_video_dir, f"{segment_name}-lip_video.mp4")
            try:
                lip_video_success, lip_video_output_file = extract_and_save_lip_video(
                    video_output_file, 
                    lip_video_output_file,
                    to_grayscale=True 
                )
                if not lip_video_success:
                    print(f"Warning: Failed to extract lip video for {segment_name}")
            except Exception as e:
                print(f"Error extracting lip video for {segment_name}: {str(e)}")
                lip_video_success = False
        # ------------------------------------------------------------------------------------------------    

        # ADD RECORD TO DATASET ------------------------------------------------------------------------------------------------    
        if audio_success or video_success:
            processed_count += 1
            record = {
                "id": segment_name,
                "meeting_id": meeting_id,
                "speaker_id": ami_speaker_id,
                "word": word,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "disfluency_type": disfluency_type,
                "is_laugh": is_laugh,
                "has_audio": audio_success,
                "has_video": video_success,
                "has_lip_video": lip_video_success
            }
            
            if audio_success:
                record["audio"] = audio_output_file
            if video_success:
                record["video"] = video_output_file
            if lip_video_success:
                record["lip_video"] = lip_video_output_file
                
            dataset_records.append(record)
        # ------------------------------------------------------------------------------------------------    
    
    print(f"\nProcessed {processed_count} segments from the CSV.")
    
    # Create dataset if requested
    if to_dataset and dataset_records:
        # Use the specific dataset path and a descriptive name
        audio_video_to_dataset(dataset_records, 
                               dataset_path=dataset_path) 
    else:
        print("Dataset creation skipped as per request or no records were processed.")

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












