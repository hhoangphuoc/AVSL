"""
This script is used to generate the source segments (audio and video)
from the original audio and video files, based on the transcript_segments timestamps
corresponding to each [meeting_id]-[speaker_id].

The source segments are saved in the following format:
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].wav
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].mp4

- audio_segment_dir: the directory to save the audio segments
- video_segment_dir: the directory to save the video segments
- transcript_segments: the transcript segments dataframe

"""

import os
import sys
import re
# Add the preprocess directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import Dataset, Audio, Video, Features, Value, DatasetDict

from constants import DATA_PATH, TRANS_SEG_PATH, SOURCE_PATH, AUDIO_PATH, VIDEO_PATH

from audio_process import segment_audio
from video_process import segment_video


transcript_segments_dir = TRANS_SEG_PATH
source_original_dir = SOURCE_PATH
audio_segment_dir = AUDIO_PATH
video_segment_dir = VIDEO_PATH


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
                    to_dataset=False
                    ):
    """
    This function is used to segment the audio and video sources based on the `transcript_segments` timestamps and 
    save the segmented audio and video resources. 
    
    If `to_dataset` is True, the sources will be saved in a HuggingFace Dataset.

    The segmented audio and video sources will be saved in `audio_segment_dir` and `video_segment_dir` respectively, with the following format:\n
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].wav (e.g. EN2001a-A-0.00-0.10-audio.wav)\n
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].mp4 (e.g. EN2001a-A-0.00-0.10-video.mp4)
    
    Args:
        transcript_segments_dir: Directory containing transcript segment files
        audio_segment_dir: Directory to save audio segments
        video_segment_dir: Directory to save video segments
        to_dataset: Whether to create a HuggingFace dataset
        dataset_path: Path to HuggingFace Dataset (default: `DATA_PATH/dataset`)
    """
    
    # Create output directories if they don't exist
    os.makedirs(audio_segment_dir, exist_ok=True)
    os.makedirs(video_segment_dir, exist_ok=True)
    
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
                
            # Get the corresponding audio and video sources for this speaker
            audio_source = ami_speakers[ami_speaker_id]['audio']
            video_source = ami_speakers[ami_speaker_id]['video']
                
            # Construct paths to the original audio and video files
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
                                video_segment_dir, 
                                f"{segment_id}-video.mp4"
                            )
                            video_success, video_output_file = segment_video(video_file, start_time, end_time, video_output_file)
                        
                        # Check for alignment issues
                        if process_audio and process_video:
                            if audio_success != video_success:
                                msg = f"Alignment issue for segment {segment_id}: audio_success={audio_success}, video_success={video_success}"
                                print(msg)
                                alignment_issues.append(msg)
                                
                        # Add to dataset records if at least one of audio or video was processed successfully
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
                                "has_video": video_success
                            }
                            
                            if audio_success:
                                record["audio"] = audio_output_file
                                
                            if video_success:
                                record["video"] = video_output_file

                            print("Record: ", record)
                                
                            dataset_records.append(record)
    
    if alignment_issues:
        print(f"\nFound {len(alignment_issues)} segments with potential alignment issues")
        with open(os.path.join(DATA_PATH, "alignment_issues.log"), "w") as f:
            for issue in alignment_issues:
                f.write(f"{issue}\n")
    
    # Create dataset if requested
    if to_dataset and dataset_records:
        dataset_path = os.path.join(DATA_PATH, "dataset")
        audio_video_to_dataset(dataset_records, meeting_ids, dataset_path)
    
    print("Source segmentation completed.")

def audio_video_to_dataset(recordings, meeting_ids, dataset_path=None):
    """
    Create a HuggingFace dataset from the processed segments (audio and video), 
    along with the transcript text.
    
    Args:
        recordings: List of dictionaries containing segment information
        meeting_ids: List of meeting IDs used for train/val/test splitting (70%, 15%, 15%)
        dataset_path: Path to HuggingFace Dataset. If None, defaults to `DATA_PATH/dataset`
    """
    print(f"Creating HuggingFace dataset with {len(recordings)} records")
    

    
    os.makedirs(dataset_path, exist_ok=True)
    
    # GENERATE THE DATASET FROM THE RECORDINGS
    df = pd.DataFrame(recordings)
    
    # Split the dataset
    meeting_ids = list(sorted(meeting_ids))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(meeting_ids)
    
    # Split meetings into train (70%), validation (15%), test (15%)
    n = len(meeting_ids)
    train_meetings = meeting_ids[:int(0.7*n)]
    val_meetings = meeting_ids[int(0.7*n):int(0.85*n)]
    test_meetings = meeting_ids[int(0.85*n):]
    
    print(f"Train meetings: {len(train_meetings)} recordings, Val meetings: {len(val_meetings)} recordings, Test meetings: {len(test_meetings)} recordings")
    
    # Create DatasetDict with splits
    datasets = DatasetDict()
    
    # Create train dataset
    train_df = df[df['meeting_id'].isin(train_meetings)]
    datasets['train'] = Dataset.from_pandas(train_df)
    
    # Create validation dataset
    val_df = df[df['meeting_id'].isin(val_meetings)]
    datasets['validation'] = Dataset.from_pandas(val_df)
    
    # Create test dataset
    test_df = df[df['meeting_id'].isin(test_meetings)]
    datasets['test'] = Dataset.from_pandas(test_df)
    
    # Add audio and video features
    for split in datasets.keys():
        # Add audio feature
        if 'audio_path' in datasets[split].features:
            datasets[split] = datasets[split].cast_column('audio', Audio(sampling_rate=16000))
        
        # Add video feature
        if 'video_path' in datasets[split].features:
            datasets[split] = datasets[split].cast_column('video', Video())
    
    # Save the dataset
    print(f"Saving dataset to {dataset_path}")
    datasets.save_to_disk(dataset_path)
    print(f"Dataset saved with {len(datasets['train'])} train recordings, {len(datasets['validation'])} validation recordings, {len(datasets['test'])} test recordings")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment audio and video sources based on transcript timestamps')
    parser.add_argument('--to_dataset', default=False, help='Create HuggingFace dataset')
    
    args = parser.parse_args()
    
    segment_sources(transcript_segments_dir, audio_segment_dir, video_segment_dir, 
                   to_dataset=True)












