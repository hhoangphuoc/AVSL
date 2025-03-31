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
from datasets import Dataset, Audio, Video, Features, Value, DatasetDict

from preprocess.constants import DATA_PATH, TRANS_SEG_PATH, SOURCE_PATH, AUDIO_PATH, VIDEO_PATH, DATASET_PATH

from audio_process import segment_audio
from video_process import segment_video, extract_and_save_lip_video


transcript_segments_dir = TRANS_SEG_PATH
source_original_dir = SOURCE_PATH
audio_segment_dir = AUDIO_PATH
video_segment_dir = VIDEO_PATH
original_video_dir = os.path.join(VIDEO_PATH, "original_videos")
lip_video_dir = os.path.join(VIDEO_PATH, "lip_videos")  # Create a subdirectory for lip videos


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

    The segmented audio and video sources will be saved in `audio_segment_dir` and `video_segment_dir` respectively, with the following format:\n
    
    For audio:
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].wav\n
    
    For video:
    - original video (Path: `VIDEO_PATH/original_videos`)
    [meeting_id]-[speaker_id]-[start_time]-[end_time]-[source_type].mp4\n
    - lip video (Path: `VIDEO_PATH/lip_videos`)
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

def audio_video_to_dataset(recordings, meeting_ids, dataset_path=None):
    """
    Create a HuggingFace dataset from the processed segments (audio, video, and lip videos), 
    along with the transcript text.
    
    Args:
        recordings: List of dictionaries containing segment information
        meeting_ids: List of meeting IDs (not used for splitting)
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
    
    parser = argparse.ArgumentParser(description='Segment audio and video sources based on transcript timestamps')
    parser.add_argument('--to_dataset', default=True, help='Create HuggingFace dataset')
    parser.add_argument('--extract_lip_videos', default=True, help='Extract lip videos from video segments')    
    args = parser.parse_args()
    
    segment_sources(transcript_segments_dir, 
                    audio_segment_dir, 
                    video_segment_dir, 
                    to_dataset=args.to_dataset,
                    extract_lip_videos=args.extract_lip_videos,
                   )












