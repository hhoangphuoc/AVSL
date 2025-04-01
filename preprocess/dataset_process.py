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

import json
import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from tqdm import tqdm

from collections import defaultdict



from preprocess.constants import (
    DATA_PATH, 
    TRANS_SEG_PATH, 
    SOURCE_PATH, 
    AUDIO_PATH, 
    VIDEO_PATH, 
    DATASET_PATH, 
    AMI_SPEAKERS
)

from audio_process import batch_segment_audio
from video_process import batch_segment_video, extract_and_save_lip_video
from utils import av_to_hf_dataset


transcript_segments_dir = TRANS_SEG_PATH
source_original_dir = SOURCE_PATH
audio_segment_dir = AUDIO_PATH # data/audio_segments
video_segment_dir = VIDEO_PATH # data/video_segments
original_video_dir = os.path.join(VIDEO_PATH, "original_videos") # data/video_segments/original_videos
lip_video_dir = os.path.join(VIDEO_PATH, "lip_videos")  # data/video_segments/lip_videos


def collect_segments_from_transcripts(transcript_segments_dir, audio_segment_dir, original_video_dir):
    """
    Collect all segments from transcript files and group them by source file.
    This function groups segments of audio/video from the same sources (i.e. [meeting_id]-[speaker_id])
    and saves them in the following format:
    - audio_segments_by_source: Dictionary mapping `audio source files` to multiple segments
    - video_segments_by_source: Dictionary mapping `video source files` to multiple segments
    
    Args:
        transcript_segments_dir: Directory containing transcript segment files
        audio_segment_dir: Directory to save audio segments
        original_video_dir: Directory to save original video segments
        
    Returns:
        Tuple of (total_segments, audio_segments_by_source, video_segments_by_source)
    """
    # Extract timestamps and text
    time_pattern = re.compile(r'\[(\d+\.\d+)-(\d+\.\d+)\]\s+(.*)') #format: [start_time-end_time] text
    
    # Data structures to group segments by their source
    audio_segments_by_source = defaultdict(list)  # {audio_file: [(start, end, output_file, text, segment_id), ...]}
    video_segments_by_source = defaultdict(list)  # {video_file: [(start, end, output_file, segment_id), ...]}
    
    # For tracking processed segments
    total_segments = 0
    
    print("Step 1: Collecting all segments from transcript files...")
    
    for file in tqdm(os.listdir(transcript_segments_dir), desc="Processing transcript files"):
        if file.endswith('.txt'):
            meeting_id, ami_speaker_id = file.split('.')[0].split('-')
            
            # Skip if speaker not in mapping
            if ami_speaker_id not in AMI_SPEAKERS:
                print(f"Warning: Speaker {ami_speaker_id} not found in mapping. Skipping file {file}")
                continue
                
            # Get the corresponding audio and video sources for this speaker
            audio_source = AMI_SPEAKERS[ami_speaker_id]['audio'] # e.g. Headset-0, Headset-1, etc.
            video_source = AMI_SPEAKERS[ami_speaker_id]['video'] # e.g. Closeup1, Closeup2, etc.
                
            # Paths to the original audio and video files
            source_audio_file = os.path.join(source_original_dir, meeting_id, 'audio', f"{meeting_id}.{audio_source}.wav") # e.g. ES2001a-Headset-0.wav
            source_video_file = os.path.join(source_original_dir, meeting_id, 'video', f"{meeting_id}.{video_source}.avi") # e.g. ES2001a-Closeup1.avi
                
            # Check if original files exist
            process_audio = os.path.exists(source_audio_file)
            process_video = os.path.exists(source_video_file)
            
            if not process_audio and not process_video:
                print(f"Skipping file {file} as neither audio nor video source exists")
                continue
                
            # PROCESS EACH LINE IN THE TRANSCRIPT FILE, WHICH CONTAINS THE START TIME, END TIME, AND TRANSCRIPT TEXT
            with open(os.path.join(transcript_segments_dir, file), 'r') as f:
                for line in tqdm(f, desc=f"Processing {file}", leave=False):
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
                        
                        # Base segment identifier
                        segment_id = f"{meeting_id}-{ami_speaker_id}-{start_time_str}-{end_time_str}"
                        
                        total_segments += 1
                        
                        # Add to appropriate source lists
                        if process_audio:
                            audio_output_file = os.path.join(audio_segment_dir, f"{segment_id}-audio.wav")
                            audio_segments_by_source[source_audio_file].append((start_time, end_time, audio_output_file, text, segment_id))
                            
                        if process_video:
                            video_output_file = os.path.join(original_video_dir, f"{segment_id}-video.mp4")
                            video_segments_by_source[source_video_file].append((start_time, end_time, video_output_file, segment_id))
    
    # SAVE THE AUDIO AND VIDEO SEGMENTS BY SOURCE IN JSON
    audio_segments_by_source_path = os.path.join(DATA_PATH, 'audio_segments_by_source.json')
    video_segments_by_source_path = os.path.join(DATA_PATH, 'video_segments_by_source.json')

    with open(audio_segments_by_source_path, 'w') as f:
        json.dump(audio_segments_by_source, f)
    with open(video_segments_by_source_path, 'w') as f:
        json.dump(video_segments_by_source, f)
    
    return total_segments, audio_segments_by_source, video_segments_by_source

def process_audio_segments(audio_segments_by_source):
    """
    Process all audio segments in batches, each batch contains multiple segments from the same audio source file.

    Args:
        audio_segments_by_source: Dictionary mapping audio source files to segments to extract
        
    Returns:
        Tuple of (audio_segment_results, successful_audio_segments)
    """
    print("\nStep 2: Processing audio files...")
    audio_segment_results = {}  # {segment_id: (success, output_file, text)}
    successful_audio_segments = 0
    
    for audio_file, segments in tqdm(audio_segments_by_source.items(), desc="Processing audio files"):
        # Extract the relevant parts for batch processing
        batch_segments = [(start, end, output) for start, end, output, _, _ in segments]
        results = batch_segment_audio(audio_file, batch_segments)
        
        # Record results for each segment
        for i, (success, output_file) in enumerate(results):
            _, _, _, text, segment_id = segments[i]
            audio_segment_results[segment_id] = (success, output_file, text)
            if success:
                successful_audio_segments += 1 # Number of successful audio segments
    
    return audio_segment_results, successful_audio_segments

def process_video_segments(video_segments_by_source):
    """
    Process all video segments in batches, each batch contains multiple segments from the same video source file (e.g. ES2001a-Closeup1.avi)
    
    Args:
        video_segments_by_source: Dictionary mapping video source files to segments to extract
        
    Returns:
        Tuple of (video_segment_results, successful_video_segments)
    """
    print("\nStep 3: Processing video files...")
    video_segment_results = {}  # {segment_id: (success, output_file)}
    successful_video_segments = 0
    
    for video_file, segments in tqdm(video_segments_by_source.items(), desc="Processing video files"):
        # Extract the relevant parts for batch processing
        batch_segments = [(start, end, output) for start, end, output, _ in segments]
        results = batch_segment_video(video_file, batch_segments)
        
        # Record results for each segment
        for i, (success, output_file) in enumerate(results):
            _, _, _, segment_id = segments[i]
            video_segment_results[segment_id] = (success, output_file)
            if success:
                successful_video_segments += 1 # Number of successful video segments
                
    return video_segment_results, successful_video_segments

def process_lip_videos(video_segment_results, 
                       lip_video_dir, 
                       use_gpu=False, 
                       use_parallel=True, 
                       batch_size=16, 
                       batch_process=True,
                       to_grayscale=True
                       ):
    """
    Extract lip videos from successful video segments. 
    This function is based on the `video_segment_results` dictionary from `process_video_segments` function. 
    The function will use these video segments to extract lip videos from that, saving them in the `lip_video_dir` directory.

    NOTE: `batch_process` using to determine video processing parallel or sequential.
    Set `batch_process=True` and `use_parallel=True` to use this feature. Optionally, set `use_gpu=True` to use GPU acceleration (default: False).

    NOTE: If `batch_process=False`, the function will process each video sequentially.

    NOTE: `to_grayscale` is used to determine if the lip videos should be saved in grayscale or color (default: True).
    Set `to_grayscale=True` to extract lip videos in grayscale. This is faster in processing, better for AV-Hubert model.
    If `to_grayscale=False`, the lip videos will be saved in color, better for visualization.
    
    Args:
        video_segment_results: Dictionary mapping segment IDs to (success, video_file) tuples
        lip_video_dir: Directory to save lip videos
        use_gpu: Whether to use GPU acceleration if available
        use_parallel: Whether to use parallel processing for lip extraction
        batch_size: Batch size for processing frames
        batch_process: Whether to process multiple videos in parallel
        
    Returns:
        Tuple of (lip_segment_results, successful_lip_segments)
    """
    print("Processing lip videos...")
    lip_segment_results = {}  # {segment_id: (success, output_file)}
    successful_lip_segments = 0
    
    # Collect all segments that have successful video
    all_lip_segments = []
    for segment_id, (success, video_file) in video_segment_results.items():
        if success:
            lip_output_file = os.path.join(lip_video_dir, f"{segment_id}-lip_video.mp4")
            all_lip_segments.append((video_file, lip_output_file, segment_id))
            
    if not all_lip_segments:
        print("No successful video segments to process for lip videos")
        return lip_segment_results, successful_lip_segments

    if batch_process:
        # Optimize: Batch process lip videos with parallel processing and GPU if available
        try:
            from video_process import batch_process_lip_videos
            
            # Extract lists for batch processing
            batch_video_paths = [x[0] for x in all_lip_segments]
            batch_lip_video_paths = [x[1] for x in all_lip_segments]
            segment_ids = [x[2] for x in all_lip_segments]
            
            # Process in batch with optimizations
            batch_kwargs = {
                'to_grayscale': to_grayscale,  # Use color for better visualization
                'use_gpu': use_gpu,
                'use_parallel': use_parallel,
                'batch_size': batch_size
            }
            
            print(f"Batch processing {len(batch_video_paths)} lip videos with optimizations...")
            results = batch_process_lip_videos(
                batch_video_paths, 
                batch_lip_video_paths,
                **batch_kwargs
            )
            
            # Record results
            for i, (success, lip_output_file) in enumerate(results):
                segment_id = segment_ids[i]
                lip_segment_results[segment_id] = (success, lip_output_file)
                if success:
                    successful_lip_segments += 1
            
            print(f"Successfully processed {successful_lip_segments}/{len(all_lip_segments)} lip videos")
            
        except Exception as e:
            print(f"Error batch processing lip videos: {str(e)}")
            print("Falling back to sequential processing...")
            # Fall back to sequential processing
            batch_process = False
    
    # If batch processing failed or was disabled, process sequentially
    if not batch_process:
        for video_file, lip_output, segment_id in tqdm(all_lip_segments, desc="Extracting lip videos"):
            try:
                lip_success, lip_output_file = extract_and_save_lip_video(
                    video_file,
                    lip_output,
                    to_grayscale=to_grayscale,  # Use color for better visualization
                    use_parallel=use_parallel,
                    batch_size=batch_size
                )
                lip_segment_results[segment_id] = (lip_success, lip_output_file)
                if lip_success:
                    successful_lip_segments += 1
            except Exception as e:
                print(f"Error extracting lip video for {segment_id}: {str(e)}")
                lip_segment_results[segment_id] = (False, None)
                
    return lip_segment_results, successful_lip_segments

def create_dataset_records(
        audio_segment_results, 
        video_segment_results, 
        lip_segment_results):
    """
    Create dataset records based on the audio, video, and lip video processing results.
    These results are stored in dictionaries, in which: 
    - keys: segment IDs (e.g. "ES2001a-A-0.00-0.10")
    - values: tuples of (success, output_file, transcript text).
    
    Args:
        audio_segment_results: Dictionary of audio processing results
        video_segment_results: Dictionary of video processing results
        lip_segment_results: Dictionary of lip video processing results
        
    Returns:
        Tuple of (dataset_records, alignment_issues)
    """
    print("\nFinalizing dataset...")
    dataset_records = []
    alignment_issues = []
    
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
                
    return dataset_records, alignment_issues

def segment_sources(transcript_segments_dir, 
                    audio_segment_dir, 
                    video_segment_dir,
                    dataset_path=DATASET_PATH, # Added default dataset path
                    to_dataset=False,
                    extract_lip_videos=True,
                    lip_video_dir=None,
                    use_gpu=True,
                    use_parallel=True,
                    batch_size=16,
                    batch_process=True
                    ):
    """
    This function is used to segment the audio and video sources based on the 
    `transcript_segments` timestamps and save the segmented audio and video resources. 
    
    NOTE: If `to_dataset` is True, the sources will be saved in a HuggingFace Dataset.
    NOTE: If `extract_lip_videos` is True, will process the lip extraction from successful video segments. 
    This stored in `lip_video_dir`, and the parameters `use_gpu`, `use_parallel`, `batch_size`, and `batch_process` will be used to determine the processing mode.

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
        dataset_path: Path to save HuggingFace Dataset (default: `DATASET_PATH`)
        to_dataset: Whether to create a HuggingFace dataset
        extract_lip_videos: Whether to extract lip regions from video segments
        lip_video_dir: Directory to save lip videos (default: VIDEO_PATH/lip_videos)
        use_gpu: Whether to use GPU acceleration if available
        use_parallel: Whether to use parallel processing for lip extraction
        batch_size: Batch size for processing frames
        batch_process: Whether to process multiple videos in parallel
    """
    
    # Create output directories if they don't exist
    os.makedirs(audio_segment_dir, exist_ok=True)
    os.makedirs(video_segment_dir, exist_ok=True)
    
    # Making original video directory if it doesn't exist
    original_video_dir = os.path.join(video_segment_dir, "original_videos")
    os.makedirs(original_video_dir, exist_ok=True)

    # Create lip video directory
    if extract_lip_videos:
        if lip_video_dir is None:
            lip_video_dir = os.path.join(video_segment_dir, "lip_videos")
        os.makedirs(lip_video_dir, exist_ok=True)
    
    # Step 1: Collect segments from transcript files
    total_segments, audio_segments_by_source, video_segments_by_source = collect_segments_from_transcripts(
        transcript_segments_dir, 
        audio_segment_dir, 
        original_video_dir
    )
    
    if total_segments == 0:
        print("No segments found in transcript files")
        return
    
    # ----------------------- Step 2: PROCESS AUDIO SEGMENTS ---------------------------------------------------
    audio_segment_results, successful_audio_segments = process_audio_segments(audio_segments_by_source)
    # Save audio segment results
    audio_segment_results_path = os.path.join(audio_segment_dir, "audio_segment_results.json")
    with open(audio_segment_results_path, "w") as f:
        json.dump(audio_segment_results, f)
    
    # ----------------------- Step 3: PROCESS VIDEO SEGMENTS ---------------------------------------------------
    video_segment_results, successful_video_segments = process_video_segments(video_segments_by_source)
    # Save video segment results
    video_segment_results_path = os.path.join(video_segment_dir, "video_segment_results.json")
    with open(video_segment_results_path, "w") as f:
        json.dump(video_segment_results, f)

    
    # ----------------------- Step 4: PROCESS LIP VIDEOS ---------------------------------------------------
    lip_segment_results = {}
    successful_lip_segments = 0
    
    if extract_lip_videos and lip_video_dir and successful_video_segments > 0:
        lip_segment_results, successful_lip_segments = process_lip_videos(
            video_segment_results, 
            lip_video_dir, 
            use_gpu, 
            use_parallel, 
            batch_size, 
            batch_process
        )

    # Save lip segment results
    lip_segment_results_path = os.path.join(lip_video_dir, "lip_segment_results.json")
    with open(lip_segment_results_path, "w") as f:
        json.dump(lip_segment_results, f)

    
    # ----------------------- Step 5: CREATE DATASET RECORDS ---------------------------------------------------
    dataset_records, alignment_issues = create_dataset_records(
        audio_segment_results, 
        video_segment_results, 
        lip_segment_results
    )
    
    # Handle alignment issues
    if alignment_issues:
        print(f"\nFound {len(alignment_issues)} segments with potential alignment issues")
        alignment_log_path = os.path.join(DATA_PATH, "transcript_alignment_issues.log")
        print(f"Saving alignment issues to {alignment_log_path}")
        with open(alignment_log_path, "w") as f:
            for issue in alignment_issues:
                f.write(f"{issue}\n")

    
    # ----------------------- Step 6: CREATE HUGGINGFACE DATASET ---------------------------------------------------
    if to_dataset and dataset_records:
        # Use a specific dataset name for transcript-based segments
        av_to_hf_dataset(dataset_records, dataset_path=dataset_path)
    

    # ----------------------- REPORT DATASET STATISTICS ---------------------------------------------------
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

#==========================================================================================================================

def hf_dataset_from_paths(source_dir=None,
                          transcript_segments_dir=None,
                          dataset_path=None):
    """
    Create a HuggingFace dataset from the processed segments (audio, video) which already exist
    in the `source_dir`. The function reads the metadata, including meeting id, speaker id, and transcript text from the `transcript_segments_dir`,
    and align it with the audio and video segments in the `source_dir`.

    NOTE: The format of the `source_dir` is as follows:

    source_dir/\n
    |_ audio_segments/\n
    |_ video_segments/\n
        |_ original_videos/\n
        |_ lip_videos/\n

    Args:
        source_dir: Path to the directory containing the processed segments (audio, video)
        transcript_segments_dir: Path to the directory containing the transcript segments
        dataset_path: Path to the HuggingFace dataset
    """
    print(f"Creating HuggingFace dataset from paths: {source_dir}")
    
    # Load the audio and video segments
    audio_segments_dir = os.path.join(source_dir, "audio_segments")
    video_segments_dir = os.path.join(source_dir, "video_segments")

    # Load the transcript segments to get the meeting id, speaker id, and transcript text
    


# ================================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment audio and video sources based on transcript timestamps')
    parser.add_argument('--transcript_segments_dir', type=str, default=TRANS_SEG_PATH, help='Directory to save transcript segments')
    parser.add_argument('--audio_segment_dir', type=str, default=AUDIO_PATH, help='Directory to save audio segments')
    parser.add_argument('--video_segment_dir', type=str, default=VIDEO_PATH, help='Directory to save video segments')
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, help='Path to save HuggingFace dataset')
    parser.add_argument('--to_dataset', action='store_true', default=True, help='Create HuggingFace dataset')
    parser.add_argument('--extract_lip_videos', action='store_true', default=True, help='Extract lip videos from video segments')
    parser.add_argument('--lip_video_dir', type=str, default=lip_video_dir, help='Directory to save lip videos')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU acceleration if available')
    parser.add_argument('--use_parallel', action='store_true', default=True, help='Use parallel processing for lip extraction')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing frames in each video')
    parser.add_argument('--batch_process', action='store_true', default=True, help='Process multiple videos in parallel batches')
    
    args = parser.parse_args()
    
    segment_sources(
        transcript_segments_dir, 
        audio_segment_dir, 
        video_segment_dir, 
        to_dataset=args.to_dataset,
        extract_lip_videos=args.extract_lip_videos,
        lip_video_dir=args.lip_video_dir,
        use_gpu=args.use_gpu,
        use_parallel=args.use_parallel,
        batch_size=args.batch_size,
        batch_process=args.batch_process
    )












