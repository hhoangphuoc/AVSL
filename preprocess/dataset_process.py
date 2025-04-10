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
from video_process import batch_segment_video
from utils import av_to_hf_dataset, av_to_hf_dataset_with_shards


transcript_segments_dir = TRANS_SEG_PATH
source_original_dir = SOURCE_PATH
audio_segment_dir = AUDIO_PATH # data/audio_segments
video_segment_dir = VIDEO_PATH # data/video_segments
original_video_dir = os.path.join(VIDEO_PATH, "original_videos") # data/video_segments/original_videos
lip_video_dir = os.path.join(VIDEO_PATH, "lip_videos")  # data/video_segments/lip_videos


#==========================================================================================================================
#                   SEGMENTING AUDIO. VIDEO AND LIP VIDEO FROM TRANSCRIPT FILES (FROM SCRATCH)
#                   This  function involves in 5 steps, corresponding to 5 functions below:
#               1. `collect_segments_from_transcripts`: Collecting all segments from transcript files
#               2. `process_audio_segments`: Processing audio segments
#               3. `process_video_segments`: Processing video segments
#               4. `process_lip_videos`: Processing lip videos from created video segments
#               5. `create_dataset_records`: Creating dataset records based of the audio, video and lip video segments sources.
#========================================================================================================================== 

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
    print("Processing audio files...")
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
    Process all video segments in batches, each batch contains multiple segments from the same video source file 
    (e.g. ES2001a-Closeup1.avi)
    
    Args:
        video_segments_by_source: Dictionary mapping video source files to segments to extract
        
    Returns:
        Tuple of (video_segment_results, successful_video_segments)
    """
    print("Processing video files...")
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
                       use_parallel=False,
                       batch_size=8, 
                       batch_process=False,
                       to_grayscale=True
                       ):
    """
    Extract lip videos from successful video segments. 
    This function is based on the `video_segment_results` dictionary from `process_video_segments` function. 
    The function will use these video segments to extract lip videos from that, saving them in the `lip_video_dir` directory.

    - NOTE: `use_parallel` parameter is deprecated and kept for backward compatibility.
    The function now uses sequential processing for stability.
    
    - NOTE: `batch_process` parameter is deprecated and kept for backward compatibility.
    The function now processes videos sequentially.

    - NOTE: `to_grayscale` is used to determine if the lip videos should be saved in grayscale or color (default: True).
    Set `to_grayscale=True` to extract lip videos in grayscale. This is faster in processing, better for AV-Hubert model.
    If `to_grayscale=False`, the lip videos will be saved in color, better for visualization.
    
    Args:
        video_segment_results: Dictionary mapping segment IDs to (success, video_file) tuples
        lip_video_dir: Directory to save lip videos
        use_parallel: Deprecated. Kept for backward compatibility.
        batch_size: Batch size for frame processing (within a single video)
        batch_process: Deprecated. Kept for backward compatibility.
        to_grayscale: Whether to extract lip videos in grayscale
        
    Returns:
        Tuple of (lip_segment_results, successful_lip_segments)
    """
    print(f"Processing lip videos with settings: to_grayscale={to_grayscale}, batch_size={batch_size}")
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

    # Process lip videos (now always sequential)
    try:
        from video_process import batch_process_lip_videos
        
        # Extract lists for sequential processing
        batch_video_paths = [x[0] for x in all_lip_segments]
        batch_lip_video_paths = [x[1] for x in all_lip_segments]
        segment_ids = [x[2] for x in all_lip_segments]
        
        # Process videos sequentially with minimal settings
        batch_kwargs = {
            'to_grayscale': to_grayscale,
            'batch_size': batch_size,
        }
        
        print(f"Processing {len(batch_video_paths)} lip videos sequentially...")
        results, successful_lip_segments = batch_process_lip_videos(
            batch_video_paths, 
            batch_lip_video_paths,
            **batch_kwargs
        )
        
        # Record results
        for i, (success, lip_output_file) in enumerate(results):
            if i < len(segment_ids):  # Ensure we don't go out of bounds
                segment_id = segment_ids[i]
                lip_segment_results[segment_id] = (success, lip_output_file)
        
        # Calculate success percentage
        success_rate = successful_lip_segments / len(all_lip_segments) * 100 if all_lip_segments else 0
        print(f"Successfully processed {successful_lip_segments}/{len(all_lip_segments)} lip videos ({success_rate:.1f}%)")
        
        # Save diagnostic information
        diagnostics = {
            "total_videos": len(all_lip_segments),
            "successful_videos": successful_lip_segments,
            "success_rate": success_rate,
            "settings": {
                "to_grayscale": to_grayscale,
                "batch_size": batch_size
            }
        }
        
        diagnostics_path = os.path.join(lip_video_dir, "lip_extraction_diagnostics.json")
        with open(diagnostics_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        
    except Exception as e:
        print(f"Error batch processing lip videos: {str(e)}")
        print("Processing will continue with individual videos...")
        # Process each video individually as fallback
        for video_file, lip_output, segment_id in tqdm(all_lip_segments, desc="Extracting lip videos"):
            try:
                from video_process import extract_and_save_lip_video
                lip_success, lip_output_file = extract_and_save_lip_video(
                    video_file,
                    lip_output,
                    to_grayscale=to_grayscale,
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
        av_to_hf_dataset(dataset_records, dataset_path=dataset_path, prefix="ami")
    

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



def ami_dataset_from_existing_segments(source_dir=DATA_PATH,
                          transcript_segments_dir=TRANS_SEG_PATH,
                          dataset_path='../data/ami_dataset',
                          include_lip_videos=False
                          ):
    """
    Create a HuggingFace dataset from the processed segments (audio, video) which already exist
    in the `source_dir`. The function reads the metadata, including meeting id, speaker id, and transcript text from the `transcript_segments_dir`,
    and align it with the audio and video segments in the `source_dir`.

    NOTE: This function work similar to `segment_sources` function, but it does not process the audio and video segments,
    and only reads the existing segments in the `source_dir`.

    NOTE: The format of the `source_dir` is as follows:

    source_dir/\n
    |_ audio_segments/\n
    |_ video_segments/\n
        |_ original_videos/\n
        |_ lips/\n
    
    Args:
        source_dir: Path to the directory containing the processed segments (audio, video)
        transcript_segments_dir: Path to the directory containing the transcript segments
        dataset_path: Path to the HuggingFace dataset
        include_lip_videos: Whether to include lip videos in the dataset (default: False)
    """
    print(f"Creating HuggingFace dataset from paths: {source_dir}")
    
    # Set default paths if not provided
    if source_dir is None:
        source_dir = os.path.dirname(DATA_PATH)
    if transcript_segments_dir is None:
        transcript_segments_dir = TRANS_SEG_PATH
    if dataset_path is None:
        dataset_path = DATASET_PATH
        
    # Create output directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)
    
    # Load the audio and video segments
    audio_segments_dir = os.path.join(source_dir, "audio_segments")
    video_segments_dir = os.path.join(source_dir, "video_segments")
    original_video_dir = os.path.join(video_segments_dir, "original_videos")
    lip_video_dir = None
    if include_lip_videos:
        lip_video_dir = os.path.join(video_segments_dir, "lips")
    
    # Check if directories exist
    if not os.path.exists(audio_segments_dir):
        print(f"Warning: Audio segments directory not found: {audio_segments_dir}")
    if not os.path.exists(original_video_dir):
        print(f"Warning: Video segments directory not found: {original_video_dir}")
    if include_lip_videos and not os.path.exists(lip_video_dir):
        print(f"Warning: Lip video segments directory not found: {lip_video_dir}")
    if not os.path.exists(transcript_segments_dir):
        print(f"Error: Transcript segments directory not found: {transcript_segments_dir}")
        return

    # Find all audio, video, and lip video files
    audio_files = {}
    video_files = {}
    lip_video_files = {}  # Initialize it regardless to avoid reference errors
    
    # Process audio files
    if os.path.exists(audio_segments_dir):
        for file in tqdm(os.listdir(audio_segments_dir), desc="Reading audio files"):
            if file.endswith("-audio.wav"):
                # Extract segment_id by removing "-audio.wav"
                segment_id = file[:-10]
                audio_files[segment_id] = os.path.join(audio_segments_dir, file) # {segment_id: audio_file} - eg. {"ES2001a-A-0.00-0.10": "audio_segments/ES2001a-A-0.00-0.10-audio.wav"}
    
    # Process video files
    if os.path.exists(original_video_dir):
        for file in tqdm(os.listdir(original_video_dir), desc="Reading video files"):
            if file.endswith("-video.mp4"):
                # Extract segment_id by removing "-video.mp4"
                segment_id = file[:-10]
                video_files[segment_id] = os.path.join(original_video_dir, file) # {segment_id: video_file} - eg. {"ES2001a-A-0.00-0.10": "video_segments/original_videos/ES2001a-A-0.00-0.10-video.mp4"}   
    
    # Process lip video files only if include_lip_videos is True
    if include_lip_videos and lip_video_dir and os.path.exists(lip_video_dir):
        for file in tqdm(os.listdir(lip_video_dir), desc="Reading lip video files"):
            if file.endswith("-lip_video.mp4"):
                # Extract segment_id by removing "-lip_video.mp4"
                segment_id = file[:-14]
                lip_video_files[segment_id] = os.path.join(lip_video_dir, file) # {segment_id: lip_video_file} - eg. {"ES2001a-A-0.00-0.10": "video_segments/lip_videos/ES2001a-A-0.00-0.10-lip_video.mp4"}
    
    print(f"Found {len(audio_files)} audio files, {len(video_files)} video files, and {len(lip_video_files) if include_lip_videos else 0} lip video files")
    
    # Extract transcript information from transcript segment files
    time_pattern = re.compile(r'\[(\d+\.\d+)-(\d+\.\d+)\]\s+(.*)') #format: [start_time-end_time] text
    transcript_info = {}  # {segment_id: transcript_text} - eg. {"ES2001a-A-0.00-0.10": "This is a test transcript"}    
    
    for file in tqdm(os.listdir(transcript_segments_dir), desc="Reading transcript files"):
        if file.endswith('.txt'):
            meeting_id, ami_speaker_id = file.split('.')[0].split('-')
            
            # Skip if speaker not in mapping
            if ami_speaker_id not in AMI_SPEAKERS:
                print(f"Warning: Speaker {ami_speaker_id} not found in mapping. Skipping file {file}")
                continue
            
            # Process each line in the transcript file
            with open(os.path.join(transcript_segments_dir, file), 'r') as f:
                for line in f:
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
                        
                        # Create segment_id
                        segment_id = f"{meeting_id}-{ami_speaker_id}-{start_time_str}-{end_time_str}"
                        transcript_info[segment_id] = text
    
    print(f"Found transcript information for {len(transcript_info)} segments")
    
    # Create dataset records by finding the intersection of transcript segments and media files
    dataset_records = []
    
    # Group audio, video, and transcript by similar key (segment_id)
    all_segment_ids = set(list(audio_files.keys()) + list(video_files.keys()) + list(transcript_info.keys()))
    
    print(f"Processing {len(all_segment_ids)} unique segment IDs")
    
    for segment_id in tqdm(all_segment_ids, desc="Creating dataset records"):
        # Check if we have either audio or video for this segment
        has_audio = segment_id in audio_files
        has_video = segment_id in video_files
        has_lip_video = include_lip_videos and segment_id in lip_video_files
        has_transcript = segment_id in transcript_info
        
        # Only include segments that have at least audio or video
        if has_audio or has_video:
            # Parse segment_id to extract metadata
            parts = segment_id.split('-')
            if len(parts) >= 4:  # Ensure we have enough parts
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
                    "transcript": transcript_info.get(segment_id, ""),
                    "has_audio": has_audio,
                    "has_video": has_video,
                    "has_lip_video": has_lip_video,
                    "has_transcript": has_transcript
                }
                
                if has_audio:
                    record["audio"] = audio_files[segment_id]
                    
                if has_video:
                    record["video"] = video_files[segment_id]
                    
                if has_lip_video:
                    record["lip_video"] = lip_video_files[segment_id]
                    
                dataset_records.append(record)
    
    print(f"Created {len(dataset_records)} dataset records")

    # Save the dataset records to a json file
    dataset_records_path = os.path.join(dataset_path, "dataset_records.json")
    with open(dataset_records_path, "w") as f:
        json.dump(dataset_records, f)
    
    # Create and save the HuggingFace dataset
    if dataset_records:
        try:
            # av_to_hf_dataset(dataset_records, dataset_path=dataset_path, prefix="ami")
            # TRY WITH SHARDS
            av_to_hf_dataset_with_shards(
                dataset_records, 
                dataset_path=dataset_path, 
                prefix="ami",
                files_per_shard=2000 #FIXME: Reduce to 2000 for limit LFS size
            )
            print(f"Dataset successfully saved to {dataset_path}")
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
    else:
        print("No valid records found, dataset creation skipped.")
    
    return dataset_records

# ================================================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment audio and video sources based on transcript timestamps')

    parser.add_argument('--mode', choices=['segment_not_exist', 'segment_exist', 'process_lip', 'process_lip_chunk'], 
                       default='segment_not_exist', 
                       help='Mode to run the script in. `segment_not_exist` to segment audio and video sources from transcript segments. `segment_exist` to skip segmentation and process from existing segments. `process_lip` to process lip videos from video segments. `process_lip_chunk` to process a chunk of videos (from a CSV file).')

    # Original segment directory parameters
    parser.add_argument('--source_dir', type=str, default=DATA_PATH, help='Original audio and video source directory')
    parser.add_argument('--transcript_segments_dir', type=str, default=TRANS_SEG_PATH, help='Directory to save transcript segments')
    parser.add_argument('--audio_segment_dir', type=str, default=AUDIO_PATH, help='Directory to save audio segments')
    parser.add_argument('--video_segment_dir', type=str, default=VIDEO_PATH, help='Directory to save video segments')


    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH, help='Path to save HuggingFace dataset')
    parser.add_argument('--to_dataset', type=bool, default=True, help='Create HuggingFace dataset')

    # Lip video extraction configuration parameters
    parser.add_argument('--extract_lip_videos', type=bool, default=True, help='Extract lip videos from video segments')
    parser.add_argument('--lip_video_dir', type=str, default=lip_video_dir, help='Directory to save lip videos')
    parser.add_argument('--use_parallel', type=bool, default=False, help='Deprecated. Kept for backward compatibility.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing frames in each video')
    parser.add_argument('--batch_process', type=bool, default=False, help='Deprecated. Kept for backward compatibility.')
    parser.add_argument('--to_grayscale', type=bool, default=True, help='Extract lip videos in grayscale')
    
    # New parameters for chunk processing
    parser.add_argument('--chunk_file', type=str, help='Path to a CSV file containing a chunk of videos to process')
    parser.add_argument('--results_dir', type=str, help='Directory to save processing results')
    
    args = parser.parse_args()
    # =============================================================================================================
    # SEGMENT AUDIO AND VIDEO SOURCES WITHOUT EXISTING SEGMENTS
    # =============================================================================================================

    if args.mode == 'segment_not_exist':   
        print(f"\nRunning mode: {args.mode.upper()}")
        print(f"Source directory: {args.source_dir}")
        print(f"Transcript segments directory: {args.transcript_segments_dir}")
        print(f"Audio segment directory: {args.audio_segment_dir}")
        print(f"Video segment directory: {args.video_segment_dir}")
        print(f"Extract lip videos: {args.extract_lip_videos}")
        print(f"Lip video directory: {args.lip_video_dir}")
        print(f"Use parallel: {args.use_parallel}")
        print(f"Batch size: {args.batch_size}")
        print(f"Batch process: {args.batch_process}")
        print(f"To dataset: {args.to_dataset}")
        print(f"Dataset path: {args.dataset_path}")
        print(f"To grayscale: {args.to_grayscale}")

        segment_sources(
            args.transcript_segments_dir, 
            args.audio_segment_dir, 
            args.video_segment_dir, 
            to_dataset=args.to_dataset,
            extract_lip_videos=args.extract_lip_videos,
            lip_video_dir=args.lip_video_dir,
            use_parallel=args.use_parallel,
            batch_size=args.batch_size,
            batch_process=args.batch_process
        )
    
    # =============================================================================================================
    # CREATE HUGGINGFACE DATASET FROM EXISTING SEGMENTS
    # ============================================================================================================= 
    elif args.mode == 'segment_exist':
        print(f"\nRunning mode: {args.mode.upper()}")
        print(f"Source directory: {args.source_dir}")
        print(f"Transcript segments directory: {args.transcript_segments_dir}")
        print(f"Dataset path: {args.dataset_path}")
        print(f"Include lip videos: {args.extract_lip_videos}")

        ami_dataset_from_existing_segments(
            source_dir=args.source_dir,
            transcript_segments_dir=args.transcript_segments_dir,
            dataset_path=args.dataset_path,
            include_lip_videos=args.extract_lip_videos
        )
    
    # =============================================================================================================
    # PROCESS LIP VIDEOS FROM EXISTING VIDEO SEGMENTS
    # ============================================================================================================= 
    elif args.mode == 'process_lip':
        print(f"\nRunning mode: {args.mode.upper()}")
        # Expect dataset_path to be provided for this mode
        if not args.dataset_path:
            print("Error: --dataset_path must be provided for 'process_lip' mode.")
            sys.exit(1)
            
        csv_path = os.path.join(args.dataset_path, "ami-segments-info.csv")
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}. Please run segmentation first.")
            sys.exit(1)
            
        print(f"Lip video directory: {args.lip_video_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"To grayscale: {args.to_grayscale}")
        print(f"Reading video list from: {csv_path}")

        # Ensure output lip directory exists
        os.makedirs(args.lip_video_dir, exist_ok=True)
        
        try:
            # Read the dataset CSV
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} records from {csv_path}")
            
            # Filter for records that have video and potentially need lip processing
            # We process all videos listed, assuming lip data might be missing or needs update
            video_records = df[df['has_video'] == True].copy()
            if 'video' not in video_records.columns:
                 print(f"Error: CSV file {csv_path} does not contain a 'video' column.")
                 sys.exit(1)
                 
            video_records.dropna(subset=['video'], inplace=True)
            print(f"Found {len(video_records)} records with video paths to process.")

            if video_records.empty:
                 print("No video records found in the CSV to process. Exiting.")
                 sys.exit(0)
                 
            # Create the input dictionary for batch_process_lip_videos
            # {segment_id: (success, video_path)}
            video_segment_results = {
                row['id']: (True, row['video']) 
                for index, row in video_records.iterrows()
            }
            
            # Process lip videos sequentially
            print(f"Processing {len(video_segment_results)} lip videos...")
            lip_segment_results, successful_lip_segments = process_lip_videos(
                video_segment_results,
                lip_video_dir=args.lip_video_dir,
                batch_size=args.batch_size,
                to_grayscale=args.to_grayscale
            )
            
            print(f"\nCompleted processing. Total successful: {successful_lip_segments}/{len(video_segment_results)}")
            
            # Save the raw results dictionary to JSON for diagnostics
            if lip_segment_results:
                lip_results_path = os.path.join(args.lip_video_dir, "lip_segment_results.json")
                print(f"Saving detailed lip results to {lip_results_path}")
                with open(lip_results_path, "w") as f:
                    serializable_results = {}
                    for segment_id, (success, output_file) in lip_segment_results.items():
                        serializable_results[segment_id] = {"success": success, "output_file": output_file}
                    json.dump(serializable_results, f, indent=2)
            
            # Update the original DataFrame with the results
            print(f"Updating DataFrame and saving back to {csv_path}...")
            
            # Function to map results back
            def get_lip_info(segment_id):
                if segment_id in lip_segment_results:
                    success, path = lip_segment_results[segment_id]
                    return pd.Series([success, path if success else None])
                return pd.Series([False, None]) # Default if not processed or failed
            
            # Apply the results to the DataFrame
            # Ensure the columns exist before assigning
            if 'has_lip_video' not in df.columns:
                df['has_lip_video'] = False
            if 'lip_video' not in df.columns:
                df['lip_video'] = None
                
            # Apply results 
            lip_updates = df['id'].apply(get_lip_info)
            df[['has_lip_video', 'lip_video']] = lip_updates
            
            # Convert lip_video to strings or NA if None
            df['lip_video'] = df['lip_video'].astype(str)
            df.loc[df['lip_video'] == 'None', 'lip_video'] = pd.NA # Use pandas NA for missing strings
            
            # Save the updated DataFrame back to the CSV
            df.to_csv(csv_path, index=False)
            print(f"Successfully updated {csv_path} with lip video information for {df['has_lip_video'].sum()} entries.")

        except FileNotFoundError:
             print(f"Error: The CSV file was not found at {csv_path}")
             sys.exit(1)
        except KeyError as e:
             print(f"Error: Missing expected column in CSV {csv_path}: {e}")
             sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred in process_lip mode: {str(e)}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            sys.exit(1)

    # =============================================================================================================
    # PROCESS A CHUNK OF LIP VIDEOS
    # =============================================================================================================
    elif args.mode == 'process_lip_chunk':
        print(f"\nRunning mode: {args.mode.upper()}")
        
        # Check required parameters
        if not args.chunk_file:
            print("Error: --chunk_file must be provided for 'process_lip_chunk' mode.")
            sys.exit(1)
            
        if not args.lip_video_dir:
            print("Error: --lip_video_dir must be provided for 'process_lip_chunk' mode.")
            sys.exit(1)
            
        # Create results directory if specified
        results_dir = args.results_dir or os.path.dirname(args.chunk_file)
        os.makedirs(results_dir, exist_ok=True)
        
        # Get chunk number from filename
        chunk_name = os.path.basename(args.chunk_file).split('.')[0]
        print(f"Processing chunk: {chunk_name}")
        print(f"Lip video directory: {args.lip_video_dir}")
        print(f"Batch size: {args.batch_size}")
        print(f"To grayscale: {args.to_grayscale}")
        
        # Ensure output lip directory exists
        os.makedirs(args.lip_video_dir, exist_ok=True)
        
        try:
            # Read the chunk CSV
            chunk_df = pd.read_csv(args.chunk_file)
            print(f"Loaded {len(chunk_df)} records from {args.chunk_file}")
            
            # Filter out any rows without video paths
            if 'video' not in chunk_df.columns:
                print(f"Error: CSV file {args.chunk_file} does not contain a 'video' column.")
                sys.exit(1)
                
            chunk_df.dropna(subset=['video'], inplace=True)
            print(f"Found {len(chunk_df)} valid video paths to process.")
            
            if chunk_df.empty:
                print("No video records found in the chunk CSV to process. Exiting.")
                sys.exit(0)
                
            # Create the input dictionary for batch_process_lip_videos
            video_segment_results = {
                row['id']: (True, row['video']) 
                for index, row in chunk_df.iterrows()
                if os.path.exists(row['video'])  # Verify file exists
            }
            
            missing_files = len(chunk_df) - len(video_segment_results)
            if missing_files > 0:
                print(f"Warning: {missing_files} video files were not found on disk")
            
            # Process lip videos
            print(f"Processing {len(video_segment_results)} lip videos...")
            start_time = pd.Timestamp.now()
            print(f"Started at: {start_time}")
            
            lip_segment_results, successful_lip_segments = process_lip_videos(
                video_segment_results,
                lip_video_dir=args.lip_video_dir,
                batch_size=args.batch_size,
                to_grayscale=args.to_grayscale
            )
            
            end_time = pd.Timestamp.now()
            processing_time = (end_time - start_time).total_seconds() / 60  # in minutes
            videos_per_minute = len(video_segment_results) / processing_time if processing_time > 0 else 0
            
            print(f"\nCompleted processing at: {end_time}")
            print(f"Total time: {processing_time:.2f} minutes")
            print(f"Processing speed: {videos_per_minute:.2f} videos/minute")
            print(f"Total successful: {successful_lip_segments}/{len(video_segment_results)} ({successful_lip_segments/len(video_segment_results)*100:.1f}%)")
            
            # Save results to both CSV and JSON
            # 1. Create a dataframe with results
            results_df = pd.DataFrame({
                'id': list(lip_segment_results.keys()),
                'success': [result[0] for result in lip_segment_results.values()],
                'lip_video': [result[1] for result in lip_segment_results.values()]
            })
            
            # 2. Save to CSV for easy merging
            results_csv = os.path.join(results_dir, f"{chunk_name}_results.csv")
            results_df.to_csv(results_csv, index=False)
            
            # 3. Save to JSON for diagnostics
            results_json = os.path.join(results_dir, f"{chunk_name}_results.json")
            with open(results_json, 'w') as f:
                results = {
                    'chunk': chunk_name,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'processing_time_minutes': processing_time,
                    'videos_per_minute': videos_per_minute,
                    'total_videos': len(video_segment_results),
                    'successful_videos': successful_lip_segments,
                    'success_rate': successful_lip_segments/len(video_segment_results) if video_segment_results else 0,
                    'settings': {
                        'to_grayscale': args.to_grayscale,
                        'batch_size': args.batch_size
                    },
                    'results': {id: {'success': result[0], 'path': result[1]} for id, result in lip_segment_results.items()}
                }
                json.dump(results, f, indent=2)
                
            print(f"Results saved to:")
            print(f"  - CSV: {results_csv}")
            print(f"  - JSON: {results_json}")
            
            # Create a status file to indicate successful completion
            with open(os.path.join(results_dir, f"{chunk_name}_COMPLETED"), 'w') as f:
                f.write(f"Completed at: {end_time}\n")
                f.write(f"Success rate: {successful_lip_segments}/{len(video_segment_results)}\n")
            
        except FileNotFoundError:
            print(f"Error: The CSV file was not found at {args.chunk_file}")
            sys.exit(1)
        except KeyError as e:
            print(f"Error: Missing expected column in CSV {args.chunk_file}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred in process_lip_chunk mode: {str(e)}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            
            # Create a status file to indicate failure
            with open(os.path.join(results_dir, f"{chunk_name}_FAILED"), 'w') as f:
                f.write(f"Failed at: {pd.Timestamp.now()}\n")
                f.write(f"Error: {str(e)}\n")
                
            sys.exit(1)










