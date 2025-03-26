"""
This script contains function to process video source input, including: 
    - segmenting video based on transcript_segments timestamps, video is resampled to 25fps, mp4 format
    - extracting lip-reading region of video with 96
"""

import ffmpeg
import os
import subprocess
import json


def segment_video(video_file, start_time, end_time, video_output_file):
    """
    Segment a video file between start_time and end_time, and save it to `video_output_file`.
    Video is resampled to 25fps and output format is mp4.
    
    Args:
        video_file: Path to the input video file
        start_time: Start time in seconds
        end_time: End time in seconds
        video_output_file: Path to save the output video (.mp4)
    
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    try:
        # Check that output file ends with .mp4
        if not video_output_file.lower().endswith('.mp4'):
            original_output = video_output_file
            video_output_file = f"{os.path.splitext(video_output_file)[0]}.mp4"
            print(f"Changing output format from {original_output} to {video_output_file}")
            
        # Check for valid timestamps
        if start_time < 0:
            print(f"Warning: Negative start time {start_time} for {video_file}, setting to 0")
            start_time = 0
            
        # Get video duration to validate end_time - FIXME: Remove this part---------------
        # probe = ffmpeg.probe(video_file)
        # video_duration = float(probe['format']['duration'])
        #---------------------------------------------------------------------------------
        
        # if end_time > video_duration:
        #     print(f"Warning: End time {end_time} exceeds video duration {video_duration} for {video_file}, truncating")
        #     end_time = video_duration
            
        if start_time >= end_time:
            print(f"Warning: Invalid video segment {start_time}-{end_time} for {video_file}")
            return False, None
            
        # Calculate duration
        duration = end_time - start_time
        
        # Use ffmpeg to segment the video with 25fps
        (
            ffmpeg
            .input(video_file, ss=start_time, t=duration)
            .output(
                video_output_file, 
                r=25,                  # Set frame rate to 25fps
                vcodec='libx264',      # Video codec
                acodec='aac',          # Audio codec
                format='mp4',          # Ensure mp4 format
                **{'copyts': None}     # Preserve timestamps for better alignment
            )
            .overwrite_output()
            .run(quiet=True)
        )
        
        # Verify the output video
        if os.path.exists(video_output_file) and os.path.getsize(video_output_file) > 0:
            return True, video_output_file
        else:
            print(f"Error: Output video file {video_output_file} is empty or doesn't exist")
            return False, None
            
    except ffmpeg.Error as e:
        print(f"Error segmenting video {video_file}: {e.stderr}")
        return False, None
    except Exception as e:
        print(f"Unexpected error processing video {video_file}: {str(e)}")
        return False, None



