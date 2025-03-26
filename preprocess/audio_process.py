import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf



def segment_audio(audio_file, start_time, end_time, output_file):
    """
    Segment an audio file between start_time and end_time, and save it to output_file.
    Audio is resampled to 16kHz.
    
    Args:
        audio_file: Path to the input audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_file: Path to save the output audio
    
    Returns:
        Tuple of (success_flag, output_file_path)
    """
    try:
        # Load audio with 16kHz sample rate
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Check valid time range
        max_time = len(audio) / sr
        if start_time < 0:
            print(f"Warning: Negative start time {start_time} for {audio_file}, setting to 0")
            start_time = 0
            
        if end_time > max_time:
            print(f"Warning: End time {end_time} exceeds audio duration {max_time:.2f}s for {audio_file}, truncating")
            end_time = max_time
            
        if start_time >= end_time:
            print(f"Warning: Invalid audio segment {start_time}-{end_time} for {audio_file}")
            return False, None
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Check if the segment is too short
        if end_sample <= start_sample:
            print(f"Warning: Segment {start_time}-{end_time} is too short")
            return False, None
            
        audio_segment = audio[start_sample:end_sample]
        
        # Save the audio segment
        sf.write(output_file, audio_segment, sr)
        
        # Verify the output audio
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return True, output_file
        else:
            print(f"Error: Output audio file {output_file} is empty or doesn't exist")
            return False, None
    
    except Exception as e:
        print(f"Error segmenting audio {audio_file}: {str(e)}")
        return False, None
