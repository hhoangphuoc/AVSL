import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf



def segment_audio(audio_file, start_time, end_time, output_file):
    """
    Segment an audio file between `start_time` and `end_time`, and save it to `output_file`.
    Audio is resampled to 16kHz.
    
    Args:
        audio_file: Path to the input audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_file: Path to save the output audio
    """
    # Load audio with 16kHz sample rate
    audio, sr = librosa.load(audio_file, sr=16000)
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # ensure sampling time is not out of range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    if start_sample >= end_sample:
        print(f"Warning: Invalid audio segment {start_time}-{end_time} for {audio_file}")
        return False
    
    audio_segment = audio[start_sample:end_sample] 
    
    # Save the audio segment
    sf.write(output_file, audio_segment, sr)
    
    return True
