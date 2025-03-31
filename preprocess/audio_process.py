import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from python_speech_features import logfbank



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

def add_noise(clean_wav, noise_wav, snr):
    """
    Add noise to clean audio signal at specified SNR.
    
    Args:
        clean_wav: Clean audio signal
        noise_wav: Noise audio signal
        snr: Signal-to-noise ratio in dB
        
    Returns:
        numpy.ndarray: Noisy audio signal
    """
    clean_wav = clean_wav.astype(np.float32)
    noise_wav = noise_wav.astype(np.float32)
    
    clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
    
    if len(clean_wav) > len(noise_wav):
        ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
        noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
    if len(clean_wav) < len(noise_wav):
        start = 0
        noise_wav = noise_wav[start: start + len(clean_wav)]
    
    noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1))
    adjusted_noise_rms = clean_rms / (10**(snr/20))
    adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
    mixed = clean_wav + adjusted_noise_wav

    # Avoid clipping noise
    max_int16 = np.iinfo(np.int16).max
    min_int16 = np.iinfo(np.int16).min
    if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
        if mixed.max(axis=0) >= abs(mixed.min(axis=0)): 
            reduction_rate = max_int16 / mixed.max(axis=0)
        else:
            reduction_rate = min_int16 / mixed.min(axis=0)
        mixed = mixed * (reduction_rate)
    
    mixed = mixed.astype(np.int16)
    return mixed

def extract_logfbank_features(audio_data, sample_rate=16000, stack_order=1):
    """
    Extract log filterbank features from audio data.
    
    Args:
        audio_data: Audio signal
        sample_rate: Sample rate of the audio
        stack_order: Number of frames to stack together
        
    Returns:
        numpy.ndarray: Log filterbank features
    """
    # Extract log filterbank features
    audio_feats = logfbank(audio_data, samplerate=sample_rate).astype(np.float32)
    
    # Stack consecutive frames if stack_order > 1
    if stack_order > 1:
        feat_dim = audio_feats.shape[1]
        # Pad if necessary
        if len(audio_feats) % stack_order != 0:
            res = stack_order - len(audio_feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(audio_feats.dtype)
            audio_feats = np.concatenate([audio_feats, res], axis=0)
        # Reshape to stack frames
        audio_feats = audio_feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
    
    return audio_feats

def audio_to_tensor(audio_features, normalize=True):
    """
    Convert audio features to tensor format suitable for AV-HuBERT.
    
    Args:
        audio_features: numpy.ndarray of log filterbank features
        normalize: Whether to apply layer normalization
        
    Returns:
        numpy.ndarray: Processed audio features
    """
    if normalize:
        # Apply layer normalization (mean=0, std=1 across feature dimension)
        mean = np.mean(audio_features, axis=1, keepdims=True)
        std = np.std(audio_features, axis=1, keepdims=True)
        audio_features = (audio_features - mean) / (std + 1e-5)
    
    return audio_features

def process_audio_for_av_hubert(audio_path, stack_order=1, normalize=True, add_noise_prob=0.0, 
                               noise_file=None, noise_snr=0):
    """
    Process a HuggingFace Audio object for AV-HuBERT Audio Encoder.
    
    Args:
        audio_path: Path to audio file
        stack_order: Number of frames to stack together
        normalize: Whether to apply layer normalization
        add_noise_prob: Probability of adding noise to audio
        noise_file: Path to noise audio file
        noise_snr: Signal-to-noise ratio for adding noise
        
    Returns:
        numpy.ndarray: Processed audio features
    """
    
    try:
        
        # Load audio with 16kHz sample rate
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Add noise if specified
        if add_noise_prob > 0 and noise_file is not None and np.random.rand() < add_noise_prob:
            noise_data, noise_sr = librosa.load(noise_file, sr=16000)
            audio = add_noise(audio, noise_data, noise_snr)
        
        # Extract log filterbank features
        audio_features = extract_logfbank_features(audio, sample_rate=sr, stack_order=stack_order)
        
        # Convert to tensor format
        audio_features = audio_to_tensor(audio_features, normalize=normalize)
        
        return audio_features
    
    except Exception as e:
        print(f"Error processing audio for AV-HuBERT: {str(e)}")
        return None

def align_audio_video_features(audio_features, video_features):
    """
    Align audio and video features to have the same length.
    
    Args:
        audio_features: Audio features array
        video_features: Video features array
        
    Returns:
        Tuple of (aligned_audio_features, aligned_video_features)
    """
    if audio_features is None or video_features is None:
        return audio_features, video_features
    
    # Get lengths
    audio_len = len(audio_features)
    video_len = len(video_features)
    
    # Align lengths
    if audio_len > video_len:
        # Truncate audio features
        audio_features = audio_features[:video_len]
    elif audio_len < video_len:
        # Truncate video features
        video_features = video_features[:audio_len]
    
    return audio_features, video_features


def process_audio_dual_encoder(audio_path, stack_order=1, normalize=True):
    """
    Process audio for both AV-HuBERT and Whisper encoders without noise addition.
    
    Args:
        audio_path: Path to audio file
        stack_order: Number of frames to stack together for AV-HuBERT features
        normalize: Whether to apply layer normalization to AV-HuBERT features
        
    Returns:
        Dict containing:
            'waveform': Audio waveform for Whisper (16kHz)
            'sample_rate': Sample rate of the waveform
            'av_hubert_features': Processed features for AV-HuBERT
    """
    # Load audio waveform
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process for AV-HuBERT (log filterbank features)
    av_hubert_features = extract_logfbank_features(audio, sample_rate=sr, stack_order=stack_order)
    av_hubert_features = audio_to_tensor(av_hubert_features, normalize=normalize)
    
    # For Whisper, we return the raw waveform at 16kHz
    # Whisper expects float32 audio with values between -1 and 1
    whisper_audio = audio.astype(np.float32)
    if whisper_audio.max() > 1.0 or whisper_audio.min() < -1.0:
        whisper_audio = whisper_audio / max(abs(whisper_audio.max()), abs(whisper_audio.min()))
    
    return {
        'waveform': whisper_audio,  
        'sample_rate': sr,
        'av_hubert_features': av_hubert_features
    }

def preprocess_audio_for_whisper(audio_path):
    """
    Preprocess audio specifically for Whisper model.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        numpy.ndarray: Audio waveform normalized for Whisper (16kHz, float32, [-1, 1])
    """
    # Load audio waveform
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Whisper expects float32 audio with values between -1 and 1
    whisper_audio = audio.astype(np.float32)
    if whisper_audio.max() > 1.0 or whisper_audio.min() < -1.0:
        whisper_audio = whisper_audio / max(abs(whisper_audio.max()), abs(whisper_audio.min()))
    
    return whisper_audio

def preprocess_audio_for_av_hubert(audio_path, stack_order=1, normalize=True):
    """
    Preprocess audio specifically for AV-HuBERT model without noise addition.
    
    Args:
        audio_path: Path to audio file
        stack_order: Number of frames to stack together
        normalize: Whether to apply layer normalization
        
    Returns:
        numpy.ndarray: Processed log filterbank features for AV-HuBERT
    """
    # Load audio waveform
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process for AV-HuBERT (log filterbank features)
    av_hubert_features = extract_logfbank_features(audio, sample_rate=sr, stack_order=stack_order)
    av_hubert_features = audio_to_tensor(av_hubert_features, normalize=normalize)
    
    return av_hubert_features
