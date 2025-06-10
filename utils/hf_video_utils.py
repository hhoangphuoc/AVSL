"""
Utility functions for handling HuggingFace Video objects in the AVSL project.
This module provides functions to extract video data from HuggingFace dataset objects
and process them for the whisper-flamingo model training pipeline.
"""

import os
import numpy as np
import cv2
from typing import Optional, Any
import traceback
from tqdm import tqdm

# Try to import decord for VideoReader support
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("Warning: decord not available. Some video loading features may not work.")


def extract_video_path_from_hf_object(video_object: Any) -> Optional[str]:
    """
    Extract video file path from a HuggingFace Video object.
    
    Args:
        video_object: HuggingFace Video object from dataset
        
    Returns:
        str: Path to the video file, or None if not found
    """
    try:
        # Method 1: Check if it's a decord.VideoReader (common in HF Video datasets)
        if hasattr(video_object, '__class__') and 'VideoReader' in str(type(video_object)):
            # Try to get the filename from the VideoReader object
            if hasattr(video_object, '_filename'):
                return video_object._filename
            if hasattr(video_object, 'filename'):
                return video_object.filename
            # Some VideoReader objects store the path in different attributes
            for attr in ['_path', 'path', '_file_path', 'file_path']:
                if hasattr(video_object, attr):
                    path = getattr(video_object, attr)
                    if path and os.path.exists(str(path)):
                        return str(path)
            print(f"Warning: decord.VideoReader object found but no accessible path attribute")
            return None
            
        # Method 2: Check if it has a 'path' attribute
        if hasattr(video_object, 'path') and video_object.path:
            return video_object.path
            
        # Method 3: Check if it's a dict with 'path' key
        if isinstance(video_object, dict) and 'path' in video_object:
            return video_object['path']
            
        # Method 5: Check for other common attributes
        for attr in ['filename', 'file_path', 'source']:
            if hasattr(video_object, attr):
                path = getattr(video_object, attr)
                if path and os.path.exists(str(path)):
                    return str(path)
                    
        print(f"Warning: Could not extract path from video object: {type(video_object)}")
        print(f"Available attributes: {[attr for attr in dir(video_object) if not attr.startswith('_')]}")
        return None
        
    except Exception as e:
        print(f"Error extracting video path from HF object: {e}")
        return None

def load_video_feats_from_decord_reader(video_reader: Any, train: bool = False,
                                       image_crop_size: int = 88,
                                       image_mean: float = 0.421,
                                       image_std: float = 0.165) -> np.ndarray:
    """
    Load video features directly from a decord.VideoReader object.
    
    Args:
        video_reader: decord.VideoReader object
        train: Whether this is for training (affects augmentation)
        image_crop_size: Size to crop images to
        image_mean: Mean for normalization
        image_std: Standard deviation for normalization
        
    Returns:
        np.ndarray: Video features with shape [T, H, W, C]
    """
    try:
        # Get the number of frames
        num_frames = len(video_reader)
        
        # Read all frames at once for efficiency
        frame_indices = list(range(num_frames))
        frames = video_reader.get_batch(frame_indices)  # Returns numpy array [T, H, W, C]
        
        # Convert to numpy if needed
        if not isinstance(frames, np.ndarray):
            frames = frames.asnumpy()
        
        # Convert to grayscale if needed (frames are usually RGB)
        if len(frames.shape) == 4 and frames.shape[3] == 3:  # [T, H, W, 3]
            # Convert RGB to grayscale
            frames = np.dot(frames[...,:3], [0.2989, 0.5870, 0.1140])
        elif len(frames.shape) == 4 and frames.shape[3] == 1:  # Already grayscale
            frames = frames.squeeze(axis=3)
        
        # Ensure frames is [T, H, W]
        if len(frames.shape) != 3:
            raise ValueError(f"Expected 3D frames array after processing, got shape: {frames.shape}")
        
        # Normalize to [0, 1] if needed
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32) / 255.0
        elif frames.max() > 1.0:
            frames = frames.astype(np.float32) / 255.0
        
        # Apply crop and normalization
        H, W = frames.shape[1], frames.shape[2]
        start_h = (H - image_crop_size) // 2
        start_w = (W - image_crop_size) // 2
        
        if start_h >= 0 and start_w >= 0:
            frames = frames[:, start_h:start_h+image_crop_size, start_w:start_w+image_crop_size]
        else:
            # Resize if image is too small
            frames_resized = []
            for frame in frames:
                frame_resized = cv2.resize(frame, (image_crop_size, image_crop_size))
                frames_resized.append(frame_resized)
            frames = np.stack(frames_resized)
        
        # Normalize with mean and std
        frames = (frames - image_mean) / image_std
        
        # Add channel dimension: [T, H, W] -> [T, H, W, 1]
        frames = np.expand_dims(frames, axis=-1)
        
        return frames
        
    except Exception as e:
        print(f"Error loading video features from decord reader: {e}")
        traceback.print_exc()
        raise e


def load_video_feats_from_hf_object(video_object: Any, train: bool = False, 
                                   image_crop_size: int = 88,
                                   image_mean: float = 0.421, 
                                   image_std: float = 0.165) -> np.ndarray:
    """
    Load video features from a HuggingFace Video object, compatible with whisper-flamingo's
    load_video_feats function.
    
    Args:
        video_object: HuggingFace Video object from dataset
        train: Whether this is for training (affects augmentation)
        image_crop_size: Size to crop images to
        image_mean: Mean for normalization
        image_std: Standard deviation for normalization
        
    Returns:
        np.ndarray: Video features with shape [T, H, W, C]
    """
    try:
        # Handle decord.VideoReader objects directly
        if hasattr(video_object, '__class__') and 'VideoReader' in str(type(video_object)):
            return load_video_feats_from_decord_reader(
                video_object, train=train,
                image_crop_size=image_crop_size,
                image_mean=image_mean,
                image_std=image_std
            )
        
        # Try to get video path for other types
        video_path = extract_video_path_from_hf_object(video_object)
        
        if video_path and os.path.exists(video_path):
            # Use the existing load_video_feats function
            try:
                from whisper_flamingo.utils import load_video_feats
                return load_video_feats(video_path, train=train, 
                                      image_crop_size=image_crop_size,
                                      image_mean=image_mean, 
                                      image_std=image_std)
            except ImportError:
                print(f"Error loading video features from HF object: {e}")

                traceback.print_exc()
                raise e 
                                                   
    except Exception as e:
        print(f"Error loading video features from HF object: {e}")
        traceback.print_exc()
        raise e

def debug_hf_video_object(video_object: Any, idx: int = 0) -> None:
    """
    Debug utility to inspect HuggingFace Video objects.
    
    Args:
        video_object: HuggingFace Video object to inspect
        idx: Index of the sample (for logging purposes)
    """
    # print(f"\n--- Debug HF Video Object (sample {idx}) ---")
    # print(f"Type: {type(video_object)}")
    
    if hasattr(video_object, '__dict__'):
        print(f"Attributes: {list(video_object.__dict__.keys())}")
    else:
        print(f"Dir: {[attr for attr in dir(video_object) if not attr.startswith('_')]}")
    
    # Check common attributes
    for attr in ['path', 'array', 'frames', 'filename', 'file_path', 'source']:
        if hasattr(video_object, attr):
            value = getattr(video_object, attr)
            print(f"  {attr}: {type(value)} - {str(value)[:100]}")
    
    # Try to convert to string
    try:
        print(f"String representation: {str(video_object)[:200]}")
    except:
        print("Could not convert to string")
    
    # print("--- End Debug ---\n") 

def validate_hf_video_object(video_object: Any, max_retries: int = 2) -> bool:
    """
    Validate if a HuggingFace Video object can be loaded without errors.
    
    Args:
        video_object: HuggingFace Video object from dataset
        max_retries: Number of retries for validation
        
    Returns:
        bool: True if video can be loaded successfully, False otherwise
    """
    if video_object is None:
        return False
        
    try:
        # Try to access basic properties first
        if hasattr(video_object, '__class__') and 'VideoReader' in str(type(video_object)):
            # For decord.VideoReader objects
            try:
                # Try to get basic info without reading frames
                if hasattr(video_object, '__len__'):
                    video_length = len(video_object)
                    if video_length <= 0:
                        return False
                
                # Try to read one frame to verify the video is actually readable
                if hasattr(video_object, '__getitem__'):
                    try:
                        _ = video_object[0]  # Try to read first frame
                        return True
                    except (RuntimeError, IndexError, OSError) as e:
                        return False
                        
            except Exception as e:
                return False
        
        # For other video object types, try basic validation
        elif hasattr(video_object, 'path'):
            video_path = video_object.path
            if not os.path.exists(video_path):
                return False
            
            # Quick file size check
            if os.path.getsize(video_path) < 1024:  # Less than 1KB is likely corrupted
                return False
                
            return True
            
        elif hasattr(video_object, '_filename'):
            video_path = video_object._filename
            if not os.path.exists(video_path):
                return False
                
            if os.path.getsize(video_path) < 1024:
                return False
                
            return True
        
        # If we can't determine the type, assume it might be valid
        return True
        
    except Exception as e:
        # Any exception during validation means the video is not usable
        return False


def safe_load_video_feats_from_hf_object(video_object: Any, train: bool = False, 
                                        image_crop_size: int = 88, 
                                        image_mean: float = 0.421,
                                        image_std: float = 0.165,
                                        max_frames: int = 1500) -> Optional[np.ndarray]:
    """
    Safely load video features from HuggingFace Video object with error handling.
    
    Args:
        video_object: HuggingFace Video object from dataset
        train: Whether in training mode (affects processing)
        image_crop_size: Size to crop video frames
        image_mean: Mean for normalization
        image_std: Standard deviation for normalization 
        max_frames: Maximum number of frames to load
        
    Returns:
        np.ndarray or None: Video features (T, H, W, C) or None if failed
    """
    try:
        # First validate the video
        if not validate_hf_video_object(video_object):
            return None
            
        # Try the original loading function
        return load_video_feats_from_hf_object(
            video_object, train, image_crop_size, image_mean, image_std
        )
        
    except Exception as e:
        # Log the error but don't crash
        print(f"Warning: Failed to load video features: {e}")
        return None


def create_robust_video_filter(dataset, video_column='video', progress_callback=None):
    """
    Create a robust filter function that identifies valid video samples.
    
    Args:
        dataset: HuggingFace dataset
        progress_callback: Optional callback for progress reporting
        
    Returns:
        tuple: (valid_indices, corrupted_files_info)
    """
    valid_indices = []
    corrupted_files = []
    
    print(f"🔍 Validating {len(dataset)} video samples...")
    
    for idx in tqdm(range(len(dataset)), desc="Validating videos"):
        try:
            sample = dataset[idx]
            video_object = sample.get(video_column, 'lip_video')
            
            if video_object is None:
                corrupted_files.append({
                    'index': idx,
                    'reason': 'video_object_is_none',
                    'file': 'unknown'
                })
                continue
                
            # Validate the video
            if validate_hf_video_object(video_object):
                valid_indices.append(idx)
            else:
                # Try to get file info for reporting
                file_info = "unknown"
                try:
                    file_info = extract_video_path_from_hf_object(video_object) or "path_unknown"
                except:
                    pass
                    
                corrupted_files.append({
                    'index': idx,
                    'reason': 'video_validation_failed',
                    'file': file_info
                })
                
        except Exception as e:
            # Extract file path if possible for better error reporting
            file_info = "unknown"
            try:
                sample = dataset[idx]
                video_object = sample.get(video_column, 'lip_video')
                if video_object:
                    file_info = extract_video_path_from_hf_object(video_object) or str(type(video_object))
            except:
                pass
                
            corrupted_files.append({
                'index': idx,
                'reason': f'exception: {str(e)}',
                'file': file_info
            })
            
        if progress_callback and idx % 100 == 0:
            progress_callback(idx, len(dataset), len(valid_indices), len(corrupted_files))
    
    print(f"✅ Validation complete: {len(valid_indices)} valid, {len(corrupted_files)} corrupted")
    
    return valid_indices, corrupted_files 