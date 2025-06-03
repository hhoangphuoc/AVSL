"""
Utilities for path management and validation across the codebase.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

# Project root path - calculate once at module import time
_this_file = os.path.abspath(__file__)
_utils_dir = os.path.dirname(_this_file)
PROJECT_ROOT = os.path.dirname(_utils_dir)

# Common subdirectories
AVSL_DIR = os.path.join(PROJECT_ROOT, 'avsl')
PREPROCESS_DIR = os.path.join(PROJECT_ROOT, 'preprocess')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
UTILS_DIR = _utils_dir
CONFIG_DIR = os.path.join(AVSL_DIR, 'config')
CHECKPOINTS_DIR = os.path.join(AVSL_DIR, 'checkpoints')
LOGS_DIR = os.path.join(AVSL_DIR, 'logs')
MODELS_DIR = os.path.join(AVSL_DIR, 'models')
SCRIPTS_DIR = os.path.join(PREPROCESS_DIR, 'scripts')

# Path verification functions
def ensure_dir_exists(path: str) -> bool:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created, False on failure
    """
    if os.path.exists(path) and os.path.isdir(path):
        return True
    
    try:
        os.makedirs(path, exist_ok=True)
        return os.path.exists(path) and os.path.isdir(path)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False

def verify_path_exists(path: str, description: str = None) -> bool:
    """
    Verify that a path exists and log an error if it doesn't.
    
    Args:
        path: Path to verify
        description: Optional description for error messages
        
    Returns:
        True if path exists, False otherwise
    """
    if not os.path.exists(path):
        path_type = "Directory" if os.path.isdir(path) else "File"
        desc_text = f"{description} " if description else ""
        print(f"❌ ERROR: {desc_text}{path_type} not found: {path}")
        return False
    return True

def verify_writable_dir(dir_path: str, description: str = None) -> bool:
    """
    Verify that a directory exists and is writable.
    
    Args:
        dir_path: Directory path
        description: Optional description for error messages
        
    Returns:
        True if directory exists and is writable, False otherwise
    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            desc_text = f"{description} " if description else ""
            print(f"❌ ERROR: Could not create {desc_text}directory: {dir_path}, error: {e}")
            return False
    
    if not os.path.isdir(dir_path):
        desc_text = f"{description} " if description else ""
        print(f"❌ ERROR: {desc_text}path exists but is not a directory: {dir_path}")
        return False
    
    # Check write permission
    try:
        test_file = os.path.join(dir_path, ".test_write_permission")
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except Exception as e:
        desc_text = f"{description} " if description else ""
        print(f"❌ ERROR: {desc_text}directory is not writable: {dir_path}, error: {e}")
        return False

def get_ami_dataset_paths(dataset_type: str) -> Dict[str, str]:
    """
    Get standard paths for AMI dataset based on dataset type.
    
    Args:
        dataset_type: Type of AMI dataset ('standard' or 'laughter')
        
    Returns:
        Dictionary of paths
    """
    if dataset_type == 'standard':
        base_dir = os.path.join(DATA_DIR, 'ami')
    elif dataset_type == 'laughter':
        base_dir = os.path.join(DATA_DIR, 'ami_laughter')
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    ensure_dir_exists(base_dir)
    
    return {
        'base_dir': base_dir,
        'dataset_dir': os.path.join(base_dir, 'dataset'),
        'audio_dir': os.path.join(base_dir, 'audio_segments'),
        'video_dir': os.path.join(base_dir, 'video_segments'),
        'lip_dir': os.path.join(base_dir, 'lip_segments'),
        'train_dir': os.path.join(base_dir, 'train'),
        'val_dir': os.path.join(base_dir, 'validation'),
        'test_dir': os.path.join(base_dir, 'test'),
        'train_clean_dir': os.path.join(base_dir, 'train_clean'),
        'val_clean_dir': os.path.join(base_dir, 'validation_clean'),
        'test_clean_dir': os.path.join(base_dir, 'test_clean'),
        'records_file': os.path.join(base_dir, 'dataset_records.json')
    }

def get_deepstore_paths() -> Dict[str, str]:
    """
    Get standard paths for deepstore data.
    
    Returns:
        Dictionary of deepstore paths
    """
    base_path = "/deepstore/datasets/hmi/speechlaugh-corpus/ami"
    
    return {
        'base_dir': base_path,
        'transcript_dir': os.path.join(base_path, 'transcripts'),
        'audio_dir': os.path.join(base_path, 'audio_segments'),
        'video_dir': os.path.join(base_path, 'video_segments'),
        'original_videos_dir': os.path.join(base_path, 'video_segments', 'original_videos'),
        'lip_videos_dir': os.path.join(base_path, 'video_segments', 'lips'),
        'laughter_dir': os.path.join(base_path, 'laughter'),
        'laughter_csv': os.path.join(base_path, 'laughter', 'ami_laugh_markers.csv'),
        'disfluency_dir': os.path.join(base_path, 'dsfl'),
        'disfluency_csv': os.path.join(base_path, 'dsfl', 'disfluency_laughter_markers.csv')
    }

def resolve_relative_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Resolve a relative path to absolute path.
    
    Args:
        path: Path to resolve
        base_dir: Base directory for relative paths (default: PROJECT_ROOT)
        
    Returns:
        Absolute path
    """
    if os.path.isabs(path):
        return path
    
    if base_dir is None:
        base_dir = PROJECT_ROOT
    
    return os.path.normpath(os.path.join(base_dir, path))

def get_model_path(model_name: str) -> str:
    """
    Get path for a pretrained model file.
    
    Args:
        model_name: Name of model
        
    Returns:
        Absolute path to model file
    """
    return os.path.join(MODELS_DIR, model_name)

def get_checkpoint_path(model_type: str, run_id: str) -> str:
    """
    Get path for model checkpoint.
    
    Args:
        model_type: Type of model (e.g., 'whisper_flamingo')
        run_id: Specific run identifier
        
    Returns:
        Path to checkpoint directory
    """
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, model_type, run_id)
    ensure_dir_exists(checkpoint_dir)
    return checkpoint_dir

def get_log_path(model_type: str, run_id: str) -> str:
    """
    Get path for log files.
    
    Args:
        model_type: Type of model
        run_id: Specific run identifier
        
    Returns:
        Path to log directory
    """
    log_dir = os.path.join(LOGS_DIR, model_type, run_id)
    ensure_dir_exists(log_dir)
    return log_dir

def build_output_file_path(output_dir: str, segment_id: str, extension: str) -> str:
    """
    Build an output file path for a segment.
    
    Args:
        output_dir: Directory to store output file
        segment_id: Unique segment identifier
        extension: File extension (without the dot)
        
    Returns:
        Full output file path
    """
    ensure_dir_exists(output_dir)
    return os.path.join(output_dir, f"{segment_id}.{extension}")

def log_disk_space(path: str, label: str = "Directory") -> Tuple[float, float]:
    """
    Log disk space usage for a path.
    
    Args:
        path: Path to check
        label: Label for the path in log output
        
    Returns:
        Tuple of (free_space_gb, total_space_gb)
    """
    try:
        import shutil
        usage = shutil.disk_usage(path)
        gb_free = usage.free / (1024**3)
        gb_total = usage.total / (1024**3)
        print(f"Disk space for {label} ({path}): {gb_free:.2f}GB free out of {gb_total:.2f}GB")
        return gb_free, gb_total
    except FileNotFoundError:
        print(f"Warning: Could not check disk space for {label}. Path not found: {path}")
        return 0, 0
    except Exception as e:
        print(f"Warning: Error checking disk space for {label} ({path}): {e}")
        return 0, 0