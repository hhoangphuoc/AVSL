"""
Utilities for configuration management across the codebase.
"""
import os
import sys
import yaml
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
import types

# Import path utils to ensure project structure is available
try:
    from utils.path_utils import (
        PROJECT_ROOT,
        CONFIG_DIR,
        ensure_dir_exists,
        verify_path_exists,
    )
except ImportError:
    # Fallback if utils.path_utils is not available
    # Define minimal versions of required functions
    def ensure_dir_exists(path: str) -> bool:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return os.path.isdir(path)
    
    def verify_path_exists(path: str, description: str = None) -> bool:
        if not os.path.exists(path):
            desc = f"{description} " if description else ""
            print(f"ERROR: {desc}path not found: {path}")
            return False
        return True
    
    # Get project root and config directory
    _this_file = os.path.abspath(__file__)
    _utils_dir = os.path.dirname(_this_file)
    PROJECT_ROOT = os.path.dirname(_utils_dir)
    # CONFIG_DIR = os.path.join(PROJECT_ROOT, 'avsl', 'config')
    CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')


def load_yaml_config(config_path: str) -> Dict:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary with configuration values
    """
    if not verify_path_exists(config_path, "Configuration"):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            print("YAML Config Loaded:", config)
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            raise

def save_yaml_config(config: Dict, output_path: str) -> bool:
    """
    Save a dictionary as a YAML configuration file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
        
    Returns:
        True if successful, False otherwise
    """
    output_dir = os.path.dirname(output_path)
    if not ensure_dir_exists(output_dir):
        return False
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Error saving YAML configuration: {e}")
        return False

def config_to_namespace(config: Dict) -> types.SimpleNamespace:
    """
    Convert a configuration dictionary to a SimpleNamespace object.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SimpleNamespace object with configuration attributes
    """
    return types.SimpleNamespace(**config)

def get_default_config(config_type: str) -> Dict:
    """
    Get default configuration values for a specific configuration type.
    
    Args:
        config_type: Type of configuration ('whisper_flamingo', 'avhubert', etc.)
        
    Returns:
        Dictionary with default configuration values
    """
    defaults = {
        'whisper_flamingo': {
            # Model configuration
            'model_name': 'large-v2',
            'dropout_rate': 0.1,
            'freeze_video_model': True,
            'freeze_video_batch_norm_stats': False,
            'video_projection_train_only': False,
            'enable_gradient_checkpointing': True,
            'precision': 16,
            'lang': 'en',
            
            # Dataset configuration
            'audio_max_length': 240000,
            'dataset_audio_max_length': 320000,
            'max_duration_filter_seconds': 20.0,
            'text_max_length': 350,
            
            # Training configuration
            'batch_size': 2,
            'eval_batch_size': 4,
            'learning_rate': 1e-5,
            'weight_decay': 0.01,
            'adam_epsilon': 1e-8,
            'num_train_steps': 8000,
            'warmup_steps': 1000,
            'gradient_accumulation_steps': 8,
            'num_worker': 8,
            'validate_every_n_batches': 2000,
            'num_devices': 1,
            'sync_batchnorm': True,
            'resume_training': False,
            
            # Audio/Video settings
            'spec_augment': 'ls-basic',
            'prob_use_av': 1.0,
            'use_av_hubert_encoder': True,
            'add_gated_x_attn': 1,
            'av_fusion': 'separate',
            
            # Paths (to be filled based on project structure)
            'train_data_path': '',
            'val_data_path': '',
            'test_data_path': '',
            'train_data_path_original': '',
            'val_data_path_original': '',
            'test_data_path_original': '',
            'video_model_ckpt': '',
            'pt_ckpt': '',
            'download_root': '',
            'log_output_dir': '',
            'check_output_dir': '',
        },
        'laugh_dataset': {
            # Processing configuration
            'extract_lip_videos': True,
            'to_grayscale': True,
            'batch_size': 8,
            'use_shards': True,
            'files_per_shard': 2000,
            'chunked': True,
            'chunk_size': 1000,
            'num_workers': 8,
            
            # Paths (to be filled based on project structure)
            'csv_path': '',
            'output_dir': '',
            'dataset_path': '',
        }
    }
    
    if config_type in defaults:
        return deepcopy(defaults[config_type])
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration dictionary
    """
    merged = deepcopy(base_config)
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override or add value
            merged[key] = deepcopy(value)
    
    return merged

def get_config(config_type: str, config_path: Optional[str] = None) -> Dict:
    """
    Get configuration, merging defaults with values from config_path.
    
    Args:
        config_type: Type of configuration ('whisper_flamingo', 'avhubert', etc.)
        config_path: Optional path to YAML configuration file to override defaults
        
    Returns:
        Configuration dictionary
    """
    # Get default configuration
    config = get_default_config(config_type)
    
    # Override with values from config_path if provided
    if config_path and verify_path_exists(config_path):
        override_config = load_yaml_config(config_path)
        config = merge_configs(config, override_config)
    
    return config

def create_argument_parser(config_type: str) -> argparse.ArgumentParser:
    """
    Create an argument parser for a specific configuration type.
    
    Args:
        config_type: Type of configuration ('whisper_flamingo', 'avhubert', etc.)
        
    Returns:
        ArgumentParser with arguments for the configuration type
    """
    if config_type == 'whisper_flamingo':
        parser = argparse.ArgumentParser(description='Train a Whisper-Flamingo model')
        parser.add_argument('config_yaml', type=str, help='Path to YAML configuration file')
        parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        parser.add_argument('--data_slice', type=float, default=None, 
                          help='Fraction of data to use (0.0-1.0), for debugging')
        parser.add_argument('--gpu', type=int, default=None, help='Specific GPU to use')
        
    elif config_type == 'laugh_dataset':
        parser = argparse.ArgumentParser(description='Create HuggingFace dataset for laughter and fluent speech segments')
        
        parser.add_argument('--csv_path', type=str, 
                          help='Path to ami_laugh_markers.csv file')
        parser.add_argument('--output_dir', type=str,
                          help='Directory to save processed segments')
        parser.add_argument('--dataset_path', type=str,
                          help='Path to save HuggingFace dataset')
        parser.add_argument('--extract_lip_videos', action='store_true', default=True,
                          help='Extract lip videos from video segments')
        parser.add_argument('--no_lip_videos', dest='extract_lip_videos', action='store_false',
                          help='Skip lip video extraction')
        parser.add_argument('--to_grayscale', action='store_true', default=True,
                          help='Convert lip videos to grayscale')
        parser.add_argument('--batch_size', type=int, default=8,
                          help='Batch size for lip extraction')
        parser.add_argument('--use_shards', action='store_true', default=True,
                          help='Use sharded dataset format')
        parser.add_argument('--no_shards', dest='use_shards', action='store_false',
                          help='Use standard dataset format')
        parser.add_argument('--files_per_shard', type=int, default=2000,
                          help='Number of files per shard')
        parser.add_argument('--chunked', action='store_true', default=False,
                          help='Process dataset in chunks with checkpointing')
        parser.add_argument('--chunk_size', type=int, default=1000,
                          help='Number of segments per chunk (for chunked processing)')
        parser.add_argument('--num_workers', type=int, default=None,
                          help='Number of worker processes for parallel lip extraction (default: auto)')
        parser.add_argument('--use_parallel', action='store_true', default=None,
                          help='Force use of parallel processing for lip extraction')
        parser.add_argument('--no_parallel', dest='use_parallel', action='store_false',
                          help='Disable parallel processing for lip extraction')
        parser.add_argument('--max_tasks_per_child', type=int, default=10,
                          help='Maximum tasks per worker process before respawning')
        parser.add_argument('--config', type=str, default=None,
                          help='Path to YAML configuration file (overrides command line arguments)')
    
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    return parser

def parse_args_with_config(config_type: str, args: Optional[List[str]] = None) -> types.SimpleNamespace:
    """
    Parse command line arguments, potentially overridden by configuration file.
    
    Args:
        config_type: Type of configuration ('whisper_flamingo', 'avhubert', etc.)
        args: Optional list of command line arguments (uses sys.argv if None)
        
    Returns:
        SimpleNamespace object with parsed configuration
    """
    parser = create_argument_parser(config_type)
    parsed_args = parser.parse_args(args)
    
    # Get default configuration
    config = get_default_config(config_type)
    
    # Override with values from config file if provided
    if hasattr(parsed_args, 'config') and parsed_args.config:
        if verify_path_exists(parsed_args.config, "Configuration"):
            override_config = load_yaml_config(parsed_args.config)
            config = merge_configs(config, override_config)
    elif hasattr(parsed_args, 'config_yaml') and parsed_args.config_yaml:
        if verify_path_exists(parsed_args.config_yaml, "Configuration"):
            override_config = load_yaml_config(parsed_args.config_yaml)
            config = merge_configs(config, override_config)
    
    # Convert parsed_args to dictionary, excluding None values and config/config_yaml
    args_dict = {
        key: value for key, value in vars(parsed_args).items() 
        if value is not None and key not in ['config', 'config_yaml']
    }
    
    # Override config with command line arguments
    config = merge_configs(config, args_dict)
    
    return config_to_namespace(config)