#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the AVHuBERT model with a sequence-to-sequence architecture for
audio-visual speech recognition on the AMI dataset.

This script supports loading configurations from YAML files. You can provide a YAML
configuration file in one of two ways:
1. As a direct argument: ./finetune_avhubert_seq2seq.py config/avhubert_large.yaml
2. With the --config_yaml flag: ./finetune_avhubert_seq2seq.py --config_yaml config/avhubert_large.yaml

Command-line arguments will override values specified in the YAML configuration.
"""
import json
import logging
import os
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import evaluate
import torch
import transformers
import torch.nn.functional as F
from transformers import (

    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,

    Speech2TextTokenizer,   
    EarlyStoppingCallback,

    # FIXME: Uncommented for Custom Processor-----------------------
    # Wav2Vec2FeatureExtractor,
    # Wav2Vec2Processor,
    ProcessorMixin,
    # AutoImageProcessor,
    #---------------------------------------------------------------
)

from transformers.trainer_utils import get_last_checkpoint

from utils import (
    load_audio_features,
    load_video_features,
    TARGET_AUDIO_SR,
    VIDEO_CROP_SIZE,
)

#------------------------------Self-implemented models----------------------------------------------------------
# from config.av_hubert_config import AVHuBERTConfig
# from models.av_hubert_seq2seq_model import AVHuBERTForConditionalGeneration
#-----------------------------Fall-back models------------------------------------------------------------------
#- FIXME: Use this if the previous doesnt work
from avhubert.src.model.avhubert2text import AV2TextForConditionalGeneration 
from avhubert.src.model.av2text_config import AV2TextConfig
#----------------------------------------------------------------------------------------------------------

import datasets
from datasets import DatasetDict, Dataset, load_from_disk

# Import missing modules for YAML support
import yaml

logger = logging.getLogger(__name__)

AVHUBERT_PRETRAINED_TOKENIZER_PATH = os.path.join("checkpoints", "hf-avhubert")  #nguyenvulebinh/AV-HuBERT-MuAViC-en"/ "vumichien/AV-HuBERT"
AVHUBERT_PRETRAINED_CONFIG_NAME = os.path.join("checkpoints", "hf-avhubert", "config.json")
HUBERT_PRETRAINED_FEATURE_EXTRACTOR_PATH = os.path.join("checkpoints", "hf-hubert", "hubert-large-ls960-ft")
RESNET_PRETRAINED_FEATURE_EXTRACTOR_PATH = os.path.join("checkpoints", "hf-resnet", "resnet")

# --------------------------------------------------------------------------
# Combined AudioVisualProcessor for Hugging Face
# - FIXME: NOT USED
class AudioVisualProcessor(ProcessorMixin):
    """
    Hugging Face processor that bundles an audio feature extractor, a video
    feature extractor and a tokenizer so the trainer can treat them as a
    single unit.
    """
    attributes = ["audio_feature_extractor", "video_feature_extractor", "tokenizer"]
    model_input_names = ["input_values", "video", "attention_mask"]

    def __init__(self, audio_feature_extractor, video_feature_extractor, tokenizer):
        self.audio_feature_extractor = audio_feature_extractor
        self.video_feature_extractor = video_feature_extractor
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # unified call – returns a merged dict for the modalities that were
    # provided.  Keys follow HF conventions: `input_values` for audio and
    # `pixel_values` for video.
    # ------------------------------------------------------------------ #
    def __call__(self, audio=None, video=None, sampling_rate=None,
                 return_tensors=None, **kwargs):
        out = {}
        if audio is not None:
            processed_audio = self.audio_feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                return_tensors=return_tensors
            )
            out.update(processed_audio)

        if video is not None:
            processed_video = self.video_feature_extractor(
                video,
                return_tensors=return_tensors
            )
            # Set key to "video" to match AVHubertModel's expected input
            if "pixel_values" in processed_video:
                out["video"] = processed_video["pixel_values"]
            else:
                # If the processor uses a different key, still set it to "video"
                # for the model's forward method
                out["video"] = next(iter(processed_video.values()))

        return out

    # delegate common helpers to the tokenizer so that generation/decoding
    # work transparently
    def pad(self, *args, **kwargs):
        return self.tokenizer.pad(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
#---------------------------------------------------------------------------------------------------------------------

@dataclass
class DataCollatorForAVSeq2Seq:
    tokenizer: Speech2TextTokenizer
    decoder_start_token_id: int # Should be config.decoder_start_token_id
    # Add these parameters that were referenced but not initialized
    max_audio_length: Optional[int] = None # Max time steps for audio features [F, T_audio]
    max_video_frames: Optional[int] = None # Max time steps for video features [C, T_video, H, W]
    pad_to_multiple_of: Optional[int] = None # General padding utility
    default_audio_F: int = 104  # logfbank with stack_order=4 has 26*4=104 features
    default_audio_T: int = 100  # arbitrary small default for audio time dimension
    default_video_C: int = 1    # default channels for video (grayscale)
    default_video_H: int = 88   # default height for video (VIDEO_CROP_SIZE)
    default_video_W: int = 88   # default width for video (VIDEO_CROP_SIZE)

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = {}
        logger.debug(f"DataCollator processing batch with {len(features)} examples")
        
        has_audio = any(feature.get("input_values") is not None for feature in features)
        has_visual = any(feature.get("video") is not None for feature in features)
        
        logger.debug(f"Batch has audio: {has_audio}, has visual: {has_visual}")

        # Process Labels first (as they are always present)
        label_features = [torch.tensor(feature["labels"]) for feature in features]
        
        # Pad labels dynamically to the longest sequence in the batch
        labels_padded = self.tokenizer.pad(
            {"input_ids": label_features}, 
            padding=True, # Pad to longest in batch
            return_tensors="pt"
        )
        batch["labels"] = labels_padded.input_ids.masked_fill(labels_padded.attention_mask == 0, -100)
        batch["attention_mask"] = labels_padded.attention_mask # This is for the decoder
        
        logger.debug(f"Processed labels with shape: {batch['labels'].shape}")

        # Process Audio
        if has_audio:
            logger.debug("Processing audio features...")
            audio_tensors = [feature.get("input_values") for feature in features]
            # Filter out None values and create placeholders for them
            processed_audio_tensors = []
            audio_lengths = []
            valid_audio_tensors = [t for t in audio_tensors if t is not None]
            
            logger.debug(f"Found {len(valid_audio_tensors)}/{len(audio_tensors)} valid audio tensors")
            
            # If no valid tensors, use defaults
            if not valid_audio_tensors:
                # Use the defaults from the class
                default_audio_F = self.default_audio_F  
                default_audio_T = self.max_audio_length if self.max_audio_length else self.default_audio_T
                logger.debug(f"Using default audio shapes: F={default_audio_F}, T={default_audio_T}")
            else:
                # Use shape from the first valid tensor
                default_audio_F = valid_audio_tensors[0].shape[0]
                default_audio_T = valid_audio_tensors[0].shape[1] if valid_audio_tensors[0].shape[1] > 0 else self.default_audio_T
                logger.debug(f"Using shape from first valid audio tensor: F={default_audio_F}, T={default_audio_T}")
            
            for idx, tensor in enumerate(audio_tensors):
                if tensor is None:
                    # Create a zero tensor with the right shape
                    processed_audio_tensors.append(torch.zeros((default_audio_F, default_audio_T), dtype=torch.float32))
                    audio_lengths.append(0) # Mark as zero length for attention mask
                else:
                    processed_audio_tensors.append(tensor) # Shape [F, T]
                    audio_lengths.append(tensor.shape[1]) # Length is T
            
            # Pad the time dimension (dim 1 for [F, T])
            max_len = max(tensor.shape[1] for tensor in processed_audio_tensors)
            logger.debug(f"Padding audio tensors to max length {max_len}")
            padded_audio_list = []
            
            for tensor in processed_audio_tensors:
                if tensor.shape[1] < max_len:
                    padding = torch.zeros((tensor.shape[0], max_len - tensor.shape[1]), dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, padding], dim=1)
                else:
                    padded_tensor = tensor
                padded_audio_list.append(padded_tensor)
            
            # Stack to create batch dimension
            batch["input_values"] = torch.stack(padded_audio_list)
            logger.debug(f"Final audio tensor shape: {batch['input_values'].shape}")
            
            # Create audio_attention_mask
            audio_attention_mask = torch.zeros(len(features), max_len, dtype=torch.long)
            for i, length in enumerate(audio_lengths):
                if length > 0:  # Only set mask for real data (not placeholders)
                    audio_attention_mask[i, :length] = 1
            batch["audio_attention_mask"] = audio_attention_mask
            logger.debug(f"Audio attention mask shape: {audio_attention_mask.shape}")
        else:
            batch["input_values"] = None
            batch["audio_attention_mask"] = None
            logger.debug("Skipped audio processing (no valid audio features)")

        # Process Video
        if has_visual:
            logger.debug("Processing video features...")
            video_tensors = [feature.get("video") for feature in features]
            processed_video_tensors = []
            video_lengths = [] # Temporal lengths
            
            # Get shapes from valid tensors or use defaults
            valid_video_tensors = [t for t in video_tensors if t is not None]
            logger.debug(f"Found {len(valid_video_tensors)}/{len(video_tensors)} valid video tensors")
            
            if not valid_video_tensors:
                # Use the defaults from the class
                # Using default channels (C), timespans (T), height (H), width (W)
                default_C = self.default_video_C  
                default_video_T = self.max_video_frames if self.max_video_frames else self.default_audio_T
                default_H = self.default_video_H
                default_W = self.default_video_W
                logger.debug(f"Using default video shapes: C={default_C}, T={default_video_T}, H={default_H}, W={default_W}")
            else:
                # Use shape from the first valid tensor
                default_C = valid_video_tensors[0].shape[0]
                default_video_T = valid_video_tensors[0].shape[1] if valid_video_tensors[0].shape[1] > 0 else self.default_audio_T
                default_H = valid_video_tensors[0].shape[2]
                default_W = valid_video_tensors[0].shape[3]
                logger.debug(f"Using shape from first valid video tensor: C={default_C}, T={default_video_T}, H={default_H}, W={default_W}")

            for tensor in video_tensors:
                if tensor is None:
                    processed_video_tensors.append(torch.zeros((default_C, default_video_T, default_H, default_W), dtype=torch.float32))
                    video_lengths.append(0)
                else:
                    processed_video_tensors.append(tensor) # Shape [C, T, H, W]
                    video_lengths.append(tensor.shape[1]) # Length is T

            # Find max temporal dimension in batch
            max_t_len = max(tensor.shape[1] for tensor in processed_video_tensors)
            logger.debug(f"Padding video tensors to max temporal length {max_t_len}")
            padded_videos_list = []
            
            # Pad each video to max length
            for tensor in processed_video_tensors:
                c, t, h, w = tensor.shape
                if t < max_t_len:
                    padding = torch.zeros((c, max_t_len - t, h, w), dtype=tensor.dtype)
                    padded_tensor = torch.cat([tensor, padding], dim=1)
                else:
                    padded_tensor = tensor
                padded_videos_list.append(padded_tensor)
            
            batch["video"] = torch.stack(padded_videos_list)
            logger.debug(f"Final video tensor shape: {batch['video'].shape}")

            # Create visual_attention_mask
            visual_attention_mask = torch.zeros(len(features), max_t_len, dtype=torch.long)
            for i, length in enumerate(video_lengths):
                if length > 0:  # Only set mask for real data (not placeholders)
                    visual_attention_mask[i, :length] = 1
            batch["visual_attention_mask"] = visual_attention_mask
            logger.debug(f"Video attention mask shape: {visual_attention_mask.shape}")
        else:
            batch["video"] = None
            batch["visual_attention_mask"] = None
            logger.debug("Skipped video processing (no valid video features)")
            
        logger.debug("DataCollator processing complete")
        return batch

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default=AVHUBERT_PRETRAINED_TOKENIZER_PATH, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=AVHUBERT_PRETRAINED_CONFIG_NAME, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    config_yaml: Optional[str] = field(
        default="config/avhubert_large.yaml", metadata={"help": "Path to YAML configuration file. Overrides other config options when provided."}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Feature extractor name or path"}
    )
    processor_name: Optional[str] = field(
        default=None, metadata={"help": "Processor name or path"}
    )
    cache_dir: Optional[str] = field(
        default="./cache/avhubert", metadata={"help": "Where to store the pretrained models"}
    )
    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": "Whether to use one of the fast tokenizer"}
    )
    model_revision: str = field(
        default="main", metadata={"help": "The specific model version to use"}
    )
    use_auth_token: bool = field(
        default=False, metadata={"help": "Will use token for HF authentication"}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the encoder"}
    )
    freeze_decoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the decoder"}
    )
    use_audio: bool = field(
        default=True, metadata={"help": "Whether to use audio modality"}
    )
    use_visual: bool = field(
        default=True, metadata={"help": "Whether to use visual modality"}
    )
    fusion_type: str = field(
        default="concat", metadata={"help": "How to fuse audio and visual features"}
    )
    debug_mode: bool = field(
        default=False, metadata={"help": "Enable debug logging for troubleshooting"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default="ami", metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use"}
    )
    dataset_cache_dir: Optional[str] = field(
        default="data", metadata={"help": "Cache directory for the dataset."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # -------------------------------------------------------------------------
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for preprocessing."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Max samples to use for training (debugging)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Max samples to use for evaluation (debugging)."}
    )

    # -------------------------------------------------------------------------
    audio_column_name: str = field(
        default="audio", metadata={"help": "The name of the dataset column containing the audio path"}
    )
    video_column_name: str = field(
        default="lip_video", metadata={"help": "The name of the dataset column containing the lip video path"}
    )
    text_column_name: str = field(
        default="transcript", metadata={"help": "The name of the dataset column containing the text"}
    )

    # -------------------------------------------------------------------------
    train_split_name: str = field(
        default="train", metadata={"help": "The name of the training data split"}
    )
    eval_split_name: str = field(
        default="validation", metadata={"help": "The name of the evaluation data split"}
    )
    test_split_name: str = field(
        default="test", metadata={"help": "The name of the test data split"}
    )

    # -------------------------------------------------------------------------
    max_source_length: Optional[int] = field(
        default=1024, metadata={"help": "Max sequence length for encoder inputs"}
    )
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Max sequence length for decoder outputs"}
    )
    num_beams: Optional[int] = field(
        default=None, metadata={"help": "Number of beams to use for evaluation. Default to 4 if not set, from training_args"} # Default from training_args
    )
    audio_drop_prob: float = field(
        default=0.0, metadata={"help": "Probability of dropping audio during training for robustness"}
    )
    visual_drop_prob: float = field(
        default=0.0, metadata={"help": "Probability of dropping visual during training for robustness"}
    )
    max_duration_in_seconds: float = field(
        default=30.0, metadata={"help": "Maximum audio duration in seconds"}
    )
    # -------------------------------------------------------------------------

    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Whether to ignore pad tokens in loss computation"}
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text"}
    )
    forced_decoder_ids: Optional[List[List[int]]] = field(
        default=None,
        metadata={
            "help": "A list of pairs of integers which indicates a mapping from generation indices to token indices "
            "that will be forced before sampling. For example, [[0, 123]] means the first generated token "
            "will always be a token of index 123."
        },
    )
    suppress_tokens: Optional[List[int]] = field(
        default=None,
        metadata={"help": "A list of tokens that will be suppressed at generation."},
    )
    audio_drop_prob: float = field(
        default=0.5, metadata={"help": "Probability of dropping audio during training for robustness"}
    )
    visual_drop_prob: float = field(
        default=0.5, metadata={"help": "Probability of dropping visual during training for robustness"}
    )
   
def validate_model_inputs(batch, config):
    """
    Validates that the prepared batch data matches the expected input format for the model.
    Logs warnings if there are potential incompatibilities.
    
    Args:
        batch: The batch of data prepared by the data collator
        config: The model configuration
    """
    if "input_values" in batch and batch["input_values"] is not None:
        # Check audio features shape
        if len(batch["input_values"].shape) != 3:
            logger.warning(f"Unexpected audio shape: {batch['input_values'].shape}. Expected [B, F, T]")
        
        # Check audio feature dimensions match config
        if batch["input_values"].shape[1] != config.audio_feat_dim:
            logger.warning(f"Audio feature dimension {batch['input_values'].shape[1]} doesn't match config.audio_feat_dim ({config.audio_feat_dim})")
    
    if "video" in batch and batch["video"] is not None:
        # Check video features shape
        if len(batch["video"].shape) != 5:
            logger.warning(f"Unexpected video shape: {batch['video'].shape}. Expected [B, C, T, H, W]")
        
        # Check video dimensions
        if batch["video"].shape[2] != batch["visual_attention_mask"].shape[1]:
            logger.warning(f"Video temporal dimension {batch['video'].shape[2]} doesn't match visual_attention_mask length {batch['visual_attention_mask'].shape[1]}")

    if "labels" in batch and batch["labels"] is not None:
        # Check if labels exceed vocab size
        if batch["labels"].max() >= config.vocab_size:
            logger.warning(f"Labels contain indices >= vocab_size ({config.vocab_size}). Max label index: {batch['labels'].max()}")
            
    # Check for attention mask consistency
    if "input_values" in batch and batch["input_values"] is not None and "audio_attention_mask" in batch:
        if batch["input_values"].shape[0] != batch["audio_attention_mask"].shape[0]:
            logger.warning(f"Batch size mismatch between input_values ({batch['input_values'].shape[0]}) and audio_attention_mask ({batch['audio_attention_mask'].shape[0]})")
        if batch["input_values"].shape[2] != batch["audio_attention_mask"].shape[1]:
            logger.warning(f"Sequence length mismatch between input_values ({batch['input_values'].shape[2]}) and audio_attention_mask ({batch['audio_attention_mask'].shape[1]})")

# Add this method to properly handle YAML configuration loading ---------------------------------------------------------
def load_config_from_yaml(yaml_path):
    """
    Load configuration parameters from a YAML file and convert them to an AV2TextConfig.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        AV2TextConfig: Configuration object with parameters from YAML
    """
    logger.info(f"Loading configuration from YAML file: {yaml_path}")
    try:
        with open(yaml_path, 'r') as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
        
        # Create a new config with default values
        config = AV2TextConfig()
        
        # Update config with YAML values
        # Map YAML keys to config attributes
        yaml_to_config_mapping = {
            # Common fields - direct mapping
            'vocab_size': 'vocab_size',
            'encoder_layers': 'encoder_layers',
            'encoder_ffn_dim': 'encoder_ffn_dim',
            'encoder_attention_heads': 'encoder_attention_heads',
            'decoder_layers': 'decoder_layers',
            'decoder_ffn_dim': 'decoder_ffn_dim',
            'decoder_attention_heads': 'decoder_attention_heads',
            'd_model': 'd_model',
            'encoder_hidden_size': 'encoder_hidden_size',
            'decoder_hidden_size': 'decoder_hidden_size',
            'dropout': 'dropout',
            'attention_dropout': 'attention_dropout',
            'activation_dropout': 'activation_dropout',
            
            # AV-specific fields
            'use_audio': 'use_audio',
            'use_visual': 'use_visual',
            'fusion_type': 'fusion_type',
            'audio_feat_dim': 'audio_feat_dim',
            
            # Other fields with different names
            'model.encoder_embed_dim': 'd_model',
            'model.audio_feat_dim': 'audio_feat_dim',
        }
        
        # Update config with YAML values according to mapping
        flat_yaml = {}
        
        # Flatten nested YAML structure
        def flatten_dict(d, parent_key=''):
            for k, v in d.items():
                key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, key)
                else:
                    flat_yaml[key] = v
        
        # Flatten the YAML structure
        flatten_dict(yaml_config)
        
        # Update config with flattened values
        for yaml_key, config_attr in yaml_to_config_mapping.items():
            if yaml_key in flat_yaml:
                setattr(config, config_attr, flat_yaml[yaml_key])
                logger.info(f"Setting config.{config_attr} = {flat_yaml[yaml_key]} from YAML")
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading YAML configuration: {e}")
        logger.info("Falling back to default configuration")
        return AV2TextConfig()
#------------------------------------------------------------------------------------------------------------------------

def prepare_dataset(
        example,
        tokenizer,
        data_args,
        model_args,
        is_training=True
    ):
    """
    Prepares a single example for the AV-HuBERT model when using dataset.map() with batched=False.
    Extracts audio and video features in the format expected by AV2TextForConditionalGeneration.
    """
    try:
        # Initial debug info
        logger.debug(f"Processing example with columns: {list(example.keys())}")
        
        # When batched=False, example is a single item, not a batch
        audio_item = example.get(data_args.audio_column_name) # example["audio"]
        video_item = example.get(data_args.video_column_name) # example["lip_video"]
        transcript = example.get(data_args.text_column_name, "") # example["transcript"]
        
        logger.debug(f"Audio item: {type(audio_item)}, Video item: {type(video_item)}, Transcript: '{transcript[:30]}...'")
        
        if transcript is None or transcript == "":
            logger.warning(f"Empty transcript found in example. Using empty string.")
            transcript = ""

        # Parse audio and video paths, handling potentially missing or invalid entries
        current_audio_path = None
        if audio_item is not None:
            current_audio_path = audio_item["path"] if isinstance(audio_item, dict) and "path" in audio_item else audio_item
            if current_audio_path and not isinstance(current_audio_path, str):
                logger.warning(f"Invalid audio path type: {type(current_audio_path)}. Setting to None.")
                current_audio_path = None
            
        current_video_path = None
        if video_item is not None:
            current_video_path = video_item["path"] if isinstance(video_item, dict) and "path" in video_item else video_item
            if current_video_path and not isinstance(current_video_path, str):
                logger.warning(f"Invalid video path type: {type(current_video_path)}. Setting to None.")
                current_video_path = None
                
        logger.debug(f"Audio path: {current_audio_path}, Video path: {current_video_path}")

        # Modality dropping for training
        original_audio_path = current_audio_path
        original_video_path = current_video_path
        
        if is_training:
            if current_audio_path and model_args.use_audio and np.random.random() < data_args.audio_drop_prob:
                current_audio_path = None
            if current_video_path and model_args.use_visual and np.random.random() < data_args.visual_drop_prob:
                current_video_path = None
            # Ensure at least one modality is present if both were initially available and dropped
            if not current_audio_path and not current_video_path:
                if original_audio_path:
                    current_audio_path = original_audio_path
                    logger.debug(f"Restored dropped audio to ensure at least one modality")
                elif original_video_path:
                    current_video_path = original_video_path
                    logger.debug(f"Restored dropped video to ensure at least one modality")
                    
        # If both modalities were dropped, log a message
        if original_audio_path and original_video_path and not current_audio_path and not current_video_path:
            logger.debug("Both modalities were dropped")

        # Process audio
        audio_feats = None
        if current_audio_path and model_args.use_audio:
            try:
                # Load audio features and convert to tensor
                audio_feats_np = load_audio_features(current_audio_path, target_sr=TARGET_AUDIO_SR)
                if audio_feats_np is not None:
                    audio_feats = torch.from_numpy(audio_feats_np.astype(np.float32))
                    logger.debug(f"Loaded audio features with shape: {audio_feats.shape}")
                    # Layer norm as in original AV-HuBERT's load_feature
                    with torch.no_grad():
                        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
                    # Expected shape by AV2TextForConditionalGeneration: [F, T] for single example
                    audio_feats = audio_feats.transpose(0, 1)  # From [T, F] to [F, T]
                    logger.debug(f"Transposed audio features to shape: {audio_feats.shape}")
                else:
                    logger.warning(f"Failed to load audio features from {current_audio_path}")
            except Exception as e:
                logger.error(f"Error loading audio {current_audio_path}: {e}")
                audio_feats = None
        
        # Process video
        video_feats = None
        if current_video_path and model_args.use_visual:
            try:
                # Load video features and convert to tensor
                video_feats_np = load_video_features(current_video_path)
                if video_feats_np is not None:
                    video_feats = torch.from_numpy(video_feats_np.astype(np.float32))
                    logger.debug(f"Loaded video features with shape: {video_feats.shape}")
                    # Expected shape: [C, T, H, W]
                    video_feats = video_feats.permute(3, 0, 1, 2)  # From [T, H, W, C] to [C, T, H, W]
                    logger.debug(f"Permuted video features to shape: {video_feats.shape}")
                else:
                    logger.warning(f"Failed to load video features from {current_video_path}")
            except Exception as e:
                logger.error(f"Error loading video {current_video_path}: {e}")
                video_feats = None

        # Check if at least one modality was loaded successfully
        if audio_feats is None and video_feats is None:
            logger.warning(f"Both audio and video features are None. This example will be processed with empty features.")

        # Align audio and video lengths (temporal dimension T) if both are available
        if audio_feats is not None and video_feats is not None:
            # audio_feats: [F, T_audio], video_feats: [C, T_video, H, W]
            t_audio = audio_feats.shape[1]
            t_video = video_feats.shape[1]
            diff = t_audio - t_video
            
            logger.debug(f"Aligning audio ({t_audio}) and video ({t_video}) temporal dimensions")
            
            # Log significant discrepancies between modality lengths
            if abs(diff) > min(t_audio, t_video) * 0.5:  # If difference is more than 50% of the shorter modality
                logger.warning(f"Large difference between audio ({t_audio}) and video ({t_video}) lengths.")
                
            if diff < 0: # video is longer
                # Pad audio features (pad the time dimension, which is dim 1)
                padding = torch.zeros((audio_feats.shape[0], -diff), dtype=audio_feats.dtype)
                audio_feats = torch.cat([audio_feats, padding], dim=1)
                logger.debug(f"Padded audio features to match video length: {audio_feats.shape}")
            elif diff > 0: # audio is longer
                # Truncate audio features to match video length
                audio_feats = audio_feats[:, :t_video]
                logger.debug(f"Truncated audio features to match video length: {audio_feats.shape}")

        # Tokenize transcript
        labels = tokenizer(transcript).input_ids
        logger.debug(f"Tokenized transcript with {len(labels)} tokens")

        # Return a dictionary for a single example
        return {
            "input_values": audio_feats,  # For audio, expected by model
            "video": video_feats,        # For video, expected by model
            "labels": labels,
        }
    except Exception as e:
        logger.error(f"Unexpected error processing example: {e}")
        # Return a placeholder with empty features but valid labels to avoid breaking the dataset
        return {
            "input_values": None,
            "video": None,
            "labels": tokenizer("").input_ids,  # Empty transcript
        }

def main():
    #=================================================================================================================
    #                               SETUP DEVICE
    #=================================================================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")


    #=================================================================================================================
    #                               PARSING ARGUMENTS
    #=================================================================================================================
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2:
        if sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        elif sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml"):
            # If we pass only one argument to the script and it's the path to a yaml file,
            # let's treat it as a config_yaml argument.
            yaml_path = os.path.abspath(sys.argv[1])
            logger.info(f"Using YAML configuration file: {yaml_path}")
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
            model_args.config_yaml = yaml_path
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #=================================================================================================================================================
    #                                       SETUP LOGGING
    #=================================================================================================================================================
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Set debug mode based on command line argument
    if model_args.debug_mode:
        log_level = logging.DEBUG
        logger.info("Debug mode enabled - setting log level to DEBUG")
    else:
        log_level = training_args.get_process_log_level()
        
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # ----------------------------------------------------------------------------------------------------------

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    #----------------------------------------------------------------------------------------------------------------
    # Set seed before initializing model
    set_seed(training_args.seed)

    #================================================================================================================
    #                                  SETUP CONFIG
    #================================================================================================================

    # Load config
    if model_args.config_yaml and os.path.exists(model_args.config_yaml):
        logger.info(f"Loading configuration from YAML file {model_args.config_yaml}")
        config = load_config_from_yaml(model_args.config_yaml)
    elif model_args.config_name:
        config = AV2TextConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AV2TextConfig()
        logger.info("You are instantiating a default `AV2TextConfig` instance")

    # Update config with model args (command-line args override YAML config)
    # Only override if the arguments were explicitly provided on command line
    for arg_name, arg_value in vars(model_args).items():
        if arg_name in ['use_audio', 'use_visual', 'fusion_type'] and arg_value is not None:
            setattr(config, arg_name, arg_value)
            logger.info(f"Overriding config.{arg_name} with command-line value: {arg_value}")
            
    # Log the effective configuration
    logger.info(f"Using config: {config}")
    logger.info(f"Audio modality: {'enabled' if config.use_audio else 'disabled'}")
    logger.info(f"Visual modality: {'enabled' if config.use_visual else 'disabled'}")
    logger.info(f"Fusion type: {config.fusion_type}")
    # ---------------------------------------------------------------------------------------------------------------------

    # ======================================================================================================================
    #                                        SETUP TOKENIZER
    # ======================================================================================================================
    # TOKENIZER: `Speech2TextTokenizer` - The tokenizer will be loaded from 
    # `AVHUBERT_PRETRAINED_TOKENIZER_PATH`
    # This is the tokenizer generated using the `Speech2TextTokenizer`
    #----------------------------------------------------------------------------------------------------------
    tokenizer = None
    
    # 2. Try loading from model_name_or_path if tokenizer is None
    if tokenizer is None and model_args.model_name_or_path:
        try:
            logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
            tokenizer = Speech2TextTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            logger.info(f"Successfully loaded tokenizer from {model_args.model_name_or_path}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {model_args.model_name_or_path}: {e}")

    # Update config with tokenizer info
    config.vocab_size = len(tokenizer)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.decoder_start_token_id = tokenizer.bos_token_id
    # ----------------------------------------------------------------------------------------------------------
    
    #============================================================================================================
    #                                              SETUP PROCESSOR
    #============================================================================================================
    # - FIXME: CHECK IF IT WORKS.

    # # Ensure output directory exists
    # os.makedirs(training_args.output_dir, exist_ok=True)

    # try:
    #     logger.info(f"Attempting to create processor from feature extractor and tokenizer")

    #     processor = AudioVisualProcessor(
    #         audio_feature_extractor=audio_feature_extractor,
    #         video_feature_extractor=video_feature_extractor,
    #         tokenizer=tokenizer
    #     )
    #     logger.info(f"Successfully loaded processor from {training_args.output_dir}")
    # except Exception as e:
    #     logger.info(f"Failed to load processor from {training_args.output_dir}: {e}. Saving components first.")
    #     # Save the loaded feature extractor and tokenizer to the output directory
    #     logger.info(f"Saving Audio feature extractor to {training_args.output_dir}")
    #     audio_feature_extractor.save_pretrained(training_args.output_dir)
    #     logger.info(f"Saving tokenizer to {training_args.output_dir}")
    #     tokenizer.save_pretrained(training_args.output_dir)
    #     # Now, load the processor from the directory where components were just saved
    #     logger.info(f"Reloading processor from {training_args.output_dir} after saving components.")
    #     # For fallback, we can still build AudioVisualProcessor (video_feature_extractor as above)
    #     video_feature_extractor = AutoImageProcessor.from_pretrained(
    #         RESNET_PRETRAINED_FEATURE_EXTRACTOR_PATH,
    #         cache_dir=model_args.cache_dir,
    #         do_normalize=True,
    #         return_attention_mask=True,
    #         local_files_only=True
    #     )
    #     processor = AudioVisualProcessor(
    #         audio_feature_extractor=audio_feature_extractor,
    #         video_feature_extractor=video_feature_extractor,
    #         tokenizer=tokenizer
    #     )
    #     logger.info(f"Successfully created and loaded processor from components saved in {training_args.output_dir}")

 
    #============================================================================================================
    #                                               SETUP MODEL
    #============================================================================================================
    # Initialize model from pretrained model
    if model_args.model_name_or_path:
        try:
            # First attempt: try loading from Hugging Face model hub
            logger.info(f"Attempting to load pretrained model from {model_args.model_name_or_path}")
            model = AV2TextForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            logger.info(f"Successfully loaded pretrained model from {model_args.model_name_or_path}")
        except Exception as e:
            # Second attempt: try loading from local checkpoint files
            logger.info(f"Failed to load model from {model_args.model_name_or_path}: {e}")
            logger.info("Attempting to load from local checkpoint files...")
            
            # Check if model files exist locally
            local_checkpoint_dir = os.path.expanduser(model_args.model_name_or_path)
            local_weights_file = os.path.join(local_checkpoint_dir, "pytorch_model.bin")
            
            if os.path.exists(local_weights_file):
                try:
                    # Load model with local file protocol
                    model = AV2TextForConditionalGeneration.from_pretrained(
                        local_checkpoint_dir,
                        config=config,
                        local_files_only=True,
                    )
                    logger.info(f"Successfully loaded model from local checkpoint: {local_checkpoint_dir}")
                except Exception as local_err:
                    raise RuntimeError(f"Failed to load model from local checkpoint: {local_err}")
            else:
                # Create a new model with the specified config
                logger.info(f"No checkpoint found at {local_checkpoint_dir}, initializing from scratch")
                model = AV2TextForConditionalGeneration(config)
    else:
        logger.info("Training new model from scratch")
        model = AV2TextForConditionalGeneration(config)
    
    # Freeze components if requested
    if model_args.freeze_encoder:
        model.freeze_encoder()
        logger.info("Encoder parameters frozen")
    
    if model_args.freeze_decoder:
        model.freeze_decoder()
        logger.info("Decoder parameters frozen")
    
    # Resizing token embeddings
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Move model to device
    model.to(device)

    #============================================================================================================
    #                                   SETUP DATASET (LOAD, SPLIT, SAVE)
    #============================================================================================================

    ami_train_path = os.path.join("data", data_args.dataset_name, "av_hubert", "train") # data/ami/av_hubert/train
    ami_val_path = os.path.join("data", data_args.dataset_name, "av_hubert", "validation") # data/ami/av_hubert/validation
    ami_test_path = os.path.join("data", data_args.dataset_name, "av_hubert", "test") # data/ami/av_hubert/test

    # Check if dataset paths exist
    paths_exist = all(os.path.exists(path) for path in [ami_train_path, ami_val_path, ami_test_path])
    
    if not paths_exist:
        logger.warning(f"One or more dataset paths don't exist. Creating directories if needed.")
        # Create dataset directories
        for path in [ami_train_path, ami_val_path, ami_test_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Here you would typically load and process the raw dataset
        # For this implementation, we'll just log an error if the paths don't exist
        if not os.path.exists(ami_train_path):
            raise FileNotFoundError(f"Training dataset not found at {ami_train_path}. Please prepare the dataset first.")

    # Try to load the datasets
    try:
        ami_train = load_from_disk(ami_train_path)
        ami_val = load_from_disk(ami_val_path)
        ami_test = load_from_disk(ami_test_path)
        
        # logger.info dataset info
        logger.info(f"Training dataset: {ami_train}")
        logger.info(f"Validation dataset: {ami_val}")
        logger.info(f"Test dataset: {ami_test}")
    except Exception as e:
        raise RuntimeError(f"Failed to load datasets: {e}")

    
    #============================================================================================================
    #                                  SETUP DATASETS AND PROCESS DATASET
    #============================================================================================================
    logger.info("Processing datasets")
    
    #------------------- PROCESS TRAINING DATASET -------------------
    if training_args.do_train:
        logger.info("Processing training dataset")
        try:
            train_dataset = ami_train.map(
                prepare_dataset,
                fn_kwargs={"tokenizer": tokenizer, "data_args": data_args, "model_args": model_args, "is_training": True},
                batched=False,  # Process one example at a time
                remove_columns=ami_train.column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Processing train dataset",
            )
            logger.info(f"Processed training dataset: {train_dataset}")
        except Exception as e:
            logger.error(f"Failed to process training dataset: {e}")
            raise RuntimeError(f"Failed to process training dataset: {e}")
    else:
        train_dataset = None

    #------------------- PROCESS VALIDATION DATASET -------------------
    if training_args.do_eval:
        logger.info("Processing validation dataset")
        try:
            eval_dataset = ami_val.map(
                prepare_dataset, 
                fn_kwargs={"tokenizer": tokenizer, "data_args": data_args, "model_args": model_args, "is_training": False},
                batched=False,  # Process one example at a time
                remove_columns=ami_val.column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Processing eval dataset",
            )
            logger.info(f"Processed validation dataset: {eval_dataset}")
        except Exception as e:
            logger.error(f"Failed to process validation dataset: {e}")
            raise RuntimeError(f"Failed to process validation dataset: {e}")
    else:
        eval_dataset = None

    
    #============================================================================================================
    #                                               SETUP DATA COLLATOR
    #============================================================================================================
    # Calculate reasonable max lengths based on dataset and configuration
    max_audio_length = int(data_args.max_duration_in_seconds * TARGET_AUDIO_SR / 4)  # Divide by 4 because of stacking
    max_video_frames = int(data_args.max_duration_in_seconds * 25)  # Assuming 25 fps
    
    # Data collator that handles batch creation
    logger.info(f"Initializing data collator with max_audio_length={max_audio_length}, max_video_frames={max_video_frames}")
    data_collator = DataCollatorForAVSeq2Seq(
        tokenizer=tokenizer,
        decoder_start_token_id=config.decoder_start_token_id, # Get from config
        max_audio_length=max_audio_length,
        max_video_frames=max_video_frames,
    )
    
    #============================================================================================================
    #                                               SETUP METRICS
    #============================================================================================================
    # Define compute metrics
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        
        # If predictions are logits, convert to ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
            
        if isinstance(pred_ids, np.ndarray) and len(pred_ids.shape) == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)
        
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        # cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        # Log a few examples for debugging
        if len(pred_str) > 0:
            logger.info("Example predictions:")
            for i in range(min(3, len(pred_str))):
                logger.info(f"  Reference: {label_str[i]}")
                logger.info(f"  Prediction: {pred_str[i]}")
                logger.info("  ---")
        
        return {"wer": wer}
    
    #=================================================================================================================
    #                                               SETUP TRAINER
    #=================================================================================================================
    # Define a callback for validating inputs
    class InputValidationCallback(transformers.TrainerCallback):
        def __init__(self, config):
            self.config = config
            self.validation_count = 0
            
        def on_step_begin(self, args, state, control, **kwargs):
            # Only validate the first 5 batches to avoid excessive logging
            if self.validation_count < 5 and kwargs.get("train_dataloader") is not None:
                try:
                    # Get a batch from the dataloader
                    batch = next(iter(kwargs["train_dataloader"]))
                    # Move to same device as model
                    batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    # Validate the batch
                    validate_model_inputs(batch, self.config)
                    self.validation_count += 1
                except Exception as e:
                    logger.warning(f"Input validation failed: {e}")
    
    # Initialize Seq2SeqTrainer
    # Note: we don't need a processor since feature extraction is done in prepare_dataset
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,  # Just need the tokenizer for generation
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            InputValidationCallback(config),
        ],
    )
    
    # Set generation parameters
    if hasattr(data_args, "generation_max_length"):
        training_args.generation_max_length = data_args.generation_max_length
    elif hasattr(data_args, "max_target_length"):
        training_args.generation_max_length = data_args.max_target_length
        
    if hasattr(data_args, "num_beams"):
        training_args.generation_num_beams = data_args.num_beams
        
    if hasattr(data_args, "forced_decoder_ids") and data_args.forced_decoder_ids:
        training_args.generation_kwargs = {
            "forced_decoder_ids": data_args.forced_decoder_ids
        }
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        
        metrics = trainer.evaluate(
            max_length=data_args.max_target_length,
            num_beams=data_args.num_beams,
            metric_key_prefix="eval",
        )
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        results.update(metrics)
    
    # Write training parameters
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "automatic-speech-recognition"}
        trainer.push_to_hub(**kwargs)
    
    #=================================================================================================================
    #                                               SAVE RESULTS
    #=================================================================================================================
    # Save results
    results_path = os.path.join(training_args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    main()