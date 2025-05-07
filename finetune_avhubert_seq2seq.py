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
    # VIDEO_MEAN,
    # VIDEO_STD,
)

from utils.data_loading import AudioVisualProcessor

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
    tokenizer: Speech2TextTokenizer  # Using tokenizer directly since we don't need a processor for feature extraction
    decoder_start_token_id: int
    padding: Union[bool, str] = True
    max_audio_length: Optional[int] = None
    max_video_frames: Optional[int] = None
    max_label_length: Optional[int] = None
    pad_to_multiple_of_audio: Optional[int] = None
    pad_to_multiple_of_visual: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    ignore_pad_token_for_loss: bool = True

    def __init__(self, tokenizer: Speech2TextTokenizer, decoder_start_token_id: int):
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Custom collator that properly handles audio and video inputs formatted specifically
        for AV2TextForConditionalGeneration. It expects inputs in the format produced by
        load_feature or our modified prepare_dataset function.
        """
        batch_audio_values = []
        batch_visual_values = []
        batch_labels = []

        #--------------------------------------------------------------------------------------
        # GET FEATURES FROM DATASET
        #-------------------------------------------------------------------------------------- 
        has_audio = any(feature.get("input_values") is not None for feature in features)
        has_visual = any(feature.get("video") is not None for feature in features)

        for feature in features:
            # Get labels
            if isinstance(feature["labels"], list):
                batch_labels.append(torch.tensor(feature["labels"]))
            else:
                batch_labels.append(feature["labels"])
            
            # GET AUDIO FEATURES ------------------------------------------------------------   
            if has_audio:
                audio = feature.get("input_values")
                if audio is None:  # Modality dropped
                    # Create zero tensor with shape [1, F, T] where F is feature dim and T is time
                    # This follows the format from AVHuBERT's load_feature function
                    feature_dim = 26 * 4  # Default logfbank features * stacking factor
                    time_dim = self.max_audio_length if self.max_audio_length else 100
                    audio = torch.zeros((1, feature_dim, time_dim), dtype=torch.float32)
                batch_audio_values.append(audio)
            
            # GET VIDEO FEATURES ------------------------------------------------------------
            if has_visual:
                video = feature.get("video")
                if video is None:  # Modality dropped
                    # Create zero tensor with shape [1, C, T, H, W] where C is channels
                    channels = 1  # Grayscale
                    time_dim = self.max_video_frames if self.max_video_frames else 100
                    height = width = VIDEO_CROP_SIZE  # Default from AVHuBERT
                    video = torch.zeros((1, channels, time_dim, height, width), dtype=torch.float32)
                batch_visual_values.append(video)

        #--------------------------------------------------------------------------------------
        # Initialize the batch dictionary
        #--------------------------------------------------------------------------------------
        batch = {}

        #--------------------------------------------------------------------------------------
        #                       PAD LABELS
        #--------------------------------------------------------------------------------------

        labels_tensor = self.tokenizer.pad(
            batch_labels,
            padding=True,
            return_tensors="pt",
        )
        # remove decoder_start_token_id if present
        # padding token id is -100
        labels = labels_tensor["input_ids"].masked_fill(labels_tensor["attention_mask"].ne(1), -100)

        if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels


        #--------------------------------------------------------------------------------------
        #                       PAD AUDIO FEATURES
        #--------------------------------------------------------------------------------------
        if has_audio:
            # Find max dims for padding
            max_batch, max_feats, max_time = 0, 0, 0
            for tensor in batch_audio_values:
                b, f, t = tensor.shape
                max_batch = max(max_batch, b)
                max_feats = max(max_feats, f)  
                max_time = max(max_time, t)
            
            # Pad and stack audio tensors
            padded_audio = []
            for tensor in batch_audio_values:
                b, f, t = tensor.shape
                # Pad time dimension to max length
                pad_time = max_time - t
                if pad_time > 0:
                    padded = F.pad(tensor, (0, pad_time, 0, 0, 0, 0), value=0.0)
                else:
                    padded = tensor
                padded_audio.append(padded)
            
            #--------------------------------------------------------------------------------------
            # TODO: - TRY WITH Wav2Vec2FeatureExtractor as part of Processor (as fallback)
            #--------------------------------------------------------------------------------------
            
            batch["input_values"] = torch.cat(padded_audio, dim=0)
        #-------------------------------------------------------------------------------------- 

        #--------------------------------------------------------------------------------------
        #                       PAD VIDEO FEATURES
        #--------------------------------------------------------------------------------------
        if has_visual:
            # Find max dims for padding
            max_batch, max_c, max_time, max_h, max_w = 0, 0, 0, 0, 0
            for tensor in batch_visual_values:
                b, c, t, h, w = tensor.shape
                max_batch = max(max_batch, b)
                max_c = max(max_c, c)
                max_time = max(max_time, t)
                max_h = max(max_h, h)
                max_w = max(max_w, w)
            
            # Pad and stack video tensors
            padded_video = []
            for tensor in batch_visual_values:
                b, c, t, h, w = tensor.shape
                # Pad time dimension to max length
                pad_time = max_time - t
                if pad_time > 0:
                    padded = F.pad(tensor, (0, 0, 0, 0, 0, pad_time, 0, 0, 0, 0), value=0.0)
                else:
                    padded = tensor
                padded_video.append(padded)
            #--------------------------------------------------------------------------------------
            # TODO: - TRY WITH AutoImageProcessor as part of Processor (as fallback)
            #--------------------------------------------------------------------------------------
            
            batch["video"] = torch.cat(padded_video, dim=0)
        #--------------------------------------------------------------------------------------

        #--------------------------------------------------------------------------------------
        #                       ADD ATTENTION MASK
        #--------------------------------------------------------------------------------------
        max_length = batch["labels"].shape[1]
        attention_mask = torch.ones(batch["labels"].shape[0], max_length, dtype=torch.long)
        attention_mask[batch["labels"] == self.tokenizer.pad_token_id] = 0
        batch["attention_mask"] = attention_mask
        #--------------------------------------------------------------------------------------
        
        return batch # batch = {"input_values": [B, F, T], "video": [B, C, T, H, W], "labels": [B, T], "attention_mask": [B, T]}

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
    if model_args.config_yaml:
        logger.info(f"Loading configuration from YAML file {model_args.config_yaml}")
        config = AV2TextConfig.from_yaml(model_args.config_yaml)
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

    ami_train_path = os.path.join("data", data_args.dataset_name, "av_hubert", "train")
    ami_val_path = os.path.join("data", data_args.dataset_name, "av_hubert", "validation")
    ami_test_path = os.path.join("data", data_args.dataset_name, "av_hubert", "test")

    ami_train = load_from_disk(ami_train_path)
    ami_val = load_from_disk(ami_val_path)
    ami_test = load_from_disk(ami_test_path)

    
    # logger.info dataset info
    logger.info(f"Training dataset: {ami_train}")
    logger.info(f"Validation dataset: {ami_val}")
    logger.info(f"Test dataset: {ami_test}")
    # -----------------------------------------------------------------------------------------------------------------
    
    #============================================================================================================
    #                                  SETUP DATASETS AND PROCESS DATASET
    #============================================================================================================
    # Define `prepare_dataset` function for batching
    def prepare_dataset(
            batch,
            tokenizer,
            data_args,
            model_args,
            is_training=True
        ):
        """
        Prepares a single batch of data for AV-HuBERT model using dataset.map().
        This replicates the functionality of load_feature in avhubert's dataset loader,
        extracting audio and video features in the format expected by AV2TextForConditionalGeneration.
        """
                
        processed_examples = []
        for i in range(len(batch["transcript"])):
            audio_item = batch["audio"][i] if "audio" in batch else None
            video_item = batch["lip_video"][i] if "lip_video" in batch else None
            transcript = batch["transcript"][i]

            current_audio_path = audio_item["path"] if isinstance(audio_item, dict) and "path" in audio_item else audio_item
            current_video_path = video_item["path"] if isinstance(video_item, dict) and "path" in video_item else video_item

            # Modality dropping for training
            if is_training:
                if current_audio_path and model_args.use_audio and np.random.random() < data_args.audio_drop_prob:
                    current_audio_path = None
                if current_video_path and model_args.use_visual and np.random.random() < data_args.visual_drop_prob:
                    current_video_path = None
                # Ensure at least one modality is present if both were initially available and dropped
                if not current_audio_path and not current_video_path:
                    original_audio_path = audio_item["path"] if isinstance(audio_item, dict) and "path" in audio_item else audio_item
                    original_video_path = video_item["path"] if isinstance(video_item, dict) and "path" in video_item else video_item
                    if original_audio_path:
                        current_audio_path = original_audio_path
                    elif original_video_path:
                        current_video_path = original_video_path

            # Process audio like in the original AVHuBERT implementation
            audio_feats = None
            if current_audio_path and model_args.use_audio:
                try:
                    audio_feats = load_audio_features(current_audio_path, target_sr=TARGET_AUDIO_SR)
                except Exception as e:
                    logger.error(f"Error loading audio {current_audio_path}: {e}")
            
            # Process video like in the original AVHuBERT implementation
            video_feats = None
            if current_video_path and model_args.use_visual:
                video_feats = load_video_features(current_video_path)  # [T, H, W, 1]
            

            # ---------------------------------------------------------------------------------------------------------
            # Align audio and video lengths as in AVHuBERT's load_feature
            if audio_feats is not None and video_feats is not None:
                diff = len(audio_feats) - len(video_feats)
                if diff < 0:
                    audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
                elif diff > 0:
                    audio_feats = audio_feats[:-diff]
            # ---------------------------------------------------------------------------------------------------------
            
            
            #---------------------------------------------------------------------------------------------------------
            # Convert audio / video features to torch tensors and normalise
            # - FIXME: CHECK IF THIS IS CORRECT.
            if audio_feats is not None:
                audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
                with torch.no_grad():
                    audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
                # Shape to [batch=1, F, T]
                audio_feats = audio_feats.permute(1, 0).unsqueeze(0)
            
            if video_feats is not None:
                video_feats = torch.from_numpy(video_feats.astype(np.float32))
                # Shape to [batch=1, C, T, H, W]
                video_feats = video_feats.permute(3, 0, 1, 2).unsqueeze(0)

            #---------------------------------------------------------------------------------------------------------
            
            
            # Tokenize transcript
            labels = tokenizer(transcript).input_ids

            processed_examples.append({
                "input_values": audio_feats,  # Expected input name for audio
                "video": video_feats,  # Expected input name for video
                "labels": labels,
            })

        batch_dict = {
            "input_values": [ex["input_values"] for ex in processed_examples],
            "video": [ex["video"] for ex in processed_examples],
            "labels": [ex["labels"] for ex in processed_examples],
        }
        return batch_dict

    #-------------------------------------------------------------------
    # Create datasets
    #-------------------------------------------------------------------
    train_dataset = None
    eval_dataset = None
    
    # Get dropout probabilities from config or command line args
    audio_drop_prob = data_args.audio_drop_prob
    visual_drop_prob = data_args.visual_drop_prob
    # Get max_audio_length and max_video_frames
    max_audio_length = int(data_args.max_duration_in_seconds * TARGET_AUDIO_SR)
    max_video_frames = int(data_args.max_duration_in_seconds * 25)     # Default: 25 fps
    
    # For AVHuBERT's load_feature-like functionality, we only need tokenizer for labels
    # The processing will be done in prepare_dataset function itself
    
    if training_args.do_train:
        train_dataset = ami_train.map(
            prepare_dataset,
            fn_kwargs={"tokenizer": tokenizer, "data_args": data_args, "model_args": model_args, "is_training": True}
        )
    if training_args.do_eval:
        eval_dataset = ami_val.map(
            prepare_dataset,
            fn_kwargs={"tokenizer": tokenizer, "data_args": data_args, "model_args": model_args, "is_training": False}
        )

    
    #============================================================================================================
    #                                               SETUP DATA COLLATOR
    #============================================================================================================
    # Data collator that handles batch creation
    data_collator = DataCollatorForAVSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_audio_length=max_audio_length,
        max_video_frames=max_video_frames,
        max_label_length=data_args.max_target_length
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
    # Initialize Seq2SeqTrainer
    # Note: we don't need a processor since feature extraction is done in prepare_dataset
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,  # Just need the tokenizer for generation
        data_collator=data_collator, #FIXME: DO WE NEED THIS?
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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