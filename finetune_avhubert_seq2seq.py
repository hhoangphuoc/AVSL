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
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoProcessor,
    AutoTokenizer,
    Speech2TextTokenizer,   
    EarlyStoppingCallback,
)

from transformers.trainer_utils import get_last_checkpoint

from utils.data_loading import AVHubertDataset, collate_audio_visual_batch, AVHubertBatch
from config.av_hubert_config import AVHuBERTConfig
from models.av_hubert_seq2seq_model import AVHuBERTForConditionalGeneration

import datasets
from datasets import DatasetDict, Dataset, load_from_disk

logger = logging.getLogger(__name__)

MODEL_NAME_OR_PATH = "nguyenvulebinh/AV-HuBERT-MuAViC-en" #"vumichien/AV-HuBERT"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default=MODEL_NAME_OR_PATH, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
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
        default="./checkpoints/hf-avhubert", metadata={"help": "Where to store the pretrained models"}
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
        default=4, metadata={"help": "Number of beams to use for evaluation."}
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


def main():
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

    #=================================================================================================================
    #                                       SETUP LOGGING
    #=================================================================================================================
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
    # -------------------------------------------------------------------------

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
    #=================================================================================================================
    # Set seed before initializing model
    set_seed(training_args.seed)

    #=================================================================================================================
    #                        SETUP DATASET (LOAD, SPLIT, SAVE)
    #=================================================================================================================
    # Load dataset from disk
    dataset_path = os.path.join("data", data_args.dataset_name, "dataset")
    ami_dataset = load_from_disk(dataset_path)

    # Split the dataset into train-val and test
    train_test_ami = ami_dataset.train_test_split(test_size=0.2) #80% train-val, 20% test
    ami_test = train_test_ami["test"]

    # Split the train-val dataset into train and validation
    train_val_ami = train_test_ami["train"].train_test_split(test_size=0.1) #70% train, 10% validation
    ami_train = train_val_ami["train"]
    ami_val = train_val_ami["test"]

    # Create a DatasetDict with the splits
    raw_datasets = DatasetDict({
        "train": ami_train,
        "validation": ami_val,
        "test": ami_test
    })
    # -------------------------------------------------------------------------

    # Save the splits to disk-------------------------------------------------
    ami_train.save_to_disk(os.path.join("data", data_args.dataset_name, "ami_train")) #data/ami/ami_train
    ami_val.save_to_disk(os.path.join("data", data_args.dataset_name, "ami_val")) #data/ami/ami_val
    ami_test.save_to_disk(os.path.join("data", data_args.dataset_name, "ami_test")) #data/ami/ami_test
    # -------------------------------------------------------------------------
    
    # Rename the splits if needed
    if data_args.train_split_name != "train" or data_args.eval_split_name != "validation":
        raw_datasets_renamed = DatasetDict()
        if data_args.train_split_name in raw_datasets:
            raw_datasets_renamed["train"] = raw_datasets[data_args.train_split_name]
        if data_args.eval_split_name in raw_datasets:
            raw_datasets_renamed["validation"] = raw_datasets[data_args.eval_split_name]
        raw_datasets = raw_datasets_renamed

    # Trim dataset if requested - NOTE: NOT IMPLEMENTED--------------------------------
    if data_args.max_train_samples is not None and "train" in raw_datasets:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None and "validation" in raw_datasets:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))
    # -------------------------------------------------------------------------

    # Print dataset info
    logger.info(f"Training examples: {len(raw_datasets['train']) if 'train' in raw_datasets else 0}")
    logger.info(f"Validation examples: {len(raw_datasets['validation']) if 'validation' in raw_datasets else 0}")
    logger.info(f"Test examples: {len(raw_datasets['test']) if 'test' in raw_datasets else 0}")
    # -----------------------------------------------------------------------------------------------------------------


    #=================================================================================================================
    #                                               SETUP CONFIG
    #=================================================================================================================

    # Load config
    if model_args.config_yaml:
        logger.info(f"Loading configuration from YAML file {model_args.config_yaml}")
        config = AVHuBERTConfig.from_yaml(model_args.config_yaml)
    # -------------------------------------------------------------------------------
    # NOTE: NOT IMPLEMENTED
    elif model_args.config_name:
        config = AVHuBERTConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # -------------------------------------------------------------------------------
    else:
        config = AVHuBERTConfig()
        logger.warning("You are instantiating a new config instance from scratch.")

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

    #===================================================================================================================
    # NOTE: MAY NOT BE NEEDED TO LOAD TOKENIZER AND FEATURE EXTRACTOR SEPARATELY
    # AS THEY ARE INCLUDED IN THE PROCESSOR 
    #===================================================================================================================
    #-------------------------------------------------------------------------
    #                SETUP TOKENIZER
    #-------------------------------------------------------------------------
    # Try to load tokenizer from different sources with fallbacks
    tokenizer = None
    
    # 1. Try loading from tokenizer_name if provided
    # NOTE: NOT IMPLEMENTED - TO BE REMOVED--------------------------------
    # if model_args.tokenizer_name:
    #     try:
    #         logger.info(f"Loading tokenizer from {model_args.tokenizer_name}")
    #         tokenizer = AutoTokenizer.from_pretrained(
    #             model_args.tokenizer_name,
    #             cache_dir=model_args.cache_dir,
    #             use_fast=model_args.use_fast_tokenizer,
    #             revision=model_args.model_revision,
    #             use_auth_token=True if model_args.use_auth_token else None,
    #         )
    #         logger.info(f"Successfully loaded tokenizer from {model_args.tokenizer_name}")
    #     except Exception as e:
    #         logger.warning(f"Failed to load tokenizer from {model_args.tokenizer_name}: {e}")
    # ------------------------------------------------------------------------
    
    # 2. Try loading from model_name_or_path if tokenizer is still None
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
    # ------------------------------------------------------------------------
    # 3. Try loading from local dictionary file if tokenizer is still None
    if tokenizer is None:
        dict_path = os.path.join("checkpoints", "avhubert", "dict.en.txt")
        tokenizer_model_path = os.path.join("checkpoints", "avhubert", "tokenizer.model")
        
        if os.path.exists(dict_path):
            try:
                logger.info(f"Loading dictionary from local file: {dict_path}")
                
                # Extract vocabulary from dictionary file
                with open(dict_path, "r", encoding="utf-8") as f:
                    vocab_list = [line.strip() for line in f if line.strip()]
                
                # Create vocabulary dictionary with special tokens
                vocab_dict = {}
                vocab_dict["<pad>"] = 0
                vocab_dict["<s>"] = 1
                vocab_dict["</s>"] = 2
                vocab_dict["<unk>"] = 3
                
                # Add vocabulary items
                for i, token in enumerate(vocab_list):
                    vocab_dict[token] = i + 4  # Start after special tokens
                
                # Save vocabulary to file for tokenizer creation
                vocab_file = os.path.join(training_args.output_dir, "vocab.json")
                os.makedirs(training_args.output_dir, exist_ok=True)
                with open(vocab_file, "w") as f:
                    import json
                    json.dump(vocab_dict, f)
                
                # Create tokenizer from saved vocabulary
                tokenizer = Speech2TextTokenizer.from_pretrained(
                    training_args.output_dir,
                    cache_dir=model_args.cache_dir,
                    use_fast=model_args.use_fast_tokenizer,
                )
                
                # Set special tokens
                tokenizer.bos_token = "<s>"
                tokenizer.eos_token = "</s>"
                tokenizer.pad_token = "<pad>"
                tokenizer.unk_token = "<unk>"
                
                logger.info(f"Successfully created tokenizer from dictionary file with {len(vocab_dict)} tokens")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from dictionary file: {e}")
        
        # If existing tokenizer.model is available, use it (but this would depend on implementation)
        elif os.path.exists(tokenizer_model_path):
            try:
                logger.info(f"Attempting to load tokenizer model from {tokenizer_model_path}")
                # Implementation would depend on the specific tokenizer model format
                # This is a placeholder - actual implementation depends on the format
                pass
            except Exception as e:
                logger.warning(f"Failed to load tokenizer model: {e}")
    # ----------------------------------------------------------------------------------------------------------
    
    # 4. If all above methods fail, create from dataset vocabulary
    if tokenizer is None:
        logger.info("Creating vocabulary from dataset")
        
        # Extract all unique characters from transcripts
        all_texts = []
        for split in raw_datasets.keys():
            if data_args.text_column_name in raw_datasets[split].column_names:
                all_texts.extend(raw_datasets[split][data_args.text_column_name])
        
        # Create vocabulary with special tokens
        vocab_list = sorted(list(set("".join(all_texts))))
        vocab_dict = {}
        
        # Add special tokens
        vocab_dict["<pad>"] = 0
        vocab_dict["<s>"] = 1
        vocab_dict["</s>"] = 2
        vocab_dict["<unk>"] = 3
        
        # Add characters
        for i, char in enumerate(vocab_list):
            if char.strip():  # Skip empty characters
                vocab_dict[char] = i + 4  # Start after special tokens
        
        # Save vocabulary to file
        vocab_file = os.path.join(training_args.output_dir, "vocab.json")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(vocab_file, "w") as f:
            import json
            json.dump(vocab_dict, f)
        
        # Create tokenizer
        tokenizer = Speech2TextTokenizer.from_pretrained(
            training_args.output_dir,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        
        # Set special tokens
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<pad>"
        tokenizer.unk_token = "<unk>"
        
        logger.info(f"Created new tokenizer from dataset with {len(vocab_dict)} tokens")
    # -------------------------------------------------------------------------

    # Update config with tokenizer info
    config.vocab_size = len(tokenizer)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.decoder_start_token_id = tokenizer.bos_token_id
    
    # #-------------------------------------------------------------------------
    # #                           SETUP FEATURE EXTRACTOR
    # #-------------------------------------------------------------------------
    # if model_args.feature_extractor_name:
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         model_args.feature_extractor_name,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    # elif model_args.model_name_or_path:
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         model_args.model_name_or_path,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #     )
    # else:
    #     # Create a default feature extractor
    #     feature_extractor = AutoFeatureExtractor.from_pretrained(
    #         "facebook/hubert-base-ls960",
    #         cache_dir=model_args.cache_dir,
    #     )
    #     feature_extractor.do_normalize = True
    #-------------------------------------------------------------------------
        
    #=================================================================================================================
    #                                               SETUP PROCESSOR
    #=================================================================================================================
    # Create processor with fallback mechanisms
    processor = None
    
    # 1. Try loading from processor_name if provided
    # NOTE: NOT IMPLEMENTED - TO BE REMOVED--------------------------------
    # if model_args.processor_name:
    #     try:
    #         logger.info(f"Loading processor from {model_args.processor_name}")
    #         processor = AutoProcessor.from_pretrained(
    #             model_args.processor_name,
    #             cache_dir=model_args.cache_dir,
    #             revision=model_args.model_revision,
    #             use_auth_token=True if model_args.use_auth_token else None,
    #         )
    #         logger.info(f"Successfully loaded processor from {model_args.processor_name}")
    #     except Exception as e:
    #         logger.warning(f"Failed to load processor from {model_args.processor_name}: {e}")
    # ------------------------------------------------------------------------

    # 2. Try loading from model_name_or_path if processor is still None
    if processor is None and model_args.model_name_or_path:
        try:
            logger.info(f"Loading processor from {model_args.model_name_or_path}")
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            logger.info(f"Successfully loaded processor from {model_args.model_name_or_path}")
        except Exception as e:
            logger.warning(f"Failed to load processor from {model_args.model_name_or_path}: {e}")
    
    # 3. Create a basic processor if all above methods fail
    if processor is None:
        logger.info("Creating new basic processor")
        processor = transformers.ProcessorMixin()
        processor.tokenizer = tokenizer
        
        # Additional processor setup can be done here if needed
        # For example, setting properties that might be needed for the specific task
        
        # Save the processor
        os.makedirs(training_args.output_dir, exist_ok=True)
        processor.save_pretrained(training_args.output_dir)
        logger.info(f"Created and saved new processor to {training_args.output_dir}")
    
    # -------------------------------------------------------------------------
    # COMPUTE MAX AUDIO AND VIDEO LENGTHS (LOAD FROM YAML IF AVAILABLE)
    # -------------------------------------------------------------------------
    # Get max_audio_length and max_video_frames from YAML if available, otherwise compute from max_duration
    max_audio_length = int(data_args.max_duration_in_seconds * 16000)  # Default: 16kHz sampling rate
    max_video_frames = int(data_args.max_duration_in_seconds * 25)     # Default: 25 fps
    
    if model_args.config_yaml:
        # Parse YAML file directly to get dataset parameters if needed
        try:
            import yaml
            with open(model_args.config_yaml, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            if 'dataset' in yaml_config:
                if 'max_audio_length' in yaml_config['dataset']:
                    max_audio_length = yaml_config['dataset']['max_audio_length']
                    logger.info(f"Using max_audio_length from YAML config: {max_audio_length}")
                    
                if 'max_video_frames' in yaml_config['dataset']:
                    max_video_frames = yaml_config['dataset']['max_video_frames']
                    logger.info(f"Using max_video_frames from YAML config: {max_video_frames}")
        except Exception as e:
            logger.warning(f"Failed to parse dataset parameters from YAML: {e}")
            logger.info(f"Using default max_audio_length: {max_audio_length}, max_video_frames: {max_video_frames}")
    # -------------------------------------------------------------------------
    
    #=================================================================================================================
    #                                               SETUP MODEL
    #=================================================================================================================
    # Initialize model from pretrained model
    if model_args.model_name_or_path:
        try:
            # First attempt: try loading from Hugging Face model hub
            logger.info(f"Attempting to load pretrained model from {model_args.model_name_or_path}")
            model = AVHuBERTForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            logger.info(f"Successfully loaded pretrained model from {model_args.model_name_or_path}")
        except Exception as e:
            # Second attempt: try loading from local checkpoint files
            logger.warning(f"Failed to load model from {model_args.model_name_or_path}: {e}")
            logger.info("Attempting to load from local checkpoint files...")
            
            try:
                # Check if local checkpoint files exist
                checkpoint_path = os.path.join("checkpoints", "avhubert", "checkpoint_best.pt")
                dict_path = os.path.join("checkpoints", "avhubert", "dict.en.txt")
                tokenizer_path = os.path.join("checkpoints", "avhubert", "tokenizer.model")
                
                if os.path.exists(checkpoint_path) and os.path.exists(dict_path) and os.path.exists(tokenizer_path):
                    logger.info(f"Found local checkpoint files: {checkpoint_path}")
                    
                    # Initialize the model with the config
                    model = AVHuBERTForConditionalGeneration(config)
                    
                    # Load the checkpoint weights
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if "model" in checkpoint:
                        checkpoint = checkpoint["model"]
                    
                    # Check for key mismatches or missing keys
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                    if missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
                    
                    logger.info(f"Successfully loaded weights from local checkpoint: {checkpoint_path}")
                    
                    # If a dictionary file exists, update the tokenizer vocabulary
                    if os.path.exists(dict_path):
                        logger.info(f"Loading dictionary from {dict_path}")
                        with open(dict_path, "r", encoding="utf-8") as f:
                            vocab = [line.strip() for line in f if line.strip()]
                        
                        # Update config with correct vocabulary size
                        config.vocab_size = len(vocab)
                        model.config.vocab_size = len(vocab)
                        
                        # Need to resize token embeddings if the vocabulary size changed
                        model.resize_token_embeddings(len(vocab))
                    
                    # If a tokenizer model exists, update the processor
                    if hasattr(tokenizer, "save_vocabulary") and os.path.exists(tokenizer_path):
                        logger.info(f"Loading tokenizer model from {tokenizer_path}")
                        tokenizer = tokenizer.__class__.from_pretrained(
                            os.path.dirname(tokenizer_path),
                            cache_dir=model_args.cache_dir,
                        )
                else:
                    raise FileNotFoundError(f"Local checkpoint files not found in {os.path.join('checkpoints', 'avhubert')}")
            except Exception as local_err:
                logger.error(f"Failed to load from local checkpoint: {local_err}")
                logger.info("Training new model from scratch")
                model = AVHuBERTForConditionalGeneration(config)
    else:
        logger.info("Training new model from scratch")
        model = AVHuBERTForConditionalGeneration(config)
    
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

    #=================================================================================================================
    #                                  SETUP DATASETS AND PROCESS DATASET
    #=================================================================================================================
    # Create datasets
    train_dataset = None
    eval_dataset = None
    
    # Get dropout probabilities from config or command line args
    audio_drop_prob = data_args.audio_drop_prob
    visual_drop_prob = data_args.visual_drop_prob
    
    # Override with YAML values if they exist
    if model_args.config_yaml:
        if hasattr(config, 'modality_dropout'):
            visual_drop_prob = config.modality_dropout
            logger.info(f"Using visual_drop_prob from YAML config: {visual_drop_prob}")
        if hasattr(config, 'audio_dropout'):
            audio_drop_prob = config.audio_dropout
            logger.info(f"Using audio_drop_prob from YAML config: {audio_drop_prob}")
    
    if training_args.do_train:
        train_dataset = AVHubertDataset(
            dataset=raw_datasets["train"],
            processor=processor,
            split="train",
            max_audio_length=max_audio_length,
            max_video_frames=max_video_frames,
            audio_drop_prob=audio_drop_prob,
            visual_drop_prob=visual_drop_prob,
        )
    
    if training_args.do_eval:
        eval_dataset = AVHubertDataset(
            dataset=raw_datasets["validation"],
            processor=processor,
            split="validation",
            max_audio_length=max_audio_length,
            max_video_frames=max_video_frames,
        )
    
    #=================================================================================================================
    #                                               SETUP DATA COLLATOR
    #=================================================================================================================
    # Data collator that handles batch creation
    data_collator = lambda examples: collate_audio_visual_batch(examples, processor)
    
    #=================================================================================================================
    #                                               SETUP METRICS
    #=================================================================================================================
    # Define compute metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
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
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}
    
    #=================================================================================================================
    #                                               SETUP TRAINER
    #=================================================================================================================
    # Initialize Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor,
        data_collator=data_collator,
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
        processor.save_pretrained(training_args.output_dir)
        
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