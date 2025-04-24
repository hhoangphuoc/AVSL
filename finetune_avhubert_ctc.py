#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the AVHuBERT model on the AMI dataset for speech recognition.
"""

import argparse
import logging
import os
import sys
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import datasets
import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    set_seed,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from utils.data_loading import AVHubertDataset, collate_audio_visual_batch, AVHubertBatch
from config.av_hubert_config import AVHuBERTConfig
from models.av_hubert_model import AVHuBERTForCTC

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
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
        default=None, metadata={"help": "Where to store the pretrained models"}
    )
    freeze_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the encoder"}
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
        default=None, metadata={"help": "HuggingFace dataset name"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Cache directory for the dataset."}
    )
    train_split_name: Optional[str] = field(
        default="train", metadata={"help": "Name of the training data split."}
    )
    eval_split_name: Optional[str] = field(
        default="validation", metadata={"help": "Name of the evaluation data split."}
    )
    audio_column_name: Optional[str] = field(
        default="audio", metadata={"help": "Column name for the audio path in the dataset."}
    )
    video_column_name: Optional[str] = field(
        default="lip_video", metadata={"help": "Column name for the lip video path in the dataset."}
    )
    text_column_name: Optional[str] = field(
        default="transcript", metadata={"help": "Column name for the text in the dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training examples to use."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of evaluation examples to use."}
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
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of workers for preprocessing."}
    )


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    raw_datasets = datasets.load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=data_args.dataset_cache_dir,
    )

    # If no validation data is available, split the training data
    if "validation" not in raw_datasets:
        logger.info("Creating validation split from training data")
        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1)
        raw_datasets = datasets.DatasetDict({
            "train": raw_datasets["train"], 
            "validation": raw_datasets["test"]
        })

    # Trim dataset if requested
    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
    if data_args.max_eval_samples is not None:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))

    # Print dataset info
    logger.info(f"Training examples: {len(raw_datasets['train'])}")
    logger.info(f"Validation examples: {len(raw_datasets['validation'])}")

    # Compute max input length
    max_audio_length = int(data_args.max_duration_in_seconds * 16000)  # 16kHz sampling rate
    max_video_frames = int(data_args.max_duration_in_seconds * 25)  # 25 fps

    # Load processor (tokenizer + feature extractor)
    if model_args.processor_name:
        processor = Wav2Vec2Processor.from_pretrained(model_args.processor_name)
    elif model_args.model_name_or_path:
        processor = Wav2Vec2Processor.from_pretrained(model_args.model_name_or_path)
    else:
        # Create processor from scratch if not available
        if model_args.tokenizer_name:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_args.tokenizer_name)
        elif model_args.model_name_or_path:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_args.model_name_or_path)
        else:
            # Create vocabulary from dataset
            logger.info("Creating vocabulary from dataset")
            vocab_dict = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
            
            # Extract unique characters from transcripts
            all_texts = []
            for split in raw_datasets.keys():
                all_texts.extend([text for text in raw_datasets[split][data_args.text_column_name]])
                
            vocab = sorted(list(set("".join(all_texts))))
            for i, char in enumerate(vocab):
                vocab_dict[char] = i + 4  # Account for special tokens
                
            # Save vocabulary to file
            vocab_file = os.path.join(training_args.output_dir, "vocab.json")
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(vocab_file, "w") as f:
                import json
                json.dump(vocab_dict, f)
                
            tokenizer = Wav2Vec2CTCTokenizer(vocab_file)
            
        # Create feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )
        
        # Create processor
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Create config
    if model_args.config_name:
        config = AVHuBERTConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AVHuBERTConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = AVHuBERTConfig()
        
    # Update config with model args
    config.vocab_size = len(processor.tokenizer)
    config.use_audio = model_args.use_audio
    config.use_visual = model_args.use_visual
    config.fusion_type = model_args.fusion_type

    # Create dataset instances
    train_dataset = AVHubertDataset(
        dataset=raw_datasets["train"],
        processor=processor,
        split="train",
        max_audio_length=max_audio_length,
        max_video_frames=max_video_frames,
        audio_drop_prob=data_args.audio_drop_prob,
        visual_drop_prob=data_args.visual_drop_prob,
    )
    
    eval_dataset = AVHubertDataset(
        dataset=raw_datasets["validation"],
        processor=processor,
        split="validation",
        max_audio_length=max_audio_length,
        max_video_frames=max_video_frames,
    )

    # Define data collator
    data_collator = lambda examples: collate_audio_visual_batch(examples, processor)

    # Initialize model
    if model_args.model_name_or_path:
        model = AVHuBERTForCTC.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AVHuBERTForCTC(config)

    # Freeze encoder if requested
    if model_args.freeze_encoder:
        model.freeze_feature_encoder()
        logger.info("Encoder parameters frozen")

    # Define compute_metrics function
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # Replace <unk> with empty string
        pred_str = [s.replace("<unk>", "") for s in pred_str]
        
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "cer": cer}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "dataset": data_args.dataset_name,
        "modalities": [],
    }
    
    if model_args.use_audio:
        kwargs["modalities"].append("audio")
    if model_args.use_visual:
        kwargs["modalities"].append("visual")
        
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()