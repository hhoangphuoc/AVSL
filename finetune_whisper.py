#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the Whisper model for automatic speech recognition using
the Transformers library.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
import evaluate
import transformers
from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    WhisperConfig,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
from datasets import DatasetDict, Dataset, load_from_disk
from utils import create_dataset_splits

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        default="openai/whisper-large-v2", 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="./checkpoints/hf-whisper", metadata={"help": "Where to store the pretrained models"}
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
    attn_dropout: float = field(
        default=0.1, metadata={"help": "Attention dropout probability"}
    )
    hidden_dropout: float = field(
        default=0.1, metadata={"help": "Hidden layer dropout probability"}
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
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for preprocessing."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Max samples to use for training (debugging)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Max samples to use for evaluation (debugging)."}
    )
    audio_column_name: str = field(
        default="audio", metadata={"help": "The name of the dataset column containing the audio path"}
    )
    text_column_name: str = field(
        default="transcript", metadata={"help": "The name of the dataset column containing the text"}
    )
    train_split_name: str = field(
        default="train", metadata={"help": "The name of the training data split"}
    )
    eval_split_name: str = field(
        default="validation", metadata={"help": "The name of the evaluation data split"}
    )
    test_split_name: str = field(
        default="test", metadata={"help": "The name of the test data split"}
    )
    max_duration_in_seconds: float = field(
        default=30.0, metadata={"help": "Maximum audio duration in seconds"}
    )
    max_target_length: Optional[int] = field(
        default=128, metadata={"help": "Max sequence length for decoder outputs"}
    )
    num_beams: Optional[int] = field(
        default=4, metadata={"help": "Number of beams to use for evaluation."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Whether to ignore pad tokens in loss computation"}
    )
    language: str = field(
        default="en", metadata={"help": "Language for Whisper model"}
    )
    task: str = field(
        default="transcribe", metadata={"help": "Task for Whisper model (transcribe, translate, etc)"}
    )

@dataclass
class WhisperFtTrainingArguments(Seq2SeqTrainingArguments):
    """
    Training arguments for Whisper ASR training
    """
    output_dir: str = field(
        default="output/whisper_ft", metadata={"help": "Output directory for Whisper ASR training"}
    )
    do_train: bool = field(
        default=True, metadata={"help": "Whether to train the model"}
    )
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to evaluate the model"}
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size for training"}
    )
    num_train_epochs: int = field(
        default=10, metadata={"help": "Number of training epochs"}
    ) 
    warmup_ratio: float = field(
        default=0.15, metadata={"help": "Warmup ratio"}
    )
    gradient_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of gradient accumulation steps"}
    )
    
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size for evaluation"}
    )
    eval_strategy: str = field(
        default="epoch", metadata={"help": "Evaluation strategy"}
    )
    eval_accumulation_steps: int = field(
        default=4, metadata={"help": "Number of evaluation accumulation steps"}
    )
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Learning rate"}
    )
    lr_scheduler_type: str = field(
        default="linear", metadata={"help": "Learning rate scheduler type"}
    )
    weight_decay: float = field(
        default=0.005, metadata={"help": "Weight decay"}
    )
    
    save_strategy: str = field(
        default="epoch", metadata={"help": "Save strategy"}
    )
    save_total_limit: int = field(
        default=3, metadata={"help": "Total number of checkpoints to save"}
    )

    logging_steps: int = field(
        default=100, metadata={"help": "Logging steps"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["tensorboard"], metadata={"help": "Report to"}
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "Load best model at end"}
    )
    metric_for_best_model: str = field(
        default="wer", metadata={"help": "Metric for best model"}
    )
    greater_is_better: bool = field(
        default=False, metadata={"help": "Whether greater is better"}
    )
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Remove unused columns"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "Resume from checkpoint"}
    )
    predict_with_generate: bool = field(
        default=True, metadata={"help": "Whether to predict with generate"}
    )
    generation_max_length: int = field(
        default=448, metadata={"help": "Max generation length"}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Whether to use gradient checkpointing"}
    )
    fp16: bool = field(
        default=True, metadata={"help": "Whether to use mixed precision training"}
    )
    adam_beta2: float = field(
        default=0.98, metadata={"help": "Adam beta 2"}
    )
    torch_empty_cache_steps: int = field(
        default=1000, metadata={"help": "Torch empty cache steps"}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether to push to hub"}
    )

class WhisperDataset(torch.utils.data.Dataset):
    """Dataset for Whisper ASR training"""
    
    def __init__(
        self, 
        dataset: Dataset, 
        processor: WhisperProcessor,
        max_duration_in_seconds: float = 30.0,
        split: str = "train",
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_duration_in_seconds = max_duration_in_seconds
        self.split = split
        self.is_train = split == "train"
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load audio
        audio = item["audio"]
        
        # Get input features
        input_features = self.processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_features.squeeze(0)
        
        # Prepare target text
        transcript = item["transcript"]
        
        # Tokenize text
        labels = self.processor.tokenizer(
            transcript,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        return {
            "input_features": input_features,
            "labels": labels
        }

class WhisperDataCollator:
    def __init__(self, processor: WhisperProcessor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id
    
    def __call__(self, batch):
        """
        Data collator for Whisper that handles batching input features
        """
        input_features = [{
            "input_features": item["input_features"]
        } for item in batch]
        labels = [{
            "input_ids": item["labels"]
        } for item in batch]
    
        # Pad input features
        input_features = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        
        # Pad labels
        labels = self.processor.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt",
        )

        # Remove decoder_start_token_id if present
        # padding token id is -100
        labels = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)

        # Remove decoder_start_token_id if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
    
        batch = {
            "input_features": input_features,
            "labels": labels,
        }
    
        return batch

def main():
    #=================================================================================================================
    #                               PARSING ARGUMENTS
    #=================================================================================================================
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, WhisperFtTrainingArguments))
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
    dataset_path = os.path.join("data", data_args.dataset_name, "dataset") #data/ami/dataset
    ami_dataset = load_from_disk(dataset_path)

    raw_datasets = create_dataset_splits(ami_dataset, 
                                         dataset_name=data_args.dataset_name, 
                                         model_name="whisper")
    

    # Trim dataset if requested
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
    #                                       SETUP MODEL AND PROCESSOR
    #=================================================================================================================
    # Load Whisper model and processor
    config = WhisperConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Update config with model args
    config.dropout = model_args.hidden_dropout
    config.attention_dropout = model_args.attn_dropout
    
    processor = WhisperProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Freeze components if requested
    if model_args.freeze_encoder:
        model.freeze_encoder()
        logger.info("Encoder parameters frozen")
    
    if model_args.freeze_decoder:
        model.freeze_decoder()
        logger.info("Decoder parameters frozen")
    
    # Set language and task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=data_args.language,
        task=data_args.task
    )
    model.config.suppress_tokens = []

    #=================================================================================================================
    #                                  SETUP DATASETS AND PROCESS DATASET
    #=================================================================================================================
    # Create custom datasets
    train_dataset = None
    eval_dataset = None
    
    if training_args.do_train:
        train_dataset = WhisperDataset(
            dataset=raw_datasets["train"],
            processor=processor,
            max_duration_in_seconds=data_args.max_duration_in_seconds,
            split="train",
        )
    
    if training_args.do_eval:
        eval_dataset = WhisperDataset(
            dataset=raw_datasets["validation"],
            processor=processor,
            max_duration_in_seconds=data_args.max_duration_in_seconds,
            split="validation",
        )
    
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
        
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}
    
    #=================================================================================================================
    #                                               SETUP TRAINER
    #=================================================================================================================
    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    # Initialize Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # Set generation parameters
    if hasattr(data_args, "max_target_length"):
        training_args.generation_max_length = data_args.max_target_length
        
    if hasattr(data_args, "num_beams"):
        training_args.generation_num_beams = data_args.num_beams
    
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
        import json
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    main()