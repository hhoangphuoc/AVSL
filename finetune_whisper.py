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
    local_files_only: bool = field(
        default=True, metadata={"help": "Whether to use local files only"}
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
        default=448, metadata={"help": "Max sequence length for decoder outputs"}
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

class WhisperDataCollator:
    def __init__(self, processor: WhisperProcessor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    
    def __call__(self, features):
        """
        Data collator for Whisper that handles batching input features
        """
        if not features:
            return None
        
        input_features = [{
            "input_features": item["input_features"]
        } for item in features] #"input_features" is the input features parsed from `WhisperDataset`
        
        # Pad input features - Only pad in DataCollator
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
        
        labels_features = [{
            "input_ids": item["labels"]
        } for item in features] #"input_ids" is the tokenized labels parsed from `WhisperDataset`
    
        # Pad labels - Only pad in DataCollator
        labels_batch = self.processor.tokenizer.pad(
            labels_features,
            padding=True,
            return_tensors="pt",
        )

        # Remove decoder_start_token_id if present
        # padding token id is -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)

        # Remove decoder_start_token_id if present
        if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

    
        return batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    #=================================================================================================================
    #                               PARSING ARGUMENTS
    #=================================================================================================================
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #=================================================================================================================
    #                                       SETUP LOGGING
    #=================================================================================================================
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO  # Ensure basicConfig sets a level so logger.info works
    )
    # Diagnostic logs:
    logger.info(f"Type of training_args: {type(training_args)}")
    logger.info(f"Does training_args have 'distributed_state' attribute? {'distributed_state' in dir(training_args)}")
    if 'distributed_state' in dir(training_args):
        logger.info(f"Value of training_args.distributed_state: {getattr(training_args, 'distributed_state', 'Error retrieving attribute')}")
    else:
        logger.info(f"training_args does NOT have 'distributed_state' attribute.")

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # -------------------------------------------------------------------------

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {device}, n_gpu: {training_args.n_gpu}, "
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
    #                                       SETUP MODEL AND PROCESSOR
    #=================================================================================================================
    # Load Whisper model and processor
    config = WhisperConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only=True if model_args.local_files_only else False,
    )
    
    # Update config with model args
    config.dropout = model_args.hidden_dropout
    config.attention_dropout = model_args.attn_dropout
    
    processor = WhisperProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only=True if model_args.local_files_only else False,
    )
    
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    new_tokens = ['<laugh>']
    tokenizer.add_tokens(new_tokens)

    laugh_token_id = tokenizer.convert_tokens_to_ids('<laugh>')
    logger.info(f"The token ID for '<laugh>' is: {laugh_token_id}") # WE FOUND THAT <laugh> TOKEN ID is 51865
    #=================================================================================================================
    #                                       SETUP MODEL
    #=================================================================================================================
    model = WhisperForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        local_files_only=True if model_args.local_files_only else False,
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
    model.resize_token_embeddings(len(tokenizer))
    model.config.suppress_tokens = []
    model.config.use_cache = False # disable caching

    model.to(device) # move model to GPUs

    #=================================================================================================================
    #                        SETUP DATASET (LOAD, SPLIT, SAVE)
    #=================================================================================================================
    # Load dataset from disk (WHEN NOT SPLITTED)
    # dataset_path = os.path.join("data", data_args.dataset_name, "dataset") #data/ami/dataset
    # ami_dataset = load_from_disk(dataset_path)

    # raw_datasets = create_dataset_splits(ami_dataset, 
    #                                      dataset_name=data_args.dataset_name, 
    #                                      model_name="whisper")
    # ------------------------------------------------------------------------------------------------------

    # Load dataset from disk (WHEN SPLITTED)
    ami_train_path = os.path.join("data", data_args.dataset_name, "whisper", "train") #data/ami/whisper/train
    ami_val_path = os.path.join("data", data_args.dataset_name, "whisper", "validation") #data/ami/whisper/validation

    ami_train = load_from_disk(ami_train_path)
    ami_val = load_from_disk(ami_val_path)

    # Print dataset info
    logger.info(f"Training examples: {len(ami_train)}")
    logger.info(f"Validation examples: {len(ami_val)}")
    # -----------------------------------------------------------------------------------------------------------------

    def prepare_dataset(batch):
        audio = batch["audio"]
        transcript = batch["transcript"].lower()


        batch["input_features"] = feature_extractor(
            raw_speech=audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0] #[0] #(n_mels, time_steps)

        labels = tokenizer(
            transcript,
        ).input_ids

        batch["labels"] = labels

        return batch
        

    #=================================================================================================================
    #                                  SETUP DATASETS AND PROCESS DATASET
    #=================================================================================================================
    # Create custom datasets
    train_dataset = None
    eval_dataset = None
    
    if training_args.do_train:
        train_dataset = ami_train.map(
            prepare_dataset, 
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=ami_train.column_names,
            desc="Preprocessing training dataset"
        )
    
    if training_args.do_eval:
        eval_dataset = ami_val.map(
            prepare_dataset, 
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=ami_val.column_names,
            desc="Preprocessing validation dataset"
        )
    
    #=================================================================================================================
    #                                               SETUP METRICS
    #=================================================================================================================
    # Define compute metrics
    wer_metric = evaluate.load("wer")
    # cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        
        # If predictions are logits, convert to ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
            
        if isinstance(pred_ids, np.ndarray) and len(pred_ids.shape) == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)
        
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        # cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    #=================================================================================================================
    #                                               SETUP TRAINER
    #=================================================================================================================
    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        max_target_length=data_args.max_target_length
    )
    # Initialize Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=processor,
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