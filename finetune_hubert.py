#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning the HuBERT model for automatic speech recognition using 
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
    TrainingArguments,
    Trainer,
    set_seed,
    AutoConfig,
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    AutoFeatureExtractor,
    HubertForCTC,
    Wav2Vec2Processor,
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
        default="facebook/hubert-large-ls960-ft", 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default="./checkpoints/hf-hubert", metadata={"help": "Where to store the pretrained models"}
    )
    model_revision: str = field(
        default="main", metadata={"help": "The specific model version to use"}
    )
    use_auth_token: bool = field(
        default=False, metadata={"help": "Will use token for HF authentication"}
    )
    freeze_feature_encoder: bool = field(
        default=False, metadata={"help": "Whether to freeze the feature encoder"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "Attention dropout probability"}
    )
    hidden_dropout: float = field(
        default=0.1, metadata={"help": "Hidden layer dropout probability"}
    )
    feat_proj_dropout: float = field(
        default=0.1, metadata={"help": "Feature projection dropout probability"}
    )
    final_dropout: float = field(
        default=0.1, metadata={"help": "Final dropout probability"}
    )
    mask_time_prob: float = field(
        default=0.05, metadata={"help": "Probability of masking time feature inputs"}
    )
    layerdrop: float = field(
        default=0.1, metadata={"help": "Probability of dropping a layer during training"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments for dataset to input our model for training and eval.
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
    chars_to_ignore: List[str] = field(
        default_factory=lambda: [",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�", ],
        metadata={"help": "Characters to remove from transcripts during preprocessing"},
    )

@dataclass
class HubertFtTrainingArguments(TrainingArguments):
    """
    Training arguments for HuBERT ASR training
    """
    # Output directory--------------------------------
    output_dir: str = field(
        default="output/hubert_ft", metadata={"help": "Output directory"}
    )
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the output directory"}
    )
    group_by_length: bool = field(
        default=True, metadata={"help": "Group by length"}
    )

    # Dataloader Configs--------------------------------
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Per device train batch size"}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Per device eval batch size"}
    )
    gradient_accumulation_steps: int = field(
        default=2, metadata={"help": "Gradient accumulation steps"}
    )
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "Evaluation strategy"}
    )
    num_train_epochs: int = field(
        default=30, metadata={"help": "Number of training epochs"}
    )

    # Computations efficiency--------------------------------
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Gradient checkpointing"}
    )
    fp16: bool = field(
        default=True, metadata={"help": "Mixed precision training"}
    )
    adam_beta2: float = field(
        default=0.98, metadata={"help": "Adam beta 2"}
    )
    torch_empty_cache_steps: int = field(
        default=1000, metadata={"help": "Torch empty cache steps"}
    )

    # Save strategy--------------------------------
    save_strategy: str = field(
        default="steps", metadata={"help": "Save strategy"}
    )
    save_steps: int = field(
        default=100, metadata={"help": "Save steps"}
    )
    eval_steps: int = field(
        default=100, metadata={"help": "Eval steps"}
    )
    logging_steps: int = field(
        default=100, metadata={"help": "Logging steps"}
    )
    # NN configs--------------------------------
    learning_rate: float = field(
        default=1e-4, metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.005, metadata={"help": "Weight decay"}
    )
    warmup_ratio: float = field(
        default=0.15, metadata={"help": "Warmup ratio"}
    )

    # Save strategy--------------------------------
    save_total_limit: int = field(
        default=3, metadata={"help": "Save total limit"}
    )
    torch_empty_cache_steps: int = field(
        default=1000, metadata={"help": "Torch empty cache steps"}
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "Load best model at end"}
    )
    metric_for_best_model: str = field(
        default="wer", metadata={"help": "Metric for best model"}
    )
    greater_is_better: bool = field(
        default=False, metadata={"help": "Greater is better"}
    )
    

class HubertDataset(torch.utils.data.Dataset):
    """Dataset for HuBERT ASR training"""
    
    def __init__(
        self, 
        dataset: Dataset, 
        processor: Wav2Vec2Processor,
        max_duration_in_seconds: float = 30.0,
        chars_to_ignore: List[str] = None,
        split: str = "train"
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_duration_in_seconds = max_duration_in_seconds
        self.chars_to_ignore = chars_to_ignore if chars_to_ignore is not None else []
        self.split = split
        self.is_train = split == "train"
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load audio
        audio = item["audio"]
        
        # Preprocess text
        transcript = item["transcript"]
        for char in self.chars_to_ignore:
            transcript = transcript.replace(char, "")
        transcript = transcript.upper()
        
        # Get input values
        input_values = self.processor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt"
        ).input_values.squeeze(0)
        
        # Process target text
        with self.processor.as_target_processor():
            labels = self.processor(transcript, return_tensors="pt").input_ids.squeeze(0)
        
        return {
            "input_values": input_values,
            "labels": labels
        }

class HubertDataCollator:
    """
    Data collator for HuBERT that handles batching input values and labels
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, batch):
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": item["input_values"]} for item in batch]
        label_features = [{"input_ids": item["labels"]} for item in batch]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def main():
    #=================================================================================================================
    #                               PARSING ARGUMENTS
    #=================================================================================================================
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, HubertFtTrainingArguments))
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

    # Check if dataset is already split or needs splitting
    if isinstance(ami_dataset, DatasetDict):
        # Dataset is already a DatasetDict, check if it has the required splits
        if "train" not in ami_dataset or "validation" not in ami_dataset or "test" not in ami_dataset:
            # Create splits if needed
            raw_datasets = create_dataset_splits(ami_dataset, 
                                                 dataset_name=data_args.dataset_name, 
                                                 model_name="hubert")
        else:
            # Use existing splits
            raw_datasets = ami_dataset
    else:
        # Dataset is a single Dataset without splits, create them
        raw_datasets = create_dataset_splits(ami_dataset, 
                                                 dataset_name=data_args.dataset_name, 
                                                 model_name="hubert")

    # Rename the splits if needed
    if data_args.train_split_name != "train" or data_args.eval_split_name != "validation":
        raw_datasets_renamed = DatasetDict()
        if data_args.train_split_name in raw_datasets:
            raw_datasets_renamed["train"] = raw_datasets[data_args.train_split_name]
        if data_args.eval_split_name in raw_datasets:
            raw_datasets_renamed["validation"] = raw_datasets[data_args.eval_split_name]
        raw_datasets = raw_datasets_renamed

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
    # Load configuration
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Update config with model args
    config.attention_dropout = model_args.attention_dropout
    config.hidden_dropout = model_args.hidden_dropout
    config.feat_proj_dropout = model_args.feat_proj_dropout
    config.final_dropout = model_args.final_dropout
    config.mask_time_prob = model_args.mask_time_prob
    config.layerdrop = model_args.layerdrop
    
    # Get feature extractor and tokenizer for the processor
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # For HuBERT we'll need to create a tokenizer from dataset vocab if it doesn't exist
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        logger.info("Creating a tokenizer from dataset vocabulary...")
        
        # Extract all unique characters from transcripts
        all_texts = []
        for split in raw_datasets.keys():
            if data_args.text_column_name in raw_datasets[split].column_names:
                all_texts.extend(raw_datasets[split][data_args.text_column_name])
        
        # Clean transcripts
        all_text = " ".join(all_texts)
        for char in data_args.chars_to_ignore:
            all_text = all_text.replace(char, "")
        all_text = all_text.upper()
        
        # Create vocabulary with special tokens
        vocab_list = sorted(list(set(all_text)))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        
        # Add special tokens
        vocab_dict["<pad>"] = len(vocab_dict)
        vocab_dict["<unk>"] = len(vocab_dict)
        vocab_dict["<s>"] = len(vocab_dict)
        vocab_dict["</s>"] = len(vocab_dict)
        
        # Save vocabulary to file
        vocab_file = os.path.join(training_args.output_dir, "vocab.json")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(vocab_file, "w") as f:
            import json
            json.dump(vocab_dict, f)
        
        # Create tokenizer from vocab file
        tokenizer = AutoTokenizer.from_pretrained(
            training_args.output_dir,
            vocab_file=vocab_file,
            do_lower_case=False,
            cache_dir=model_args.cache_dir,
        )
    
    # Create processor from feature extractor and tokenizer
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # Update config with tokenizer info
    config.vocab_size = len(tokenizer)
    
    # Load model
    model = HubertForCTC.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Freeze feature encoder if requested
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()
        logger.info("Feature encoder parameters frozen")

    #=================================================================================================================
    #                                  SETUP DATASETS AND PROCESS DATASET
    #=================================================================================================================
    # Create custom datasets
    train_dataset = None
    eval_dataset = None
    
    if training_args.do_train:
        train_dataset = HubertDataset(
            dataset=raw_datasets["train"],
            processor=processor,
            max_duration_in_seconds=data_args.max_duration_in_seconds,
            chars_to_ignore=data_args.chars_to_ignore,
            split="train",
        )
    
    if training_args.do_eval:
        eval_dataset = HubertDataset(
            dataset=raw_datasets["validation"],
            processor=processor,
            max_duration_in_seconds=data_args.max_duration_in_seconds,
            chars_to_ignore=data_args.chars_to_ignore,
            split="validation",
        )
    
    #=================================================================================================================
    #                                               SETUP METRICS
    #=================================================================================================================
    # Define compute metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer, "cer": cer}
    
    #=================================================================================================================
    #                                               SETUP TRAINER
    #=================================================================================================================
    data_collator = HubertDataCollator(processor=processor)
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
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
        
        metrics = trainer.evaluate()
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