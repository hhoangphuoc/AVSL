# YAML Configuration for AVHuBERT Fine-tuning

This document describes how to use YAML configuration files with the AVHuBERT fine-tuning scripts.

## Overview

The AVHuBERT model supports loading configurations from YAML files. This provides a more flexible and modular way to manage model configurations than command-line arguments, especially for complex models with many parameters.

## YAML Configuration Structure

The YAML configuration file is structured into logical sections:

```yaml
common:
  # General training settings
  fp16: true
  log_format: json
  log_interval: 200
  seed: 1337

model:
  # Model architecture configuration
  use_audio: true
  use_visual: true
  modality_fuse: concat
  # ... other model parameters

tokenizer:
  # Tokenizer parameters
  vocab_size: 10000
  # ... other tokenizer parameters

training:
  # Training hyperparameters
  max_update: 400000
  lr: 0.002
  # ... other training parameters

criterion:
  # Loss and optimization parameters
  label_smoothing: 0.1
  # ... other criterion parameters

dataset:
  # Dataset parameters
  max_tokens: 1000
  max_audio_length: 480000  # 30 seconds at 16kHz
  # ... other dataset parameters
```

## Using YAML Configuration Files

You can provide a YAML configuration file to the fine-tuning script in two ways:

### 1. Direct Argument

Pass the YAML file as the only argument to the script:

```bash
python finetune_avhubert_seq2seq.py config/avhubert_large.yaml
```

### 2. With the `--config_yaml` Flag

Use the `--config_yaml` flag to specify the path to the YAML file:

```bash
python finetune_avhubert_seq2seq.py --config_yaml config/avhubert_large.yaml
```

## Command-line Overrides

Command-line arguments will override values specified in the YAML configuration. This allows for quick experimentation without modifying the YAML file.

Example:

```bash
python finetune_avhubert_seq2seq.py \
    --config_yaml config/avhubert_large.yaml \
    --use_audio true \
    --use_visual false \
    --fusion_type sum
```

In this example, the `use_audio`, `use_visual`, and `fusion_type` values from the command line will override those in the YAML file.

## Example Script

A sample script for fine-tuning with a YAML configuration is provided in `scripts/finetune_avhubert_with_yaml.sh`:

```bash
#!/bin/bash
# Example script for fine-tuning AVHuBERT using a YAML configuration

# Set variables
CONFIG_YAML="config/avhubert_large.yaml"
OUTPUT_DIR="output/avhubert_ft_yaml"
DATASET_NAME="ami_corpus"  # Replace with your dataset name
NUM_GPUS=1
BATCH_SIZE=8
GRAD_ACCUM=4
LR=2e-5
NUM_EPOCHS=10

# Run fine-tuning with YAML configuration
python finetune_avhubert_seq2seq.py \
    --config_yaml $CONFIG_YAML \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET_NAME \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --fp16 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --logging_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model "wer" \
    --greater_is_better false \
    --predict_with_generate \
    --do_train \
    --do_eval
```

## Available Configuration Files

The following YAML configuration files are available:

- `config/avhubert_large.yaml`: Configuration for the large AVHuBERT model with default parameters based on the original Meta AI implementation.

You can create your own configuration files by modifying these templates.