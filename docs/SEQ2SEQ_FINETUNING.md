# Fine-tuning AV-HuBERT with Sequence-to-Sequence Architecture

This guide explains how to fine-tune the audio-visual HuBERT model using a sequence-to-sequence architecture for audio-visual speech recognition (AVSR) on the AMI Dataset.

## Overview

The sequence-to-sequence approach for speech recognition uses an encoder-decoder architecture instead of CTC. This offers several advantages:

- Language model capabilities are integrated in the decoder
- Better handling of complex linguistic patterns
- Improved accuracy for long transcriptions
- The ability to generate structured outputs

## Prerequisites

Ensure you have:
- Installed all requirements: `pip install -r requirements.txt`
- Prepared the AMI dataset in HuggingFace format with audio and lip video paths

## Key Components

1. **Fixed Seq2Seq Model**: `models/fixed_av_seq2seq_model.py` 
   - Provides properly implemented sequence-to-sequence models for AVSR
   - Includes `AVHuBERTForSpeech2Text` and `AVHuBERTForConditionalGeneration`
   - Fixes alignment issues between encoder and decoder

2. **Finetuning Script**: `finetune_avhubert_seq2seq.py`
   - Uses HuggingFace's Seq2SeqTrainer for efficient training
   - Supports both audio-only, visual-only, and multi-modal training
   - Implements beam search for optimal decoding

## Fine-tuning Procedure

### 1. Basic Fine-tuning

Run the following command to fine-tune AVHuBERT for sequence-to-sequence AVSR:

```bash
python finetune_avhubert_seq2seq.py \
  --output_dir ./results/avhubert-seq2seq-ami \
  --dataset_name data/ami/dataset \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 20 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --dataloader_num_workers 4 \
  --fp16 \
  --predict_with_generate \
  --max_target_length 128 \
  --num_beams 4
```

### 2. Using Only One Modality

#### Audio-only Fine-tuning:

```bash
python finetune_avhubert_seq2seq.py \
  --output_dir ./results/avhubert-seq2seq-ami-audio-only \
  --dataset_name data/ami/dataset \
  --use_audio True \
  --use_visual False \
  --do_train \
  --do_eval \
  --predict_with_generate
```

#### Visual-only Fine-tuning:

```bash
python finetune_avhubert_seq2seq.py \
  --output_dir ./results/avhubert-seq2seq-ami-visual-only \
  --dataset_name data/ami/dataset \
  --use_audio False \
  --use_visual True \
  --do_train \
  --do_eval \
  --predict_with_generate
```

### 3. Modality Dropout for Robustness

To improve model robustness to missing modalities, use modality dropout during training:

```bash
python finetune_avhubert_seq2seq.py \
  --output_dir ./results/avhubert-seq2seq-ami-robust \
  --dataset_name data/ami/dataset \
  --audio_drop_prob 0.1 \
  --visual_drop_prob 0.1 \
  --do_train \
  --do_eval \
  --predict_with_generate
```

### 4. Fine-tuning from Pre-trained Model

To continue training from a pre-trained model:

```bash
python finetune_avhubert_seq2seq.py \
  --output_dir ./results/avhubert-seq2seq-ami-continued \
  --model_name_or_path /path/to/pretrained/model \
  --dataset_name data/ami/dataset \
  --do_train \
  --do_eval \
  --predict_with_generate
```

### 5. Component Freezing

You can freeze specific components to focus training on particular parts of the model:

```bash
# Freeze encoder (train only decoder)
python finetune_avhubert_seq2seq.py \
  --output_dir ./results/avhubert-seq2seq-ami-encoder-frozen \
  --model_name_or_path /path/to/pretrained/model \
  --dataset_name data/ami/dataset \
  --freeze_encoder True \
  --do_train \
  --do_eval \
  --predict_with_generate
  
# Freeze decoder (train only encoder)
python finetune_avhubert_seq2seq.py \
  --output_dir ./results/avhubert-seq2seq-ami-decoder-frozen \
  --model_name_or_path /path/to/pretrained/model \
  --dataset_name data/ami/dataset \
  --freeze_decoder True \
  --do_train \
  --do_eval \
  --predict_with_generate
```

## Key Differences from CTC Fine-tuning

The sequence-to-sequence approach differs from CTC-based training in several ways:

1. **Training Process**:
   - Uses teacher forcing during training
   - Handles sequence generation with an autoregressive decoder
   - Requires special tokens (BOS, EOS) for proper sequence handling

2. **Loss Calculation**:
   - Uses cross-entropy loss instead of CTC loss
   - Properly handles padding in sequences

3. **Evaluation**:
   - Uses beam search for better output quality
   - Requires `predict_with_generate=True` for proper metrics calculation

4. **Performance Metrics**:
   - Same metrics (WER/CER), but typically better results on complex language

## Generation Parameters

The sequence-to-sequence model supports various generation parameters:

- `num_beams`: Controls the beam search width (default: 4)
- `max_target_length`: Maximum length of generated transcripts (default: 128)
- `forced_decoder_ids`: Force specific tokens at the beginning of generation
- `suppress_tokens`: Prevent specific tokens from being generated

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size or use gradient accumulation
   - Try reducing `max_duration_in_seconds` to process shorter clips

2. **Slow Training**:
   - Increase `dataloader_num_workers` for faster data loading
   - Consider using a smaller model version

3. **Poor Performance**:
   - Ensure your vocabulary covers all characters in your dataset
   - Try adjusting learning rate and beam search parameters
   - Increase training time or use a pretrained model

## Reference Architecture

The sequence-to-sequence implementation is based on HuggingFace's Speech2Text models but adapted for audio-visual inputs. Key changes include:

1. **Multi-modal encoder**: Handles both audio and visual inputs
2. **Modality fusion**: Combines features from both modalities
3. **Custom cross-attention**: Properly aligns encoder-decoder dimensions
4. **Generation capabilities**: Includes all necessary methods for text generation