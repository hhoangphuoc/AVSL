# Fine-tuning AV-HuBERT on AMI Dataset

This guide explains how to fine-tune the AV-HuBERT model on the AMI Dataset using the provided scripts.

## Prerequisites

Ensure you have:
- Installed all requirements: `pip install -r requirements.txt`
- Prepared the AMI dataset in HuggingFace format with audio and lip video paths

## Data Preparation

The dataset should contain:
- `audio`: Path to the audio file
- `lip_video`: Path to the lip video file
- `transcript`: Text transcript for ASR training

## Fine-tuning Procedure

### 1. Basic Fine-tuning

Run the following command to fine-tune AV-HuBERT for ASR on the AMI dataset:

```bash
python finetune_avhubert_ami.py \
  --output_dir ./results/avhubert-ami \
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
  --fp16
```

### 2. Using Only One Modality

#### Audio-only Fine-tuning:

```bash
python finetune_avhubert_ami.py \
  --output_dir ./results/avhubert-ami-audio-only \
  --dataset_name data/ami/dataset \
  --use_audio True \
  --use_visual False \
  --do_train \
  --do_eval
```

#### Visual-only Fine-tuning:

```bash
python finetune_avhubert_ami.py \
  --output_dir ./results/avhubert-ami-visual-only \
  --dataset_name data/ami/dataset \
  --use_audio False \
  --use_visual True \
  --do_train \
  --do_eval
```

### 3. Modality Dropout for Robustness

To improve model robustness to missing modalities, use modality dropout during training:

```bash
python finetune_avhubert_ami.py \
  --output_dir ./results/avhubert-ami-robust \
  --dataset_name data/ami/dataset \
  --audio_drop_prob 0.1 \
  --visual_drop_prob 0.1 \
  --do_train \
  --do_eval
```

### 4. Fine-tuning from Pre-trained Model

To continue training from a pre-trained model:

```bash
python finetune_avhubert_ami.py \
  --output_dir ./results/avhubert-ami-continued \
  --model_name_or_path /path/to/pretrained/model \
  --dataset_name data/ami/dataset \
  --do_train \
  --do_eval
```

### 5. Freezing the Encoder

If you want to fine-tune only the decoder while keeping the encoder frozen:

```bash
python finetune_avhubert_ami.py \
  --output_dir ./results/avhubert-ami-frozen-encoder \
  --model_name_or_path /path/to/pretrained/model \
  --dataset_name data/ami/dataset \
  --freeze_encoder True \
  --do_train \
  --do_eval
```

## Understanding the Code

### Key Components

1. **Data Loading**: `utils/data_loading.py` implements the dataset class and preprocessing logic
2. **Model Configuration**: `config/av_hubert_config.py` contains the model configuration
3. **Model Architecture**: `models/av_hubert_model.py` implements the AV-HuBERT model

### Handling Missing Modalities

The code efficiently handles cases where one modality (audio or video) might be missing:

- During training, use `audio_drop_prob` and `visual_drop_prob` to randomly drop modalities
- During inference, the model automatically adapts based on available inputs

### Fine-tuning Hyperparameters

Recommended hyperparameters for fine-tuning:

- Learning rate: 3e-5 to 5e-5
- Batch size: 8-16 (depending on available GPU memory)
- Training epochs: 15-30 (with early stopping)
- Fusion type: "concat" (default) or "add"

## Evaluation

The model is evaluated using Word Error Rate (WER) and Character Error Rate (CER) metrics. Lower values indicate better performance.

## Model Outputs

The fine-tuned model can be used for:
- ASR (Automatic Speech Recognition)
- Audio-Visual Speech Recognition
- Testing robustness against noisy inputs (by dropping one modality)

## Reference

For more details on the architecture, refer to the `README-technical.md` file.