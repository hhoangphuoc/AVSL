#!/bin/bash
#SBATCH --job-name=hubert_ft
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:ampere:1
#SBATCH --output=logs/hubert_ft_%j.log
#SBATCH --error=logs/hubert_ft_%j.err
#SBATCH --time=240:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl


# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
module load nvidia/cuda-11.8
module load nvidia/nvtop 


# Print some useful information
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)" # log hostname
echo "Working Directory = $(pwd)"
echo "Name of nodes used          : "$SLURM_JOB_NODELIST
echo "Gpu devices                 : "$CUDA_VISIBLE_DEVICES
echo "Starting worker: "

echo "Number of CPU cores = $(nproc)"
echo "SLURM_CPUS_PER_TASK:          "$SLURM_CPUS_PER_TASK


# Activate conda environment if needed (adjust to your environment setup)
source activate .venv
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate .venv

# Go to project root directory
cd /home/s2587130/AVSL

# Set variables
MODEL_NAME_OR_PATH="checkpoints/hf-hubert/hubert-large-ls960-ft"  # HuBERT large fine-tuned on LibriSpeech
OUTPUT_DIR="output/hubert_ft"
DATASET_NAME="ami"
CACHE_DIR="./checkpoints/hf-hubert/"
BATCH_SIZE=16  # Larger batch size for CTC model
GRAD_ACCUM=2
LR=3e-5
NUM_EPOCHS=15
MAX_DURATION=30.0  # Maximum duration in seconds

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Run fine-tuning
python finetune_hubert.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --cache_dir $CACHE_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET_NAME \
    --dataset_cache_dir "data" \
    --max_duration_in_seconds $MAX_DURATION \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --fp16 \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --metric_for_best_model "wer" \
    --greater_is_better false \
    --do_train \
    --do_eval \
    --freeze_feature_encoder False \
    --attention_dropout 0.1 \
    --hidden_dropout 0.1 \
    --feat_proj_dropout 0.1 \
    --final_dropout 0.1 \
    --mask_time_prob 0.05 \
    --layerdrop 0.1