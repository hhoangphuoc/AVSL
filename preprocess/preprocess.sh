#!/bin/bash
#SBATCH --job-name=preprocess-dsfl-ami-dataset                    # Job name
#SBATCH -c 16                                               # Number of cores
#SBATCH --mem=16G                                           # Request 32GB memory
#SBATCH --gres=gpu:ampere:1                                  # Request 1 GPU
#SBATCH --time=72:00:00                                      # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                          # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

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

# Activate your environment (if applicable)
source activate .venv

# RUNNING MODE: TRANSCRIPT SEGMENTATION ------------------------------------------------------------    
# python dataset_process.py \
#     --transcript_segments_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcript_segments" \
#     --audio_segment_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/audio_segments" \
#     --video_segment_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/video_segments" \
#     --dataset_path "../data/ami_dataset" \
#     --lip_video_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/lip_videos" \
    # --use_gpu True \
    # --use_parallel True \
    # --batch_size 16 \
    # --batch_process True

# RUNNING MODE: DISFLUENCY/LAUGHTER SEGMENTATION ------------------------------------------------------------    
python dsfl_dataset_process.py \
    --dsfl_laugh_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl" \
    --dataset_path "../data/dsfl/dataset" \
    # --use_parallel True \
    # --batch_size 16 \
    # --batch_process True \
    # --to_grayscale True