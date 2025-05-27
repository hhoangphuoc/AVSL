#!/bin/bash
#SBATCH --job-name=ami_process-to-dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=../output/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Activate your environment if needed
source activate .venv
cd ../preprocess
#=================================================================================================
# RUN OTHER PROCESSING MODES
#================================================================================================= 


# RUNNING MODE: PROCESS NOT EXISTING SEGMENTS -----------------------------------------------------
# python dataset_process.py \
#     --mode segment_not_exist \
#     --transcript_segments_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcript_segments" \
#     --audio_segment_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/audio_segments" \
#     --video_segment_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/video_segments" \
#     --dataset_path "../data/ami_dataset" \
#     --lip_video_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/lip_videos" \
#     --batch_size 8 \
#     --to_grayscale True

# RUNNING MODE: PROCESS EXISTING SEGMENTS   -----------------------------------------------------
python dataset_process.py \
    --mode segment_exist \
    --source_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami" \
    --transcript_segments_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcript_segments" \
    --dataset_path "../data/ami/dataset" \
    --extract_lip_videos True \
