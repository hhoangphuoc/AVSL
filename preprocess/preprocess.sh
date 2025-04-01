#!/bin/bash
#SBATCH --job-name=preprocess-ami-lip-videos               # Job name
#SBATCH -c 8                                               # Number of cores (8 is enough for sequential)
#SBATCH --mem=64G                                          # Request 64GB memory
#SBATCH --gres=gpu:ampere:1                                # Request 1 GPU
#SBATCH --time=120:00:00                                   # Set higher walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                         # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl        # Your email address

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
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi # Display GPU info

# Activate your environment (if applicable)
source activate .venv

# RUNNING MODE: PROCESS LIP VIDEOS WITH CHUNKING AND RESUMPTION
python process_in_chunks.py \
    --csv_path "../data/ami_dataset/ami-segments-info.csv" \
    --output_dir "../data/ami_dataset/lip_processing" \
    --lip_video_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/lip_videos" \
    --chunk_size 1000 \
    --batch_size 8 \
    --filter_processed \
    --to_grayscale

# If you want to test with a smaller number of videos:
# python process_in_chunks.py \
#     --csv_path "../data/ami_dataset/ami-segments-info.csv" \
#     --output_dir "../data/ami_dataset/lip_processing" \
#     --lip_video_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/lip_videos" \
#     --chunk_size 1000 \
#     --batch_size 8 \
#     --filter_processed \
#     --to_grayscale \
#     --max_videos 50  # Process only 50 videos for testing

# OTHER PROCESSING MODES (COMMENTED OUT)
# ------------------------------------------------------------

# RUNNING MODE: PROCESS NOT EXISTING SEGMENTS
# python dataset_process.py \
#     --mode segment_not_exist \
#     --transcript_segments_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcript_segments" \
#     --audio_segment_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/audio_segments" \
#     --video_segment_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/video_segments" \
#     --dataset_path "../data/ami_dataset" \
#     --lip_video_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/av_segments/lip_videos" \
#     --batch_size 8 \
#     --to_grayscale True

# RUNNING MODE: PROCESS EXISTING SEGMENTS  
# python dataset_process.py \
#     --mode segment_exist \
#     --source_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami" \
#     --transcript_segments_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcript_segments" \
#     --dataset_path "../data/ami_dataset" 

# RUNNING MODE: PROCESS LIP VIDEOS (ORIGINAL METHOD)
# python dataset_process.py \
#     --mode process_lip \
#     --dataset_path "../data/ami_dataset" \
#     --lip_video_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/lip_videos" \
#     --batch_size 8 \
#     --to_grayscale True  

# RUNNING MODE: DISFLUENCY/LAUGHTER SEGMENTATION  
# python dsfl_dataset_process.py \
#     --dsfl_laugh_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl_laugh" \
#     --dataset_path "../data/dsfl_laugh/dataset" \
#     --extract_lip_videos \
#     --batch_size 8 \
#     --to_grayscale True