#!/bin/bash
#SBATCH --job-name=dsfl_process-dataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=../output/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Activate your environment if needed
source activate .venv
cd ../preprocess

# RUNNING MODE: DISFLUENCY/LAUGHTER SEGMENTATION  ---------------------------------------------
python dsfl_dataset_process.py \
    --dsfl_laugh_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl" \
    --dataset_path "../deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/dataset" \
    --include_lip_videos True