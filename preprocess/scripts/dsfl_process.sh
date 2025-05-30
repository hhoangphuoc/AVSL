#!/bin/bash
#SBATCH --job-name=dsfl_process-ami_laugh
#SBATCH --output=logs/process-ami_laugh-%j.out
#SBATCH --error=logs/process-ami_laugh-%j.err
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

mkdir -p logs

source /etc/profile.d/modules.sh

module purge # clean the environment before loading new modules
module load nvidia/cuda-11.8
module load nvidia/nvtop 

# Set the python environment you want to use for your code (whisper-flamingo)-----------------
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
#----------------------------------------------------------------------------------------------

cd /home/s2587130/AVSL/preprocess

# RUNNING MODE: DISFLUENCY/LAUGHTER TO CSV  ---------------------------------------------------
# --use_disfluency False
python disfluency_laughter_process.py \
    --input "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts" \
    --output "/deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_laugh" \
    --use_disfluency False

# RUNNING MODE: DISFLUENCY/LAUGHTER SEGMENTATION  ---------------------------------------------
# python dsfl_dataset_process.py \
#     --dsfl_laugh_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl" \
#     --dataset_path "/deepstore/datasets/hmi/speechlaugh-corpus/ami/dsfl/dataset" \
#     --include_lip_videos True