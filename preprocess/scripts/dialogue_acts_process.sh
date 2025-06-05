#!/bin/bash
#SBATCH --job-name=dialogue_acts_process-ami
#SBATCH --output=../logs/process-dialogue_acts-%j.out
#SBATCH --error=../logs/process-dialogue_acts-%j.err
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

# RUNNING MODE: DIALOGUE ACTS TO CSV  --------------------------------------------------------
python dialogue_acts_process.py \
    --input "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts" \
    --output "/deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_laughter/dialogue_acts_laughter" \
    --dialogue_acts_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts/dialogueActs" \
    --da_types_file "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts/ontologies/da-types.xml" \
    --include_adjacency_pairs

# Alternative: Skip adjacency pairs processing if only dialogue acts are needed
# python dialogue_acts_process.py \
#     --input "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts" \
#     --output "/deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_dialogue_acts" \
#     --dialogue_acts_dir "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts/dialogueActs" \
#     --da_types_file "/deepstore/datasets/hmi/speechlaugh-corpus/ami/transcripts/ontologies/da-types.xml" \
#     --skip_adjacency_pairs 