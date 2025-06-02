#!/bin/bash
set -e
#SBATCH --job-name=whisper_flamingo_ft  
#SBATCH --output=logs/whisper_flamingo_ft_%j.log
#SBATCH --error=logs/whisper_flamingo_ft_%j.err
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=1 # assert ntasks_per_node == cfg.distributed_world_size // nnodes
#SBATCH --time=240:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Define project root and log directory for clarity
PROJECT_ROOT_ABS="/home/s2587130/AVSL"
LOG_DIR_ABS="${PROJECT_ROOT_ABS}/avsl/logs"

mkdir -p "${LOG_DIR_ABS}"

# Initialize environment module system (if not already done by login shell)
source /etc/profile.d/modules.sh

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


## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

# Set PYTHONPATH for project-specific modules
export PYTHONPATH=${PROJECT_ROOT_ABS}/whisper_flamingo:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT_ABS}/whisper_flamingo/av_hubert:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT_ABS}/whisper_flamingo/av_hubert/fairseq:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT_ABS}/utils:${PYTHONPATH}


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_ROOT_ABS}/avsl"

# srun python -u whisper_ft_muavic_video.py config/visual/v_en_large.yaml
python -u whisper_flamingo_ft_ami.py ${PROJECT_ROOT_ABS}/config/ami_whisper_flamingo_large.yaml
