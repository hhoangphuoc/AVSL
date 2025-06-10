#!/bin/bash
#SBATCH --job-name=whisper_flamingo_ft  
#SBATCH --output=/home/s2587130/AVSL/avsl/logs/whisper_flamingo_ft_%j.log
#SBATCH --error=/home/s2587130/AVSL/avsl/logs/whisper_flamingo_ft_%j.err
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a100
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

# GPU and CUDA diagnostics
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_GPUS: $SLURM_GPUS"
nvidia-smi
echo "PyTorch CUDA availability check:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"


## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

# Verify conda environment
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set PYTHONPATH for project-specific modules
export PYTHONPATH=${PROJECT_ROOT_ABS}/whisper_flamingo:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT_ABS}/whisper_flamingo/av_hubert:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT_ABS}/whisper_flamingo/av_hubert/fairseq:${PYTHONPATH}
export PYTHONPATH=${PROJECT_ROOT_ABS}/utils:${PYTHONPATH}

# CUDA and PyTorch environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_DISABLE=0

# Ensure CUDA libraries are in path
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8

cd "${PROJECT_ROOT_ABS}/avsl"

# Ensure all necessary directories exist
mkdir -p "${PROJECT_ROOT_ABS}/avsl/logs"
mkdir -p "${PROJECT_ROOT_ABS}/avsl/output/train_whisper_flamingo_ft"
mkdir -p "${PROJECT_ROOT_ABS}/avsl/checkpoints/whisper_flamingo_ft"

echo "Starting training script..."
echo "Config file: ${PROJECT_ROOT_ABS}/config/ami_whisper_flamingo_large.yaml"
echo "Working directory: $(pwd)"

# Run the training script
python -u whisper_flamingo_ft_ami.py ${PROJECT_ROOT_ABS}/config/ami_whisper_flamingo_large.yaml
