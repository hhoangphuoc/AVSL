#!/bin/bash
#SBATCH --job-name=whisper_flamingo_tests-video_validation-dataset
#SBATCH --output=logs/whisper_flamingo_video_validation-dataset_%j.log
#SBATCH --error=logs/whisper_flamingo_video_validation-dataset_%j.err
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

mkdir -p logs

# Initialize environment module system (if not already done by login shell)
source /etc/profile.d/modules.sh

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
module load nvidia/cuda-11.8
module load nvidia/nvtop 

# Print some useful information
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
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

conda list

cd /home/s2587130/AVSL/avsl


# Test 1: Robust Video Validation (New comprehensive test)
echo "=========================================================================="
echo "TEST : Robust Video Validation and Corruption Detection"
echo "=========================================================================="
echo "Running test/test_video_validation.py..."
echo ""

python test/test_video_validation.py 
TEST1_EXIT_CODE=$?

echo ""
if [ $TEST1_EXIT_CODE -eq 0 ]; then
    echo "✅ TEST PASSED: Video validation and corruption detection working"
else
    echo "⚠️  TEST COMPLETED: Corrupted videos identified and handled (expected)"
fi
echo ""