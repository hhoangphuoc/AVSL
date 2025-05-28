#!/bin/bash
#SBATCH --job-name=comprehensive_test_whisper_flamingo_ft  
#SBATCH --output=logs/comprehensive_test_%j.log
#SBATCH --error=logs/comprehensive_test_%j.err
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1 
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Comprehensive Test Runner for Whisper-Flamingo AMI Fine-tuning
# Based on successful AV-HuBERT task registration fix

# Print job information
echo "=================================================================="
echo "SLURM Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=================================================================="

mkdir -p logs

# Initialize environment module system (if not already done by login shell)
source /etc/profile.d/modules.sh

# Load modules (adjust versions as needed)
module purge # clean the environment before loading new modules
module load nvidia/cuda-11.8
module load nvidia/nvtop 

## Set the python environment you want to use for your code
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo "=================================================================="
echo "Environment Information:"
echo "Python: $(which python)"
echo "CUDA Version: $(nvcc --version | grep release)"
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=================================================================="

conda list

echo "====================================================================="
echo " Whisper-Flamingo AMI Fine-tuning Comprehensive Test"
echo "====================================================================="
echo ""

SCRIPT_DIR="/home/s2587130/AVSL/avsl/scripts/"
PROJECT_ROOT="/home/s2587130/AVSL/avsl"


echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root directory where the test script is located
cd "$PROJECT_ROOT"
echo "Changed to project root: $(pwd)"
echo ""

# Run the test with optional config file
CONFIG_FILE="${1:-config/ami_whisper_flamingo_large.yaml}"

echo "Starting comprehensive test..."
echo "Config file: $CONFIG_FILE"
echo ""

# Verify key files exist before running
echo "File verification:"
echo "  Test script: $(test -f test_whisper_flamingo.py && echo "EXISTS" || echo "MISSING")"
echo "  Config file: $(test -f "$CONFIG_FILE" && echo "EXISTS" || echo "MISSING")"
echo "  Training script: $(test -f whisper_flamingo_ft_ami.py && echo "EXISTS" || echo "MISSING")"
echo ""

# Run the test and capture output
if [ -f "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
    python test_whisper_flamingo.py "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
else
    echo "Config file not found, running with defaults..."
    python test_whisper_flamingo.py 2>&1 | tee "$LOG_FILE"
fi

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "====================================================================="
echo "Test completed with exit code: $EXIT_CODE"
echo "Full log saved to: $LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] All tests passed! Ready for training."
    echo ""
    echo "To start training, run:"
    echo "  cd $PROJECT_ROOT"
    echo "  python whisper_flamingo_ft_ami.py $CONFIG_FILE"
    echo ""
    echo "Or submit training job with:"
    echo "  sbatch scripts/train/whisper_flamingo_ft.sh"
else
    echo "Some tests failed. Check logs/comprehensive_test.log for details."
    echo ""
    echo "To view the full log:"
    echo "  cat $LOG_FILE"
    echo ""
    echo "To view only errors:"
    echo "  grep -E '(|ERROR|Failed)' $LOG_FILE"
fi

echo "====================================================================="
echo "Job completed at: $(date)"
exit $EXIT_CODE 