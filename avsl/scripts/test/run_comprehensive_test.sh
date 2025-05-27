#!/bin/bash
#SBATCH --job-name=comprehensive_test_whisper_flamingo_ft  
#SBATCH --output=logs/comprehensive_test_%j.log
#SBATCH --error=logs/comprehensive_test_%j.err
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Comprehensive Test Runner for Whisper-Flamingo AMI Fine-tuning
# Based on successful AV-HuBERT task registration fix


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

conda list

echo "====================================================================="
echo "🚀 Whisper-Flamingo AMI Fine-tuning Comprehensive Test Suite"
echo "====================================================================="
echo ""

# Current directory----------------------------------------------------------
# SCRIPT_DIR would be: /home/s2587130/AVSL/avsl/scripts/test
# PROJECT_ROOT would be: /home/s2587130/AVSL/avsl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"
echo ""

# Set up log file
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/comprehensive_test_${TIMESTAMP}.log"

echo "Test log will be saved to: $LOG_FILE"
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
    echo "🎉 SUCCESS: All tests passed! Ready for training."
    echo ""
    echo "To start training, run:"
    echo "  cd $PROJECT_ROOT"
    echo "  python whisper_flamingo_ft_ami.py $CONFIG_FILE"
else
    echo "❌ Some tests failed. Check the log for details."
    echo ""
    echo "To view the full log:"
    echo "  cat $LOG_FILE"
    echo ""
    echo "To view only errors:"
    echo "  grep -E '(✗|❌|💥|ERROR|Failed)' $LOG_FILE"
fi

echo "====================================================================="
exit $EXIT_CODE 