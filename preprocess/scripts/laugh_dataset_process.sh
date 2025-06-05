#!/bin/bash
#SBATCH --job-name=laugh_dataset_process-chunked(fluent_laughter-balanced)
#SBATCH --output=../logs/laugh_process-chunked-balance-fixed-%j.log
#SBATCH --error=../logs/laugh_process-chunked-balance-fixed-%j.err
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:ampere:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Create logs directory if it doesn't exist
mkdir -p /home/s2587130/AVSL/preprocess/logs

# Load modules
source /etc/profile.d/modules.sh
module purge
module load nvidia/cuda-11.8
module load nvidia/nvtop

# Set the python environment (whisper-flamingo)
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

# Print environment info
echo "============================================"
echo "Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "============================================"

# Set Python environment variables
export PYTHONUNBUFFERED=1

# Change to preprocess directory
cd /home/s2587130/AVSL/preprocess

# echo "All dependencies verified!"

# Step 1: Run tests
# echo "================================================================================================================"
# echo "Step 1: Running tests"
# echo "================================================================================================================"

# # Run simple CSV test first
# echo "Running simple CSV structure test..."
# python test/test_laugh_simple.py
# if [ $? -ne 0 ]; then
#     echo "Simple CSV test failed!"
#     exit 1
# fi

# # Run full test suite
# echo "Running full test suite..."
# python test/test_laugh_dataset.py
# TEST_RESULT=$?

# if [ $TEST_RESULT -ne 0 ]; then
#     echo "Tests failed with exit code $TEST_RESULT"
#     echo "Aborting processing."
#     exit 1
# fi

# echo "All tests passed!"

# Step 2: Process dataset
echo "================================================================================================================"
echo "Step 2: Processing laughter dataset"
echo "================================================================================================================"

# Set config path
CONFIG_PATH="/home/s2587130/AVSL/config/laugh_dataset_process.yaml"

# Validate config path
if [ ! -r "$CONFIG_PATH" ]; then
    echo "WARNING: Config file not readable: $CONFIG_PATH"
    echo "Will use default configuration or command-line arguments."
    CONFIG_ARG=""
else
    CONFIG_ARG="--config $CONFIG_PATH"
    echo "Using configuration file: $CONFIG_PATH"
fi

# Parse command line arguments for processing mode
PROCESSING_MODE="chunked"  # standard or chunked
CHUNK_SIZE=1000
NUM_WORKERS=8
BATCH_SIZE=16
BALANCE=True
ADDITIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --chunked)
      PROCESSING_MODE="chunked"
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS --chunked"
      shift
      ;;
    --chunk_size=*)
      CHUNK_SIZE="${1#*=}"
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS --chunk_size=$CHUNK_SIZE"
      shift
      ;;
    --workers=*)
      NUM_WORKERS="${1#*=}"
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS --num_workers=$NUM_WORKERS"
      shift
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS --batch_size=$BATCH_SIZE"
      shift
      ;;
    --output_dir=*)
      OUTPUT_DIR="${1#*=}"
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS --output_dir=$OUTPUT_DIR"
      shift
      ;;
    --dataset_path=*)
      DATASET_PATH="${1#*=}"
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS --dataset_path=$DATASET_PATH"
      shift
      ;;
    --balance)
      ADDITIONAL_ARGS="$ADDITIONAL_ARGS --balance"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

echo "============================================"
echo "Processing configuration:"
echo "Mode: $PROCESSING_MODE"
echo "Chunk size: $CHUNK_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Balance: $BALANCE"
echo "Additional arguments: $ADDITIONAL_ARGS"
echo "============================================"

# Run the processing script with the config file and any additional arguments
echo "Running laugh_dataset_process.py with configuration..."
python laugh_dataset_process.py $CONFIG_ARG $ADDITIONAL_ARGS

PROCESS_RESULT=$?

# Check processing result
if [ $PROCESS_RESULT -eq 0 ]; then
    echo "================================================================================================================"
    echo "Processing completed successfully!"
    echo "End time: $(date)"
    echo "================================================================================================================"
    
else
    echo "================================================================================================================"
    echo "Processing failed with exit code $PROCESS_RESULT"
    echo "End time: $(date)"
    echo "================================================================================================================"
    exit $PROCESS_RESULT
fi

# Print final statistics
echo "================================================================================================================"
echo "Final Statistics:"
echo "================================================================================================================"

if [ -f "$OUTPUT_DIR/dataset_records.json" ]; then
    python -c "
import json
with open('$OUTPUT_DIR/dataset_records.json', 'r') as f:
    records = json.load(f)
print(f'Total records: {len(records)}')

# Count by type
types = {}
for r in records:
    t = r.get('disfluency_type', 'unknown')
    types[t] = types.get(t, 0) + 1
print(f'By type: {types}')

# Count media availability
has_audio = sum(1 for r in records if r.get('audio'))
has_video = sum(1 for r in records if r.get('video_path'))
has_lip = sum(1 for r in records if r.get('lip_video'))
print(f'Has audio: {has_audio}/{len(records)} ({has_audio/len(records)*100:.1f}%)')
print(f'Has video: {has_video}/{len(records)} ({has_video/len(records)*100:.1f}%)')
print(f'Has lip video: {has_lip}/{len(records)} ({has_lip/len(records)*100:.1f}%)')
"
fi

exit 0 