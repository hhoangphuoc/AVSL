#!/bin/bash
#SBATCH --job-name=laugh_dataset_process-chunked
#SBATCH --output=logs/laugh_process-chunked-%j.log
#SBATCH --error=logs/laugh_process-chunked-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:ampere:1
#SBATCH --constraint=a40
#SBATCH --time=240:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Create logs directory if it doesn't exist
mkdir -p logs

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

# Install/verify dependencies
echo "Checking and installing dependencies..."

# Install dlib if not already installed (handle the installation issue)
echo "Checking dlib installation..."
if ! python -c "import dlib" 2>/dev/null; then
    echo "Installing dlib..."
    # Try different installation methods
    pip install dlib-bin 2>/dev/null || \
    pip install dlib --no-cache-dir 2>/dev/null || \
    conda install -c conda-forge dlib -y 2>/dev/null || \
    echo "Warning: Could not install dlib automatically"
fi

# Test all dependencies
echo "Testing all dependencies..."
python test_dlib_install.py
if [ $? -ne 0 ]; then
    echo "Dependency check failed. Attempting to install missing packages..."
    
    # Try to install any missing packages
    pip install pandas numpy tqdm librosa soundfile opencv-python datasets ffmpeg-python --upgrade
    
    # Test again
    python test_dlib_install.py
    if [ $? -ne 0 ]; then
        echo "Failed to install all dependencies. Please check the error messages above."
        exit 1
    fi
fi

echo "All dependencies verified!"

# Step 1: Run tests
echo "============================================"
echo "Step 1: Running tests"
echo "============================================"

# Run simple CSV test first
echo "Running simple CSV structure test..."
python test_laugh_simple.py
if [ $? -ne 0 ]; then
    echo "Simple CSV test failed!"
    exit 1
fi

# Run full test suite
echo "Running full test suite..."
python test_laugh_dataset.py
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo "Tests failed with exit code $TEST_RESULT"
    echo "Aborting processing."
    exit 1
fi

echo "All tests passed!"

# Step 2: Process dataset
echo "============================================"
echo "Step 2: Processing laughter dataset"
echo "============================================"

# Set processing parameters
# CSV_PATH="/home/s2587130/AVSL/preprocess/ami_laugh_markers.csv"
# OUTPUT_DIR="/home/s2587130/AVSL/data/ami_laugh/segments"
CSV_PATH="/deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_laughter/ami_laugh_markers.csv"
OUTPUT_DIR="/deepstore/datasets/hmi/speechlaugh-corpus/ami/ami_laughter/segments"
DATASET_PATH="/home/s2587130/AVSL/data/ami_laughter/dataset"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATASET_PATH"

# Parse command line arguments for processing mode
PROCESSING_MODE="standard"  # standard or chunked
CHUNK_SIZE=1000
NUM_WORKERS=8
BATCH_SIZE=16

while [[ $# -gt 0 ]]; do
  case $1 in
    --chunked)
      PROCESSING_MODE="chunked"
      shift
      ;;
    --chunk_size=*)
      CHUNK_SIZE="${1#*=}"
      shift
      ;;
    --workers=*)
      NUM_WORKERS="${1#*=}"
      shift
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

echo "Processing configuration:"
echo "Mode: $PROCESSING_MODE"
echo "Chunk size: $CHUNK_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"

if [ "$PROCESSING_MODE" = "chunked" ]; then
    # Use built-in chunked processing with checkpointing
    echo "Using built-in chunked processing with checkpointing..."
    
    python laugh_dataset_process.py \
        --csv_path "$CSV_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_path "$DATASET_PATH" \
        --chunked \
        --chunk_size "$CHUNK_SIZE" \
        --extract_lip_videos \
        --to_grayscale \
        --batch_size "$BATCH_SIZE" \
        --use_shards \
        --files_per_shard 2000 \
        --num_workers "$NUM_WORKERS"
    
    PROCESS_RESULT=$?
    
else
    # Standard processing - run the full dataset at once
    echo "Using standard processing..."
    
    python laugh_dataset_process.py \
        --csv_path "$CSV_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --dataset_path "$DATASET_PATH" \
        --extract_lip_videos \
        --to_grayscale \
        --batch_size "$BATCH_SIZE" \
        --use_shards \
        --files_per_shard 2000 \
        --num_workers "$NUM_WORKERS"
    
    PROCESS_RESULT=$?
fi

# Check processing result
if [ $PROCESS_RESULT -eq 0 ]; then
    echo "============================================"
    echo "Processing completed successfully!"
    echo "End time: $(date)"
    echo "============================================"
    
else
    echo "============================================"
    echo "Processing failed with exit code $PROCESS_RESULT"
    echo "End time: $(date)"
    echo "============================================"
    exit $PROCESS_RESULT
fi

# Print final statistics
echo "============================================"
echo "Final Statistics:"
echo "============================================"

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