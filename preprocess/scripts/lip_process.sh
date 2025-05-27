#!/bin/bash
#SBATCH --job-name=ami-lip_extraction-multiprocessing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=72:00:00
#SBATCH --output=../output/slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Activate your environment if needed
source activate .venv
cd ../preprocess

#=================================================================================================
# RUN PARALLEL PROCESSING / MULTIPROCESSING FOR LIP VIDEO EXTRACTION
#================================================================================================= 

echo "Starting lip video extraction with GPU acceleration"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Set Python environment variables
export PYTHONUNBUFFERED=1

# Configure system
NUM_WORKERS=8  # Leave some cores for system processes
NUM_TASKS_PER_WORKER=5  # Process 5 videos before respawning worker
CHUNK_SIZE=1000  # Process 1000 videos per chunk
BATCH_SIZE=16   # Number of frames to process in a batch

# Default is to use multiprocessing
USE_SEQUENTIAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --sequential)
      USE_SEQUENTIAL=true
      shift
      ;;
    --chunk_size=*)
      CHUNK_SIZE="${1#*=}"
      shift
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --workers=*)
      NUM_WORKERS="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting processing with configuration:"
echo "Mode: $([ "$USE_SEQUENTIAL" = true ] && echo "Sequential" || echo "Multiprocessing with $NUM_WORKERS workers")"
echo "Tasks per worker: $NUM_TASKS_PER_WORKER"
echo "Chunk size: $CHUNK_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Adaptive memory: Enabled"

# Build command
CMD="python process_in_chunks.py \
  --csv_path \"../data/ami/ami-segments-info.csv\" \
  --output_dir \"../data/ami/lip_processing\" \
  --lip_video_dir \"/deepstore/datasets/hmi/speechlaugh-corpus/ami/video_segments/lips\" \
  --chunk_size $CHUNK_SIZE \
  --batch_size $BATCH_SIZE \
  --filter_processed \
  --to_grayscale"

# Add mode-specific arguments
if [ "$USE_SEQUENTIAL" = true ]; then
  CMD="$CMD --use_sequential"
else
  CMD="$CMD --num_workers $NUM_WORKERS --max_tasks_per_child $NUM_TASKS_PER_WORKER"
fi

# Print and execute the command
echo "Executing: $CMD"
eval $CMD

exit_status=$?

echo "Processing completed with exit status: $exit_status"

if [ $exit_status -eq 0 ]; then
  echo "Job completed successfully!"
else
  echo "Job failed with exit code $exit_status"
fi

exit $exit_status
