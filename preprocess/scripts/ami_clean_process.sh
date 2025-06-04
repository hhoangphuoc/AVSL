#!/bin/bash
#SBATCH --job-name=ami_clean_process
#SBATCH --output=../logs/ami_clean_process-%j.log
#SBATCH --error=../logs/ami_clean_process-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl

# Create logs directory if it doesn't exist
mkdir -p /home/s2587130/AVSL/preprocess/logs

# Load necessary modules
source /etc/profile.d/modules.sh
module purge
module load nvidia/cuda-11.8

# Set up Python environment
PYTHON_VIRTUAL_ENVIRONMENT=whisper-flamingo
CONDA_ROOT=/home/s2587130/miniconda3/
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

# Print job information
echo "============================================"
echo "Job Information:"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to project directory
cd /home/s2587130/AVSL/preprocess

# Default configuration
CSV_PATH="/deepstore/datasets/hmi/speechlaugh-corpus/ami/dataset/ami-segments-info.csv"
OUTPUT_DIR="/home/s2587130/AVSL/preprocess/output/ami_clean_process"
DATASET_PATH="/home/s2587130/AVSL/data/ami/ami_clean"


echo "============================================"
echo "Processing Configuration:"
echo "CSV Path: $CSV_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Dataset Path: $DATASET_PATH"
echo "============================================"

# Check prerequisites
echo "Checking prerequisites..."

# Check if CSV file exists
if [ ! -f "$CSV_PATH" ]; then
    echo "ERROR: CSV file not found: $CSV_PATH"
    exit 1
fi

# # Check if output directory can be created
# mkdir -p "$OUTPUT_DIR"
# if [ $? -ne 0 ]; then
#     echo "ERROR: Cannot create output directory: $OUTPUT_DIR"
#     exit 1
# fi

# Check if dataset directory can be created
if [ -n "$DATASET_PATH" ]; then
    mkdir -p "$DATASET_PATH"
    if [ $? -ne 0 ]; then
        echo "ERROR: Cannot create dataset directory: $DATASET_PATH"
        exit 1
    fi
fi

# Check available disk space
echo "Checking disk space..."
df -h "$OUTPUT_DIR"
if [ -n "$DATASET_PATH" ]; then
    df -h "$DATASET_PATH"
fi

# Run the optimized processing script
echo "================================================================================================================"
echo "Starting AMI segments dataset processing..."
echo "================================================================================================================"

# Run Python script with properly quoted arguments
echo "Running command: python ami_clean_dataset_process.py --csv_path \"$CSV_PATH\" --output_dir \"$OUTPUT_DIR\" --dataset_path \"$DATASET_PATH\""

if [ -n "$DATASET_PATH" ]; then
    python ami_clean_dataset_process.py --csv_path "$CSV_PATH" --output_dir "$OUTPUT_DIR" --dataset_path "$DATASET_PATH"
else
    python ami_clean_dataset_process.py --csv_path "$CSV_PATH" --output_dir "$OUTPUT_DIR"
fi

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
    
    # Show error information
    echo "Error diagnostics:"
    echo "- Check log files for detailed error messages"
    echo "- Verify input file format and paths"
    echo "- Check available disk space and memory"
    echo "- Ensure all dependencies are properly installed"
    
    exit $PROCESS_RESULT
fi

# Cleanup if requested
if [ "$1" = "--cleanup" ]; then
    echo "Cleaning up temporary files..."
    find "$OUTPUT_DIR" -name "*.tmp" -delete 2>/dev/null || true
fi

echo "Job completed successfully!"
exit 0 