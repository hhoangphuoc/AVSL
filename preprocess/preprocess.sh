#!/bin/bash
#SBATCH --job-name=preprocess-AMI-Corpus-with-lip-videos                    # Job name
#SBATCH -c 16                                               # Number of cores
#SBATCH --mem=16G                                           # Request 8GB memory
#SBATCH --time=24:00:00                                      # Set a walltime limit
#SBATCH --mail-type=BEGIN,END,FAIL                          # Email status changes
#SBATCH --mail-user=hohoangphuoc@student.utwente.nl  # Your email address

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

# Activate your environment (if applicable)
source activate .venv

python dataset_process.py