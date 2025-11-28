#!/bin/bash

#SBATCH --job-name=fast-dlm-run-base
#SBATCH --account=cse585f25_class
#SBATCH --partition=spgpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60g
#SBATCH --gpus=a40:1

module load cuda
module load cudnn
module load uv

mkdir -p logs
LOG_FILE="logs/test_generate_$(date +%Y%m%d_%H%M%S).log"
uv run python llada/generate.py >> "$LOG_FILE" 2>&1