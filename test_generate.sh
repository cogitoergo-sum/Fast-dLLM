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

uv run python llada/generate.py >> test_generate.log 2>&1