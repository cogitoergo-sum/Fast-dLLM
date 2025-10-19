#!/bin/bash

#SBATCH --job-name=fast-dlm-run-Parallel
#SBATCH --account=cse585f25_class
#SBATCH --partition=spgpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60g
#SBATCH --gpus=a40:1

module load cuda
module load cudnn
module load uv

uv run python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"

if [ $? -eq 0 ]; then
    echo "CUDA available for PyTorch"
else
    echo "Failed to use CUDA for PyTorch, exiting..."
    exit
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
limit=128

# Parallel decoding - fast generation with 8 steps using confidence threshold
uv run accelerate launch llada/eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--limit ${limit} \
--output_path ./results_gsm8k_parallel \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,save_dir=./test_gsm8k_parallel,show_speed=True
