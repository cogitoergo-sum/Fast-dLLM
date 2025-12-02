#!/bin/bash

#SBATCH --job-name=fast-dlm-run-base
#SBATCH --account=cse585f25_class
#SBATCH --partition=spgpu
#SBATCH --time=05:00:00
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

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=128
num_fewshot=5
steps=$((length / block_length))
model_path='GSAI-ML/LLaDA-8B-Instruct'
# Run a quick test with limit=2
echo "Running quick test for eval_llada.py..."
load_results_path='results/samples_gsm8k_2025-11-20T00-22-22.127800.jsonl'
uv run accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},threshold=0.9,show_speed=True,load_results_path=${load_results_path} \
--limit 128 \

if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed!"
    exit 1
fi
