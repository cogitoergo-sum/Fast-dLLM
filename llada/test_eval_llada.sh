#!/bin/bash

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model_path='GSAI-ML/LLaDA-8B-Instruct'

# Run a quick test with limit=2
echo "Running quick test for eval_llada.py..."
load_results_path='results/samples_gsm8k_2025-11-19T23-07-40.279237.jsonl'
# accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
accelerate launch --cpu eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,load_results_path=${load_results_path} \
--limit 2

if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed!"
    exit 1
fi
