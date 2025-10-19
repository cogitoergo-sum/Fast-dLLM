module load cuda/12.8.1
module load cudnn/12.8-v9.10.0
module load uv
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
limit=32

# Baseline - full quality generation with 256 steps
uv run accelerate launch llada/eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--limit ${limit} \
--output_path ./results_gsm8k \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},save_dir=./test_gsm8k,show_speed=True

# Parallel decoding - fast generation with 8 steps using confidence threshold
uv run accelerate launch llada/eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--limit ${limit} \
--output_path ./results_gsm8k_parallel \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,save_dir=./test_gsm8k_parallel,show_speed=True
