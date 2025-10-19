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

uv run accelerate launch llada/eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--limit 32 \
--output_path ./results_gsm8k \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},save_dir=./test_gsm8k,show_speed=True
