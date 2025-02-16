# gpu setup
export CUDA_VISIBLE_DEVICES=$1
task=$2

python run_glue.py \
    --mode train \
    --method btplora \
    --dataset $task \
    --model_type meta-llama/Llama-3.2-3B-Instruct \
    --max_input_len 256 \
    --max_output_len 3 \
    --num_side_tokens 16 \
    --num_warmup_step_ratio 0.06 \
    --batch_size 16 \
    --peft_hidden_size 64 \
    --gradient_accumulation_steps 1 \
    --lr 2e-4 \
    --num_epochs 10 \
    --do_eval \
    --do_train \
    --bf16 \
    --eval_interval 0.1 \
    --seed 24


#models
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
# meta-llama/Llama-2-13b-hf

