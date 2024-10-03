accelerate config
# export CUDA_LAUNCH_BLOCKING=1
# export WANDB_DISABLED=true
export HF_HUB_CACHE="YOUR_HF_HUB_CACHE_PATH"

DATASET=beauty

LOG_DIR=YOUR_LOG_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR
BASE_MODEL=YOUR_BASE_MODEL_PATH

OUTPUT_DIR=${OUTPUT_DIR}/target/llama-7B_lora

nohup accelerate launch finetune_llama.py \
    --llama \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --data_path ../data \
    --learning_rate 0.0001 \
    --train_batch_size 256 \
    --micro_batch_size 32 \
    --epochs 20 \
    --index_file .LCRec-1e-3lr.json \
    --cutoff_len 512 \
    --train_on_inputs \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "['q_proj','v_proj']" \
    --only_train_response \
    --wandb_project AtSpeed-target \
    --wandb_run_name AtSpeed-target-response \
    --wandb_watch all \
    &> ${LOG_DIR}/train_finetune_llama_7B.log &
