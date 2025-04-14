# accelerate config
# export CUDA_LAUNCH_BLOCKING=1
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1
export HF_HUB_CACHE="YOUR_HF_HUB_CACHE_PATH"



DATASET=beauty
TRAIN_TEACHER_DATA=YOUR_TRAIN_TEACHER_DATA
EVAL_TEACHER_DATA=YOUR_EVAL_TEACHER_DATA

LOG_DIR=YOUR_LOG_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR
TARGET_MODEL=YOUR_TARGET_MODEL_PATH
BASE_MODEL=YOUR_BASE_MODEL_PATH

OUTPUT_DIR=${OUTPUT_DIR}/${dataset}

GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u generate_teacher_data.py \
    --llama \
    --filter_items \
    --index_file .LCRec-1e-3lr.json \
    --dataset $DATASET \
    --base_model ${BASE_MODEL} \
    --target_model ${TARGET_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --micro_batch_size 1 \
    --beam_size 20 \
    &> ${LOG_DIR}/${DATASET}/generate_teacher_data.log &
