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

MODEL_CLASS=AtSpeedRModel  # WordKDModel, TVDKDModel, SeqKDModel, AtSpeeSModel
ALPHA=0.7
# constrained_loss=""
# constrained_softmax=""
constrained_loss="--constrained_loss"
constrained_softmax="--constrained_softmax"
TEMPERATURE=0.001
WD=0.1

PORT=7270
SUFFIX=${MODEL_CLASS:0:-5}_${ALPHA}${constrained_loss}${constrained_softmax}_temp${TEMPERATURE}_wd${WD}
OUTPUT_DIR=OUTPUT_DIR/${dataset}/llama-68M_${SUFFIX}

nohup accelerate launch \
    --config_file accelerate.yaml \
    --main_process_port ${PORT} \
    train.py \
    --llama \
    --target_model ${TARGET_MODEL} \
    --base_model ${BASE_MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --dataset ${DATASET} \
    --learning_rate 0.001 \
    --train_batch_size 256 \
    --micro_batch_size 64 \
    --epochs 20 \
    --index_file .LCRec-1e-3lr.json \
    --cutoff_len 512 \
    --train_on_inputs \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "['q_proj','v_proj']" \
    --only_train_response \
    --wandb_watch all \
    --wandb_project AtSpeed-draft \
    --wandb_run_name AtSpeed-${SUFFIX}\
    --train_data ${TRAIN_TEACHER_DATA} \
    --valid_data ${EVAL_TEACHER_DATA} \
    --model_class ${MODEL_CLASS} \
    --alpha ${ALPHA} \
    ${constrained_loss} \
    ${constrained_softmax} \
    --temperature_softmax ${TEMPERATURE} \
    --weight_decay ${WD} \
    &>  ${LOG_DIR}/${DATASET}/train_${SUFFIX}.log &
