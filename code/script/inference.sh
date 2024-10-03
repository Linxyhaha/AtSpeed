export PYTHONUNBUFFERED=1



DATASET=beauty

LOG_DIR=YOUR_LOG_DIR
OUTPUT_DIR=YOUR_OUTPUT_DIR
DRAFT_MODEL=DRAFT_MODEL_PATH
TARGET_BASE_MODEL=TARGET_BASE_MODEL_PATH
TARGET_CKPT_PATH=TARGET_CKPT_PATH
DRATF_MODEL_NAME=DRAFT_MODEL_NAME
TARGET_MODEL_NAME=TARGET_MODEL_NAME

draft_beam_size=40
run_beam_sizes="[20,10]"
do_sample=""
# do_sample="--do_sample"
seed=2025

GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u inference.py \
    --llama \
    --filter_items \
    --index_file .LCRec-1e-3lr.json \
    --dataset $DATASET \
    --draft_model  $DRAFT_MODEL \
    --TARGET_BASE_MODEL $TARGET_BASE_MODEL_PATH \
    --target_ckpt_path $TARGET_CKPT_PATH \
    --draft_model_name $DRAFT_MODEL_NAME \
    --target_model_name $TARGET_MODEL_NAME \
    --run_beam_sizes $run_beam_sizes \
    --draft_beam_size $draft_beam_size \
    $do_sample \
    --seed $seed \
&> ${LOG_DIR}/${DATASET}/infer_${TARGET_MODEL_NAME}_${DRAFT_MODEL_NAME}_${run_beam_sizes}-${draft_beam_size}${do_sample}_seed${seed}.log &

