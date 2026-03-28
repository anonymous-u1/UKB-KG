#!/bin/bash

MODE="eval"
TRAIN_DATA="entity_type_tag/data/train_data_35w.txt"
EVAL_DATA="entity_type_tag/data/eval_data_35w.txt"
TEST_DATA="entity_type_tag/data/test_data_35w.txt"
SAVE_DIR="entity_type_tag/save/save_35w_256"
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=256
GRADIENT_ACCUMULATION_STEPS=1
LR=2e-5
EPOCH=15
LOG="${SAVE_DIR}/logs/${MODE}_$(date '+%Y-%m-%d-%H:%M').log"

export WANDB_PROJECT="UKB"
export WANDB_RUN_GROUP="biobert_classifier"

mkdir -p "${SAVE_DIR}/logs"
exec > >(tee -a "$LOG") 2>&1
set -x

# --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
nohup env CUDA_VISIBLE_DEVICES=5 WANDB_DISABLED=false python entity_type_tag/02_train.py \
  --mode "$MODE" \
  --train_data_path "$TRAIN_DATA" \
  --eval_data_path "$EVAL_DATA" \
  --test_data_path "$TEST_DATA" \
  --save_dir "$SAVE_DIR" \
  --max_length 64 \
  --per_device_train_batch_size $TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $EVAL_BATCH_SIZE \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --fp16 \
  --early_stopping \
  >> "$LOG" 2>&1 &