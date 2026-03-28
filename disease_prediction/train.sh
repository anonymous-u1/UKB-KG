#!/bin/bash

DATA_TYPE="ene" # choices=['binary', 'ene']
KGE="TransE"
SAVE_DIR="disease_prediction/save/nn5_gpt-5_minimal_12r_${KGE}_${DATA_TYPE}"
DATA_DIR="disease_prediction/data/data_gpt-5_minimal_12r_${KGE}"
TRAIN_ENE_PATH="${DATA_DIR}/avr_ene_12r_train.npy"
VAL_ENE_PATH="${DATA_DIR}/avr_ene_12r_val.npy"
TRAIN_FEATURE_PATH="disease_prediction/data/data_raw/features_train.npy"
VAL_FEATURE_PATH="disease_prediction/data/data_raw/features_val.npy"
TRAIN_LABEL_PATH="disease_prediction/data/data_raw/labels_train.npy"
VAL_LABEL_PATH="disease_prediction/data/data_raw/labels_val.npy"
HIDDEN_SIZE=150
BATCH_SIZE=128
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.0
DEBUG=0
LOG="${SAVE_DIR}/train.log"

mkdir -p "$SAVE_DIR"
echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG"

nohup python -m disease_prediction.train \
    -s "$SAVE_DIR" \
    --data_type "$DATA_TYPE" \
    --train_feature_path "$TRAIN_FEATURE_PATH" \
    --val_feature_path "$VAL_FEATURE_PATH" \
    --train_ene_path "$TRAIN_ENE_PATH" \
    --val_ene_path "$VAL_ENE_PATH" \
    --train_label_path "$TRAIN_LABEL_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    -hs $HIDDEN_SIZE \
    -bs $BATCH_SIZE \
    -lr $LEARNING_RATE \
    -wd $WEIGHT_DECAY \
    --debug $DEBUG \
    >> "$LOG" 2>&1 &