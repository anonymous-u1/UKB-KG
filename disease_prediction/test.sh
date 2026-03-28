#!/bin/bash

DATA_TYPE="ene" # choices=['binary', 'ene']
KGE="TransE"
SAVE_DIR="disease_prediction/save/nn5_gpt-5_minimal_12r_${KGE}_${DATA_TYPE}"
DATA_DIR="disease_prediction/data/data_gpt-5_minimal_12r_${KGE}"
VAL_ENE_PATH="${DATA_DIR}/avr_ene_12r_val.npy"
TEST_ENE_PATH="${DATA_DIR}/avr_ene_12r_test.npy"
VAL_FEATURE_PATH="disease_prediction/data/data_raw/features_val.npy"
TEST_FEATURE_PATH="disease_prediction/data/data_raw/features_test.npy"
TRAIN_LABEL_PATH="disease_prediction/data/data_raw/labels_train.npy"
VAL_LABEL_PATH="disease_prediction/data/data_raw/labels_val.npy"
TEST_LABEL_PATH="disease_prediction/data/data_raw/labels_test.npy"
HIDDEN_SIZE=150
BATCH_SIZE=128
DEBUG=0
LOG="${SAVE_DIR}/test.log"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') ==========" >> "$LOG"


nohup python -m disease_prediction.test \
    --save_dir "$SAVE_DIR" \
    --test_name "$TEST_NAME" \
    --data_type "$DATA_TYPE" \
    --val_feature_path "$VAL_FEATURE_PATH" \
    --test_feature_path "$TEST_FEATURE_PATH" \
    --val_ene_path "$VAL_ENE_PATH" \
    --test_ene_path "$TEST_ENE_PATH" \
    --train_label_path "$TRAIN_LABEL_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    --test_label_path "$TEST_LABEL_PATH" \
    -hs $HIDDEN_SIZE \
    -bs $BATCH_SIZE \
    --debug $DEBUG \
    >> "$LOG" 2>&1 &