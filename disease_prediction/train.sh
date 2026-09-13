#!/usr/bin/env bash

set -euo pipefail


# ====================== parameters ======================

DATA_TYPE="ene"  # choices: binary, ene
KGE="TransE"

TIMESTAMP=$(date '+%Y%m%d_%H%M')

SAVE_DIR="disease_prediction/save/nn5_gpt-5-minimal_12r_${KGE}_${DATA_TYPE}"

DATA_DIR="disease_prediction/data/data_gpt-5-minimal_12r_${KGE}"

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


# ====================== helper functions ======================

check_file() {
    local file_path="$1"

    if [[ ! -f "$file_path" ]]; then
        echo "[ERROR] File not found: $file_path" >&2
        exit 1
    fi
}


# ====================== validate configuration ======================

check_file "$TRAIN_LABEL_PATH"
check_file "$VAL_LABEL_PATH"

if [[ "$DATA_TYPE" == "binary" ]]; then

    check_file "$TRAIN_FEATURE_PATH"
    check_file "$VAL_FEATURE_PATH"

    FEATURE_ARGS=(
        --train_feature_path "$TRAIN_FEATURE_PATH"
        --val_feature_path "$VAL_FEATURE_PATH"
    )

elif [[ "$DATA_TYPE" == "ene" ]]; then

    check_file "$TRAIN_ENE_PATH"
    check_file "$VAL_ENE_PATH"

    FEATURE_ARGS=(
        --train_ene_path "$TRAIN_ENE_PATH"
        --val_ene_path "$VAL_ENE_PATH"
    )

else

    echo "[ERROR] Invalid DATA_TYPE: $DATA_TYPE" >&2
    echo "[ERROR] Allowed values: binary, ene" >&2
    exit 1

fi


# ====================== prepare output ======================

mkdir -p "$SAVE_DIR"

{
    echo "============================================================"
    echo "Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Data Type  : $DATA_TYPE"
    echo "KGE        : $KGE"
    echo "Save Dir   : $SAVE_DIR"
    echo "============================================================"
} >> "$LOG"


# ====================== run ======================

nohup python -u -m disease_prediction.train \
    -s "$SAVE_DIR" \
    --data_type "$DATA_TYPE" \
    "${FEATURE_ARGS[@]}" \
    --train_label_path "$TRAIN_LABEL_PATH" \
    --val_label_path "$VAL_LABEL_PATH" \
    -hs "$HIDDEN_SIZE" \
    -bs "$BATCH_SIZE" \
    -lr "$LEARNING_RATE" \
    -wd "$WEIGHT_DECAY" \
    --debug "$DEBUG" \
    >> "$LOG" 2>&1 &

PID=$!


# ====================== log process info ======================

{
    echo "PID        : $PID"
    echo "Log File   : $LOG"
    echo "============================================================"
} >> "$LOG"

echo "[INFO] Training started successfully."
echo "[INFO] PID: $PID"
echo "[INFO] Save directory: $SAVE_DIR"
echo "[INFO] Log: $LOG"