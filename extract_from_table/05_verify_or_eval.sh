#!/bin/bash

LLM="gpt-5"
EFFORT="minimal"
CHUNK=0 # CHUNK=0 means no chunking, all files
NUMBER_OF_CHUNK=6
CHUNK_SIZE=1000

MODE='verify' # verify or eval
ABLATION_WO_VERIFIER=0 # 1 means ablation wo verifier

INPUT_DIR="save/save_table/triple/triples_gpt-5_minimal.json"
FOLDER_DIR="PMC_Data/rel_tables"

LOG_BASE="save/save_table/log/${MODE}_triples_${LLM}_${EFFORT}"
if [ "$CHUNK" -eq 0 ]; then
    LOG="${LOG_BASE}.log"
else
    LOG="${LOG_BASE}_${CHUNK}.log"
fi

EVALUATION_PROMPT="evaluation_table_prompt_structured"

exec >> "$LOG" 2>&1
set -x
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

nohup python -m extract_from_table.05_verify_or_eval \
    -i "$INPUT_DIR" \
    -f "$FOLDER_DIR" \
    -c $CHUNK \
    -n $NUMBER_OF_CHUNK \
    --chunk_size $CHUNK_SIZE \
    --llm "$LLM" \
    --effort "$EFFORT" \
    --mode "$MODE" \
    --ablation_wo_verifier $ABLATION_WO_VERIFIER \
    --evaluation_prompt "$EVALUATION_PROMPT" \
    >> "$LOG" 2>&1 &