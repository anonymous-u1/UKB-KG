#!/bin/bash

LLM="gpt-5"
EFFORT="minimal"
CHUNK=0 # CHUNK=0 means no chunking, all files
NUMBER_OF_CHUNK=4
CHUNK_SIZE=1000

MODE="baseline" # triple or baseline
REL_FOLDER_DIR="PMC_Data/rel_tables"
BASE_FOLDER_DIR="PMC_Data/baseline_tables"
REL_SAVE_PATH="save/save_table/triple/triples_${LLM}_${EFFORT}.json"
BASE_SAVE_PATH="save/save_table/baseline/baseline_${LLM}_${EFFORT}.json"
LOG_BASE="save/save_table/log/${MODE}_${LLM}_${EFFORT}"
if [ "$CHUNK" -eq 0 ]; then
    LOG="${LOG_BASE}.log"
else
    LOG="${LOG_BASE}_${CHUNK}.log"
fi

GET_TRIPLE_PROMPT="extract_triples_from_tables_structured"
GET_BASE_PROMPT="extract_baseline_from_tables_structured"

exec >> "$LOG" 2>&1
set -x
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

nohup python -m extract_from_table.03_extract_from_tables \
    --mode "$MODE" \
    --rel_save_path "$REL_SAVE_PATH" \
    --baseline_save_path "$BASE_SAVE_PATH" \
    --rel_folder_dir "$REL_FOLDER_DIR" \
    --baseline_folder_dir "$BASE_FOLDER_DIR" \
    --llm "$LLM" \
    --effort "$EFFORT" \
    -c $CHUNK \
    -n $NUMBER_OF_CHUNK \
    --chunk_size $CHUNK_SIZE \
    --get_triple_prompt "$GET_TRIPLE_PROMPT" \
    --get_baseline_prompt "$GET_BASE_PROMPT" \
    >> "$LOG" 2>&1 &