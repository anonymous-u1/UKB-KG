#!/bin/bash

LLM="gpt-5"
EFFORT="minimal"
CHUNK=0 # CHUNK=0 means no chunking, all files
NUMBER_OF_CHUNK=8
CHUNK_SIZE=1000

FOLDER_DIR="PMC_Data/tables"
REL_TABLE_SAVE_DIR="PMC_Data/rel_tables"
BASE_TABLE_SAVE_DIR="baseline_tables"
RESULT_SAVE_PATH="save/save_table/select/select_table_result.json"
LOG_BASE="save/save_table/log/select_tables"
if [ "$CHUNK" -eq 0 ]; then
    LOG="${LOG_BASE}.log"
else
    LOG="${LOG_BASE}_${CHUNK}.log"
fi

SELECT_TABLE_PROMPT="select_table_prompt_structured"

exec >> "$LOG" 2>&1
set -x
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

nohup python -m extract_from_table.02_select_tables \
    -f "$FOLDER_DIR" \
    -c $CHUNK \
    -n $NUMBER_OF_CHUNK \
    --chunk_size $CHUNK_SIZE \
    --llm $LLM \
    --effort $EFFORT \
    --rel_table_save_dir "$REL_TABLE_SAVE_DIR" \
    --baseline_table_save_dir "$BASE_TABLE_SAVE_DIR" \
    --result_save_path "$RESULT_SAVE_PATH" \
    --select_table_prompt "$SELECT_TABLE_PROMPT" \
    >> "$LOG" 2>&1 &