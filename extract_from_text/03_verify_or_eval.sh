#!/bin/bash

LLM="gpt-5"
EFFORT="minimal"
CHUNK=0 # CHUNK=0 means no chunking, all files
NUMBER_OF_CHUNK=9
CHUNK_SIZE=1000

MODE='verify' # verify or eval
ABLATION_WO_VERIFIER=0 # 1 means ablation wo verifier
ABLATION_WO_NER=0
ABLATION_0SHOT=0
ABLATION_5SHOT=0
ABLATION_CM=0

INPUT_DIR="save/save_sample/triple/triples_gpt-5_minimal.json"
FOLDER_DIR="PMC_Data/xml"

LOG_BASE="save/save_sample/log/${MODE}_triples_${LLM}_${EFFORT}"
if [ "$CHUNK" -eq 0 ]; then
    LOG="${LOG_BASE}.log"
else
    LOG="${LOG_BASE}_${CHUNK}.log"
fi

EVALUATION_PROMPT="evaluation_prompt_structured"
TITLE_FILE="PMC_Data/csv-ukb-filter.csv"

exec >> "$LOG" 2>&1
set -x
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

nohup python -m extract_from_text.03_verify_or_eval \
    -i "$INPUT_DIR" \
    -f "$FOLDER_DIR" \
    -c $CHUNK \
    -n $NUMBER_OF_CHUNK \
    --chunk_size $CHUNK_SIZE \
    --llm "$LLM" \
    --effort "$EFFORT" \
    --mode "$MODE" \
    --ablation_wo_verifier $ABLATION_WO_VERIFIER \
    --ablation_wo_ner $ABLATION_WO_NER \
    --ablation_0shot $ABLATION_0SHOT \
    --ablation_5shot $ABLATION_5SHOT \
    --ablation_cm $ABLATION_CM \
    --evaluation_prompt "$EVALUATION_PROMPT" \
    --title_file "$TITLE_FILE" \
    >> "$LOG" 2>&1 &