#!/bin/bash

LLM="gpt-5"
EFFORT="minimal"
CHUNK=0 # CHUNK=0 means no chunking, all files
NUMBER_OF_CHUNK=5
CHUNK_SIZE=100

PAPER_LIST="PMC_Data/pmcid-ukb-xml-sample.txt"
FOLDER_DIR="PMC_Data/xml"

SAVE_BASE="save/save_sample"
TRIPLE_BASE="${SAVE_BASE}/triple"
CONTENT_BASE="${SAVE_BASE}/log_content"
ENTITY_BASE="${SAVE_BASE}/log_entity"
LOG_BASE="${SAVE_BASE}/log"
mkdir -p "$TRIPLE_BASE"
mkdir -p "$CONTENT_BASE"
mkdir -p "$ENTITY_BASE"
mkdir -p "$LOG_BASE"
SAVE_DIR="${TRIPLE_BASE}/triples_${LLM}_${EFFORT}.json"
LOG_CONTENT="${CONTENT_BASE}/content_${LLM}_${EFFORT}.txt"
LOG_ENTITY="${ENTITY_BASE}/entities_${LLM}_${EFFORT}.txt"

if [ "$CHUNK" -eq 0 ]; then
    LOG="${LOG_BASE}/${LLM}_${EFFORT}.log"
else
    LOG="${LOG_BASE}/${LLM}_${EFFORT}_chunk${CHUNK}.log"
fi

ENTITY_FILTER_PROMPT="entity_filter_prompt_structured"
GET_TRIPLE_PROMPT="get_triple_prompt_structured"
ABLATION_WO_NER=0 # 1 means ablation_wo_ner=true

TITLE_FILE="PMC_Data/csv-ukb-filter.csv"

exec >> "$LOG" 2>&1
set -x
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

nohup python -m extract_from_text.01_extract \
    -s "$SAVE_DIR" \
    -l "$PAPER_LIST" \
    -f "$FOLDER_DIR" \
    -c $CHUNK \
    -n $NUMBER_OF_CHUNK \
    --llm "$LLM" \
    --effort "$EFFORT" \
    --chunk_size $CHUNK_SIZE \
    --log_content "$LOG_CONTENT" \
    --log_entity "$LOG_ENTITY" \
    --entity_filter_prompt "$ENTITY_FILTER_PROMPT" \
    --get_triple_prompt "$GET_TRIPLE_PROMPT" \
    --ablation_wo_ner $ABLATION_WO_NER \
    --title_file "$TITLE_FILE" \
    >> "$LOG" 2>&1 &