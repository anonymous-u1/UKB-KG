#!/bin/bash

# ====================== parameters ======================
LLM="gpt-5"
EFFORT="minimal"
DATA="pubmedqa"
METHOD="rag"
TEMPLATE="${DATA}_${METHOD}"
PARAMS=3 # k1k2k3，retrieve params
SCORE=2 # w1w2w3，score combination
CHUNK=0 # CHUNK=0 means no chunking, all files

if [[ "$METHOD" == *"rag"* ]]; then
    NAME="${LLM}_${EFFORT}_${DATA}_${METHOD}_param-${PARAMS}_score-${SCORE}_chunk-${CHUNK}"
elif [[ "$METHOD" == *"cot"* || "$METHOD" == *"none"* ]]; then
    NAME="${LLM}_${EFFORT}_${DATA}_${METHOD}_chunk-${CHUNK}"
fi

# The knowledge graph this run retrieves from: $SAVE_ROOT/$RUN_NAME of the
# run_postprocess.sh run that built it.
KG_DIR="save/sample"

DATA_PATH="rag/data/${DATA}/test_set.json"
SAVE_PATH="rag/save/${NAME}.json"
PROMPT_PATH="rag/save/prompt/${NAME}.txt"
RESPONSE_PATH="rag/save/response/${NAME}.txt"
TRIPLE_CSV_PATH="${KG_DIR}/csv/triples_with_attributes.csv"
UKB_NODE_EMB_PATH="${KG_DIR}/csv/node_emb.npz"
DATA_ENTITY_EMB_PATH="rag/data/${DATA}/test_set_entity_map_emb.npz"
LOG="rag/log/${NAME}_$(date '+%m-%d-%H-%M').log"

mkdir -p rag/log rag/save rag/save/prompt rag/save/response

exec > >(tee -a "$LOG") 2>&1
set -x
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

nohup python -m rag.02_rag \
    --llm "$LLM" \
    --effort "$EFFORT" \
    --data "$DATA" \
    --method "$METHOD" \
    --template "$TEMPLATE" \
    --params $PARAMS \
    --score $SCORE \
    --chunk $CHUNK \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --response_path "$RESPONSE_PATH" \
    --triple_csv_path "$TRIPLE_CSV_PATH" \
    --ukb_node_emb "$UKB_NODE_EMB_PATH" \
    --data_entity_emb "$DATA_ENTITY_EMB_PATH" \
    >> "$LOG" 2>&1 &