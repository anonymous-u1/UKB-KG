#!/bin/bash

LLM="gpt-5"
EFFORT="minimal"
MODE='rule_filter' # llm_filter or refine or rule_filter

TRIPLE_PATH="save/save_table/triple/triples_gpt-5_minimal.json"
LOG="save/save_table/log/${MODE}_triples_gpt-5_minimal.log"

FILTER_PROMPT="triple_filter_prompt_structured"
REFINE_PROMPT="triple_refine_prompt_structured"

exec >> "$LOG" 2>&1
set -x
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

nohup python -m extract_from_text.02_filter_or_refine \
    -p "$TRIPLE_PATH" \
    -m "$MODE" \
    --llm "$LLM" \
    --effort "$EFFORT" \
    --filter_prompt "$FILTER_PROMPT" \
    --refine_prompt "$REFINE_PROMPT" \
    >> "$LOG" 2>&1 &