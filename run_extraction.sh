#!/bin/bash
# Triple extraction and refinement, end to end, for both text and tables.
#
#   ./run_extraction.sh                            # full run with configs/default.sh
#   ./run_extraction.sh --config configs/my.sh     # different settings
#   ./run_extraction.sh --source text              # text pipeline only
#   ./run_extraction.sh --from verify              # resume at a stage
#   ./run_extraction.sh --only baseline            # a single stage
#
# Stages, in order:
#   text  : extract llm_filter refine rule_filter verify
#   table : parse select extract llm_filter refine rule_filter verify [baseline]
#
# Every stage is resumable: rerunning one skips the articles already present in
# its output, so an interrupted run is continued by invoking the same command.
#
# To detach:  nohup ./run_extraction.sh > run.log 2>&1 &

set -euo pipefail

CONFIG="configs/default.sh"
SOURCE="both"
FROM=""
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --from)   FROM="$2";   shift 2 ;;
        --only)   ONLY="$2";   shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# shellcheck source=configs/default.sh
source "$CONFIG"
source "$(dirname "$0")/scripts/lib.sh"

RUN_DIR="${SAVE_ROOT}/${RUN_NAME}"
TEXT_DIR="${RUN_DIR}/text"
TABLE_DIR_OUT="${RUN_DIR}/table"
LOG_DIR="${RUN_DIR}/log"
mkdir -p "$TEXT_DIR" "$TABLE_DIR_OUT" "$LOG_DIR"

write_run_config "$RUN_DIR/run_config.json" "$CONFIG"

banner "Run '${RUN_NAME}'  llm=${LLM} effort=${EFFORT} shards=${NUM_SHARDS}"
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"

WO_NER_FLAG=()
if [[ "${ABLATION_WO_NER}" == "1" ]]; then
    WO_NER_FLAG=(--ablation-wo-ner)
fi

TITLE_ARG=()
if [[ -n "${TITLE_FILE}" ]]; then
    TITLE_ARG=(--title-file "$TITLE_FILE")
fi

# --------------------------------------------------------------- text stages

text_extract() {
    run_sharded "${TEXT_DIR}/01_extract" "${TEXT_DIR}/01_extract.json" \
        text_extract \
        python -m extract_from_text.01_extract \
            --paper-list "$PAPER_LIST" \
            --xml-dir "$XML_DIR" \
            --llm "$LLM" --effort "$EFFORT" \
            --entity-filter-prompt "$ENTITY_FILTER_PROMPT" \
            --get-triple-prompt "$GET_TRIPLE_PROMPT" \
            --spacy-model "$SPACY_MODEL" \
            --log-content "${LOG_DIR}/text_content.txt" \
            --log-entity "${LOG_DIR}/text_entities.txt" \
            "${TITLE_ARG[@]}" "${WO_NER_FLAG[@]}"
}

text_llm_filter() { filter_stage "$TEXT_DIR" text llm_filter 01_extract 02_llm_filter; }
text_refine()     { filter_stage "$TEXT_DIR" text refine     02_llm_filter 03_refine; }
text_rule_filter() { filter_stage "$TEXT_DIR" text rule_filter 03_refine 04_rule_filter; }

text_verify() {
    run_sharded_verify "${TEXT_DIR}/05_verify" \
        "${TEXT_DIR}/05_verify.json" "${TEXT_DIR}/05_judgements.json" \
        text_verify \
        python -m extract_from_text.03_verify_or_eval \
            --input "${TEXT_DIR}/04_rule_filter.json" \
            --xml-dir "$XML_DIR" \
            --mode verify \
            --llm "$LLM" --effort "$EFFORT" \
            --evaluation-prompt "$TEXT_EVAL_PROMPT" \
            "${TITLE_ARG[@]}"
}

# -------------------------------------------------------------- table stages

table_parse() {
    banner "table: parse"
    log_to "${LOG_DIR}/table_parse.log" \
        python -m extract_from_table.01_parse_tables \
            --xml-dir "$XML_DIR" \
            --output-dir "$TABLE_DIR" \
            --paper-list "$PAPER_LIST"
}

table_select() {
    run_sharded "${TABLE_DIR_OUT}/00_select" "${TABLE_DIR_OUT}/00_select.json" \
        table_select \
        python -m extract_from_table.02_select_tables \
            --table-dir "$TABLE_DIR" \
            --relation-table-dir "$RELATION_TABLE_DIR" \
            --baseline-table-dir "$BASELINE_TABLE_DIR" \
            --llm "$LLM" --effort "$EFFORT" \
            --select-table-prompt "$SELECT_TABLE_PROMPT"
}

table_extract() {
    run_sharded "${TABLE_DIR_OUT}/01_extract" "${TABLE_DIR_OUT}/01_extract.json" \
        table_extract \
        python -m extract_from_table.03_extract_from_tables \
            --mode triples \
            --table-dir "$RELATION_TABLE_DIR" \
            --llm "$LLM" --effort "$EFFORT" \
            --get-triple-prompt "$TABLE_TRIPLE_PROMPT"
}

table_llm_filter()  { filter_stage "$TABLE_DIR_OUT" table llm_filter 01_extract 02_llm_filter; }
table_refine()      { filter_stage "$TABLE_DIR_OUT" table refine     02_llm_filter 03_refine; }
table_rule_filter() { filter_stage "$TABLE_DIR_OUT" table rule_filter 03_refine 04_rule_filter; }

table_verify() {
    run_sharded_verify "${TABLE_DIR_OUT}/05_verify" \
        "${TABLE_DIR_OUT}/05_verify.json" "${TABLE_DIR_OUT}/05_judgements.json" \
        table_verify \
        python -m extract_from_table.04_verify_or_eval \
            --input "${TABLE_DIR_OUT}/04_rule_filter.json" \
            --table-dir "$RELATION_TABLE_DIR" \
            --mode verify \
            --llm "$LLM" --effort "$EFFORT" \
            --evaluation-prompt "$TABLE_EVAL_PROMPT"
}

table_baseline() {
    run_sharded "${TABLE_DIR_OUT}/baseline" "${TABLE_DIR_OUT}/baseline.json" \
        table_baseline \
        python -m extract_from_table.03_extract_from_tables \
            --mode baseline \
            --table-dir "$BASELINE_TABLE_DIR" \
            --llm "$LLM" --effort "$EFFORT" \
            --get-baseline-prompt "$BASELINE_PROMPT"
}

# ------------------------------------------------------------------ dispatch

TEXT_STAGES=(extract llm_filter refine rule_filter verify)
TABLE_STAGES=(parse select extract llm_filter refine rule_filter verify baseline)

run_chain() {
    local prefix="$1"; shift
    local stages=("$@")
    local started=0
    [[ -z "$FROM" ]] && started=1

    for stage in "${stages[@]}"; do
        if [[ -n "$ONLY" ]]; then
            [[ "$stage" == "$ONLY" ]] && "${prefix}_${stage}"
            continue
        fi
        [[ "$stage" == "$FROM" ]] && started=1
        [[ "$started" -eq 1 ]] && "${prefix}_${stage}"
    done
}

case "$SOURCE" in
    text)  run_chain text  "${TEXT_STAGES[@]}" ;;
    table) run_chain table "${TABLE_STAGES[@]}" ;;
    both)
        run_chain text  "${TEXT_STAGES[@]}"
        run_chain table "${TABLE_STAGES[@]}"
        ;;
    *) echo "--source must be text, table or both" >&2; exit 1 ;;
esac

banner "Extraction finished"
echo "Text triples:  ${TEXT_DIR}/05_verify.json"
echo "Table triples: ${TABLE_DIR_OUT}/05_verify.json"
echo
echo "Next: ./run_postprocess.sh --config ${CONFIG}"
