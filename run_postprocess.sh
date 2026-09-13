#!/bin/bash
# Turn verified triples into the final knowledge graph CSV.
#
#   ./run_postprocess.sh                          # full chain
#   ./run_postprocess.sh --config configs/my.sh
#   ./run_postprocess.sh --from csv               # resume at a stage
#   ./run_postprocess.sh --only attributes
#
# Stages, in order:
#   merge       text + table verified triples -> one triple set
#   umls_link   entity alignment to UMLS canonical names
#   process     case unification, dedup, semantic typing
#   bios        pull in BIOS edges between entities we already have, and type them
#               (skipped when ENABLE_BIOS_COMPLETION=0)
#   csv         flatten to the edge-list CSV
#   attributes  add per-edge confidence scores from publication metadata
#
# Requires run_extraction.sh to have finished for this RUN_NAME.

set -euo pipefail

CONFIG="configs/default.sh"
FROM=""
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --from)   FROM="$2";   shift 2 ;;
        --only)   ONLY="$2";   shift 2 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# shellcheck source=configs/default.sh
source "$CONFIG"
source "$(dirname "$0")/scripts/lib.sh"

RUN_DIR="${SAVE_ROOT}/${RUN_NAME}"
TEXT_DIR="${RUN_DIR}/text"
TABLE_DIR_OUT="${RUN_DIR}/table"
MERGED_DIR="${RUN_DIR}/merged"
CSV_DIR="${RUN_DIR}/csv"
LOG_DIR="${RUN_DIR}/log"
mkdir -p "$MERGED_DIR" "$CSV_DIR" "$LOG_DIR"

MERGED="${MERGED_DIR}/06_merged.json"
LINKED="${MERGED_DIR}/07_umls_linked.json"
PROCESSED="${MERGED_DIR}/08_processed.json"
BIOS_RAW="${MERGED_DIR}/09_bios_completion.json"
BIOS_TYPED="${MERGED_DIR}/09_bios_completion_typed.json"
FINAL_JSON="${MERGED_DIR}/10_triples.json"
CSV_RAW="${CSV_DIR}/triples.csv"
CSV_FINAL="${CSV_DIR}/triples_with_attributes.csv"

banner "Post-processing run '${RUN_NAME}'"
echo "Config: $CONFIG"
echo "Output: $RUN_DIR"

# ------------------------------------------------------------------- stages

stage_merge() {
    banner "merge"
    local inputs=()
    [[ -f "${TEXT_DIR}/05_verify.json" ]]  && inputs+=("${TEXT_DIR}/05_verify.json")
    [[ -f "${TABLE_DIR_OUT}/05_verify.json" ]] && inputs+=("${TABLE_DIR_OUT}/05_verify.json")
    [[ ${#inputs[@]} -gt 0 ]] || die "No verified triples found under ${RUN_DIR}; run run_extraction.sh first."

    log_to "${LOG_DIR}/merge.log" \
        python -m Proprocess.00_merge_triples \
            --inputs "${inputs[@]}" \
            --output "$MERGED" \
            --stats-output "${MERGED_DIR}/06_merge_stats.txt"
}

stage_umls_link() {
    banner "umls_link"
    log_to "${LOG_DIR}/umls_link.log" \
        python -m Proprocess.01_umls_link \
            --input "$MERGED" \
            --output "$LINKED" \
            --threshold "$UMLS_THRESHOLD"
}

stage_process() {
    banner "process"
    log_to "${LOG_DIR}/process.log" \
        python -m Proprocess.03_process \
            --input "$LINKED" \
            --output "$PROCESSED" \
            --mode ukb \
            --model-dir "$ENTITY_TYPE_MODEL"
}

stage_bios() {
    if [[ "$ENABLE_BIOS_COMPLETION" != "1" ]]; then
        banner "bios (skipped: ENABLE_BIOS_COMPLETION=0)"
        return 0
    fi
    [[ -f "$BIOS_CONCEPTS" ]] || die "BIOS concepts not found: ${BIOS_CONCEPTS} (see README > Data Requirements, or set ENABLE_BIOS_COMPLETION=0)"
    [[ -f "$BIOS_TRIPLETS" ]] || die "BIOS triplets not found: ${BIOS_TRIPLETS}"

    banner "bios completion"
    log_to "${LOG_DIR}/bios_completion.log" \
        python -m Proprocess.02_bios_completion \
            --input "$LINKED" \
            --output "$BIOS_RAW" \
            --bios-concepts "$BIOS_CONCEPTS" \
            --bios-triplets "$BIOS_TRIPLETS" \
            --cache-dir "${RUN_DIR}/bios_cache"

    banner "bios typing"
    log_to "${LOG_DIR}/bios_typing.log" \
        python -m Proprocess.03_process \
            --input "$BIOS_RAW" \
            --output "$BIOS_TYPED" \
            --mode completion \
            --model-dir "$ENTITY_TYPE_MODEL"
}

stage_csv() {
    banner "csv"
    local inputs=("$PROCESSED")
    if [[ "$ENABLE_BIOS_COMPLETION" == "1" && -f "$BIOS_TYPED" ]]; then
        inputs+=("$BIOS_TYPED")
    fi

    log_to "${LOG_DIR}/json_to_csv.log" \
        python -m Proprocess.04_json_to_csv \
            --inputs "${inputs[@]}" \
            --output-csv "$CSV_RAW" \
            --merged-json "$FINAL_JSON"
}

stage_attributes() {
    [[ -f "$CSV_RAW" ]] || die "Missing ${CSV_RAW}; run the 'csv' stage first."
    [[ -f "$PUBLICATION_METADATA" ]] || die "Publication metadata not found: ${PUBLICATION_METADATA} (see README > Data Requirements)"

    banner "attributes"
    log_to "${LOG_DIR}/add_attributes.log" \
        python -m Proprocess.05_add_attributes \
            --input-csv "$CSV_RAW" \
            --output-csv "$CSV_FINAL" \
            --source-file "$PUBLICATION_METADATA" \
            --current-year "$CURRENT_YEAR"
}

# ------------------------------------------------------------------ dispatch

STAGES=(merge umls_link process bios csv attributes)

started=0
[[ -z "$FROM" ]] && started=1

for stage in "${STAGES[@]}"; do
    if [[ -n "$ONLY" ]]; then
        [[ "$stage" == "$ONLY" ]] && "stage_${stage}"
        continue
    fi
    [[ "$stage" == "$FROM" ]] && started=1
    [[ "$started" -eq 1 ]] && "stage_${stage}"
done

banner "Post-processing finished"
echo "Knowledge graph JSON: ${FINAL_JSON}"
echo "Knowledge graph CSV : ${CSV_FINAL}"
