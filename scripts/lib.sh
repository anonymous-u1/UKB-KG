#!/bin/bash
# Helpers shared by run_extraction.sh and run_postprocess.sh.
#
# The important one is run_sharded: it launches $NUM_SHARDS copies of a stage
# concurrently, each writing its own shard file, then concatenates the shards
# into the single record file the next stage reads.

banner() {
    echo
    echo "=============================================================="
    echo "  $*  ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "=============================================================="
}

die() {
    echo "❌ $*" >&2
    exit 1
}

# Must match utils.pipeline.shard_path so the Python side finds the same files.
shard_file() {
    local stage_dir="$1" shard="$2"
    if [[ "$NUM_SHARDS" -le 1 ]]; then
        echo "${stage_dir}/all.json"
    else
        printf '%s/shard_%03d_of_%03d.json\n' "$stage_dir" "$shard" "$NUM_SHARDS"
    fi
}

wait_all() {
    local pid failed=0
    for pid in "$@"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    return "$failed"
}

log_to() {
    local log="$1"; shift
    "$@" 2>&1 | tee -a "$log"
}

# run_sharded <stage_dir> <merged_output> <label> <command...>
#
# The command must accept --output/--shard/--num-shards; those are appended.
run_sharded() {
    local stage_dir="$1"; shift
    local merged="$1"; shift
    local label="$1"; shift

    banner "$label"
    mkdir -p "$stage_dir"

    local pids=() shard
    for shard in $(seq 1 "$NUM_SHARDS"); do
        "$@" \
            --output "$(shard_file "$stage_dir" "$shard")" \
            --shard "$shard" --num-shards "$NUM_SHARDS" \
            > "${LOG_DIR}/${label}.shard${shard}.log" 2>&1 &
        pids+=("$!")
        echo "  launched shard ${shard}/${NUM_SHARDS} (pid ${pids[-1]}) → ${LOG_DIR}/${label}.shard${shard}.log"
    done

    wait_all "${pids[@]}" \
        || die "${label}: a shard failed, see ${LOG_DIR}/${label}.shard*.log"

    log_to "${LOG_DIR}/${label}.log" \
        python -m utils.pipeline merge-shards --stage-dir "$stage_dir" --output "$merged"
}

# run_sharded_verify <stage_dir> <verified_out> <judgement_out> <label> <command...>
#
# Verification writes two files per shard, so it needs its own launcher.
run_sharded_verify() {
    local stage_dir="$1"; shift
    local verified="$1"; shift
    local judgements="$1"; shift
    local label="$1"; shift

    banner "$label"
    mkdir -p "${stage_dir}/verified" "${stage_dir}/judgements"

    local pids=() shard
    for shard in $(seq 1 "$NUM_SHARDS"); do
        "$@" \
            --verified-output "$(shard_file "${stage_dir}/verified" "$shard")" \
            --judgement-output "$(shard_file "${stage_dir}/judgements" "$shard")" \
            --shard "$shard" --num-shards "$NUM_SHARDS" \
            > "${LOG_DIR}/${label}.shard${shard}.log" 2>&1 &
        pids+=("$!")
        echo "  launched shard ${shard}/${NUM_SHARDS} (pid ${pids[-1]}) → ${LOG_DIR}/${label}.shard${shard}.log"
    done

    wait_all "${pids[@]}" \
        || die "${label}: a shard failed, see ${LOG_DIR}/${label}.shard*.log"

    log_to "${LOG_DIR}/${label}.log" \
        python -m utils.pipeline merge-shards --stage-dir "${stage_dir}/verified" --output "$verified"
    log_to "${LOG_DIR}/${label}.log" \
        python -m utils.pipeline merge-shards --stage-dir "${stage_dir}/judgements" --output "$judgements"
}

# filter_stage <run_subdir> <label> <mode> <input_stage> <output_stage>
#
# The filter/refine stages are shared by the text and table pipelines: the
# record format is identical, so only the directory differs.
filter_stage() {
    local dir="$1" label="$2" mode="$3" in_stage="$4" out_stage="$5"
    local input="${dir}/${in_stage}.json"
    local output="${dir}/${out_stage}.json"

    [[ -f "$input" ]] || die "${label} ${mode}: missing input ${input}"

    if [[ "$mode" == "rule_filter" ]]; then
        # Deterministic rules, no API calls, fast enough to stay single-process.
        banner "${label}: ${mode}"
        log_to "${LOG_DIR}/${label}_${mode}.log" \
            python -m extract_from_text.02_filter_or_refine \
                --input "$input" --output "$output" --mode "$mode"
    else
        run_sharded "${dir}/${out_stage}" "$output" "${label}_${mode}" \
            python -m extract_from_text.02_filter_or_refine \
                --input "$input" --mode "$mode" \
                --llm "$LLM" --effort "$EFFORT" \
                --filter-prompt "$FILTER_PROMPT" \
                --refine-prompt "$REFINE_PROMPT"
    fi
}

# Record which settings produced a run, next to its outputs.
write_run_config() {
    local out="$1" cfg="$2"
    cat > "$out" <<EOF
{
  "config_file": "${cfg}",
  "created": "$(date '+%Y-%m-%d %H:%M:%S')",
  "run_name": "${RUN_NAME}",
  "llm": "${LLM}",
  "effort": "${EFFORT}",
  "num_shards": ${NUM_SHARDS},
  "paper_list": "${PAPER_LIST}",
  "xml_dir": "${XML_DIR}",
  "prompts": {
    "entity_filter": "${ENTITY_FILTER_PROMPT}",
    "get_triple": "${GET_TRIPLE_PROMPT}",
    "filter": "${FILTER_PROMPT}",
    "refine": "${REFINE_PROMPT}",
    "text_eval": "${TEXT_EVAL_PROMPT}",
    "select_table": "${SELECT_TABLE_PROMPT}",
    "table_triple": "${TABLE_TRIPLE_PROMPT}",
    "table_eval": "${TABLE_EVAL_PROMPT}",
    "baseline": "${BASELINE_PROMPT}"
  },
  "spacy_model": "${SPACY_MODEL}",
  "entity_type_model": "${ENTITY_TYPE_MODEL}",
  "umls_threshold": ${UMLS_THRESHOLD},
  "enable_bios_completion": ${ENABLE_BIOS_COMPLETION},
  "current_year": ${CURRENT_YEAR},
  "ablation_wo_ner": ${ABLATION_WO_NER}
}
EOF
    echo "Wrote run config to ${out}"
}
