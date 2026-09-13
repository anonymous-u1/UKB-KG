"""Stage 2 (table): pick out the tables worth extracting from.

One LLM call per article classifies its tables into two buckets, and the
selected tables are written back out so the later stages only pay for tables
that can actually yield something:

  relation tables  contain extractable biomedical relationships -> triples
  baseline tables  describe the study cohort's demographics    -> baseline info

A table can land in both buckets, or neither.
"""

import argparse
import io
import os
import re
import sys

import prompts
from pydantic import BaseModel

from llm.openai_chat import chat_structured
from utils import pipeline
from utils.pmc_xml import count_tokens

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdout.reconfigure(line_buffering=True)


class TableSelectionForm(BaseModel):
    Relation_Tables: list[str]
    Baseline_Tables: list[str]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table-dir", required=True, help="Output of 01_parse_tables.")
    p.add_argument("--relation-table-dir", required=True)
    p.add_argument("--baseline-table-dir", required=True)
    p.add_argument("--output", required=True, help="Record file of selections.")
    p.add_argument("--llm", required=True)
    p.add_argument("--effort", required=True)
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--select-table-prompt", default="select_table_prompt_structured")
    return p.parse_args()


def extract_tables_by_ids(txt_content, selected_ids):
    """Slice out the "### Table N:" blocks named in `selected_ids`."""
    blocks = re.split(r"(?=^##+\s*Table\s+\d+:)", txt_content, flags=re.MULTILINE)
    if not blocks:
        return []

    extracted = []
    for tid in selected_ids:
        match = re.search(r"(\d+)", tid)
        if not match:
            continue
        num = match.group(1)
        for block in blocks:
            if re.search(rf"^##+\s*Table\s+{num}\b", block, flags=re.MULTILINE):
                extracted.append(block.strip())
                break
    return extracted


def select_tables(args, file_names):
    os.makedirs(args.relation_table_dir, exist_ok=True)
    os.makedirs(args.baseline_table_dir, exist_ok=True)

    done = pipeline.open_output(args.output)
    print(f"✅ {len(done)} files already processed, {len(file_names)} in this shard.")

    valid_files = 0
    total_rel = total_baseline = total_tokens = 0

    for file_name in file_names:
        if file_name in done:
            continue

        print(file_name)
        file_path = os.path.join(args.table_dir, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            table_content = f.read().strip()

        try:
            prompt = getattr(prompts, args.select_table_prompt).replace(
                "<<tables>>", table_content
            )
            response = chat_structured(args.llm, prompt, TableSelectionForm, args.effort)
            selected_rel_ids = response["Relation_Tables"]
            selected_baseline_ids = response["Baseline_Tables"]
            tokens_count = count_tokens(prompt)
            print(f"Prompt has {tokens_count} tokens")
        except Exception as e:
            print(f"❌ {e}")
            continue

        if selected_rel_ids is None or selected_baseline_ids is None:
            continue
        print(
            f"{file_name} → Rel Selected: {selected_rel_ids}; "
            f"Baseline Selected: {selected_baseline_ids}"
        )

        result = {"Relation_Tables": [], "Baseline_Tables": []}
        for ids, key, out_dir in (
            (selected_rel_ids, "Relation_Tables", args.relation_table_dir),
            (selected_baseline_ids, "Baseline_Tables", args.baseline_table_dir),
        ):
            if not ids:
                continue
            result[key] = ids
            extracted = extract_tables_by_ids(table_content, ids)
            if not extracted:
                print(f"❌ No valid {key} found for {file_name}.")
                continue
            with open(os.path.join(out_dir, file_name), "w", encoding="utf-8") as f:
                f.write("\n\n".join(extracted))

        pipeline.append_records(args.output, [{file_name: result}])

        valid_files += 1
        total_rel += len(selected_rel_ids)
        total_baseline += len(selected_baseline_ids)
        total_tokens += tokens_count

        print(
            f"✅ Saved filtered tables for {file_name} "
            f"({len(selected_rel_ids)} rel tables, {len(selected_baseline_ids)} baseline tables)\n"
        )

    print("\n========== SUMMARY ==========")
    print(f"Valid files: {valid_files}")
    print(f"Total selected rel tables across all files: {total_rel}")
    print(f"Total selected baseline tables across all files: {total_baseline}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_files > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_files:.2f}")
    print(f"Results saved to: {args.output}")
    print(f"Filtered rel tables saved to: {args.relation_table_dir}")
    print(f"Filtered baseline tables saved to: {args.baseline_table_dir}")
    print("=============================\n")


if __name__ == "__main__":
    args = parse_args()

    all_file_names = sorted(
        f for f in os.listdir(args.table_dir) if f.endswith(".txt")
    )
    file_names = pipeline.shard_items(all_file_names, args.shard, args.num_shards)

    print(f"Using LLM: {args.llm} (effort={args.effort})")
    print(
        f"Shard {args.shard}/{args.num_shards}: "
        f"{len(file_names)} of {len(all_file_names)} table files"
    )
    print(f"Output: {args.output}")

    select_tables(args, file_names)
