"""Stage 3 (table): extract triples, or cohort baseline info, from tables.

Two modes over the two buckets that 02_select_tables produced:

  triples   relation tables -> triples, in the same record format as the text
            pipeline, so they share the downstream filter/refine/verify stages
  baseline  baseline tables -> structured study-cohort demographics
"""

import argparse
import inspect
import io
import os
import sys
import time

import prompts
from pydantic import BaseModel

from llm.openai_chat import chat_structured
from utils import pipeline
from utils.pmc_xml import count_tokens

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdout.reconfigure(line_buffering=True)


class TripleForm(BaseModel):
    Entity1: str
    Relation: str
    Entity2: str


class TripleExtractionForm(BaseModel):
    Triples: list[TripleForm]


class InfoForm(BaseModel):
    cohort_description: str
    cohort_name: str
    sample_size: str
    mean_age: str
    age_distribution: str
    gender_distribution: str
    racial_distribution: str
    educational_attainment: str
    employment_status: str


class BaselineInfoForm(BaseModel):
    Cohorts: list[InfoForm]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["triples", "baseline"])
    p.add_argument("--table-dir", required=True, help="Selected tables for this mode.")
    p.add_argument("--output", required=True)
    p.add_argument("--llm", required=True)
    p.add_argument("--effort", required=True)
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--get-triple-prompt", default="extract_triples_from_tables_structured")
    p.add_argument("--get-baseline-prompt", default="extract_baseline_from_tables_structured")
    p.add_argument("--flush-every", type=int, default=10)
    return p.parse_args()


def get_triples(table_content, args):
    try:
        prompt = getattr(prompts, args.get_triple_prompt).replace(
            "<<tables>>", table_content
        )
        response = chat_structured(args.llm, prompt, TripleExtractionForm, args.effort)
        tokens_count = count_tokens(prompt)

        triples = response["Triples"]
        if not triples:
            return None, None

        return [
            {
                "head": {"name": t["Entity1"]},
                "relation": {"name": t["Relation"]},
                "tail": {"name": t["Entity2"]},
            }
            for t in triples
        ], tokens_count
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None


def get_baseline_info(table_content, args):
    try:
        prompt = getattr(prompts, args.get_baseline_prompt).replace(
            "<<tables>>", table_content
        )
        response = chat_structured(args.llm, prompt, BaselineInfoForm, args.effort)
        tokens_count = count_tokens(prompt)

        baseline_info = response["Cohorts"]
        if not baseline_info:
            return None, None
        return baseline_info, tokens_count
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None


def extract(args, file_names):
    done = pipeline.open_output(args.output)

    valid_articles = 0
    total_tokens = 0
    batch = []
    start = time.time()

    for file_name in file_names:
        # Key on the article, not the table file, so text and table triples merge.
        file_name_xml = file_name.replace(".txt", ".xml")
        if file_name_xml in done:
            continue

        print(file_name)
        with open(os.path.join(args.table_dir, file_name), "r", encoding="utf-8") as f:
            table_content = f.read()

        if not table_content:
            continue

        if args.mode == "triples":
            payload, tokens_count = get_triples(table_content, args)
        else:
            payload, tokens_count = get_baseline_info(table_content, args)

        if not payload:
            continue

        batch.append({file_name_xml: payload})
        valid_articles += 1
        total_tokens += tokens_count

        if len(batch) >= args.flush_every:
            pipeline.append_records(args.output, batch)
            batch = []
            print(f"Average tokens per valid table: {total_tokens / valid_articles:.2f}")

    pipeline.append_records(args.output, batch)

    runtime = (time.time() - start) / 60
    print("\n====================== Extraction Summary ======================")
    print(f"Total valid articles: {valid_articles}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_articles > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
    print("================================================================\n")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = parse_args()

    all_file_names = sorted(f for f in os.listdir(args.table_dir) if f.endswith(".txt"))
    file_names = pipeline.shard_items(all_file_names, args.shard, args.num_shards)

    print(f"Using LLM: {args.llm} (effort={args.effort}), mode: {args.mode}")
    print(
        f"Shard {args.shard}/{args.num_shards}: "
        f"{len(file_names)} of {len(all_file_names)} table files"
    )
    print(f"Output: {args.output}")

    extract(args, file_names)
