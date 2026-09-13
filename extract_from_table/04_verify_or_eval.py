"""Stage 4 (table): self-verification, and LLM-as-judge evaluation.

Same contract as extract_from_text/03_verify_or_eval.py -- the only difference
is the evidence shown to the judge: the selected markdown tables rather than
the article prose.
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


class ResultForm(BaseModel):
    Result: str


class EvaluationForm(BaseModel):
    Evaluations: list[ResultForm]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Triples to judge.")
    p.add_argument("--judgement-output", required=True)
    p.add_argument("--verified-output", default="", help="Required for --mode verify.")
    p.add_argument("--table-dir", required=True, help="Selected relation tables.")
    p.add_argument("--mode", required=True, choices=["verify", "eval"])
    p.add_argument("--llm", required=True)
    p.add_argument("--effort", default="minimal")
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--evaluation-prompt", default="evaluation_table_prompt_structured")
    args = p.parse_args()
    if args.mode == "verify" and not args.verified_output:
        p.error("--verified-output is required for --mode verify")
    return args


def judge_article(table_content, triples, args):
    """Label every triple of one article. Returns ({index: label}, tokens)."""
    triples_text = "".join(
        f"{i + 1}. [ {t['head']['name']} | {t['relation']['name']} | {t['tail']['name']} ]\n"
        for i, t in enumerate(triples)
    )
    if not triples_text:
        return None, 0

    prompt = (
        getattr(prompts, args.evaluation_prompt)
        .replace("<<table>>", table_content)
        .replace("<<triples>>", triples_text)
    )
    response = chat_structured(args.llm, prompt, EvaluationForm, args.effort)

    tokens_count = count_tokens(prompt)
    print(f"Prompt has {tokens_count} tokens")

    # A judgement list of the wrong length cannot be aligned to the triples.
    if len(response["Evaluations"]) != len(triples):
        return None, tokens_count

    return {i: r["Result"] for i, r in enumerate(response["Evaluations"])}, tokens_count


def run(args, records):
    done = pipeline.open_output(args.judgement_output)
    if args.mode == "verify":
        pipeline.open_output(args.verified_output)

    print(f"Loaded {len(records)} triple records from {args.input}")

    valid_articles = 0
    total_tokens = 0
    start = time.time()

    for record in records:
        file_name = next(iter(record.keys()))
        if file_name in done:
            continue

        print(f"Evaluating: {file_name}")
        triples = record[file_name]

        table_path = os.path.join(args.table_dir, file_name.replace(".xml", ".txt"))
        if not os.path.isfile(table_path):
            print(f"⚠️  Table file not found: {table_path}")
            continue

        with open(table_path, "r", encoding="utf-8") as f:
            table_content = f.read()
        if not table_content:
            continue

        try:
            labels, tokens_count = judge_article(table_content, triples, args)
            if labels is None:
                continue

            if args.mode == "verify":
                verified = [triples[i] for i, v in labels.items() if v == "Correct"]
                if verified:
                    pipeline.append_records(
                        args.verified_output, [{file_name: verified}]
                    )
                    pipeline.append_records(
                        args.judgement_output, [{file_name: labels}]
                    )
                    print(
                        f"Verified {len(verified)}/{len(triples)} triples for {file_name}"
                    )
            else:
                pipeline.append_records(args.judgement_output, [{file_name: labels}])

            valid_articles += 1
            total_tokens += tokens_count
            print(
                f"Average tokens per valid article: {total_tokens / valid_articles:.2f}"
            )
        except Exception as e:
            print(f"❌ {e}")
            print(inspect.currentframe().f_lineno)
            continue

    runtime = (time.time() - start) / 60
    print(f"\n====================== {args.mode} Summary ======================")
    print(f"Total valid articles: {valid_articles}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_articles > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
    print("================================================================\n")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = parse_args()

    all_records = pipeline.read_records(args.input)
    records = pipeline.shard_items(all_records, args.shard, args.num_shards)

    print(f"Mode: {args.mode}, judge: {args.llm}")
    print(
        f"Shard {args.shard}/{args.num_shards}: "
        f"{len(records)} of {len(all_records)} articles"
    )
    print(f"Input: {args.input}")
    print(f"Judgement output: {args.judgement_output}")
    if args.mode == "verify":
        print(f"Verified output:  {args.verified_output}")

    run(args, records)
