"""Recall evaluation: does the extracted triple set cover a reference set?

The reference set is produced by running the same extraction pipeline with a
stronger model / higher reasoning effort. A judge LLM labels each reference
triple Covered / Not Covered against the extracted triples for that article.
"""

import argparse
import io
import sys
import time

import prompts
from pydantic import BaseModel

from llm.openai_chat import chat_structured
from utils import pipeline

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
    p.add_argument("--reference", required=True, help="Reference (GT) triples.")
    p.add_argument("--input", required=True, help="Extracted triples to score.")
    p.add_argument("--output", default="", help="Defaults to <input>_recalleval.json.")
    p.add_argument("--llm", default="gpt-5.4")
    p.add_argument("--effort", default="medium")
    p.add_argument("--prompt", default="recall_evaluation_prompt_structured")
    return p.parse_args()


def render_triples(triples) -> str:
    return "".join(
        f"{i + 1}. [ {t['head']['name']} | {t['relation']['name']} | {t['tail']['name']} ]\n"
        for i, t in enumerate(triples)
    )


def evaluate(args, output_path):
    done = pipeline.open_output(output_path)

    reference = pipeline.read_records(args.reference)
    print(f"Loaded {len(reference)} reference records from {args.reference}")

    extracted = pipeline.read_records(args.input)
    print(f"Loaded {len(extracted)} extracted records from {args.input}")
    extracted_by_id = {pipeline.record_key(r): next(iter(r.values())) for r in extracted}

    valid_articles = 0
    start = time.time()

    for record in reference:
        file_name = pipeline.record_key(record)
        if file_name in done:
            continue

        print(f"Evaluating: {file_name}")
        gt_triples = record[file_name]
        input_triples = extracted_by_id.get(file_name)

        if not gt_triples or not input_triples:
            print("❌ gt triples or input triples not found.")
            continue

        try:
            prompt = (
                getattr(prompts, args.prompt)
                .replace("<<gt_triples>>", render_triples(gt_triples))
                .replace("<<extracted_triples>>", render_triples(input_triples))
            )
            response = chat_structured(args.llm, prompt, EvaluationForm, args.effort)

            if len(response["Evaluations"]) != len(gt_triples):
                continue

            labels = {i: r["Result"] for i, r in enumerate(response["Evaluations"])}
            pipeline.append_records(output_path, [{file_name: labels}])
            valid_articles += 1
        except Exception as e:
            print(f"❌ {e}")
            continue

    runtime = (time.time() - start) / 60
    print(f"Done! Total valid articles: {valid_articles}")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = parse_args()
    output_path = args.output or args.input.replace(".json", "_recalleval.json")
    print(
        f"Reference: {args.reference}\nInput: {args.input}\nOutput: {output_path}\n"
        f"Judge: {args.llm} (effort={args.effort})\n"
    )
    evaluate(args, output_path)
