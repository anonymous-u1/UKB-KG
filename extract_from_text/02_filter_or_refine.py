"""Stage 2: filter and revise candidate triples.

Three modes, run in this order:

  llm_filter   drop triples the LLM judges not to express a biomedical relation
  refine       let the LLM rewrite heads/relations/tails into canonical form
  rule_filter  apply deterministic surface-form rules (no LLM calls)

The same script handles both text- and table-derived triples: the record format
is identical, so only --input/--output differ.
"""

import argparse
import inspect
import io
import re
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

LLM_MODES = ("llm_filter", "refine")


class TripleForm(BaseModel):
    Entity1: str
    Relation: str
    Entity2: str


class TripleListForm(BaseModel):
    Triples: list[TripleForm]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", required=True, choices=["llm_filter", "refine", "rule_filter"])
    p.add_argument("--llm", default="")
    p.add_argument("--effort", default="")
    p.add_argument("--filter-prompt", default="triple_filter_prompt_structured")
    p.add_argument("--refine-prompt", default="triple_refine_prompt_structured")
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--flush-every", type=int, default=10)
    args = p.parse_args()
    if args.mode in LLM_MODES and not args.llm:
        p.error(f"--llm is required for mode {args.mode}")
    if args.mode == "rule_filter" and args.num_shards > 1:
        # rule_filter makes no API calls and reports corpus-level counts, so it
        # is always run over the whole file at once.
        p.error("rule_filter does not support sharding")
    return args


# ------------------------------------------------------- rule_filter mode


def count_words(s: str) -> int:
    return len(re.split(r"[ _]+", s.strip()))


def is_pure_digit_or_symbol(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    return re.search(r"[A-Za-z]", s) is None


def should_exclude_triple(head: str, tail: str, relation: str) -> bool:
    """Surface-form rules for triples that cannot be valid KG edges."""
    if not head.strip() or not tail.strip() or not relation.strip():
        return True
    if head == tail:
        return True
    if count_words(head) > 8 or count_words(tail) > 8 or len(relation) > 35 or len(head) < 2 or len(tail) < 2 or len(relation) < 2:
        return True
    if is_pure_digit_or_symbol(head) or is_pure_digit_or_symbol(tail) or is_pure_digit_or_symbol(relation):
        return True
    if re.search(r'[0-9@#$%^&*+=<>?!|]', relation) or re.search(r'[@#$%^&?!|]', head) or re.search(r'[@#$%^&?!|]', tail):
        return True
    if re.search(r'\(.*?\)', relation) or '"' in relation or '/' in relation:
        return True
    return False


def safe_get_str(d: dict, *keys) -> str:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return ""
        d = d[k]
    return str(d).strip() if d is not None else ""


def rule_filter(input_path, output_path):
    records = pipeline.read_records(input_path)

    filtered = []
    total_triples = kept_triples = 0
    kept_papers = 0

    for record in records:
        key, triples = next(iter(record.items()))
        valid = []

        for triple in triples:
            head = safe_get_str(triple, "head", "name")
            tail = safe_get_str(triple, "tail", "name")
            relation = safe_get_str(triple, "relation", "name")
            total_triples += 1

            if not should_exclude_triple(head, tail, relation):
                valid.append(triple)
                kept_triples += 1

        if valid:
            filtered.append({key: valid})
            kept_papers += 1

    pipeline.write_records(output_path, filtered)

    print("=== Filtering complete. ===")
    print(f"Total papers:  {len(records):,}")
    print(f"Kept papers:   {kept_papers:,}")
    print(f"Total triples: {total_triples:,}")
    print(f"Kept triples:  {kept_triples:,}")
    print(f"Filtered file saved to: {output_path}")


# ---------------------------------------------------------- LLM modes


def call_llm(triples_text, mode, args):
    """Returns (triples, prompt_tokens) or (None, None)."""
    try:
        prompt_name = args.refine_prompt if mode == "refine" else args.filter_prompt
        prompt = getattr(prompts, prompt_name).replace("<<triples>>", triples_text)
        response = chat_structured(args.llm, prompt, TripleListForm, args.effort)

        tokens_count = count_tokens(prompt)
        print(f"Prompt has {tokens_count} tokens")

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


def render_triples(triples) -> str:
    return "".join(
        f"{j + 1}. [ {t['head']['name']} | {t['relation']['name']} | {t['tail']['name']} ]\n"
        for j, t in enumerate(triples)
    )


def llm_filter_or_refine(input_path, output_path, mode, args):
    done = pipeline.open_output(output_path)

    all_records = pipeline.read_records(input_path)
    records = pipeline.shard_items(all_records, args.shard, args.num_shards)
    print(f"Loaded {len(all_records)} triple records from {input_path}")
    print(
        f"Shard {args.shard}/{args.num_shards}: {len(records)} articles in this shard"
    )

    valid_articles = 0
    total_in = total_out = total_tokens = 0
    batch = []
    start = time.time()

    for record in records:
        file_name = next(iter(record.keys()))
        if file_name in done:
            continue

        print(f"[{mode}]: {file_name}")
        triples = record[file_name]
        total_in += len(triples)

        triples_text = render_triples(triples)
        if not triples_text:
            continue

        new_triples, tokens_count = call_llm(triples_text, mode, args)
        if not new_triples:
            continue

        batch.append({file_name: new_triples})
        valid_articles += 1
        total_tokens += tokens_count
        total_out += len(new_triples)
        print(f"Refined {len(new_triples)}/{len(triples)} triples for {file_name}")

        if len(batch) >= args.flush_every:
            pipeline.append_records(output_path, batch)
            batch = []
            print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")

    pipeline.append_records(output_path, batch)

    runtime = (time.time() - start) / 60
    print(f"\n====================== {mode} Summary ======================")
    print(f"Total valid articles: {valid_articles}")
    print(f"Total triples in:  {total_in}")
    print(f"Total triples out: {total_out}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_articles > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
    print("================================================================\n")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = parse_args()
    print(f"Mode: {args.mode}\nInput:  {args.input}\nOutput: {args.output}")

    if args.mode == "rule_filter":
        rule_filter(args.input, args.output)
    else:
        llm_filter_or_refine(args.input, args.output, args.mode, args)
