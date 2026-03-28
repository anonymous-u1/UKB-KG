import io
import os
import sys
import json
import re
import argparse
import tiktoken
import prompts
from llm.openai_chat import *
from pathlib import Path
from pydantic import BaseModel
import time
import inspect

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)
encoder = tiktoken.encoding_for_model('gpt-4o')


class TripleForm(BaseModel):
    Entity1: str
    Relation: str
    Entity2: str


class TripleListForm(BaseModel):
    Triples: list[TripleForm]


def args_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--triple_path', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, required=True)
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('--effort', type=str, required=True)
    parser.add_argument('--filter_prompt', type=str, required=True)
    parser.add_argument('--refine_prompt', type=str, required=True)
    args = parser.parse_args()
    return args


def count_words(s: str) -> int:
    return len(re.split(r"[ _]+", s.strip()))


def is_pure_digit_or_symbol(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    return re.search(r'[A-Za-z]', s) is None


def should_exclude_triple(head: str, tail: str, relation: str) -> bool:
    if not head.strip() or not tail.strip() or not relation.strip():
        return True
    if head == tail:
        return True
    if count_words(head) > 8 or count_words(tail) > 8 or len(relation) > 35 or len(head) < 2 or len(tail) < 2 or len(
            relation) < 2:
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


def filter_triples(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data_points = json.load(f)

    filtered_data = []
    total_triples = kept_triples = 0
    total_papers = len(data_points)
    kept_papers = 0

    for data_point in data_points:
        # if i == 50:
        #     break
        key, triples = next(iter(data_point.items()))
        valid_triples = []

        for triple in triples:
            head = safe_get_str(triple, 'head', 'name')
            tail = safe_get_str(triple, 'tail', 'name')
            relation = safe_get_str(triple, 'relation', 'name')
            total_triples += 1

            if not should_exclude_triple(head, tail, relation):
                valid_triples.append(triple)
                kept_triples += 1

        if valid_triples:
            filtered_data.append({key: valid_triples})
            kept_papers += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"=== Filtering complete. ===")
    print(f"Total papers:  {total_papers:,}")
    print(f"Kept papers:   {kept_papers:,}")
    print(f"Total triples: {total_triples:,}")
    print(f"Kept triples:  {kept_triples:,}")
    print(f"Filtered file saved to: {output_path}")


def save_cache(cache_file, save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data.extend(cache_file)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_triples(triples_text, mode):
    try:
        if mode == "refine":
            llm_prompt = getattr(prompts, args.refine_prompt).replace('<<triples>>', triples_text)
        elif mode == "llm_filter":
            llm_prompt = getattr(prompts, args.filter_prompt).replace('<<triples>>', triples_text)
        response = chat_structured(args.llm, llm_prompt, TripleListForm, args.effort)
        tokens = encoder.encode(llm_prompt)
        tokens_count = len(tokens)
        print(f'Prompt has {tokens_count} tokens')

        triples = response['Triples']
        new_triples = {}
        if triples and len(triples) > 0:
            for index, triple in enumerate(triples):
                triple_key = f"triple{index}"
                new_triples[triple_key] = {
                    "head": {"name": triple['Entity1']},
                    "relation": {"name": triple['Relation']},
                    "tail": {"name": triple['Entity2']}
                }
            return new_triples, tokens_count
        else:
            return None, None
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None


def refine(input_path, output_path, mode):
    if not os.path.isfile(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[]')
        print(f"Created {output_path}.\n")
        refined_files = []
    else:
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            refined_files = [next(iter(data)) for data in saved_data] if saved_data else []

    with open(input_path, "r", encoding='utf-8') as f:
        triples_data = json.load(f)
    print(f"Loaded {len(triples_data)} triple files from {input_path}")

    valid_articles = 0
    total_triples_count = 0
    total_refined_triples_count = 0
    total_tokens = 0

    cache_batch = []
    save_dict = {}

    start = time.time()
    for i, triples_dict in enumerate(triples_data, start=1):
        file_name = next(iter(triples_dict.keys()))
        if file_name in refined_files:
            continue

        print(f"[{mode}]: {file_name}")
        triples = triples_dict[file_name]
        triples_count = len(triples)
        total_triples_count += triples_count

        triples_text = ""
        for j, triple in enumerate(triples):
            triples_text += str(j + 1) + ". [ " + triple["head"]["name"] + " | " + triple["relation"][
                "name"] + " | " + triple["tail"]["name"] + " ]\n"

        if not triples_text:
            continue

        refined_triples, tokens_count = get_triples(triples_text, mode)

        if refined_triples:
            save_dict[file_name] = []
            for triple in refined_triples.values():
                save_dict[file_name].append(triple)

            cache_batch.append(save_dict)
            save_dict = {}
            valid_articles += 1
            total_tokens += tokens_count
            refined_triples_count = len(refined_triples)
            total_refined_triples_count += refined_triples_count
            print(f"Refined {refined_triples_count}/{triples_count} triples for {file_name}")

        if i % 10 == 0:
            save_cache(cache_batch, output_path)
            cache_batch = []
            save_dict = {}
            if valid_articles > 0:
                print(f"Average tokens per valid table: {total_tokens / valid_articles:.2f}")

    if cache_batch:
        save_cache(cache_batch, output_path)

    end = time.time()
    runtime = (end - start) / 60
    print("\n====================== Verification Summary ======================")
    print(f"Total valid articles: {valid_articles}")
    print(f"Total triples: {total_triples_count}")
    print(f"Total refined triples: {total_refined_triples_count}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_articles > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
    print("================================================================\n")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = args_argument()
    triple_path = args.triple_path
    mode = args.mode

    if mode == 'llm_filter':
        input_path = triple_path
        output_path = input_path.replace('.json', f'_filter1.json')
    elif mode == 'refine':
        input_path = triple_path.replace('.json', f'_filter1.json')
        output_path = input_path.replace('.json', f'_refine.json')
    elif mode == 'rule_filter':
        input_path = triple_path.replace('.json', f'_filter1_refine.json')
        output_path = input_path.replace('.json', f'_filter2.json')
    else:
        raise Exception(f"Unrecognized mode: {mode}")
    print(f"Input Path: {input_path}\nOutput Path: {output_path}")

    if mode == 'rule_filter':
        filter_triples(input_path, output_path)
    else:
        refine(input_path, output_path, mode)