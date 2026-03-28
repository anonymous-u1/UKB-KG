import scispacy
import spacy
import os
import io
import sys
import re
import json
import prompts
import argparse
import tiktoken
from pydantic import BaseModel
from llm.openai_chat import *
from typing import Optional, List
from pathlib import Path
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)
nlp = spacy.load("en_core_sci_scibert")
encoder = tiktoken.encoding_for_model('gpt-4o')


def args_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--rel_save_path', type=str, required=True, help='Path to save the triples.')
    parser.add_argument('--baseline_save_path', type=str, required=True, help='Path to save the baseline info.')
    parser.add_argument('--rel_folder_dir', type=str, required=True, help='Path to the rel tables.')
    parser.add_argument('--baseline_folder_dir', type=str, required=True, help='Path to the baseline tables.')
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('--effort', type=str, required=True)
    parser.add_argument('-c', '--chunk', type=int, required=True)
    parser.add_argument('-n', '--number_of_chunk', type=int, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--get_triple_prompt', type=str, required=True)
    parser.add_argument('--get_baseline_prompt', type=str, required=True)
    args = parser.parse_args()
    return args


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
    
    
def get_triples(table_content):
    try:
        get_triple_prompt = getattr(prompts, args.get_triple_prompt).replace('<<tables>>', table_content)
        response = chat_structured(args.llm, get_triple_prompt, TripleExtractionForm, args.effort)
        tokens = encoder.encode(get_triple_prompt)
        tokens_count = len(tokens)

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


def get_baseline_info(table_content):
    try:
        get_baseline_prompt = getattr(prompts, args.get_baseline_prompt).replace('<<tables>>', table_content)
        response = chat_structured(args.llm, get_baseline_prompt, BaselineInfoForm, args.effort)

        tokens = encoder.encode(get_baseline_prompt)
        tokens_count = len(tokens)

        baseline_info = response['Cohorts']
        if baseline_info and len(baseline_info) > 0:
            return baseline_info, tokens_count
        else:
            return None, None
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None


def save_cache(cache_file, save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data.extend(cache_file)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract(save_path, folder_dir, file_names):
    if not os.path.isfile(save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('[]')
        print(f"create {save_path}.\n")
        file_names_saved = []
    else:
        with open(save_path, 'r', encoding='utf-8') as f:
            files = json.load(f)
            file_names_saved = [next(iter(file)) for file in files] if files else []

    valid_articles = 0
    total_tokens = 0

    cache_batch = []
    save_dict = {}

    start = time.time()
    for i, file_name in enumerate(file_names, start=1):
        file_name_xml = file_name.replace(".txt", ".xml")
        if file_name_xml in file_names_saved:
            continue
            
        print(file_name)
        table_path = os.path.join(folder_dir, file_name)
        with open(table_path, "r", encoding="utf-8") as f:
            table_content = f.read()

        if not table_content:
            continue

        if args.mode == "triple":
            triples, tokens_count = get_triples(table_content)
            if triples:
                save_dict[file_name_xml] = []
                for triple in triples.values():
                    save_dict[file_name_xml].append(triple)

                cache_batch.append(save_dict)
                save_dict = {}
                valid_articles += 1
                total_tokens += tokens_count
        elif args.mode == "baseline":
            baseline_info, tokens_count = get_baseline_info(table_content)
            if baseline_info:
                cache_batch.append({file_name_xml: baseline_info})
                valid_articles += 1
                total_tokens += tokens_count

        if i % 10 == 0:
            save_cache(cache_batch, save_path)
            cache_batch = []
            save_dict = {}
            if valid_articles > 0:
                print(f"Average tokens per valid table: {total_tokens / valid_articles:.2f}")

    if cache_batch:
        save_cache(cache_batch, save_path)

    end = time.time()
    runtime = (end - start) / 60
    print("\n====================== Extraction Summary ======================")
    print(f"Total valid articles: {valid_articles}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_articles > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
    print("================================================================\n")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = args_argument()
    if args.mode == "triple":
        save_path = args.rel_save_path
        folder_dir = args.rel_folder_dir
    elif args.mode == "baseline":
        save_path = args.baseline_save_path
        folder_dir = args.baseline_folder_dir
    else:
        raise ValueError("Unknown mode")

    all_file_names = os.listdir(folder_dir)
    all_file_names.sort()
    print(f"Using LLM: {args.llm}")

    chunk_size = args.chunk_size
    if args.chunk == 0:
        file_names = all_file_names
        save_path = save_path
        print("File: all (no chunking)")
    else:
        start_idx = (args.chunk - 1) * chunk_size
        end_idx = start_idx + chunk_size if args.chunk < args.number_of_chunk else None
        print(f"File: {start_idx}-{'' if end_idx is None else end_idx}")
        file_names = all_file_names[start_idx:end_idx]

        suffix = f"_{(end_idx or len(all_file_names))}"
        save_path = save_path.replace('.json', f'{suffix}.json')

    print(f"Mode: {args.mode}. Total files: {len(file_names)}. Save path:{save_path}")

    extract(save_path, folder_dir, file_names)