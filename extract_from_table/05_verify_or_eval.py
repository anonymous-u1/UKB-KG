import io
import os
import sys
import re
import json
import pandas as pd
import numpy as np
import openai
import argparse
import tiktoken
import prompts
from lxml import etree
from llm.openai_chat import *
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
import pubmed_parser as pp
import time
import inspect

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)
encoder = tiktoken.encoding_for_model('gpt-4o')

class ResultForm(BaseModel):
    Result: str
class EvaluationForm(BaseModel):
    Evaluations: list[ResultForm]


def args_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Path to save the triples.')
    parser.add_argument('-f', '--folder_dir', type=str, required=True, help='Path to the papers.')
    parser.add_argument('-c', '--chunk', type=int, required=True)
    parser.add_argument('-n', '--number_of_chunk', type=int, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('--effort', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--ablation_wo_verifier', type=int, required=True)
    parser.add_argument('--evaluation_prompt', type=str, required=True)
    args = parser.parse_args()
    return args


def save_cache(cache_file, save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data.extend(cache_file)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def verify(input_path, output_verified_path, output_evaluation_path, folder_path, mode):
    if not os.path.isfile(output_evaluation_path):
        with open(output_evaluation_path, 'w', encoding='utf-8') as f:
            f.write('[]')
        print(f"Created {output_evaluation_path}.\n")
        evaluated_files = []
    else:
        with open(output_evaluation_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            evaluated_files = [next(iter(data)) for data in saved_data] if saved_data else []

    if mode == "verify":
        if not os.path.isfile(output_verified_path):
            with open(output_verified_path, 'w', encoding='utf-8') as f:
                f.write('[]')
            print(f"Created {output_verified_path}.\n")

    with open(input_path, "r", encoding='utf-8') as f:
        triples_data = json.load(f)
    print(f"Loaded {len(triples_data)} triple files from {input_path}")

    valid_articles = 0
    total_tokens = 0

    start = time.time()
    for triples_dict in triples_data:
        file_name = next(iter(triples_dict.keys()))
        file_name_txt = file_name.replace(".xml", ".txt")
        if file_name in evaluated_files:
            continue

        print(f"Evaluating: {file_name}")

        table_path = os.path.join(folder_path, file_name_txt)
        with open(table_path, "r", encoding="utf-8") as f:
            table_content = f.read()

        if not table_content:
            continue

        triples = triples_dict[file_name]
        triples_count = len(triples)
        triple_eval_dict = {}

        try:
            triples_text = ""
            for i, triple in enumerate(triples):
                triples_text += str(i + 1) + ". [ " + triple["head"]["name"] + " | " + triple["relation"][
                    "name"] + " | " + triple["tail"]["name"] + " ]\n"

            if triples_text != "":
                eval_prompt = getattr(prompts, args.evaluation_prompt).replace('<<table>>', table_content).replace('<<triples>>', triples_text)
                response = chat_structured(llm_name, eval_prompt, EvaluationForm, args.effort)

                tokens = encoder.encode(eval_prompt)
                tokens_count = len(tokens)
                print(f'Prompt has {tokens_count} tokens')

                if len(response['Evaluations']) != triples_count:
                    continue

                for i, result in enumerate(response['Evaluations']):
                    triple_eval_dict[i] = result['Result']

                if mode == "verify":
                    verified_idxs = [k for k, v in triple_eval_dict.items() if v == "Correct"]
                    if len(verified_idxs) > 0:
                        verified_triples = [triples[i] for i in verified_idxs]
                        verified_data = [{file_name: verified_triples}]
                        eval_data = [{file_name: triple_eval_dict}]

                        save_cache(verified_data, output_verified_path)
                        save_cache(eval_data, output_evaluation_path)
                        print(f"Verified {len(verified_triples)}/{triples_count} triples for {file_name}")
                else:
                    eval_data = [{file_name: triple_eval_dict}]
                    save_cache(eval_data, output_evaluation_path)

                valid_articles += 1
                total_tokens += tokens_count

                if valid_articles != 0:
                    print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
        except Exception as e:
            print(f"❌ {e}")
            print(inspect.currentframe().f_lineno)
            continue

    end = time.time()
    runtime = (end - start) / 60
    print("\n====================== Verification Summary ======================")
    print(f"Total valid articles: {valid_articles}")
    print(f"Total tokens counted: {total_tokens}")
    if valid_articles > 0:
        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")
    print("================================================================\n")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    args = args_argument()
    input_dir = args.input_dir
    folder_path = args.folder_dir
    llm_name = args.llm
    mode = args.mode
    ablation_wo_verifier = args.ablation_wo_verifier

    # region Chunk Processing
    chunk_size = args.chunk_size
    if args.chunk == 0:
        # 处理单个文件（对应三元组保存时没有分块）
        if mode == 'verify' and ablation_wo_verifier == 0:
            input_path = input_dir.replace('.json', '_filter1_refine_filter2.json')
        elif mode == 'eval' and ablation_wo_verifier == 1:
            input_path = input_dir
        elif mode == 'eval' and ablation_wo_verifier == 0:
            input_path = input_dir.replace('.json', '_filtered_gpt-5verified.json')
        else:
            print(f"Uncorrect setting: mode - {mode} or ablation_wo_verifier - {ablation_wo_verifier}")
        print("File: all (no chunking)")
    else:
        start_idx = (args.chunk - 1) * chunk_size
        end_idx = start_idx + chunk_size if args.chunk < args.number_of_chunk else None
        print(f"File: {start_idx}-{'' if end_idx is None else end_idx}")
        total_files = sum(len(files) for _, _, files in os.walk(folder_path))

        # 保存路径
        if mode == 'verify' and ablation_wo_verifier == 0:
            suffix = f"_{(end_idx or total_files)}_filter1_refine_filter2.json"
        elif mode == 'eval' and ablation_wo_verifier == 1:
            suffix = f"_{(end_idx or total_files)}.json"
        elif mode == 'eval' and ablation_wo_verifier == 0:
            suffix = f"_{(end_idx or total_files)}_filtered_gpt-5verified.json"
        else:
            print(f"Uncorrect setting: mode - {mode} or ablation_wo_verifier - {ablation_wo_verifier}")
        input_path = input_dir.replace('.json', f'{suffix}')
    # endregion

    output_verified_path = input_path.replace('.json', f'_{llm_name}verified.json')
    if mode == 'verify':
        output_evaluation_path = input_path.replace('.json', f'_{llm_name}eval.json')
    else:
        output_evaluation_path = input_path.replace('.json', f'_{llm_name}eval.json')
    print(
        f"Input Path: {input_path}\nOutput Evaluation Path: {output_evaluation_path}\nOutput Verified Path: {output_verified_path}")

    verify(input_path, output_verified_path, output_evaluation_path, folder_path, mode)