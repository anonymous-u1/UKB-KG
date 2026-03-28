import os
import pandas as pd
import re
import numpy as np
from llm.openai_chat import *
import prompts
import json
import argparse
import sys
import io
import tiktoken
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)

class ResultForm(BaseModel):
    Result: str
class EvaluationForm(BaseModel):
    Evaluations: list[ResultForm]

def save_cache(cache_file, save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data.extend(cache_file)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def verify(gt_triple_path, input_triple_path, output_path, llm, effort):
    if not os.path.isfile(output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[]')
        print(f"Created {output_path}.\n")
        evaluated_files = []
    else:
        with open(output_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            evaluated_files = [next(iter(data)) for data in saved_data] if saved_data else []

    with open(gt_triple_path, "r", encoding='utf-8') as f:
        gt_data = json.load(f)
    print(f"Loaded {len(gt_data)} GT triple files from {gt_triple_path}")
    with open(input_triple_path, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    print(f"Loaded {len(input_data)} Input triple files from {input_triple_path}")

    valid_articles = 0

    start = time.time()
    for gt_file_data in gt_data:
        file_name = next(iter(gt_file_data.keys()))
        if file_name in evaluated_files:
            continue

        print(f"Evaluating: {file_name}")
        gt_triples = gt_file_data[file_name]
        gt_triples_count = len(gt_triples)
        for input_file_data in input_data:
            if file_name == next(iter(input_file_data.keys())):
                input_triples = input_file_data[file_name]

        triple_eval_dict = {}

        if not gt_triples or not input_triples:
            print("❌ gt triples or input triples not found.")
            continue
        try:
            gt_triples_text = ""
            input_triples_text = ""
            for i, triple in enumerate(gt_triples):
                gt_triples_text += str(i + 1) + ". [ " + triple["head"]["name"] + " | " + triple["relation"][
                    "name"] + " | " + triple["tail"]["name"] + " ]\n"
            for i, triple in enumerate(input_triples):
                input_triples_text += str(i + 1) + ". [ " + triple["head"]["name"] + " | " + triple["relation"][
                    "name"] + " | " + triple["tail"]["name"] + " ]\n"
            if gt_triples_text != "" and input_triples_text != "":
                eval_prompt = getattr(prompts, "recall_evaluation_prompt_structured").replace('<<gt_triples>>', gt_triples_text).replace('<<extracted_triples>>', input_triples_text)
                response = chat_structured(llm, eval_prompt, EvaluationForm, effort)
                if len(response['Evaluations']) != gt_triples_count:
                    continue

                for i, result in enumerate(response['Evaluations']):
                    triple_eval_dict[i] = result['Result']

                eval_data = [{file_name: triple_eval_dict}]
                save_cache(eval_data, output_path)

                valid_articles += 1
        except Exception as e:
            print(f"❌ {e}")
            continue

    end = time.time()
    runtime = (end - start) / 60
    print(f"Done! Total valid articles: {valid_articles}")
    print(f"Run time: {runtime:.2f} minutes.\n")


if __name__ == "__main__":
    gt_triple_path = "save/save_ablation/triple/triples_gpt-5.4_medium_filter1_refine_filter2_gpt-5.4verified.json"
    input_triple_path = "save/save_ablation/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98.json"
    output_path = input_triple_path.replace('.json', '_recalleval.json')
    print(f"GT input Path: {gt_triple_path}\nTriple input Path: {input_triple_path}\nOutput Evaluation Path: {output_path}\n")

    llm = "gpt-5.4"
    effort = "medium"
    verify(gt_triple_path, input_triple_path, output_path, llm, effort)