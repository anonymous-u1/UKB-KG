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
    parser.add_argument('--ablation_wo_ner', type=int, required=True)
    parser.add_argument('--ablation_0shot', type=int, required=True)
    parser.add_argument('--ablation_5shot', type=int, required=True)
    parser.add_argument('--ablation_cm', type=int, required=True)
    parser.add_argument('--evaluation_prompt', type=str, required=True)
    parser.add_argument('--title_file', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_json_markdown(markdown_content):
    if not isinstance(markdown_content, str):
        print("❌ Input is not a string")
        return None

    content = markdown_content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON in markdown block: {e}")
            return None

    match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON in code block: {e}")
            return None

    print("❌ No valid JSON found")
    return None

def build_xml_index(directory: Path) -> dict:
    """
    Build an index {filename: full_path} for fast lookup of XML files under a directory.
    """
    xml_index = {}
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(".xml"):
                xml_index[fname] = os.path.join(root, fname)
    return xml_index


def find_xml_file(pmcid: str, xml_index: dict) -> Optional[str]:
    """ Find XML path by PMCID using pre-built index. """
    if not pmcid.endswith(".xml"):
        pmcid += ".xml"
    return xml_index.get(pmcid)


def get_title(file_name):
    papers = pd.read_csv(args.title_file, encoding="ISO-8859-1")
    if len(papers[papers['PMCID'] == file_name[:-4]]) > 1:
        title = papers[papers['PMCID'] == file_name[:-4]]['Title'].tolist()[0]
        return title
    else:
        title = papers[papers['PMCID'] == file_name[:-4]]['Title'].item()
        return title

def extract_abstract(tree):
    abstract_texts = tree.xpath('//abstract//text()')
    full_abstract = ' '.join([text.strip() for text in abstract_texts if text.strip()])
    if full_abstract != '':
        return full_abstract
    return None

_ROMAN = r"(?:[ivxlcdm]+)"  # roman numerals
_NUM   = r"(?:\d+(?:\.\d+)*)"

_PREFIX_RE = re.compile(
    rf"^\s*(?:"
    rf"{_NUM}"                 # 4 or 4.1 or 4.1.2
    rf"|{_ROMAN}"              # iv / IV
    rf"|[a-zA-Z]"              # A / a
    rf")\s*[\)\].:\-–—]*\s+",   # followed by ).:- and spaces
    flags=re.IGNORECASE
)

def _norm_title(s: str) -> str:
    """Normalize section titles for robust matching."""
    if not s:
        return ""
    s = s.strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = _PREFIX_RE.sub("", s)
    return s.strip().lower()

def _sec_title_text(sec_node) -> str:
    """Get the visible title text of a <sec> (join all title text)."""
    title_parts = sec_node.xpath("./title//text()")
    title = " ".join(t.strip() for t in title_parts if t and t.strip())
    return title.strip()

def _find_first_major_sec(tree, aliases):
    """
    Find the first <sec> whose <title> matches any alias (case-insensitive).
    aliases: list[str] already normalized or raw. We'll normalize internally.
    """
    alias_norm = {_norm_title(a) for a in aliases}

    for sec in tree.xpath("//sec[title]"):
        t = _sec_title_text(sec)
        if _norm_title(t) in alias_norm:
            return sec
    return None

def _find_all_major_secs(tree, aliases):
    alias_norm = {_norm_title(a) for a in aliases}
    out = []
    for sec in tree.xpath("//sec[title]"):
        t = _sec_title_text(sec)
        if _norm_title(t) in alias_norm:
            out.append(sec)
    return out

def build_major_subsection_dict(tree, major_aliases):
    major_secs = _find_all_major_secs(tree, major_aliases)
    if not major_secs:
        return None, set()

    titles = []
    for major_sec in major_secs:
        secs = [major_sec] + major_sec.xpath(".//sec[title]")
        for s in secs:
            title = _sec_title_text(s)
            if title:
                titles.append(title)

    seen = set()
    uniq_titles = []
    for t in titles:
        nt = _norm_title(t)
        if nt and nt not in seen:
            uniq_titles.append(t)
            seen.add(nt)

    if not uniq_titles:
        major_title = _sec_title_text(major_sec) or major_aliases[0]
        uniq_titles = [major_title]

    sub_dict = {t: None for t in uniq_titles}
    sub_norm_set = {_norm_title(t) for t in uniq_titles}
    return sub_dict, sub_norm_set

def get_section(xml_path: str, file_name):
    chapter_dict = {}
    tree = etree.parse(xml_path)

    try:
        xml_parse_data = pp.parse_pubmed_xml(xml_path)
        title = xml_parse_data['full_title']
        abstract = xml_parse_data['abstract']
    except Exception:
        print("XML parsing error")
        title = get_title(file_name)
        abstract = extract_abstract(tree)

    if title is not None and title.strip() != '':
        chapter_dict['Title'] = title.strip()
    if abstract is not None and abstract.strip() != '':
        chapter_dict['Abstract'] = abstract.strip()

    discussion_aliases = ["discussion", "discussions"]
    results_aliases = ["result", "results"]
    conclusion_aliases = ["conclusion", "conclusions"]

    discussion_dict, discussion_norm_set = build_major_subsection_dict(tree, discussion_aliases)

    results_dict, results_norm_set = (None, set())
    if discussion_dict is None:
        results_dict, results_norm_set = build_major_subsection_dict(tree, results_aliases)

    conclusion_dict, conclusion_norm_set = build_major_subsection_dict(tree, conclusion_aliases)

    paragraphs = pp.parse_pubmed_paragraph(xml_path, all_paragraph=True)

    def _collect_by_norm_set(norm_set):
        buf = []
        for p in paragraphs:
            sec = p.get("section")
            if not sec:
                continue
            if _norm_title(sec) in norm_set:
                txt = p.get("text", "")
                if txt and txt.strip():
                    buf.append(txt.strip())
        return "\n".join(buf).strip() if buf else None

    if discussion_dict is not None:
        discussion_text = _collect_by_norm_set(discussion_norm_set)
        if discussion_text:
            chapter_dict["Discussion"] = discussion_text
    elif results_dict is not None:
        results_text = _collect_by_norm_set(results_norm_set)
        if results_text:
            chapter_dict["Results"] = results_text

    if conclusion_dict is not None:
        conclusion_text = _collect_by_norm_set(conclusion_norm_set)
        if conclusion_text:
            chapter_dict["Conclusion"] = conclusion_text

    return chapter_dict if chapter_dict else None


def get_abstract_and_conclusion(xml_path, file_name):
    chapter_dict = get_section(xml_path, file_name)

    if chapter_dict is None or len(chapter_dict) < 2:
        return ''
    else:
        abstract_and_conclusion = ''
        for k, v in chapter_dict.items():
            abstract_and_conclusion += f'## {k}:\n{v}\n'
        tokens = encoder.encode(abstract_and_conclusion)
        tokens_count = len(tokens)
        print(f'Abstract and Conclusion has {tokens_count} tokens')

        max_context_length = 20000
        if tokens_count > max_context_length:
            print(f"⚠️ Text too long: {tokens_count} tokens, truncating to {max_context_length} tokens.")
            abstract_and_conclusion = encoder.decode(tokens[:max_context_length])

        return abstract_and_conclusion


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

    print("Building XML index...")
    xml_index = build_xml_index(Path(folder_path))
    print(f"✅ Indexed {len(xml_index)} XML files under {folder_path}")

    valid_articles = 0
    total_tokens = 0

    start = time.time()
    for triples_dict in triples_data:
        file_name = next(iter(triples_dict.keys()))
        if file_name in evaluated_files:
            continue

        print(f"Evaluating: {file_name}")
        triples = triples_dict[file_name]
        triples_count = len(triples)
        triple_eval_dict = {}

        xml_path = find_xml_file(file_name, xml_index)
        if xml_path is None:
            continue

        try:
            abstract_and_conclusion = get_abstract_and_conclusion(xml_path, file_name)

            if abstract_and_conclusion is not None and abstract_and_conclusion != '':
                triples_text = ""
                for i, triple in enumerate(triples):
                    triples_text += str(i + 1) + ". [ " + triple["head"]["name"] + " | " + triple["relation"][
                        "name"] + " | " + triple["tail"]["name"] + " ]\n"

                if triples_text != "":
                    eval_prompt = getattr(prompts, args.evaluation_prompt).replace('<<text>>', abstract_and_conclusion).replace('<<triples>>', triples_text)
                    if llm_name == "deepseek":
                        response = chat_deepseek(eval_prompt)
                        response = parse_json_markdown(response)
                    elif llm_name == "qwen":
                        response = chat_qwen(eval_prompt)
                        response = parse_json_markdown(response)
                    else:
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
    ablation_wo_ner = args.ablation_wo_ner
    ablation_0shot = args.ablation_0shot
    ablation_5shot = args.ablation_5shot
    ablation_cm = args.ablation_cm

    chunk_size = args.chunk_size
    if args.chunk == 0:
        if mode == 'verify' and ablation_wo_verifier == 0:
            if ablation_0shot == 1:
                input_path = input_dir.replace('.json', '_0shot_filter1_refine_filter2.json')
            elif ablation_5shot == 1:
                input_path = input_dir.replace('.json', '_5shot_filter1_refine_filter2.json')
            elif ablation_wo_ner == 1:
                input_path = input_dir.replace('.json', '_wo_ner_filter1_refine_filter2.json')
            elif ablation_cm == 1:
                raise ValueError("wrong mode")
            else:
                input_path = input_dir.replace('.json', '_filter1_refine_filter2.json')
        elif mode == 'eval' and ablation_wo_verifier == 0:
            if ablation_0shot == 1:
                input_path = input_dir.replace('.json', '_0shot_filter1_refine_filter2_gpt-5verified_umlslinked98.json')
            elif ablation_5shot == 1:
                input_path = input_dir.replace('.json', '_5shot_filter1_refine_filter2_gpt-5verified_umlslinked98.json')
            elif ablation_wo_ner == 1:
                input_path = input_dir.replace('.json', '_wo_ner_filter1_refine_filter2_gpt-5verified_umlslinked98.json')
            elif ablation_cm == 1:
                input_path = input_dir.replace('.json', '_filter1_refine_filter2_crossmodelverified_umlslinked98.json')
            else:
                input_path = input_dir.replace('.json', '_filter1_refine_filter2_gpt-5verified_umlslinked98.json')
        elif mode == 'eval' and ablation_wo_verifier == 1:
            if ablation_cm == 1:
                raise ValueError("wrong mode")
            else:
                input_path = input_dir.replace('.json', '_filter1_refine_filter2_umlslinked98.json')
        else:
            print(f"Uncorrect setting: mode - {mode} or ablation_wo_verifier - {ablation_wo_verifier}")
        print("File: all (no chunking)")
    else:
        start_idx = (args.chunk - 1) * chunk_size
        end_idx = start_idx + chunk_size if args.chunk < args.number_of_chunk else None
        print(f"File: {start_idx}-{'' if end_idx is None else end_idx}")
        total_files = sum(len(files) for _, _, files in os.walk(folder_path))

        if mode == 'verify' and ablation_wo_verifier == 0:
            if ablation_0shot == 1:
                suffix = f'_0shot_{(end_idx or total_files)}_filter1_refine_filter2.json'
            elif ablation_5shot == 1:
                suffix = f'_5shot_{(end_idx or total_files)}_filter1_refine_filter2.json'
            elif ablation_wo_ner == 1:
                suffix = f'_wo_ner_{(end_idx or total_files)}_filter1_refine_filter2.json'
            elif ablation_cm == 1:
                raise ValueError("wrong mode")
            else:
                suffix = f'_{(end_idx or total_files)}_filter1_refine_filter2.json'
        elif mode == 'eval' and ablation_wo_verifier == 0:
            if ablation_0shot == 1:
                suffix = f'_0shot_{(end_idx or total_files)}_filter1_refine_filter2_gpt-5verified_umlslinked98.json'
            elif ablation_5shot == 1:
                suffix = f'_5shot_{(end_idx or total_files)}_filter1_refine_filter2_gpt-5verified_umlslinked98.json'
            elif ablation_wo_ner == 1:
                suffix = f'_wo_ner_{(end_idx or total_files)}_filter1_refine_filter2_gpt-5verified_umlslinked98.json'
            elif ablation_cm == 1:
                suffix = f'_{(end_idx or total_files)}_filter1_refine_filter2_crossmodelverified_umlslinked98.json'
            else:
                suffix = f'_{(end_idx or total_files)}_filter1_refine_filter2_gpt-5verified_umlslinked98.json'
        elif mode == 'eval' and ablation_wo_verifier == 1:
            if ablation_cm == 1:
                raise ValueError("wrong mode")
            else:
                suffix = f"_{(end_idx or total_files)}_filter1_refine_filter2_umlslinked98.json"
        else:
            print(f"Uncorrect setting: mode - {mode} or ablation_wo_verifier - {ablation_wo_verifier}")
        input_path = input_dir.replace('.json', f'{suffix}')

    output_verified_path = input_path.replace('.json', f'_{llm_name}verified.json')
    output_evaluation_path = input_path.replace('.json', f'_{llm_name}eval.json')
    print(f"Input Path: {input_path}\nOutput Evaluation Path: {output_evaluation_path}\nOutput Verified Path: {output_verified_path}")

    verify(input_path, output_verified_path, output_evaluation_path, folder_path, mode)