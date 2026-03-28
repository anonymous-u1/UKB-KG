import io
import os
import sys
import re
import json
import scispacy
import spacy
import pandas as pd
import numpy as np
import openai
import ast
import markdown_to_json
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
nlp = spacy.load("en_core_sci_scibert")
encoder = tiktoken.encoding_for_model('gpt-4o')

class EntityFilterForm(BaseModel):
    entities: list[str]

class TripleForm(BaseModel):
    Entity1: str
    Relation: str
    Entity2: str
class TripleExtractionForm(BaseModel):
    Triples: list[TripleForm]


def args_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_dir', type=str, required=True, help='Path to save the triples.')
    parser.add_argument('-l', '--paper_list', type=str, required=True, help='Path to the paper list from which triples need to be extracted.')
    parser.add_argument('-f', '--folder_dir', type=str, required=True, help='Path to the papers.')
    parser.add_argument('-c', '--chunk', type=int, required=True)
    parser.add_argument('-n', '--number_of_chunk', type=int, required=True)
    parser.add_argument('--llm', type=str, required=True)
    parser.add_argument('--effort', type=str, required=True)
    parser.add_argument('--chunk_size', type=int, required=True)
    parser.add_argument('--log_content', type=str, required=True)
    parser.add_argument('--log_entity', type=str, required=True)
    parser.add_argument('--entity_filter_prompt', type=str, required=True)
    parser.add_argument('--get_triple_prompt', type=str, required=True)
    parser.add_argument('--title_file', type=str, required=True)
    parser.add_argument('--ablation_wo_ner', type=int, required=True)
    args = parser.parse_args()
    return args


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


def ner(text):
    try:
        doc = nlp(text)
        entities = doc.ents

        def remove_duplicate_text(items):
            seen_texts = set()
            unique_items = []
            for item in items:
                if item.text not in seen_texts:
                    unique_items.append(item)
                    seen_texts.add(item.text)
            return unique_items

        unique_entities = remove_duplicate_text(entities)

        noun_entities = []
        for ent in unique_entities:
            if any(token.pos_ in {"NOUN", "PROPN"} for token in ent):
                noun_entities.append(ent)
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None

    if len(noun_entities) == 0:
        return None, None
    try:
        entity_filter_prompt = getattr(prompts, args.entity_filter_prompt).replace('<<entities>>', str(noun_entities))
        tokens = encoder.encode(entity_filter_prompt)
        tokens_count = len(tokens)
        response = chat_structured(args.llm, entity_filter_prompt, EntityFilterForm, args.effort)
        filtered_entities = [entity.lower() for entity in response['entities']]
        # 提取实体及其位置
        entities_with_positions = [(ent.text, ent.start_char, ent.end_char) for ent in doc.ents if ent.text.lower() in filtered_entities]
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None

    return entities_with_positions, tokens_count


def mark_entities(text, entities):
    marked_text = text
    offset = 0
    for entity, start, end in entities:
        marked_text = (
            marked_text[:start + offset]
            + "<ent>"
            + marked_text[start + offset:end + offset]
            + "</ent>"
            + marked_text[end + offset:]
        )
        offset += len("<ent>") + len("</ent>")
    return marked_text


def get_triples(text):
    try:
        get_triple_prompt = getattr(prompts, args.get_triple_prompt).replace('<<text>>', text)
        tokens = encoder.encode(get_triple_prompt)
        tokens_count = len(tokens)
        response = chat_structured(args.llm, get_triple_prompt, TripleExtractionForm, args.effort)
        triples = response['Triples']
        new_triples = {}
        if triples is not None and len(triples) > 0:
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


def save_cache(cache_file, save_path):
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        data.extend(cache_file)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract(triple_save_path, folder_path, file_names):
    if not os.path.isfile(triple_save_path):
        with open(triple_save_path, 'w', encoding='utf-8') as f:
            f.write('[]')
        print(f"create {triple_save_path}.\n")
        file_names_saved = []
    else:
        with open(triple_save_path, 'r', encoding='utf-8') as f:
            files = json.load(f)
            file_names_saved = [next(iter(file)) for file in files] if files else []

    print("Building XML index...")
    xml_index = build_xml_index(Path(folder_path))
    print(f"✅ Indexed {len(xml_index)} XML files under {folder_path}")

    valid_articles = 0
    total_tokens = 0

    start = time.time()
    for i, file_name in enumerate(file_names):
        triple_cache = []
        triple_save_dict = {}
        if file_name not in file_names_saved:
            print(file_name)
            xml_path = find_xml_file(file_name, xml_index)
            if xml_path is None:
                continue

            try:
                abstract_and_conclusion = get_abstract_and_conclusion(xml_path, file_name)
            except Exception as e:
                print(f"❌ {e}")
                print(inspect.currentframe().f_lineno)
                continue

            if abstract_and_conclusion is not None and abstract_and_conclusion != '':
                if args.ablation_wo_ner == 1:
                    print("Ablation without NER")
                    if i <= 10:
                        with open(log_content_path, 'a', encoding='utf-8') as file:
                            file.write(file_name + '\n' + '### CONTENTS! ###' + '\n' + abstract_and_conclusion + '\n')

                    triples, tokens_count = get_triples(abstract_and_conclusion)
                else:
                    entities, filter_token_count = ner(abstract_and_conclusion)
                    if entities is None:
                        continue

                    if i <= 10:
                        with open(log_entity_path, 'a', encoding='utf-8') as file:
                            file.write(file_name + '\n' + str(entities) + '\n')

                    marked_text = mark_entities(abstract_and_conclusion, entities)
                    if i <= 10:
                        with open(log_content_path, 'a', encoding='utf-8') as file:
                            file.write(file_name + '\n' + '### CONTENTS! ###' + '\n' + marked_text + '\n')

                    triples, extract_tokens_count = get_triples(marked_text)
                    if extract_tokens_count is not None:
                        tokens_count = filter_token_count + extract_tokens_count

                if triples is not None and len(triples) > 0:
                    triple_save_dict[file_name] = []
                    for triple in triples.values():
                        triple_save_dict[file_name].append(triple)

                    triple_cache.append(triple_save_dict)
                    valid_articles += 1
                    total_tokens += tokens_count

                    save_cache(triple_cache, triple_save_path)

                    if valid_articles != 0:
                        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")

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
    save_dir = args.save_dir
    folder_path = args.folder_dir
    paper_list = args.paper_list
    # all_file_names = os.listdir(folder_path)
    with open(paper_list, 'r', encoding='utf-8') as f:
        all_file_names = [line.strip() for line in f if line.strip()]
    all_file_names.sort()
    print(f"Using LLM: {args.llm}")

    chunk_size = args.chunk_size
    if args.chunk == 0:
        file_names = all_file_names
        triple_save_path = save_dir
        log_content_path = args.log_content
        log_entity_path = args.log_entity
        print("File: all (no chunking)")
    else:
        start_idx = (args.chunk - 1) * chunk_size
        end_idx = start_idx + chunk_size if args.chunk < args.number_of_chunk else None
        print(f"File: {start_idx}-{'' if end_idx is None else end_idx}")
        file_names = all_file_names[start_idx:end_idx]

        suffix = f"_{(end_idx or len(all_file_names))}"
        triple_save_path = save_dir.replace('.json', f'{suffix}.json')

        log_content_path = args.log_content.replace('.txt', f'{suffix}.txt')
        log_entity_path = args.log_entity.replace('.txt', f'{suffix}.txt')

    print(f"Triple save path:{triple_save_path}")

    extract(triple_save_path, folder_path, file_names)