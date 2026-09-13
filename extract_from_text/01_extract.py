"""Stage 1 (text): extract candidate triples from article text.

For each article: pull Title/Abstract/Discussion(or Results)/Conclusion out of
the JATS XML, run scispaCy NER, ask the LLM to keep only the biomedically
meaningful entity mentions, wrap the survivors in <ent> tags, then ask the LLM
for triples over the marked-up text.

With --ablation-wo-ner the NER and entity-filter steps are skipped and the
plain article text is sent straight to the triple prompt.
"""

import argparse
import inspect
import io
import sys
import time

import prompts
import spacy
from pydantic import BaseModel

from llm.openai_chat import chat_structured
from utils import pipeline
from utils.pmc_xml import (
    build_xml_index,
    count_tokens,
    find_xml_file,
    get_article_text,
    read_paper_list,
)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stdout.reconfigure(line_buffering=True)

# Number of leading articles whose intermediate NER / prompt text is logged.
N_DEBUG_ARTICLES = 10


class EntityFilterForm(BaseModel):
    entities: list[str]


class TripleForm(BaseModel):
    Entity1: str
    Relation: str
    Entity2: str


class TripleExtractionForm(BaseModel):
    Triples: list[TripleForm]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, help="Record file for this shard.")
    p.add_argument("--paper-list", required=True, help="Newline-separated PMCID list.")
    p.add_argument("--xml-dir", required=True, help="Root of the PMC XML tree.")
    p.add_argument("--llm", required=True)
    p.add_argument("--effort", required=True)
    p.add_argument("--shard", type=int, default=1)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--log-content", default="", help="Optional prompt-text dump.")
    p.add_argument("--log-entity", default="", help="Optional NER dump.")
    p.add_argument("--entity-filter-prompt", default="entity_filter_prompt_structured")
    p.add_argument("--get-triple-prompt", default="get_triple_prompt_structured")
    p.add_argument("--title-file", default="", help="Metadata CSV for title fallback.")
    p.add_argument("--spacy-model", default="en_core_sci_scibert")
    p.add_argument("--ablation-wo-ner", action="store_true")
    p.add_argument("--flush-every", type=int, default=10)
    return p.parse_args()


def filter_entities(nlp, text, llm, effort, entity_filter_prompt):
    """NER over `text`, then keep only the mentions the LLM considers relevant.

    Returns (entities_with_positions, prompt_tokens) or (None, None).
    """
    try:
        doc = nlp(text)

        seen_texts = set()
        unique_entities = []
        for ent in doc.ents:
            if ent.text not in seen_texts:
                unique_entities.append(ent)
                seen_texts.add(ent.text)

        noun_entities = [
            ent
            for ent in unique_entities
            if any(token.pos_ in {"NOUN", "PROPN"} for token in ent)
        ]
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None

    if len(noun_entities) == 0:
        return None, None

    try:
        prompt = getattr(prompts, entity_filter_prompt).replace(
            "<<entities>>", str(noun_entities)
        )
        tokens_count = count_tokens(prompt)
        response = chat_structured(llm, prompt, EntityFilterForm, effort)
        filtered = [entity.lower() for entity in response["entities"]]
        entities_with_positions = [
            (ent.text, ent.start_char, ent.end_char)
            for ent in doc.ents
            if ent.text.lower() in filtered
        ]
    except Exception as e:
        print(f"❌ {e}")
        print(inspect.currentframe().f_lineno)
        return None, None

    return entities_with_positions, tokens_count


def mark_entities(text, entities):
    """Wrap each entity span in <ent>...</ent>, keeping offsets in sync."""
    marked_text = text
    offset = 0
    for entity, start, end in entities:
        marked_text = (
            marked_text[: start + offset]
            + "<ent>"
            + marked_text[start + offset : end + offset]
            + "</ent>"
            + marked_text[end + offset :]
        )
        offset += len("<ent>") + len("</ent>")
    return marked_text


def get_triples(text, llm, effort, get_triple_prompt):
    """Ask the LLM for triples. Returns (triples, prompt_tokens) or (None, None)."""
    try:
        prompt = getattr(prompts, get_triple_prompt).replace("<<text>>", text)
        tokens_count = count_tokens(prompt)
        response = chat_structured(llm, prompt, TripleExtractionForm, effort)
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


def log_to(path, *parts):
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def extract(args, nlp, file_names):
    done = pipeline.open_output(args.output)

    print(f"Building XML index under {args.xml_dir} ...")
    xml_index = build_xml_index(args.xml_dir)
    print(f"✅ Indexed {len(xml_index)} XML files")

    valid_articles = 0
    total_tokens = 0
    batch = []
    start = time.time()

    for i, file_name in enumerate(file_names):
        if file_name in done:
            continue

        print(file_name)
        xml_path = find_xml_file(file_name, xml_index)
        if xml_path is None:
            continue

        try:
            article_text = get_article_text(xml_path, file_name, args.title_file)
        except Exception as e:
            print(f"❌ {e}")
            print(inspect.currentframe().f_lineno)
            continue

        if not article_text:
            continue

        if args.ablation_wo_ner:
            print("Ablation without NER")
            if i <= N_DEBUG_ARTICLES:
                log_to(args.log_content, file_name, "### CONTENTS! ###", article_text)
            triples, tokens_count = get_triples(
                article_text, args.llm, args.effort, args.get_triple_prompt
            )
        else:
            entities, filter_tokens = filter_entities(
                nlp,
                article_text,
                args.llm,
                args.effort,
                args.entity_filter_prompt,
            )
            if entities is None:
                continue

            if i <= N_DEBUG_ARTICLES:
                log_to(args.log_entity, file_name, str(entities))

            marked_text = mark_entities(article_text, entities)
            if i <= N_DEBUG_ARTICLES:
                log_to(args.log_content, file_name, "### CONTENTS! ###", marked_text)

            triples, extract_tokens = get_triples(
                marked_text, args.llm, args.effort, args.get_triple_prompt
            )
            tokens_count = (
                filter_tokens + extract_tokens if extract_tokens is not None else None
            )

        if not triples:
            continue

        batch.append({file_name: triples})
        valid_articles += 1
        total_tokens += tokens_count

        if len(batch) >= args.flush_every:
            pipeline.append_records(args.output, batch)
            batch = []

        print(f"Average tokens per valid article: {total_tokens / valid_articles:.2f}")

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

    all_file_names = read_paper_list(args.paper_list)
    file_names = pipeline.shard_items(all_file_names, args.shard, args.num_shards)

    print(f"Using LLM: {args.llm} (effort={args.effort})")
    print(
        f"Shard {args.shard}/{args.num_shards}: "
        f"{len(file_names)} of {len(all_file_names)} articles"
    )
    print(f"Output: {args.output}")

    nlp = None if args.ablation_wo_ner else spacy.load(args.spacy_model)

    extract(args, nlp, file_names)
