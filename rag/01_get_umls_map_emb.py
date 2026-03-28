import json
import torch
from transformers import AutoTokenizer, AutoModel
from utils.umls import umls_map
from llm.openai_chat import *
from pydantic import BaseModel
import scispacy
import spacy
import prompts
import numpy as np
import os
import re
import ast
from tqdm import tqdm


datatype = "pubmedqa"
data_path = f'rag/data/{datatype}/test_set.json'
entity_filter_prompt = "entity_filter_prompt_structured"
entity_path = f'rag/data/{datatype}/test_set_entity.json'
entity_umls_map_path = f"rag/data/{datatype}/test_set_entity_umls_map.json"
emb_path = f'rag/data/{datatype}/test_set_entity_map_emb.npz'

llm = "gpt-5"
effort = "minimal"
nlp = spacy.load("en_core_sci_scibert")


class EntityFilterForm(BaseModel):
    entities: list[str]

def ner(text):
    try:
        doc = nlp(text)
        entities = doc.ents

        seen = set()
        noun_entities = []
        for ent in entities:
            if ent.text in seen:
                continue
            if any(tok.pos_ in {"NOUN", "PROPN"} for tok in ent):
                noun_entities.append(ent)
                seen.add(ent.text)
    except Exception as e:
        print(f"❌ NER error: {e}")
        return None

    if not noun_entities:
        print("no entities found")
        return None

    try:
        ent_texts = [ent.text for ent in noun_entities]
        prompt = getattr(prompts, entity_filter_prompt).replace('<<entities>>', str(ent_texts))
        response = chat_structured(llm, prompt, EntityFilterForm, effort)
        filtered_entities = [entity.lower() for entity in response['entities']]
        if not filtered_entities:
            print("no filtered entities found")
            return None
        final_entities = [ent for ent in ent_texts if ent in filtered_entities]
    except Exception as e:
        print(f"❌ LLM filter error: {e}")
        return None

    return final_entities


# --------------------
#   EXTRACT ENTITY
# --------------------
data = json.load(open(data_path, encoding='utf-8'))

if os.path.exists(entity_path):
    results = json.load(open(entity_path))
else:
    results = {}

for idx, details in tqdm(data.items()):
    if idx in results:
        continue
    question = details.get('QUESTION', '')
    if not question:
        continue

    entities = ner(question)
    if entities:
        results[idx] = entities

json.dump(results, open(entity_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


# --------------------
#      UMLS MAP
# --------------------
umls_threshold = 0.98
umls_mapper = umls_map(threshold=umls_threshold)

if os.path.exists(entity_umls_map_path):
    mapped_data = json.load(open(entity_umls_map_path))
else:
    mapped_data = {}

for idx, entities in tqdm(results.items()):
    if idx in mapped_data:
        continue
    entities_mapped = [umls_mapper(entity) for entity in entities]
    mapped_data[idx] = entities_mapped

json.dump(mapped_data, open(entity_umls_map_path, 'w'), indent=2, ensure_ascii=False)

# --------------------
#       EMBED
# --------------------
model_path = "model/biobert_v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

print("[INFO] BioBERT model loaded.")

def get_entity_embeddings(entities):
    inputs = tokenizer(
        entities,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    return embeddings  # shape: (N, hidden_dim)

if os.path.exists(emb_path):
    embeddings_dict = dict(np.load(emb_path, allow_pickle=True))
else:
    embeddings_dict = {}

for idx, entities in tqdm(mapped_data.items()):
    if idx in embeddings_dict:
        continue
    embs = get_entity_embeddings(entities)
    embeddings_dict[idx] = embs

np.savez(emb_path, **embeddings_dict)
print(f"[INFO] Saved embeddings to {emb_path}")