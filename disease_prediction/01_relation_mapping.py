import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from disease_prediction import config


# Environment setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = config.BIOBERT_PATH

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)
model.eval()

print(f"[INFO] BioBERT loaded on {device}")


# Configuration

triples_path = config.TRIPLES_JSON

mapped_triples_path = config.MAPPED_TRIPLES_JSON
mapping_path = config.RELATION_MAPPING_JSON
txt_triples_path = config.MAPPED_TRIPLES_TXT

target_relations = [
    "physically_related_to",
    "spatially_related_to",
    "temporally_related_to",
    "conceptually_related_to",
    "affects",
    "brings_about",
    "performs",
    "occurs_in",
    "uses",
    "indicates",
    "result_of",
    "not_associated_with",
]


# Load relation types from triples

print("[INFO] Loading triples...")

with open(triples_path, "r", encoding="utf-8") as f:
    data = json.load(f)

original_relations = set()

for paper in data:
    for triples in paper.values():
        for tri in triples:
            rel_name = tri["relation"]["name"].strip()

            if rel_name:
                original_relations.add(rel_name)

original_relations = sorted(original_relations)

print(f"[INFO] Found {len(original_relations)} unique relations in dataset.")


def preprocess_relation(relation):
    """
    Convert relation names to natural-language text.

    Example:
        physically_related_to -> physically related to
    """
    return relation.replace("_", " ").strip()


def get_relation_vectors(relations, batch_size=16):
    """
    Encode relations with BioBERT in batches using attention-mask mean pooling
    to exclude padding tokens.
    """
    vectors = []

    for i in tqdm(
        range(0, len(relations), batch_size),
        desc="Encoding relations"
    ):
        batch_relations = relations[i:i + batch_size]

        batch_texts = [
            preprocess_relation(rel)
            for rel in batch_relations
        ]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        summed_embeddings = torch.sum(hidden_states * mask, dim=1)
        valid_token_count = torch.sum(mask, dim=1).clamp(min=1e-9)

        batch_vectors = (summed_embeddings / valid_token_count)

        vectors.append(batch_vectors.cpu().numpy())

    return np.vstack(vectors)


# Compute relation embeddings

print("[INFO] Computing BioBERT embeddings for target relations...")

target_vectors = get_relation_vectors(target_relations)

print("[INFO] Computing BioBERT embeddings for original relations...")

original_vectors = get_relation_vectors(original_relations)


def normalize_vectors(vectors):
    """
    L2-normalize vectors so cosine similarity can be computed by dot product.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    norms = np.clip(norms, a_min=1e-12, a_max=None)

    return vectors / norms


target_vectors_norm = normalize_vectors(target_vectors)
original_vectors_norm = normalize_vectors(original_vectors)


# Map original relations to target relations

print("[INFO] Mapping relations...")

similarity_matrix = np.matmul(
    original_vectors_norm,
    target_vectors_norm.T
)

best_indices = np.argmax(similarity_matrix, axis=1)

best_scores = np.max(similarity_matrix, axis=1)

relation_mapping = {}

for rel, best_idx in zip(original_relations, best_indices):
    relation_mapping[rel] = target_relations[best_idx]


print(f"[INFO] Average similarity = {np.mean(best_scores):.4f}")


# Mapping summary

print("[INFO] Mapping summary:")

mapping_counts = {
    relation: 0
    for relation in target_relations
}

for mapped_relation in relation_mapping.values():
    mapping_counts[mapped_relation] += 1

for target_relation in target_relations:
    print(f"   {target_relation:<25} {mapping_counts[target_relation]:>5d} mapped")


# Update relation names in JSON

print("[INFO] Updating triples...")

unmapped_relations = set()

for paper in tqdm(data, desc="Updating JSON triples"):
    for triples in paper.values():
        for tri in triples:

            old_rel = tri["relation"]["name"].strip()

            mapped_rel = relation_mapping.get(old_rel)

            if mapped_rel is not None:
                tri["relation"]["name"] = mapped_rel
            else:
                unmapped_relations.add(old_rel)


if unmapped_relations:
    print(f"[WARNING] {len(unmapped_relations)} relations were not mapped:")

    for rel in sorted(unmapped_relations):
        print(f"   {rel}")


# Save mapped triples

with open(mapped_triples_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[INFO] Mapped triples saved to {mapped_triples_path}")


# Save relation mapping

with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(relation_mapping, f, indent=2, ensure_ascii=False)

print(f"[INFO] Relation mapping table saved to {mapping_path}")

print("[INFO] Done ✅")
