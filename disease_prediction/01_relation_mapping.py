import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "model/biobert_v1.2"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)
model.eval()
print(f"[INFO] BioBERT loaded on {device}")

triples_path = "save/save_sample/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_bioscompleted.json"
mapped_triples_path = triples_path.replace(".json", "_mapped_12r.json")
mapping_path = triples_path.replace(".json", "_mapping_12r.json")
txt_triples_path = mapped_triples_path.replace(".json", ".txt")

target_relations = [
    "physically_related_to", "spatially_related_to", "temporally_related_to",
    "conceptually_related_to", "affects", "brings_about", "performs",
    "occurs_in", "uses", "indicates", "result_of", "not_associated_with"
]


print("[INFO] Loading triples...")
with open(triples_path, "r", encoding="utf-8") as f:
    data = json.load(f)

original_relations = set()
for paper in data:
    for _, triples in paper.items():
        for tri in triples:
            rel_name = tri["relation"]["name"].strip()
            if rel_name:
                original_relations.add(rel_name)

original_relations = sorted(list(original_relations))
print(f"[INFO] Found {len(original_relations)} unique relations in dataset.")


def get_relation_vectors(relations, batch_size=16):
    vectors = []
    for i in tqdm(range(0, len(relations), batch_size), desc="Encoding relations"):
        batch = relations[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        vectors.append(batch_vec)
    return np.vstack(vectors)


print("[INFO] Computing BioBERT embeddings for relations ...")
target_vectors = get_relation_vectors(target_relations)
original_vectors = get_relation_vectors(original_relations)

target_relation_vectors = dict(zip(target_relations, target_vectors))
original_relation_vectors = dict(zip(original_relations, original_vectors))


def map_relation(relation_vector, target_relation_vectors):
    sims = [
        cosine_similarity(relation_vector.reshape(1, -1), tvec.reshape(1, -1))[0, 0]
        for tvec in target_relation_vectors.values()
    ]
    best_idx = int(np.argmax(sims))
    best_match = list(target_relation_vectors.keys())[best_idx]
    return best_match, sims[best_idx]

print("[INFO] Mapping relations ...")
relation_mapping = {}
similarity_scores = []

for rel, vec in tqdm(original_relation_vectors.items(), desc="Mapping original relations"):
    best_match, sim = map_relation(vec, target_relation_vectors)
    relation_mapping[rel] = best_match
    similarity_scores.append(sim)

print(f"[INFO] Average similarity = {np.mean(similarity_scores):.4f}")
print("[INFO] Mapping summary:")
for tgt in target_relations:
    count = sum(1 for v in relation_mapping.values() if v == tgt)
    print(f"   {tgt:<25} {count:>5d} mapped")


print("[INFO] Updating triples ...")
for paper in tqdm(data, desc="Updating JSON triples"):
    for _, triples in paper.items():
        for tri in triples:
            old_rel = tri["relation"]["name"].strip()
            if old_rel in relation_mapping:
                tri["relation"]["name"] = relation_mapping[old_rel]
            else:
                print(f"❌ Relation {old_rel} not mapped.")

with open(mapped_triples_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"[INFO] Mapped triples saved to {mapped_triples_path}")


with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(relation_mapping, f, indent=2, ensure_ascii=False)
print(f"[INFO] Relation mapping table saved to {mapping_path}")

print("[INFO] Done ✅")