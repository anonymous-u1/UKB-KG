import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from utils.umls import umls_map as umls_mapper
import warnings
import os
from tqdm import tqdm
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

phecode_path = 'disease_prediction/data/phecode_def_abbr.csv'
node_emb_path = 'save/csv/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_bioscompleted_node_emb.npz'
entities_dict_path = "disease_prediction/KGE-RotatE/data/ukb_gpt-5_minimal_12r/entities.dict"
output_path = "disease_prediction/data/phecode_node_map.csv"

# config and load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model_path = "model/biobert_v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()
print("[INFO] BioBERT model loaded.")

phecode_df = pd.read_csv(phecode_path)
node_emb = np.load(node_emb_path, allow_pickle=True)
graph_entities = node_emb["entities"]
graph_embeddings = node_emb["embeddings"]

graph_embeddings = graph_embeddings / np.linalg.norm(graph_embeddings, axis=1, keepdims=True)

# UMLS Link
umls_threshold = 0.98
umls_fn = umls_mapper(threshold=umls_threshold)
phecode_df["phenotype_umls"] = phecode_df["phenotype"].apply(umls_fn)

# Embedding
def embed_texts(texts, tokenizer, model, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding phenotypes"):
        batch = [str(t) for t in texts[i:i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_emb)
    return np.vstack(embeddings)

print(f"[INFO] Generating embeddings for {len(phecode_df)} phenotypes ...")
phenotype_embeddings = embed_texts(phecode_df["phenotype_umls"].tolist(), tokenizer, model, device, batch_size=32)

print("[INFO] Matching phenotypes to graph nodes ...")
phenotype_embeddings = phenotype_embeddings / np.linalg.norm(phenotype_embeddings, axis=1, keepdims=True)

similarity_list = []
node_list = []

for emb in tqdm(phenotype_embeddings, desc="Computing similarities"):
    sims = np.dot(graph_embeddings, emb)
    max_index = np.argmax(sims)
    similarity_list.append(sims[max_index])
    node_list.append(graph_entities[max_index])

phecode_df["node"] = node_list
phecode_df["similarity"] = similarity_list

# Load entity-id mapping
entity2id = {}

with open(entities_dict_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            eid, entity_name = parts
            entity2id[entity_name.strip()] = int(eid)

print(f"[INFO] Loaded {len(entity2id):,} entities from {entities_dict_path}")

# Map node → entity_id
def get_entity_id(node_name):
    if not isinstance(node_name, str):
        return -1
    return entity2id.get(node_name.strip(), -1)

phecode_df["entity_id"] = phecode_df["node"].apply(get_entity_id)

missing_count = (phecode_df["entity_id"] == -1).sum()
print(f"[INFO] entity_id mapping finished, {missing_count} nodes did not match the ID.")


phecode_df.to_csv(output_path, index=False)
print(f"[INFO] Saved results with entity_id to: {output_path}")
print("[INFO] Done ✅")