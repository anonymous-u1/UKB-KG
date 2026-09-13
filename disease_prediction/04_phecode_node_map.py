import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from utils.umls import umls_map as umls_mapper

from disease_prediction import config


warnings.filterwarnings("ignore", category=FutureWarning)


# Configuration

phecode_path = config.PHECODE_DEF_CSV

node_emb_path = config.NODE_EMB_NPZ

entities_dict_path = config.ENTITIES_DICT

output_path = config.PHECODE_NODE_MAP_CSV

model_path = config.BIOBERT_PATH

umls_threshold = 0.98
batch_size = 32


# Load device and model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()

print("[INFO] BioBERT model loaded.")


# Load data

print("[INFO] Loading data...")

phecode_df = pd.read_csv(phecode_path)

required_columns = {"phenotype"}
missing_columns = required_columns - set(phecode_df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

node_emb = np.load(node_emb_path, allow_pickle=True)

graph_entities = node_emb["entities"]
graph_embeddings = np.asarray(node_emb["embeddings"], dtype=np.float32)

if len(graph_entities) != len(graph_embeddings):
    raise ValueError(
        "Number of graph entities does not match number of graph embeddings."
    )

print(f"[INFO] Loaded {len(graph_entities):,} graph entities.")


def l2_normalize(vectors, eps=1e-12):
    """L2-normalize a 2D vector matrix."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)

    return vectors / norms


graph_embeddings = l2_normalize(graph_embeddings)

print("[INFO] Graph embeddings normalized.")


# Map phenotypes to UMLS terms

print(f"[INFO] Mapping phenotypes to UMLS (threshold={umls_threshold})...")

umls_fn = umls_mapper(threshold=umls_threshold)
phecode_df["phenotype_umls"] = phecode_df["phenotype"].apply(umls_fn)

print("[INFO] UMLS mapping completed.")


def clean_text(value):
    """Convert a value to a non-empty stripped string or return None."""
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return text


original_phenotypes = phecode_df["phenotype"].apply(clean_text)
umls_phenotypes = phecode_df["phenotype_umls"].apply(clean_text)

phenotypes_for_embedding = []
umls_fallback_count = 0

for original_text, umls_text in zip(original_phenotypes, umls_phenotypes):
    if umls_text is not None:
        phenotypes_for_embedding.append(umls_text)
    elif original_text is not None:
        phenotypes_for_embedding.append(original_text)
        umls_fallback_count += 1
    else:
        phenotypes_for_embedding.append(None)

invalid_indices = [
    i
    for i, text in enumerate(phenotypes_for_embedding)
    if text is None
]

if invalid_indices:
    raise ValueError(
        f"{len(invalid_indices)} phenotype entries are empty after both UMLS "
        f"mapping and original-phenotype fallback. "
        f"Example row indices: {invalid_indices[:10]}"
    )

phecode_df["phenotype_for_embedding"] = phenotypes_for_embedding

print(
    f"[INFO] UMLS mapping fallback to original phenotype: "
    f"{umls_fallback_count:,} entries"
)


def embed_texts(texts, tokenizer, model, device, batch_size=32):
    """Generate BioBERT embeddings using attention-mask mean pooling."""
    embeddings = []

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding phenotypes"
    ):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)

        hidden_states = outputs.last_hidden_state
        attention_mask = (
            inputs["attention_mask"]
            .unsqueeze(-1)
            .type_as(hidden_states)
        )

        summed_embeddings = (hidden_states * attention_mask).sum(dim=1)
        valid_token_count = attention_mask.sum(dim=1).clamp(min=1e-9)
        batch_embeddings = summed_embeddings / valid_token_count

        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


# Generate phenotype embeddings

print(
    f"[INFO] Generating embeddings for "
    f"{len(phecode_df):,} phenotypes..."
)

phenotype_embeddings = embed_texts(
    texts=phecode_df["phenotype_for_embedding"].tolist(),
    tokenizer=tokenizer,
    model=model,
    device=device,
    batch_size=batch_size
)

phenotype_embeddings = np.asarray(
    phenotype_embeddings,
    dtype=np.float32
)

print(
    f"[INFO] Phenotype embedding shape: "
    f"{phenotype_embeddings.shape}"
)

phenotype_embeddings = l2_normalize(phenotype_embeddings)


# Compute cosine similarities

print("[INFO] Computing phenotype-node similarities...")

similarity_matrix = np.matmul(
    phenotype_embeddings,
    graph_embeddings.T
)

print(
    f"[INFO] Similarity matrix shape: "
    f"{similarity_matrix.shape}"
)

max_indices = np.argmax(similarity_matrix, axis=1)

row_indices = np.arange(len(phenotype_embeddings))

max_similarities = similarity_matrix[
    row_indices,
    max_indices
]

matched_nodes = graph_entities[max_indices]

phecode_df["node"] = matched_nodes
phecode_df["similarity"] = max_similarities

del similarity_matrix


# Load entity-to-ID mapping

entity2id = {}

with open(entities_dict_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) == 2:
            eid, entity_name = parts
            entity2id[entity_name.strip()] = int(eid)

print(
    f"[INFO] Loaded {len(entity2id):,} entities from "
    f"{entities_dict_path}"
)


def get_entity_id(node_name):
    """Return the KGE entity ID for a node name."""
    if not isinstance(node_name, str):
        return -1

    return entity2id.get(node_name.strip(), -1)


phecode_df["entity_id"] = phecode_df["node"].apply(get_entity_id)

missing_count = (phecode_df["entity_id"] == -1).sum()

print(
    f"[INFO] entity_id mapping completed. "
    f"{missing_count:,} nodes were not found."
)


# Save results

phecode_df.to_csv(output_path, index=False)

print(f"[INFO] Saved results with entity_id to: {output_path}")
print("[INFO] Done ✅")