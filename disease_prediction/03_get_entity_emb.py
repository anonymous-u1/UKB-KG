import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from disease_prediction import config


# Configuration

data_csv_path = config.TRIPLES_CSV

entity_emb_npz_path = config.NODE_EMB_NPZ

model_path = config.BIOBERT_PATH
batch_size = 32


# Load entities from CSV

print("[INFO] Loading CSV...")

df = pd.read_csv(data_csv_path)

required_columns = {"HeadName", "TailName"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

entity_series = pd.concat([df["HeadName"], df["TailName"]], ignore_index=True)

n_raw_entities = len(entity_series)

entity_series = entity_series.dropna()
entity_series = entity_series.astype(str).str.strip()
entity_series = entity_series[entity_series != ""]

# Sort unique entities to ensure deterministic output ordering.
entities = sorted(entity_series.unique().tolist())

print(f"[INFO] Raw head/tail entries: {n_raw_entities:,}")
print(f"[INFO] Valid unique entities: {len(entities):,}")

if not entities:
    raise ValueError("No valid entities found after filtering.")


# Load BioBERT

print("[INFO] Loading BioBERT...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
model.eval()

print(f"[INFO] BioBERT loaded on {device}")


def get_entity_embeddings(entities, tokenizer, model, device, batch_size=32):
    """Encode entity names with BioBERT using attention-mask mean pooling."""
    embeddings = []

    for i in tqdm(range(0, len(entities), batch_size), desc="Encoding entities"):
        batch_entities = entities[i:i + batch_size]

        inputs = tokenizer(
            batch_entities,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
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


# Generate entity embeddings

entity_embeddings = get_entity_embeddings(
    entities=entities,
    tokenizer=tokenizer,
    model=model,
    device=device,
    batch_size=batch_size
)

print(f"[INFO] Embedding shape: {entity_embeddings.shape}")


# Save embeddings

np.savez(
    entity_emb_npz_path,
    entities=np.asarray(entities),
    embeddings=entity_embeddings
)

print(f"[INFO] Embeddings saved to: {entity_emb_npz_path}")
print("[INFO] Done ✅")