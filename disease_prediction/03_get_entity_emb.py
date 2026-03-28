import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

data_csv_path = 'save/csv/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_bioscompleted.csv'
entity_emb_npz_path = 'save/csv/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_bioscompleted_node_emb.npz'

# ================== Load ==================
df = pd.read_csv(data_csv_path)

entities = pd.concat([df['HeadName'], df['TailName']], ignore_index=True).astype(str).unique()
entities = sorted(list(entities))

print(f"[INFO] Total unique entities: {len(entities)}")

model_path = "model/biobert_v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("[INFO] BioBERT model loaded.")


# ================== Encode Entities ==================
def get_entity_embeddings(entities, tokenizer, model, device, batch_size=32):
    embeddings = []

    for i in tqdm(range(0, len(entities), batch_size), desc="Encoding entities in batches"):
        batch_entities = entities[i: i + batch_size]
        # batch tokenization
        inputs = tokenizer(
            batch_entities,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_emb)

    return np.vstack(embeddings)

entity_embeddings = get_entity_embeddings(entities, tokenizer, model, device, batch_size=32)

np.savez(entity_emb_npz_path, entities=np.array(entities), embeddings=entity_embeddings)

print(f"[INFO] Embeddings saved to: {entity_emb_npz_path}")
print("[INFO] Done.")
