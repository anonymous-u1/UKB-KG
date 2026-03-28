import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Load data
train_feature = np.load('disease_prediction/data/data_raw/features_train.npy') # (195289, 1560)
val_feature = np.load('disease_prediction/data/data_raw/features_val.npy') # (55797, 1560)
test_feature = np.load('disease_prediction/data/data_raw/features_test.npy') # (27899, 1560)

model_name = 'TransE'
phecodes = np.load('disease_prediction/data/phecodes.npy') # Shape: (1560,)
ene = np.load(f'disease_prediction/KGE-RotatE/models/{model_name}_ukb_gpt-5_minimal_12r/entity_embedding.npy') # Shape: (117637, 1000)
map_df = pd.read_csv('disease_prediction/data/phecode_node_map.csv')

phecode_to_entity_id = dict(zip(map_df['phecode'], map_df['entity_id']))

# Construct phecode → embedding mapping
embedding_dim = ene.shape[1]
phecode_emb_matrix = np.zeros((len(phecodes), embedding_dim), dtype=np.float32) # (1560, 1000)

for idx, phecode in enumerate(phecodes):
    entity_id = phecode_to_entity_id.get(phecode, -1)
    if entity_id != -1:
        phecode_emb_matrix[idx] = ene[int(entity_id)]
    else:
        phecode_emb_matrix[idx] = 0.0
        print(f"No entity ID matching phecode {phecode} found")

print(f"[INFO] phecode embedding matrix built: {phecode_emb_matrix.shape}")


# Compute average embedding
def compute_average_embeddings(feature_matrix, phecode_emb_matrix):
    sums = feature_matrix @ phecode_emb_matrix  # (N, P) × (P, D) → (N, D)

    counts = feature_matrix.sum(axis=1).reshape(-1, 1)
    counts[counts == 0] = 1

    result = sums / counts
    return result

print("[INFO] Computing average embeddings ...")
train_feature_ene = compute_average_embeddings(train_feature, phecode_emb_matrix)
val_feature_ene = compute_average_embeddings(val_feature, phecode_emb_matrix)
test_feature_ene = compute_average_embeddings(test_feature, phecode_emb_matrix)
print("[INFO] Done.")


save_dir = f'disease_prediction/data/data_gpt-5_minimal_12r_{model_name}'
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'avr_ene_12r_train.npy'), train_feature_ene)
np.save(os.path.join(save_dir, 'avr_ene_12r_val.npy'), val_feature_ene)
np.save(os.path.join(save_dir, 'avr_ene_12r_test.npy'), test_feature_ene)

print(f"[INFO] Saved results to {save_dir}")