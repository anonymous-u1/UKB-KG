import os

import numpy as np
import pandas as pd

from disease_prediction import config


# Configuration

train_feature_path = os.path.join(config.RAW_DIR, "features_train.npy")
val_feature_path = os.path.join(config.RAW_DIR, "features_val.npy")
test_feature_path = os.path.join(config.RAW_DIR, "features_test.npy")

phecodes_path = config.PHECODES_NPY
map_path = config.PHECODE_NODE_MAP_CSV

model_name = "TransE"

entity_embedding_path = config.kge_entity_embedding(model_name)

save_dir = config.ene_output_dir(model_name)


# Load data

print("[INFO] Loading data...")

train_feature = np.load(train_feature_path)
val_feature = np.load(val_feature_path)
test_feature = np.load(test_feature_path)

phecodes = np.load(phecodes_path)
ene = np.load(entity_embedding_path)
map_df = pd.read_csv(map_path)

print(f"[INFO] Train feature shape: {train_feature.shape}")
print(f"[INFO] Val feature shape:   {val_feature.shape}")
print(f"[INFO] Test feature shape:  {test_feature.shape}")
print(f"[INFO] Number of phecodes:  {len(phecodes):,}")
print(f"[INFO] KGE embedding shape: {ene.shape}")


def check_feature_dimension(name, feature_matrix, phecodes):
    """Check that the feature matrix matches the number of Phecodes."""
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D matrix, but got shape {feature_matrix.shape}"
        )

    if feature_matrix.shape[1] != len(phecodes):
        raise ValueError(
            f"{name} has {feature_matrix.shape[1]} columns, "
            f"but phecodes contains {len(phecodes)} entries."
        )


check_feature_dimension("train_feature", train_feature, phecodes)
check_feature_dimension("val_feature", val_feature, phecodes)
check_feature_dimension("test_feature", test_feature, phecodes)

print(
    "[INFO] Feature dimension check passed: "
    "all feature matrices match phecodes."
)


# Validate Phecode mapping

if "phecode" not in map_df.columns:
    raise ValueError("Column 'phecode' is missing from mapping CSV.")

if "entity_id" not in map_df.columns:
    raise ValueError("Column 'entity_id' is missing from mapping CSV.")

duplicate_mask = map_df["phecode"].duplicated(keep=False)
duplicate_rows = map_df[duplicate_mask]

if len(duplicate_rows) > 0:
    duplicate_phecodes = duplicate_rows["phecode"].drop_duplicates().tolist()
    print(
        f"[WARNING] Found {len(duplicate_phecodes):,} duplicated Phecodes "
        f"in mapping CSV."
    )
    print(f"[WARNING] Example duplicated Phecodes: {duplicate_phecodes[:10]}")
else:
    print("[INFO] Duplicate Phecode check passed: no duplicates found.")

phecode_to_entity_id = dict(zip(map_df["phecode"], map_df["entity_id"]))


# Build Phecode-to-KGE embedding matrix

if ene.ndim != 2:
    raise ValueError(
        f"KGE entity embedding must be 2D, but got shape {ene.shape}"
    )

embedding_dim = ene.shape[1]
num_entities = ene.shape[0]

phecode_emb_matrix = np.zeros(
    (len(phecodes), embedding_dim),
    dtype=np.float32
)

# mapped_mask[i] = 1 for a valid KGE mapping and 0 otherwise.
mapped_mask = np.zeros(len(phecodes), dtype=np.float32)

lookup_failed_phecodes = []
nan_entity_id_phecodes = []
negative_entity_id_phecodes = []
out_of_range_phecodes = []
invalid_entity_id_phecodes = []

for idx, phecode in enumerate(phecodes):
    entity_id = phecode_to_entity_id.get(phecode, None)

    if entity_id is None:
        lookup_failed_phecodes.append(phecode)
        continue

    if pd.isna(entity_id):
        nan_entity_id_phecodes.append(phecode)
        continue

    try:
        entity_id = int(entity_id)
    except (TypeError, ValueError):
        invalid_entity_id_phecodes.append(phecode)
        continue

    if entity_id < 0:
        negative_entity_id_phecodes.append(phecode)
        continue

    if entity_id >= num_entities:
        out_of_range_phecodes.append((phecode, entity_id))
        continue

    phecode_emb_matrix[idx] = ene[entity_id].astype(
        np.float32,
        copy=False
    )
    mapped_mask[idx] = 1.0


# Report mapping results

mapped_count = int(mapped_mask.sum())
unmapped_count = len(phecodes) - mapped_count

print(
    f"[INFO] Phecode mapping summary: "
    f"{mapped_count:,}/{len(phecodes):,} mapped successfully."
)

if lookup_failed_phecodes:
    print(
        f"[WARNING] {len(lookup_failed_phecodes):,} Phecodes were not found "
        f"in phecode_node_map."
    )
    print(
        "[WARNING] This may indicate missing mappings or "
        "Phecode data-type mismatch."
    )
    print(
        f"[WARNING] Example lookup failures: "
        f"{lookup_failed_phecodes[:10]}"
    )

if nan_entity_id_phecodes:
    print(
        f"[WARNING] {len(nan_entity_id_phecodes):,} "
        f"Phecodes have NaN entity_id."
    )
    print(
        f"[WARNING] Example NaN entity_id Phecodes: "
        f"{nan_entity_id_phecodes[:10]}"
    )

if negative_entity_id_phecodes:
    print(
        f"[WARNING] {len(negative_entity_id_phecodes):,} "
        f"Phecodes have negative entity_id."
    )
    print(
        f"[WARNING] Example negative entity_id Phecodes: "
        f"{negative_entity_id_phecodes[:10]}"
    )

if invalid_entity_id_phecodes:
    print(
        f"[WARNING] {len(invalid_entity_id_phecodes):,} "
        f"Phecodes have invalid entity_id values."
    )
    print(
        f"[WARNING] Example invalid entity_id Phecodes: "
        f"{invalid_entity_id_phecodes[:10]}"
    )

if out_of_range_phecodes:
    print(
        f"[WARNING] {len(out_of_range_phecodes):,} Phecodes have entity_id "
        f"outside KGE embedding range [0, {num_entities - 1}]."
    )
    print(
        f"[WARNING] Example out-of-range mappings: "
        f"{out_of_range_phecodes[:10]}"
    )

if unmapped_count > 0:
    print(
        f"[WARNING] Total unmapped/invalid Phecodes: "
        f"{unmapped_count:,}"
    )
    print(
        "[WARNING] Ideally all Phecodes should have valid KG entity mappings. "
        "Please check the warnings above."
    )
else:
    print("[INFO] All Phecodes have valid KGE entity mappings.")

print(
    f"[INFO] Phecode embedding matrix built: "
    f"{phecode_emb_matrix.shape}"
)


def compute_average_embeddings(
    feature_matrix,
    phecode_emb_matrix,
    mapped_mask,
    split_name
):
    """
    Compute the equal-weight average KGE embedding over mapped Phecodes
    for each sample. Unmapped Phecodes are excluded from the denominator.
    """
    feature_float = feature_matrix.astype(np.float32, copy=False)

    sums = feature_float @ phecode_emb_matrix
    sums = sums.astype(np.float32, copy=False)

    counts = feature_float @ mapped_mask.reshape(-1, 1)
    counts = counts.astype(np.float32, copy=False)

    zero_count_mask = counts[:, 0] == 0
    zero_count_num = int(zero_count_mask.sum())

    if zero_count_num > 0:
        print(
            f"[WARNING] {split_name}: {zero_count_num:,} samples have no "
            f"mapped Phecodes and will receive zero embeddings."
        )

    result = np.zeros_like(sums, dtype=np.float32)

    np.divide(sums, counts, out=result, where=(counts > 0))
    return result


# Compute average KGE embeddings

print("[INFO] Computing average KGE embeddings ...")

train_feature_ene = compute_average_embeddings(
    train_feature,
    phecode_emb_matrix,
    mapped_mask,
    "Train"
)

val_feature_ene = compute_average_embeddings(
    val_feature,
    phecode_emb_matrix,
    mapped_mask,
    "Validation"
)

test_feature_ene = compute_average_embeddings(
    test_feature,
    phecode_emb_matrix,
    mapped_mask,
    "Test"
)

print("[INFO] Average KGE embeddings computed.")
print(
    f"[INFO] Train embeddings: {train_feature_ene.shape}, "
    f"dtype={train_feature_ene.dtype}"
)
print(
    f"[INFO] Val embeddings:   {val_feature_ene.shape}, "
    f"dtype={val_feature_ene.dtype}"
)
print(
    f"[INFO] Test embeddings:  {test_feature_ene.shape}, "
    f"dtype={test_feature_ene.dtype}"
)


# Save results

os.makedirs(save_dir, exist_ok=True)
np.save(
    os.path.join(save_dir, "avr_ene_12r_train.npy"),
    train_feature_ene
)
np.save(
    os.path.join(save_dir, "avr_ene_12r_val.npy"),
    val_feature_ene
)
np.save(
    os.path.join(save_dir, "avr_ene_12r_test.npy"),
    test_feature_ene
)

print(f"[INFO] Saved results to: {save_dir}")
print("[INFO] Done ✅")