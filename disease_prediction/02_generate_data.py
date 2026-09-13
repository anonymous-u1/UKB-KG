import json
import os
import random
from tqdm import tqdm

from disease_prediction import config


# Configuration

triples_path = config.MAPPED_TRIPLES_JSON

data_dir = config.KGE_DATA_DIR

entities_dict_path = config.ENTITIES_DICT
relations_dict_path = config.RELATIONS_DICT
train_path = os.path.join(data_dir, "train.txt")
valid_path = os.path.join(data_dir, "valid.txt")
test_path = os.path.join(data_dir, "test.txt")

train_ratio = 0.9
valid_ratio = 0.0999
test_ratio = 0.0001

SEED = 42

os.makedirs(data_dir, exist_ok=True)


# Validate split ratios

ratio_sum = train_ratio + valid_ratio + test_ratio

if abs(ratio_sum - 1.0) > 1e-8:
    raise ValueError(
        f"train_ratio + valid_ratio + test_ratio must equal 1.0, but got {ratio_sum}"
    )

print(f"[INFO] Random seed: {SEED}")
print(
    f"[INFO] Split ratios: train={train_ratio}, "
    f"valid={valid_ratio}, test={test_ratio}"
)


# Load triples

print("[INFO] Loading triples...")

with open(triples_path, "r", encoding="utf-8") as f:
    data = json.load(f)

triples = []
empty_triples_count = 0

for paper in tqdm(data, desc="Loading triples"):
    for triples_list in paper.values():
        for tri in triples_list:
            h = tri["head"]["name"].strip()
            r = tri["relation"]["name"].strip()
            t = tri["tail"]["name"].strip()

            if not h or not r or not t:
                empty_triples_count += 1
                continue

            triples.append((h, r, t))

print(f"[INFO] Loaded triples: {len(triples):,}")

if empty_triples_count > 0:
    print(
        f"[WARNING] Skipped {empty_triples_count:,} triples with "
        f"empty head/relation/tail."
    )


# Deduplicate triples

n_before_dedup = len(triples)

# Sort after deduplication to ensure deterministic ordering before shuffling.
triples = sorted(set(triples))

n_after_dedup = len(triples)

print(f"[INFO] Unique triples: {n_after_dedup:,}")
print(
    f"[INFO] Removed duplicate triples: "
    f"{n_before_dedup - n_after_dedup:,}"
)

if not triples:
    raise ValueError("No valid triples found after filtering.")


# Build entity and relation dictionaries

entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
relations = sorted({r for _, r, _ in triples})

entity2id = {
    entity: idx
    for idx, entity in enumerate(entities)
}

relation2id = {
    relation: idx
    for idx, relation in enumerate(relations)
}

with open(entities_dict_path, "w", encoding="utf-8") as f:
    for entity, idx in entity2id.items():
        f.write(f"{idx}\t{entity}\n")

with open(relations_dict_path, "w", encoding="utf-8") as f:
    for relation, idx in relation2id.items():
        f.write(f"{idx}\t{relation}\n")

print(f"[INFO] Saved entities.dict: {len(entity2id):,} entities")
print(f"[INFO] Saved relations.dict: {len(relation2id):,} relations")


def calculate_split_sizes(
    n_total,
    train_ratio,
    valid_ratio,
    test_ratio
):
    """
    Calculate integer train/validation/test sizes using the
    largest-remainder method while preserving the total sample size.
    """
    ratios = [train_ratio, valid_ratio, test_ratio]
    raw_sizes = [n_total * ratio for ratio in ratios]
    sizes = [int(size) for size in raw_sizes]
    remainder = n_total - sum(sizes)

    fractional_parts = [
        raw_sizes[i] - sizes[i]
        for i in range(len(sizes))
    ]

    order = sorted(
        range(len(sizes)),
        key=lambda i: fractional_parts[i],
        reverse=True
    )

    for i in order[:remainder]:
        sizes[i] += 1

    return sizes


n_total = len(triples)

n_train_target, n_valid_target, n_test_target = calculate_split_sizes(
    n_total,
    train_ratio,
    valid_ratio,
    test_ratio
)

print("[INFO] Target split sizes:")
print(f"       Train: {n_train_target:,}")
print(f"       Valid: {n_valid_target:,}")
print(f"       Test : {n_test_target:,}")


# Shuffle triples reproducibly

rng = random.Random(SEED)
rng.shuffle(triples)


# Ensure full entity coverage in training

print("[INFO] Building training set with full entity coverage...")

coverage_train = []
remaining_triples = []
train_entities = set()

for triple in triples:
    h, r, t = triple

    if h not in train_entities or t not in train_entities:
        coverage_train.append(triple)
        train_entities.add(h)
        train_entities.add(t)
    else:
        remaining_triples.append(triple)

print(
    f"[INFO] Triples required for full entity coverage: "
    f"{len(coverage_train):,}"
)
print(
    f"[INFO] Entities covered in train: "
    f"{len(train_entities):,} / {len(entities):,}"
)

if train_entities != set(entities):
    missing_entities = set(entities) - train_entities
    raise RuntimeError(
        f"Entity coverage failed. {len(missing_entities)} entities "
        f"are missing from training set."
    )


# Complete the train/validation/test split

if len(coverage_train) <= n_train_target:
    n_extra_train = n_train_target - len(coverage_train)

    train_set = coverage_train + remaining_triples[:n_extra_train]
    remaining_after_train = remaining_triples[n_extra_train:]

    valid_set = remaining_after_train[:n_valid_target]
    test_set = remaining_after_train[
        n_valid_target: n_valid_target + n_test_target
    ]

else:
    print(
        f"[WARNING] Full entity coverage requires {len(coverage_train):,} "
        f"triples, which is larger than target train size "
        f"{n_train_target:,}."
    )
    print(
        "[WARNING] Training set will therefore be larger than "
        "the requested train ratio."
    )

    train_set = coverage_train
    remaining_after_train = remaining_triples

    eval_ratio_sum = valid_ratio + test_ratio

    if eval_ratio_sum > 0:
        test_fraction = test_ratio / eval_ratio_sum
        n_test_actual = round(len(remaining_after_train) * test_fraction)
    else:
        n_test_actual = 0

    n_valid_actual = len(remaining_after_train) - n_test_actual

    valid_set = remaining_after_train[:n_valid_actual]
    test_set = remaining_after_train[n_valid_actual:]


# Validate final splits

if len(train_set) + len(valid_set) + len(test_set) != n_total:
    raise RuntimeError(
        f"Split size mismatch: train + valid + test != total. "
        f"{len(train_set) + len(valid_set) + len(test_set)} != {n_total}"
    )

train_entities_final = {
    entity
    for h, _, t in train_set
    for entity in (h, t)
}

valid_entities = {
    entity
    for h, _, t in valid_set
    for entity in (h, t)
}

test_entities = {
    entity
    for h, _, t in test_set
    for entity in (h, t)
}

missing_valid_entities = valid_entities - train_entities_final
missing_test_entities = test_entities - train_entities_final

if missing_valid_entities:
    raise RuntimeError(
        f"{len(missing_valid_entities)} validation entities do not "
        f"appear in training set."
    )

if missing_test_entities:
    raise RuntimeError(
        f"{len(missing_test_entities)} test entities do not "
        f"appear in training set."
    )

print(
    "[INFO] Entity coverage check passed: "
    "all valid/test entities appear in train."
)


# Save train/validation/test triples

def save_triples(filename, triples_list):
    with open(filename, "w", encoding="utf-8") as f:
        for h, r, t in triples_list:
            f.write(f"{h}\t{r}\t{t}\n")


save_triples(train_path, train_set)
save_triples(valid_path, valid_set)
save_triples(test_path, test_set)


# Final statistics

print("[INFO] Final split:")
print(f"       Train: {len(train_set):,} ({len(train_set) / n_total:.4%})")
print(f"       Valid: {len(valid_set):,} ({len(valid_set) / n_total:.4%})")
print(f"       Test : {len(test_set):,} ({len(test_set) / n_total:.4%})")

print("[INFO] Files saved:")
print(f"       {train_path}")
print(f"       {valid_path}")
print(f"       {test_path}")

print("[INFO] Done ✅")