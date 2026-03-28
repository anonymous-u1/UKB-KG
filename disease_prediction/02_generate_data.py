import json
import os
import random
from tqdm import tqdm


triples_path = "save/save_sample/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_bioscompleted_mapped_12r.json"
data_dir = 'disease_prediction/KGE-RotatE/data/ukb_gpt-5_minimal_12r'
entities_dict_path = os.path.join(data_dir, "entities.dict")
relations_dict_path = os.path.join(data_dir, "relations.dict")
train_path = os.path.join(data_dir, "train.txt")
valid_path = os.path.join(data_dir, "valid.txt")
test_path = os.path.join(data_dir, "test.txt")

train_ratio = 0.9
valid_ratio = 0.0999
test_ratio = 0.0001

with open(triples_path, "r", encoding="utf-8") as f:
    data = json.load(f)

triples = []

for paper in tqdm(data, desc="Loading triples"):
    for _, triples_list in paper.items():
        for tri in triples_list:
            h = tri["head"]["name"].strip()
            r = tri["relation"]["name"].strip()
            t = tri["tail"]["name"].strip()
            triples.append((h, r, t))

triples = list(set(triples))
print(f"✅ Total unique triples: {len(triples)}")


# Construct entity and relation dictionary
entities = sorted({h for h, _, t in triples} | {t for h, _, t in triples})
relations = sorted({r for _, r, _ in triples})

entity2id = {ent: i for i, ent in enumerate(entities)}
relation2id = {rel: i for i, rel in enumerate(relations)}

with open(entities_dict_path, "w", encoding="utf-8") as f:
    for ent, idx in entity2id.items():
        f.write(f"{idx}\t{ent}\n")

with open(relations_dict_path, "w", encoding="utf-8") as f:
    for rel, idx in relation2id.items():
        f.write(f"{idx}\t{rel}\n")

print(f"✅ Saved entities.dict ({len(entity2id)}) and relations.dict ({len(relation2id)})")


# split dataset
random.shuffle(triples)
n_total = len(triples)
n_train = int(n_total * train_ratio)
n_valid = int(n_total * valid_ratio)

train_set = triples[:n_train]
valid_set = triples[n_train:n_train + n_valid]
test_set = triples[n_train + n_valid:]

def save_triples(filename, triples_list):
    with open(filename, "w", encoding="utf-8") as f:
        for h, r, t in triples_list:
            f.write(f"{h}\t{r}\t{t}\n")

save_triples(train_path, train_set)
save_triples(valid_path, valid_set)
save_triples(test_path, test_set)

print(f"✅ Files saved: train.txt ({len(train_set)}), valid.txt ({len(valid_set)}), test.txt ({len(test_set)})")
