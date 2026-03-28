import json
import os
from tqdm import tqdm


ukb_triple_path = "save/save_sample/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98.json"
bios_concept_path = "BIOS/bios_v3_release/CoreData/ConceptTerms.txt"
bios_triplet_path = "BIOS/bios_v3_release/CoreData/RelationTriplets.txt"
cache_dir = "Proprocess/bios_cache_umlslinked98"
output_path = "save/save_bios/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_completion.json"

os.makedirs(cache_dir, exist_ok=True)

# ============= Step 1. Build term → CID mapping =============
def build_term_to_cid(concept_path, cache_dir):
    cache_file = os.path.join(cache_dir, "term2cid.json")
    if os.path.exists(cache_file):
        print(f"Loading cached term→CID mapping from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    term2cid = {}
    with open(concept_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, total=46333006, desc="Build BIOS mappings:"):
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            cid, _, term, lang, tty = parts[:5]
            if lang != "ENG":
                continue
            term_lower = term.lower().strip()
            term2cid[term_lower] = cid

    print(f"✅ Built term→CID mapping: {len(term2cid):,} entries.")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(term2cid, f)
    return term2cid


# ============= Step 2. CID Extract entities from UKB-KG and map them to CID =============
def map_ukb_entities_to_cid(ukb_path, term2cid, cache_dir):
    cache_entity2cid = os.path.join(cache_dir, "entity2cid.json")
    cache_pairs = os.path.join(cache_dir, "ukb_pairs_cid.json")

    if os.path.exists(cache_entity2cid) and os.path.exists(cache_pairs):
        print(f"Loading cached entity2cid and ukb_pairs_cid")
        with open(cache_entity2cid, "r", encoding="utf-8") as f:
            entity2cid = json.load(f)
        with open(cache_pairs, "r", encoding="utf-8") as f:
            ukb_pairs_cid = {tuple(p) for p in json.load(f)}
        return entity2cid, ukb_pairs_cid

    with open(ukb_path, "r", encoding="utf-8") as f:
        ukb_data = json.load(f)

    entity2cid = {}
    ukb_pairs_cid = set()

    for paper in ukb_data:
        for pmc, triples in paper.items():
            for tri in triples:
                h = tri["head"]["name"].strip()
                t = tri["tail"]["name"].strip()
                h_lower, t_lower = h.lower(), t.lower()

                if h_lower in term2cid and h not in entity2cid:
                    entity2cid[h] = term2cid[h_lower]
                if t_lower in term2cid and t not in entity2cid:
                    entity2cid[t] = term2cid[t_lower]

                if h_lower in term2cid and t_lower in term2cid:
                    ukb_pairs_cid.add((term2cid[h_lower], term2cid[t_lower]))

    print(f"✅ Mapped {len(entity2cid):,} UKB entities to BIOS CIDs.")
    print(f"✅ Total unique UKB CID pairs: {len(ukb_pairs_cid):,}")

    with open(cache_entity2cid, "w", encoding="utf-8") as f:
        json.dump(entity2cid, f, ensure_ascii=False, indent=2)
    with open(cache_pairs, "w", encoding="utf-8") as f:
        json.dump(list(ukb_pairs_cid), f)

    return entity2cid, ukb_pairs_cid


# ============= Step 3. Filter and complete relationships from BIOS =============
def filter_bios_edges_by_cid(bios_path, entity2cid, ukb_pairs_cid, output_path):
    """
    Return：[{"BIOS_completion": [ {...}, {...} ]}]
    """
    cid_set = set(entity2cid.values())
    name_by_cid = {v: k for k, v in entity2cid.items()}

    bios_completion = []
    count_total = 0
    count_match = 0

    with open(bios_path, "r", encoding="utf-8", errors="ignore") as bios_f:
        for line in tqdm(bios_f, total=99849856, desc="Filtering BIOS Triplets (CID)", ncols=100):
            parts = line.strip().split("|")
            if len(parts) < 7:
                continue
            _, head_cid, _, rel_id, rel_name, tail_cid, _ = parts
            count_total += 1

            if rel_name == 'is a' or rel_name == 'reverse is a':
                continue

            if head_cid not in cid_set or tail_cid not in cid_set:
                continue

            if (head_cid, tail_cid) in ukb_pairs_cid:
                continue

            head_name = name_by_cid.get(head_cid, "N/A")
            tail_name = name_by_cid.get(tail_cid, "N/A")

            if head_name == "N/A" or tail_name == "N/A":
                continue

            record = {
                "head": {"name": head_name, "CID": head_cid},
                "relation": {"name": rel_name, "RELID": rel_id},
                "tail": {"name": tail_name, "CID": tail_cid}
            }
            bios_completion.append(record)
            count_match += 1

    final_output = [{"BIOS_completion": bios_completion}]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n=== Filtering Complete (CID-based, case-preserved) ===")
    print(f"Total BIOS lines scanned: {count_total:,}")
    print(f"New edges found: {count_match:,}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    term2cid = build_term_to_cid(bios_concept_path, cache_dir)
    entity2cid, ukb_pairs_cid = map_ukb_entities_to_cid(ukb_triple_path, term2cid, cache_dir)
    filter_bios_edges_by_cid(bios_triplet_path, entity2cid, ukb_pairs_cid, output_path)
