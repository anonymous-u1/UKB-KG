"""KG fusion: pull in BIOS edges between entities the KG already contains.

BIOS is licensed separately and is not redistributed here; see the Data
Requirements section of the README. The term->CID index is cached under
--cache-dir because building it means a full pass over a ~46M-line file.
"""

import argparse
import json
import os

from tqdm import tqdm

from utils import pipeline

# Only used to give tqdm a total; a wrong value costs nothing but the ETA.
N_CONCEPT_LINES = 46_333_006
N_TRIPLET_LINES = 99_849_856

SKIPPED_RELATIONS = {"is a", "reverse is a"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="UMLS-linked KG triples.")
    p.add_argument("--output", required=True)
    p.add_argument("--bios-concepts", required=True, help="BIOS ConceptTerms.txt")
    p.add_argument("--bios-triplets", required=True, help="BIOS RelationTriplets.txt")
    p.add_argument("--cache-dir", required=True)
    return p.parse_args()


def build_term_to_cid(concept_path, cache_dir):
    """{lowercased English term: CID} over the BIOS concept vocabulary."""
    cache_file = os.path.join(cache_dir, "term2cid.json")
    if os.path.exists(cache_file):
        print(f"Loading cached term→CID mapping from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    term2cid = {}
    with open(concept_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, total=N_CONCEPT_LINES, desc="Build BIOS mappings:"):
            parts = line.strip().split("|")
            if len(parts) < 5:
                continue
            cid, _, term, lang, tty = parts[:5]
            if lang != "ENG":
                continue
            term2cid[term.lower().strip()] = cid

    print(f"✅ Built term→CID mapping: {len(term2cid):,} entries.")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(term2cid, f)
    return term2cid


def map_ukb_entities_to_cid(ukb_path, term2cid, cache_dir):
    """Map our entities onto BIOS CIDs, and record which pairs already connect."""
    cache_entity2cid = os.path.join(cache_dir, "entity2cid.json")
    cache_pairs = os.path.join(cache_dir, "ukb_pairs_cid.json")

    if os.path.exists(cache_entity2cid) and os.path.exists(cache_pairs):
        print("Loading cached entity2cid and ukb_pairs_cid")
        with open(cache_entity2cid, "r", encoding="utf-8") as f:
            entity2cid = json.load(f)
        with open(cache_pairs, "r", encoding="utf-8") as f:
            ukb_pairs_cid = {tuple(p) for p in json.load(f)}
        return entity2cid, ukb_pairs_cid

    entity2cid = {}
    ukb_pairs_cid = set()

    for paper in pipeline.read_records(ukb_path):
        for _, triples in paper.items():
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


def filter_bios_edges_by_cid(bios_path, entity2cid, ukb_pairs_cid, output_path):
    """Scan BIOS triplets, keeping new edges between entities we already have."""
    cid_set = set(entity2cid.values())
    name_by_cid = {v: k for k, v in entity2cid.items()}

    bios_completion = []
    count_total = 0

    with open(bios_path, "r", encoding="utf-8", errors="ignore") as bios_f:
        for line in tqdm(
            bios_f, total=N_TRIPLET_LINES, desc="Filtering BIOS Triplets (CID)", ncols=100
        ):
            parts = line.strip().split("|")
            if len(parts) < 7:
                continue
            _, head_cid, _, rel_id, rel_name, tail_cid, _ = parts
            count_total += 1

            if rel_name in SKIPPED_RELATIONS:
                continue
            if head_cid not in cid_set or tail_cid not in cid_set:
                continue
            if (head_cid, tail_cid) in ukb_pairs_cid:
                continue

            head_name = name_by_cid.get(head_cid, "N/A")
            tail_name = name_by_cid.get(tail_cid, "N/A")
            if head_name == "N/A" or tail_name == "N/A":
                continue

            bios_completion.append(
                {
                    "head": {"name": head_name, "CID": head_cid},
                    "relation": {"name": rel_name, "RELID": rel_id},
                    "tail": {"name": tail_name, "CID": tail_cid},
                }
            )

    # Wrapped as a single pseudo-article so downstream stages can treat these
    # edges with exactly the same code as the literature-derived ones.
    pipeline.write_records(output_path, [{"BIOS_completion": bios_completion}])

    print("\n=== Filtering Complete (CID-based, case-preserved) ===")
    print(f"Total BIOS lines scanned: {count_total:,}")
    print(f"New edges found: {len(bios_completion):,}")
    print(f"Results saved to: {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"Input:  {args.input}\nOutput: {args.output}\nCache:  {args.cache_dir}")

    term2cid = build_term_to_cid(args.bios_concepts, args.cache_dir)
    entity2cid, ukb_pairs_cid = map_ukb_entities_to_cid(
        args.input, term2cid, args.cache_dir
    )
    filter_bios_edges_by_cid(args.bios_triplets, entity2cid, ukb_pairs_cid, args.output)


if __name__ == "__main__":
    main()
