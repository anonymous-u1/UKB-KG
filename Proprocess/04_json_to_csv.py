"""Flatten the typed triples into the edge-list CSV that is the KG itself.

Steps:
  1. Concatenate the literature-derived and BIOS-derived triples. Case
     unification is re-run over the union.
  2. Write one row per triple.
  3. Resolve type conflicts.
  4. Score and deduplicate.
"""

import argparse
import csv
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from utils import pipeline

FIELDNAMES = [
    "HeadName",
    "HeadType",
    "HeadCID",
    "TailName",
    "TailType",
    "TailCID",
    "RelName",
    "RelType",
    "RELID",
    "PMCID",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Typed record files (literature triples, and optionally BIOS edges).",
    )
    p.add_argument("--output-csv", required=True)
    p.add_argument("--merged-json", default="", help="Optional merged JSON dump.")
    return p.parse_args()


def load_and_unify(paths, merged_path):
    print("\n[Step 1] Loading JSON triples from:")
    for path in paths:
        print(f"  - {path}")

    data = []
    for path in paths:
        data.extend(pipeline.read_records(path))
    print(f"[Step 1] Total triple records merged: {len(data)}")

    print("\n[Step 2] Unify triples that differ only in case")
    counter = Counter()
    for paper in data:
        for _, triples in paper.items():
            for tri in triples:
                counter[tri["head"]["name"].strip()] += 1
                counter[tri["tail"]["name"].strip()] += 1

    groups = defaultdict(list)
    for ent in counter:
        groups[ent.lower()].append(ent)

    replace_map = {}
    for variants in groups.values():
        if len(variants) == 1:
            continue
        best = max(variants, key=lambda x: counter[x])
        for v in variants:
            replace_map[v] = best

    print(f"[Step 2] Found {len(replace_map)} entity variants to unify.")

    for paper in data:
        for _, triples in paper.items():
            for tri in triples:
                for role in ("head", "tail"):
                    name = tri[role]["name"].strip()
                    tri[role]["name"] = replace_map.get(name, name)
                tri["relation"]["name"] = tri["relation"]["name"].strip()

    print("[Step 2] Example replacement mappings (first 10):")
    for k, v in list(replace_map.items())[:10]:
        print(f"  {k} → {v}")

    if merged_path:
        pipeline.write_records(merged_path, data)
        print(f"[Step 2] Merged JSON saved to: {merged_path}")

    return data


def build_row(triple, article_id):
    head, tail, relation = triple["head"], triple["tail"], triple["relation"]
    return {
        "HeadName": head["name"],
        "HeadType": head["type"],
        "HeadCID": head.get("CID", "none"),
        "TailName": tail["name"],
        "TailType": tail["type"],
        "TailCID": tail.get("CID", "none"),
        # Relation names become Neo4j relationship types, which cannot
        # contain spaces or hyphens.
        "RelName": relation["name"].replace(" ", "_").replace("-", "_"),
        "RelType": relation["type"],
        "RELID": relation.get("RELID", "none"),
        # BIOS-derived edges are keyed 'BIOS_completion', not a PMCID.
        "PMCID": article_id[:-4] if article_id.endswith(".xml") else "none",
    }


def triples_to_csv(data, csv_path):
    print(f"\n[Step 3] Converting triples to CSV: {csv_path}")
    count = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for paper in data:
            for article_id, triples in paper.items():
                for tri in triples:
                    writer.writerow(build_row(tri, article_id))
                    count += 1
    print(f"[Step 3] CSV writing done. Total rows: {count}")


def process_csv(path):
    print(f"\n[Step 4] Process CSV: {path}")
    df = pd.read_csv(path)
    print(f"[Step 4] Loaded {len(df)} rows")

    print("[Step 4] Resolving type conflicts ...")
    entity_type_counter = defaultdict(Counter)
    for prefix in ("Head", "Tail"):
        for name, t in zip(df[f"{prefix}Name"], df[f"{prefix}Type"]):
            entity_type_counter[name][t] += 1

    conflicts = sum(1 for cnt in entity_type_counter.values() if len(cnt) > 1)
    print(f"[Step 4] Found {conflicts} conflicting entities (will be unified)")

    name_to_major_type = {
        name: counter.most_common(1)[0][0]
        for name, counter in entity_type_counter.items()
    }
    for prefix in ("Head", "Tail"):
        df[f"{prefix}Type"] = df[f"{prefix}Name"].map(name_to_major_type)

    # Evidence count per triple, measured before duplicates are dropped.
    df["Frequency_Score"] = df.groupby(["HeadName", "RelName", "TailName"])[
        "HeadName"
    ].transform("count")
    c_max = df["Frequency_Score"].max()
    df["Frequency_Score"] = np.log1p(df["Frequency_Score"]) / np.log1p(c_max)

    before = len(df)
    df_unique = df.drop_duplicates(subset=["HeadName", "RelName", "TailName"])
    print(f"[Step 4] Removed duplicates: {before - len(df_unique)} rows")

    df_unique.to_csv(path, index=False)
    print(f"[Step 4] Clean CSV saved to: {path}")
    print("[Done ✅] All processing finished.\n")


if __name__ == "__main__":
    args = parse_args()
    triples = load_and_unify(args.inputs, args.merged_json)
    triples_to_csv(triples, args.output_csv)
    process_csv(args.output_csv)
