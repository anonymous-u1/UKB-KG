"""Entity normalization, deduplication and semantic typing.

Three steps, in this order:

  1. Case unification (--mode ukb only). Mentions that differ only in case are
     collapsed onto the most frequent variant, so "Type 2 Diabetes" and
     "type 2 diabetes" become one node. The most frequent form is used rather
     than lowercasing everything, to keep acronyms and gene symbols intact.
  2. Deduplication of identical triples within an article.
  3. Semantic typing. Every entity is assigned one of the semantic groups
     by the classifier trained in entity_type_tag/, and each relation is typed
     by its endpoint pair (e.g. "CHEM-DISO").

BIOS-derived edges (--mode completion) skip step 1: their names come from the
BIOS vocabulary and are already canonical.
"""

import argparse
from collections import Counter, defaultdict

from entity_type_tag.entity_type import EntityTypeClassifier
from utils import pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--mode",
        required=True,
        choices=["ukb", "completion"],
        help="'ukb' for literature triples, 'completion' for BIOS-derived edges.",
    )
    p.add_argument("--model-dir", default="entity_type_tag/save/save_35w_256")
    return p.parse_args()


def iter_triples(records):
    for record in records:
        for _, triples in record.items():
            for triple in triples:
                yield triple


def unify_case(records, label):
    """Collapse case-only variants of an entity onto its most frequent form."""
    counter = Counter()
    for triple in iter_triples(records):
        counter[triple["head"]["name"].strip()] += 1
        counter[triple["tail"]["name"].strip()] += 1

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

    print(f"[Unify - {label}] Found {len(replace_map)} entity variants to unify.")
    print(f"[Unify - {label}] Example replacements (first 10):")
    for k, v in list(replace_map.items())[:10]:
        print(f"  {k} → {v}")

    for triple in iter_triples(records):
        for role in ("head", "tail"):
            name = triple[role]["name"].strip()
            triple[role]["name"] = replace_map.get(name, name)
        triple["relation"]["name"] = triple["relation"]["name"].strip()

    return records


def deduplicate(records, label):
    """Drop repeated (head, relation, tail) within each article."""
    before = after = 0
    deduped = []

    for record in records:
        key, triples = next(iter(record.items()))
        before += len(triples)

        seen = set()
        unique = []
        for triple in triples:
            triple_key = (
                triple["head"]["name"].strip(),
                triple["relation"]["name"].strip(),
                triple["tail"]["name"].strip(),
            )
            if triple_key not in seen:
                seen.add(triple_key)
                unique.append(triple)

        after += len(unique)
        deduped.append({key: unique})

    print(f"[Deduplication - {label}] Total entries: {len(deduped)}")
    print(f"[Deduplication - {label}] Triples before: {before}")
    print(f"[Deduplication - {label}] Triples after:  {after}")
    print(f"[Deduplication - {label}] Duplicates removed: {before - after}")
    return deduped


def assign_types(records, clf, label):
    """Type every entity once, then stamp the types onto every triple."""
    entity_set = set()
    for triple in iter_triples(records):
        entity_set.add(triple["head"]["name"])
        entity_set.add(triple["tail"]["name"])

    entity_list = sorted(entity_set)
    print(f"[Typing - {label}] Classifying {len(entity_list)} unique entities ...")
    entity2type = clf.predict_dict(entity_list)

    stats = defaultdict(set)
    for triple in iter_triples(records):
        h = triple["head"]["name"]
        t = triple["tail"]["name"]
        triple["head"]["type"] = entity2type[h]
        triple["tail"]["type"] = entity2type[t]
        triple["relation"]["type"] = f"{entity2type[h]}-{entity2type[t]}"
        stats[entity2type[h]].add(h)
        stats[entity2type[t]].add(t)

    print(f"[Typing - {label}] === Entity count per class ===")
    for k in sorted(stats):
        print(f"  {k}: {len(stats[k])}")

    return records


def main():
    args = parse_args()
    label = "UKB" if args.mode == "ukb" else "Completion"
    print(f"Mode: {args.mode}\nInput:  {args.input}\nOutput: {args.output}")

    records = pipeline.read_records(args.input)

    if args.mode == "ukb":
        records = unify_case(records, label)

    records = deduplicate(records, label)

    clf = EntityTypeClassifier(model_dir=args.model_dir)
    records = assign_types(records, clf, label)

    pipeline.write_records(args.output, records)
    print(f"✅ Saved processed file to {args.output}")


if __name__ == "__main__":
    main()
