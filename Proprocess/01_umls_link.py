"""Entity alignment: rewrite entity mentions to UMLS canonical names.
"""

import argparse

from utils import pipeline
from utils.umls import umls_map


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.98)
    p.add_argument("--print-examples", type=int, default=50)
    return p.parse_args()


def safe_get_str(d: dict, *keys) -> str:
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return ""
        d = d[k]
    return str(d).strip() if d is not None else ""


def main():
    args = parse_args()
    print(f"Input:  {args.input}\nOutput: {args.output}\nThreshold: {args.threshold}")

    umls_mapper = umls_map(threshold=args.threshold)
    records = pipeline.read_records(args.input)

    linked_data = []
    total_triples = linked_entities = 0
    cache = {}
    example_count = 0

    for record in records:
        key, triples = next(iter(record.items()))
        updated = []

        for triple in triples:
            head = safe_get_str(triple, "head", "name")
            tail = safe_get_str(triple, "tail", "name")
            relation = safe_get_str(triple, "relation", "name")
            if not head or not tail or not relation:
                print(f"❌ Can not find head/tail/relation in {key}")

            for name in (head, tail):
                if name not in cache:
                    cache[name] = umls_mapper(name)

            head_linked = cache[head]
            tail_linked = cache[tail]

            if example_count < args.print_examples and (
                head_linked != head or tail_linked != tail
            ):
                print(f"{head} → Linked: {head_linked}")
                print(f"{tail} → Linked: {tail_linked}")
                example_count += 1

            if head_linked != head:
                triple["head"]["name"] = head_linked
                linked_entities += 1
            if tail_linked != tail:
                triple["tail"]["name"] = tail_linked
                linked_entities += 1

            updated.append(triple)
            total_triples += 1

        linked_data.append({key: updated})

    if len(records) != len(linked_data):
        print("❌ The number of papers before and after the linking are different.")

    pipeline.write_records(args.output, linked_data)

    print("=== Linking Summary ===")
    print(f"Total papers:  {len(records):,}")
    print(f"Total triples: {total_triples:,}")
    print(f"Unique entities looked up: {len(cache):,}")
    print(f"Linked entities: {linked_entities:,}")
    print(f"Saved linked file: {args.output}")


if __name__ == "__main__":
    main()
