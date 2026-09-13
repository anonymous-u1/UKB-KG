"""Merge verified triples from the text and table pipelines into one set.
"""

import argparse

from utils import pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Verified record files to merge (e.g. the text and table outputs).",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--stats-output", default="", help="Optional stats report path.")
    return p.parse_args()


def normalize_article_id(article_id: str) -> str:
    """Both pipelines key on PMCxxxxxxx.xml; tolerate a bare PMCID."""
    article_id = str(article_id).strip()
    if article_id and not article_id.endswith(".xml"):
        article_id += ".xml"
    return article_id


def triple_key(triple):
    return (
        triple["head"]["name"].strip(),
        triple["relation"]["name"].strip(),
        triple["tail"]["name"].strip(),
    )


def merge(input_paths):
    merged = {}
    per_file = []

    for path in input_paths:
        records = pipeline.read_records(path)
        per_file.append(
            {
                "file": path,
                "articles": len(records),
                "triples": pipeline.count_triples(records),
            }
        )
        for record in records:
            for article_id, triples in record.items():
                merged.setdefault(normalize_article_id(article_id), []).extend(triples)

    before = sum(len(t) for t in merged.values())

    for article_id, triples in merged.items():
        seen = set()
        unique = []
        for triple in triples:
            key = triple_key(triple)
            if key not in seen:
                seen.add(key)
                unique.append(triple)
        merged[article_id] = unique

    after = sum(len(t) for t in merged.values())
    return merged, per_file, before, after


def main():
    args = parse_args()

    merged, per_file, before, after = merge(args.inputs)
    records = [{k: merged[k]} for k in sorted(merged)]
    pipeline.write_records(args.output, records)

    lines = ["===== Per-input file stats ====="]
    for stat in per_file:
        lines.append(
            f"{stat['file']}\n"
            f"  articles: {stat['articles']}\n"
            f"  triples : {stat['triples']}"
        )
    lines += [
        "",
        "===== Merged output stats =====",
        f"merged articles            : {len(records)}",
        f"triples before dedup       : {before}",
        f"triples after dedup        : {after}",
        f"duplicate triples removed  : {before - after}",
        f"output                     : {args.output}",
    ]
    report = "\n".join(lines)
    print(report)

    if args.stats_output:
        with open(args.stats_output, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nStats report saved to {args.stats_output}")


if __name__ == "__main__":
    main()
