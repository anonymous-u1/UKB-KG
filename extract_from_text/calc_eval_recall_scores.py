"""Recall metrics from a Covered/Not Covered judgement file.

Input comes from 04_recall_eval.py, which labels each reference triple by
whether the extracted set covers it.
"""

import argparse

from utils import pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Judgement file from 04_recall_eval.")
    p.add_argument("--output", default="", help="Defaults to <input>_metrics.txt.")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = args.output or args.input.replace(".json", "_metrics.txt")

    per_article = []
    for entry in pipeline.read_records(args.input):
        for _, label_dict in entry.items():
            per_article.append(
                {
                    "covered": sum(v == "Covered" for v in label_dict.values()),
                    "not_covered": sum(v == "Not Covered" for v in label_dict.values()),
                }
            )

    macro_recall = 0.0
    micro_covered = micro_total = valid_files = 0

    for stats in per_article:
        total_gt = stats["covered"] + stats["not_covered"]
        if total_gt > 0:
            macro_recall += stats["covered"] / total_gt
            micro_covered += stats["covered"]
            micro_total += total_gt
            valid_files += 1

    result_lines = [
        "=" * 60,
        f"Total Files Evaluated:        {len(per_article)}",
        f"Valid Files (with GT):        {valid_files}",
        f"Avg. GT triples per file:     {micro_total / valid_files if valid_files else 0:.2f}",
        "-" * 60,
        f"Macro Recall:              {macro_recall / valid_files if valid_files else 0:.4f}",
        "-" * 60,
        f"Micro Recall:              {micro_covered / micro_total if micro_total else 0:.4f}",
        "=" * 60,
    ]

    print("\n".join(result_lines))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
