"""Precision metrics from a Correct/Incorrect judgement file.

Macro rates average per-article precision; micro rates pool all triples.
"""

import argparse

from utils import pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Judgement file from 03_verify_or_eval.")
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
                    "correct": sum(v == "Correct" for v in label_dict.values()),
                    "incorrect": sum(v == "Incorrect" for v in label_dict.values()),
                }
            )

    macro_correct = macro_incorrect = 0.0
    micro_correct = micro_incorrect = valid_files = 0

    for stats in per_article:
        total = stats["correct"] + stats["incorrect"]
        if total > 0:
            macro_correct += stats["correct"] / total
            macro_incorrect += stats["incorrect"] / total
            micro_correct += stats["correct"]
            micro_incorrect += stats["incorrect"]
            valid_files += 1

    total_files = len(per_article)
    total_triples = micro_correct + micro_incorrect

    result_lines = [
        "=" * 60,
        f"Total Files Evaluated:         {total_files}",
        f"Valid Files (with triples):    {valid_files}",
        f"Avg. triples per file:         {total_triples / valid_files if valid_files else 0:.2f}",
        f"Avg. correct triples per file: {micro_correct / valid_files if valid_files else 0:.2f}",
        "-" * 60,
        f"✅ Macro Correct Rate:         {macro_correct / total_files if total_files else 0:.4f}",
        f"❌ Macro Incorrect Rate:       {macro_incorrect / total_files if total_files else 0:.4f}",
        "-" * 60,
        f"✅ Micro Correct Rate:         {micro_correct / total_triples if total_triples else 0:.4f}",
        f"❌ Micro Incorrect Rate:       {micro_incorrect / total_triples if total_triples else 0:.4f}",
        "=" * 60,
    ]

    print("\n".join(result_lines))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
