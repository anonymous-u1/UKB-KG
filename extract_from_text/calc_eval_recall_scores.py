import json
import ipdb

triple_path = r'save/save_ablation/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_gpt-5.4eval.json'
output_path = triple_path.replace('.json', '_metrics.txt')

with open(triple_path, 'r', encoding='utf-8') as f:
    evaluation_data = json.load(f)

# ====================== Count covered / not covered ======================
evaluation_stats = []
for entry in evaluation_data:
    for _, label_dict in entry.items():
        covered = sum(label == "Covered" for label in label_dict.values())
        not_covered = sum(label == "Not Covered" for label in label_dict.values())
        evaluation_stats.append({
            "covered": covered,
            "not_covered": not_covered
        })

# ====================== Aggregate ======================
macro_recall = 0.0
micro_covered = 0
micro_total = 0
valid_files = 0

for stats in evaluation_stats:
    total_gt = stats["covered"] + stats["not_covered"]
    if total_gt > 0:
        macro_recall += stats["covered"] / total_gt
        micro_covered += stats["covered"]
        micro_total += total_gt
        valid_files += 1

# ====================== Compute metrics ======================
total_files = len(evaluation_stats)
avg_gt_triples_per_file = micro_total / valid_files if valid_files else 0

macro_recall_rate = macro_recall / valid_files if valid_files else 0
micro_recall_rate = micro_covered / micro_total if micro_total else 0

# ====================== Print results ======================
result_lines = [
    "=" * 60,
    f"Total Files Evaluated:        {total_files}",
    f"Valid Files (with GT):        {valid_files}",
    f"Avg. GT triples per file:     {avg_gt_triples_per_file:.2f}",
    "-" * 60,
    f"Macro Recall:              {macro_recall_rate:.4f}",
    "-" * 60,
    f"Micro Recall:              {micro_recall_rate:.4f}",
    "=" * 60,
]

print("\n".join(result_lines))

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(result_lines))