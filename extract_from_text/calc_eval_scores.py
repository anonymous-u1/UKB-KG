import json
import ipdb

triple_path = r'save/save_ablation/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5verified_umlslinked98_gpt-5.4eval.json'
output_path = triple_path.replace('.json', '_metrics.txt')

with open(triple_path, 'r', encoding='utf-8') as f:
    evaluation_data = json.load(f)

# ====================== Count correct / incorrect ======================
evaluation_stats = []
for entry in evaluation_data:
    for _, label_dict in entry.items():
        correct = sum(label == "Correct" for label in label_dict.values())
        incorrect = sum(label == "Incorrect" for label in label_dict.values())
        evaluation_stats.append({"correct": correct, "incorrect": incorrect})

# ====================== Aggregate ======================
macro_correct = 0.0
macro_incorrect = 0.0
micro_correct = 0
micro_incorrect = 0
valid_files = 0

for stats in evaluation_stats:
    total = stats["correct"] + stats["incorrect"]
    if total > 0:
        macro_correct += stats["correct"] / total
        macro_incorrect += stats["incorrect"] / total
        micro_correct += stats["correct"]
        micro_incorrect += stats["incorrect"]
        valid_files += 1

# ====================== Compute metrics ======================
total_files = len(evaluation_stats)
total_triples = micro_correct + micro_incorrect
avg_triples_per_file = total_triples / valid_files if valid_files else 0
avg_correct_triples_per_file = micro_correct / valid_files if valid_files else 0

macro_correct_rate = macro_correct / total_files if total_files else 0
macro_incorrect_rate = macro_incorrect / total_files if total_files else 0
micro_correct_rate = micro_correct / total_triples if total_triples else 0
micro_incorrect_rate = micro_incorrect / total_triples if total_triples else 0

# ====================== Print results ======================
result_lines = [
    "=" * 60,
    f"Total Files Evaluated:         {total_files}",
    f"Valid Files (with triples):    {valid_files}",
    f"Avg. triples per file:         {avg_triples_per_file:.2f}",
    f"Avg. correct triples per file: {avg_correct_triples_per_file:.2f}",
    "-" * 60,
    f"✅ Macro Correct Rate:         {macro_correct_rate:.4f}",
    f"❌ Macro Incorrect Rate:       {macro_incorrect_rate:.4f}",
    "-" * 60,
    f"✅ Micro Correct Rate:         {micro_correct_rate:.4f}",
    f"❌ Micro Incorrect Rate:       {micro_incorrect_rate:.4f}",
    "=" * 60,
]

print("\n".join(result_lines))

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(result_lines))