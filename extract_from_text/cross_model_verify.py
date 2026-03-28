import os
import json
from collections import defaultdict
from typing import List, Dict, Set


# =========================
#       CONFIG
# =========================
VOTE_FILES = [
    "save/save_ablation/triple/triples_gpt-5_minimal_filter1_refine_filter2_gpt-5eval.json",
    "save/save_ablation/triple/triples_gpt-5_minimal_filter1_refine_filter2_deepseekeval.json",
    "save/save_ablation/triple/triples_gpt-5_minimal_filter1_refine_filter2_qweneval.json",
]
TRIPLE_FILE = "save/save_ablation/triple/triples_gpt-5_minimal_filter1_refine_filter2.json"

FINAL_JUDGEMENT_OUT = "save/save_ablation/triple/triple/triples_gpt-5_minimal_filter1_refine_filter2_crossmodeleval.json"
FILTERED_TRIPLES_OUT = "save/save_ablation/triple/triple/triples_gpt-5_minimal_filter1_refine_filter2_crossmodelverified.json"

MIN_CORRECT_VOTES = 2


# =========================
#   LOAD JUDGEMENTS
# =========================
def load_single_vote_file(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for item in data:
        for doc_id, judgement_dict in item.items():
            result[doc_id] = judgement_dict

    return result


def get_common_doc_ids(vote_dicts: List[Dict[str, Dict[str, str]]]) -> Set[str]:
    if not vote_dicts:
        return set()

    common_doc_ids = set(vote_dicts[0].keys())
    for vd in vote_dicts[1:]:
        common_doc_ids &= set(vd.keys())

    return common_doc_ids

def load_vote_files_intersection_by_docid(paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    return:
    {
      doc_id: {
        triple_idx: [Correct, Incorrect, ...]
      }
    }
    """
    vote_dicts = [load_single_vote_file(path) for path in paths]
    common_doc_ids = get_common_doc_ids(vote_dicts)

    votes = defaultdict(lambda: defaultdict(list))

    for vd in vote_dicts:
        for doc_id in common_doc_ids:
            judgement_dict = vd[doc_id]
            for idx, label in judgement_dict.items():
                votes[doc_id][idx].append(label)

    print(f"Total common doc_ids: {len(common_doc_ids)}")
    return votes


# =========================
#   MAJORITY VOTING
# =========================
def majority_vote(votes: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, str]]:
    """
    ≥ MIN_CORRECT_VOTES => Correct
    else => Incorrect
    """
    final = {}

    for doc_id, triple_votes in votes.items():
        final[doc_id] = {}
        for idx, labels in triple_votes.items():
            correct_cnt = sum(l == "Correct" for l in labels)
            final[doc_id][idx] = (
                "Correct" if correct_cnt >= MIN_CORRECT_VOTES else "Incorrect"
            )

    return final


# =========================
#   FILTER TRIPLES
# =========================
def filter_triples(
    triple_file: str,
    final_judgement: Dict[str, Dict[str, str]],
) -> Dict[str, List[dict]]:
    with open(triple_file, "r", encoding="utf-8") as f:
        triples = json.load(f)

    filtered = {}

    for item in triples:
        for doc_id, triple_list in item.items():
            if doc_id not in final_judgement:
                continue

            keep = []
            for i, triple in enumerate(triple_list):
                idx = str(i + 1)  # 注意：判断文件是 1-based
                if final_judgement[doc_id].get(idx) == "Correct":
                    keep.append(triple)

            if keep:
                filtered[doc_id] = keep

    return filtered


def dict_to_list(data: Dict) -> List[dict]:
    """
    {doc_id: content} → [{doc_id: content}, ...]
    """
    return [{doc_id: content} for doc_id, content in data.items()]


def ensure_parent_dir(filepath: str):
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    # 1. load only common doc_ids & vote
    votes = load_vote_files_intersection_by_docid(VOTE_FILES)
    final_judgement_dict = majority_vote(votes)

    # save final judgement
    final_judgement_list = dict_to_list(final_judgement_dict)
    ensure_parent_dir(FINAL_JUDGEMENT_OUT)
    with open(FINAL_JUDGEMENT_OUT, "w", encoding="utf-8") as f:
        json.dump(final_judgement_list, f, indent=2, ensure_ascii=False)

    # 2. filter triples
    filtered_triples_dict = filter_triples(TRIPLE_FILE, final_judgement_dict)

    filtered_triples_list = dict_to_list(filtered_triples_dict)
    ensure_parent_dir(FILTERED_TRIPLES_OUT)
    with open(FILTERED_TRIPLES_OUT, "w", encoding="utf-8") as f:
        json.dump(filtered_triples_list, f, indent=2, ensure_ascii=False)

    print(f"Saved final judgement to {FINAL_JUDGEMENT_OUT}")
    print(f"Saved filtered triples to {FILTERED_TRIPLES_OUT}")


if __name__ == "__main__":
    main()
