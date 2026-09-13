"""Cross-model verification: majority vote over several judges' labels.

Ablation for the single-model self-verifier. Each --votes file is a judgement
file produced by 03_verify_or_eval.py with --mode eval under a different judge
model. Only articles judged by *every* model take part, so all triples get the
same number of votes.
"""

import argparse
from collections import defaultdict

from utils import pipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--votes", nargs="+", required=True, help="Judgement files.")
    p.add_argument("--triples", required=True, help="Triples the votes refer to.")
    p.add_argument("--judgement-output", required=True)
    p.add_argument("--verified-output", required=True)
    p.add_argument(
        "--min-correct-votes",
        type=int,
        default=2,
        help="Votes needed to keep a triple (default: majority of 3).",
    )
    return p.parse_args()


def load_votes(paths):
    """{doc_id: {triple_idx: [label, ...]}} over doc_ids present in every file."""
    per_file = []
    for path in paths:
        per_file.append(
            {pipeline.record_key(r): next(iter(r.values())) for r in pipeline.read_records(path)}
        )

    common = set(per_file[0])
    for d in per_file[1:]:
        common &= set(d)

    votes = defaultdict(lambda: defaultdict(list))
    for d in per_file:
        for doc_id in common:
            for idx, label in d[doc_id].items():
                votes[doc_id][idx].append(label)

    print(f"Total common doc_ids: {len(common)}")
    return votes


def majority_vote(votes, min_correct_votes):
    return {
        doc_id: {
            idx: "Correct"
            if sum(l == "Correct" for l in labels) >= min_correct_votes
            else "Incorrect"
            for idx, labels in triple_votes.items()
        }
        for doc_id, triple_votes in votes.items()
    }


def filter_triples(triples_path, judgement):
    """Keep triples voted Correct.

    Judgement keys are the triple's 0-based position, matching what
    03_verify_or_eval.py writes.
    """
    kept = {}
    for record in pipeline.read_records(triples_path):
        for doc_id, triple_list in record.items():
            if doc_id not in judgement:
                continue
            keep = [
                triple
                for i, triple in enumerate(triple_list)
                if judgement[doc_id].get(str(i)) == "Correct"
            ]
            if keep:
                kept[doc_id] = keep
    return kept


def main():
    args = parse_args()

    votes = load_votes(args.votes)
    judgement = majority_vote(votes, args.min_correct_votes)
    pipeline.write_records(
        args.judgement_output, [{k: v} for k, v in judgement.items()]
    )

    kept = filter_triples(args.triples, judgement)
    pipeline.write_records(args.verified_output, [{k: v} for k, v in kept.items()])

    print(f"Saved final judgement to {args.judgement_output}")
    print(f"Saved filtered triples to {args.verified_output}")
    print(f"Articles kept: {len(kept)}")


if __name__ == "__main__":
    main()
