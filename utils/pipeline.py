"""Sharding and record IO shared by every pipeline stage.

Every stage of the pipeline consumes and produces the same record format:

    [ {"PMC1234567.xml": [ <triple>, ... ]}, ... ]

one entry per article, and every stage needs the same three things: split the
work list so shards can run concurrently, skip articles that a previous
(possibly interrupted) run already finished, and append results incrementally
so that a crash never loses more than the current batch.

Shard boundaries do not affect results: articles are processed independently,
so `--num-shards` is purely a knob for wall-clock time.

Also usable as a CLI, which is how the driver scripts merge shards:

    python -m utils.pipeline merge-shards --stage-dir DIR --output FILE
"""

import argparse
import json
import math
import os
from pathlib import Path

SHARD_GLOB = "shard_*.json"


# ---------------------------------------------------------------- sharding


def shard_items(items, shard: int, num_shards: int) -> list:
    """Return the contiguous slice of `items` belonging to `shard` (1-based).

    The final shard absorbs the remainder, so every item is covered exactly
    once for any `num_shards`.
    """
    items = list(items)
    if num_shards <= 1:
        return items
    if not 1 <= shard <= num_shards:
        raise ValueError(f"shard {shard} out of range 1..{num_shards}")

    size = math.ceil(len(items) / num_shards)
    start = (shard - 1) * size
    end = start + size if shard < num_shards else None
    return items[start:end]


def shard_path(stage_dir, shard: int, num_shards: int) -> Path:
    """Output file for one shard of a stage."""
    stage_dir = Path(stage_dir)
    if num_shards <= 1:
        return stage_dir / "all.json"
    return stage_dir / f"shard_{shard:03d}_of_{num_shards:03d}.json"


def merge_shards(stage_dir, output_path) -> int:
    """Concatenate every shard file of a stage into one record file."""
    stage_dir = Path(stage_dir)
    shard_files = sorted(stage_dir.glob(SHARD_GLOB)) or sorted(
        stage_dir.glob("all.json")
    )
    if not shard_files:
        raise FileNotFoundError(f"No shard files under {stage_dir}")

    print(f"Merging {len(shard_files)} shard file(s) from {stage_dir}:")
    records = []
    for f in shard_files:
        part = read_records(f)
        print(f"  └─ {f.name}: {len(part)} records")
        records.extend(part)

    write_records(output_path, records)
    print(f"✅ Merged {len(records)} records into {output_path}")
    return len(records)


# -------------------------------------------------------------- record IO


def read_records(path) -> list:
    """Read a record file, returning [] when it does not exist yet."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_records(path, records) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def record_key(record) -> str:
    """The article id a record is keyed by."""
    return next(iter(record))


def open_output(path) -> set:
    """Create the output file if absent; return the ids already finished.

    This is what makes every stage resumable: rerunning a stage after an
    interruption skips the articles already present in the output.
    """
    path = Path(path)
    if not os.path.isfile(path):
        write_records(path, [])
        print(f"Created {path}.")
        return set()

    done = {record_key(r) for r in read_records(path)}
    print(f"Resuming {path}: {len(done)} records already done.")
    return done


def append_records(path, records) -> None:
    """Append records to an existing record file."""
    if not records:
        return
    data = read_records(path)
    data.extend(records)
    write_records(path, data)


def count_triples(records) -> int:
    return sum(len(triples) for r in records for triples in r.values())


# -------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("merge-shards", help="Concatenate a stage's shard files.")
    m.add_argument("--stage-dir", required=True)
    m.add_argument("--output", required=True)

    s = sub.add_parser("stats", help="Print article/triple counts of a record file.")
    s.add_argument("--input", required=True)

    args = parser.parse_args()

    if args.command == "merge-shards":
        merge_shards(args.stage_dir, args.output)
    elif args.command == "stats":
        records = read_records(args.input)
        print(f"{args.input}")
        print(f"  articles: {len(records)}")
        print(f"  triples : {count_triples(records)}")


if __name__ == "__main__":
    main()
