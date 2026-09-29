"""Create deterministic artificially-sparsified sequential datasets.

The operation removes interactions while preserving the original user IDs and
the complete item universe.  Every user keeps at least ``min_len`` interactions
and every item keeps at least one occurrence, so changes in the experiment are
not caused by changing the catalogue or the domain.

Example:
    python sparsify_sequences.py --input ml-1m.txt --output ml-1m-s50.txt \
        --keep-rate 0.50 --seed 2026
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read_sequences(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [int(x) for x in line.split()]
        rows.append((values[0], values[1:]))
    if not rows:
        raise ValueError(f"No sequences found in {path}")
    return rows


def sparsify(rows, keep_rate, seed, min_len):
    if not 0 < keep_rate <= 1:
        raise ValueError("keep_rate must be in (0, 1]")
    rng = random.Random(seed)
    selected = []
    occurrences = defaultdict(list)
    kept_items = set()

    for row_idx, (_, items) in enumerate(rows):
        if len(items) < min_len:
            raise ValueError(f"User has only {len(items)} interactions, below min_len")
        keep = {i for i in range(len(items)) if rng.random() < keep_rate}
        if len(keep) < min_len:
            remaining = [i for i in range(len(items)) if i not in keep]
            keep.update(rng.sample(remaining, min_len - len(keep)))
        selected.append(keep)
        for pos, item in enumerate(items):
            occurrences[item].append((row_idx, pos))
        kept_items.update(items[pos] for pos in keep)

    # Repair coverage deterministically.  The item universe remains identical
    # to the source even at aggressive sparsification rates.
    for item in sorted(occurrences):
        if item not in kept_items:
            candidates = occurrences[item]
            row_idx, pos = candidates[rng.randrange(len(candidates))]
            selected[row_idx].add(pos)
            kept_items.add(item)

    out = []
    for (uid, items), keep in zip(rows, selected):
        out.append((uid, [item for pos, item in enumerate(items) if pos in keep]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--keep-rate", type=float, required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--min-len", type=int, default=3)
    args = p.parse_args()

    rows = read_sequences(args.input)
    out = sparsify(rows, args.keep_rate, args.seed, args.min_len)
    source_items = {item for _, items in rows for item in items}
    output_items = {item for _, items in out for item in items}
    if source_items != output_items:
        raise AssertionError("Item universe changed during sparsification")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join("{} {}".format(uid, " ".join(map(str, items)))
                               for uid, items in out) + "\n", encoding="utf-8")
    meta = {
        "input": str(Path(args.input).resolve()),
        "output": str(output.resolve()),
        "keep_rate": args.keep_rate,
        "seed": args.seed,
        "min_len": args.min_len,
        "users": len(rows),
        "source_interactions": sum(len(items) for _, items in rows),
        "output_interactions": sum(len(items) for _, items in out),
        "source_items": len(source_items),
        "output_items": len(output_items),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
