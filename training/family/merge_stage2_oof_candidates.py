#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local OOF Stage2 candidates into original training row order.")
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--candidate_name", required=True)
    parser.add_argument("--output_jsonl", required=True)
    args = parser.parse_args()
    fold_root = Path(args.fold_root).resolve()
    merged = {}
    for fold_dir in sorted(fold_root.glob("fold_*")):
        import numpy as np

        global_indices = np.load(fold_dir / "val_global_row_indices.npy")
        with (fold_dir / args.candidate_name).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                local_row = int(record["row_index"])
                global_row = int(global_indices[local_row])
                record["row_index"] = global_row
                record["oof_fold"] = fold_dir.name
                merged[global_row] = record
    expected = sum(len(__import__("numpy").load(path / "val_global_row_indices.npy")) for path in fold_root.glob("fold_*"))
    if len(merged) != expected or set(merged) != set(range(expected)):
        raise RuntimeError(f"OOF coverage mismatch: merged={len(merged)} expected={expected}")
    output = Path(args.output_jsonl).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in range(expected):
            handle.write(json.dumps(merged[row], ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(merged), "output": str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
