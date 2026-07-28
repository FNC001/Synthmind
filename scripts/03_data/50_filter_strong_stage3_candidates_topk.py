#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    return obj

def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)

def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize_split_name(split: str) -> str:
    s = str(split).strip().lower()
    mapping = {"tr":"train","train":"train","va":"val","valid":"val","validation":"val","val":"val","te":"test","test":"test"}
    if s not in mapping:
        raise ValueError(f"Unsupported split: {split}")
    return mapping[s]

def group_key(row: Dict[str, Any]) -> Tuple[str, int]:
    mk = str(row.get("material_key", row.get("material_id", row.get("sample_id", ""))))
    pr = int(row.get("parent_precursor_rank", 0))
    return mk, pr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--keep_topk", type=int, required=True)
    parser.add_argument("--splits", type=str, default="val,test")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    splits = [normalize_split_name(x.strip()) for x in args.splits.split(",") if x.strip()]
    keep_topk = int(args.keep_topk)

    overall = {
        "mode": "filter_strong_stage3_candidates_topk",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "keep_topk": keep_topk,
        "splits": splits,
        "split_stats": {},
    }

    for split in splits:
        in_path = input_dir / f"{split}_candidates.jsonl"
        if not in_path.exists():
            raise FileNotFoundError(f"Missing file: {in_path}")

        rows = read_jsonl(in_path)
        groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(group_key(r), []).append(r)

        kept: List[Dict[str, Any]] = []
        for key, grp in groups.items():
            grp = sorted(grp, key=lambda x: int(x.get("condition_rank", 10**9)))
            for new_rank, r in enumerate(grp[:keep_topk]):
                rr = dict(r)
                rr["condition_rank"] = int(new_rank)
                rr["stage3_model"] = f"{rr.get('stage3_model', 'stage3')}|topk{keep_topk}"
                kept.append(rr)

        out_path = output_dir / f"{split}_candidates.jsonl"
        write_jsonl(out_path, kept)
        overall["split_stats"][split] = {
            "input_rows": len(rows),
            "output_rows": len(kept),
            "n_groups": len(groups),
            "output_path": str(out_path),
        }
        print(f"[{split}] input={len(rows)} groups={len(groups)} output={len(kept)} keep_topk={keep_topk}")

    write_json(output_dir / "candidate_schema.json", {
        "mode": "filter_strong_stage3_candidates_topk",
        "keep_topk": keep_topk,
        "notes": [
            "Keep first K candidates per (material_key, parent_precursor_rank), ordered by condition_rank ascending.",
            "condition_rank is reassigned from 0 after filtering."
        ],
    })
    write_json(output_dir / "candidate_summary.json", overall)
    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")

if __name__ == "__main__":
    main()
