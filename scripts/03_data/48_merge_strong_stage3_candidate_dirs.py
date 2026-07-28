#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
48_merge_strong_stage3_candidate_dirs.py

将多个 strong stage3 candidate 目录做并集、去重，并重新分配 condition_rank。

输入目录格式要求：
  <input_dir>/{split}_candidates.jsonl

输出：
  <output_dir>/{split}_candidates.jsonl
  <output_dir>/candidate_schema.json
  <output_dir>/candidate_summary.json
"""

from __future__ import annotations

import argparse
import json
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
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_split_name(split: str) -> str:
    s = str(split).strip().lower()
    mapping = {
        "tr": "train",
        "train": "train",
        "va": "val",
        "valid": "val",
        "validation": "val",
        "val": "val",
        "te": "test",
        "test": "test",
    }
    if s not in mapping:
        raise ValueError(f"Unsupported split: {split}")
    return mapping[s]


def rounded_cont_key(cont: Dict[str, Any], ndigits: int) -> Tuple[Tuple[str, Any], ...]:
    items = []
    for k in sorted(cont.keys()):
        v = cont[k]
        if isinstance(v, float):
            v = round(v, ndigits)
        items.append((k, v))
    return tuple(items)


def disc_key(disc: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple((k, disc[k]) for k in sorted(disc.keys()))


def resolve_candidates_file(input_dir: Path, split: str) -> Path:
    p = input_dir / f"{split}_candidates.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def merge_split(
    input_dirs: List[Path],
    split: str,
    round_ndigits: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = normalize_split_name(split)

    all_rows: List[Dict[str, Any]] = []
    per_source_counts = {}

    for input_dir in input_dirs:
        p = resolve_candidates_file(input_dir, split)
        rows = read_jsonl(p)
        per_source_counts[str(input_dir)] = len(rows)
        for r in rows:
            rr = dict(r)
            rr["_source_dir"] = str(input_dir)
            all_rows.append(rr)

    # dedup within (material_key, parent_precursor_rank) by rounded continuous + discrete conditions
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in all_rows:
        mk = str(r.get("material_key", r.get("material_id", r.get("sample_id", ""))))
        pr = int(r.get("parent_precursor_rank", 0))
        groups.setdefault((mk, pr), []).append(r)

    merged_rows: List[Dict[str, Any]] = []
    n_duplicates_removed = 0

    for (mk, pr), rows in groups.items():
        seen = {}
        ordered_unique = []
        for r in rows:
            key = (
                mk,
                pr,
                disc_key(r.get("disc_conditions", {}) or {}),
                rounded_cont_key(r.get("cont_conditions", {}) or {}, round_ndigits),
            )
            if key in seen:
                n_duplicates_removed += 1
                seen[key]["_merged_from"].append(r.get("_source_dir"))
                continue
            rr = dict(r)
            rr["_merged_from"] = [r.get("_source_dir")]
            seen[key] = rr
            ordered_unique.append(rr)

        # stable sort: by stage3_score descending, then source, then old rank
        ordered_unique.sort(
            key=lambda x: (
                -(float(x.get("stage3_score", 0.0)) if x.get("stage3_score") is not None else 0.0),
                str(x.get("_source_dir", "")),
                int(x.get("condition_rank", 0)),
            )
        )

        for new_rank, r in enumerate(ordered_unique):
            out = dict(r)
            out["condition_rank"] = int(new_rank)
            # preserve provenance in stage3_model
            model = str(out.get("stage3_model", ""))
            sources = sorted(set([str(s) for s in out.get("_merged_from", []) if s]))
            if sources:
                out["stage3_model"] = f"{model}|merged"
                out["merged_from"] = sources
            out.pop("_source_dir", None)
            out.pop("_merged_from", None)
            merged_rows.append(out)

    summary = {
        "split": split,
        "n_input_rows_total": len(all_rows),
        "n_output_rows": len(merged_rows),
        "n_duplicates_removed": n_duplicates_removed,
        "n_groups": len(groups),
        "per_source_counts": per_source_counts,
    }
    return merged_rows, summary


def main():
    parser = argparse.ArgumentParser(description="Merge strong stage3 candidate dirs by union + dedup.")
    parser.add_argument("--input_dirs", type=str, required=True,
                        help="Comma-separated stage3 candidate dirs")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", type=str, default="val,test")
    parser.add_argument("--round_ndigits", type=int, default=3)
    args = parser.parse_args()

    input_dirs = [Path(x.strip()).expanduser().resolve() for x in args.input_dirs.split(",") if x.strip()]
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    splits = [normalize_split_name(x.strip()) for x in args.splits.split(",") if x.strip()]

    overall = {
        "mode": "merged_strong_stage3_candidate_union",
        "input_dirs": [str(x) for x in input_dirs],
        "round_ndigits": int(args.round_ndigits),
        "splits": splits,
        "split_stats": {},
    }

    for split in splits:
        rows, info = merge_split(input_dirs=input_dirs, split=split, round_ndigits=int(args.round_ndigits))
        out_path = output_dir / f"{split}_candidates.jsonl"
        write_jsonl(out_path, rows)
        overall["split_stats"][split] = {
            **info,
            "output_path": str(out_path),
        }
        print(f"[{split}] input_total={info['n_input_rows_total']} output={info['n_output_rows']} dup_removed={info['n_duplicates_removed']}")

    schema = {
        "mode": "merged_strong_stage3_candidate_union",
        "required_fields": [
            "sample_id", "material_id", "material_key",
            "parent_precursor_rank", "condition_rank",
            "disc_conditions", "disc_condition_indices",
            "cont_conditions", "stage3_score", "stage3_model", "source_split"
        ],
        "notes": [
            "Union of multiple strong stage3 candidate dirs.",
            "Dedup key = (material_key, parent_precursor_rank, rounded cont_conditions, disc_conditions).",
            "condition_rank is reassigned after merge.",
        ],
    }
    write_json(output_dir / "candidate_schema.json", schema)
    write_json(output_dir / "candidate_summary.json", overall)
    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
