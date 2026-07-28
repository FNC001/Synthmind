#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    mapping = {"tr":"train","train":"train","va":"val","val":"val","validation":"val","valid":"val","te":"test","test":"test"}
    if s not in mapping:
        raise ValueError(f"Unsupported split: {split}")
    return mapping[s]


def resolve_candidates_file(input_dir: Path, split: str) -> Path:
    p = input_dir / f"{split}_candidates.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return p


def precursor_key(precursor_set: Sequence[Any]) -> Tuple[str, ...]:
    return tuple(sorted(str(x) for x in (precursor_set or [])))


def score_value(row: Dict[str, Any]) -> float:
    v = row.get("stage2_score")
    if v is None or v == "":
        return float("-inf")
    try:
        return float(v)
    except Exception:
        return float("-inf")


def merge_split(input_dirs: List[Path], split: str) -> tuple[list[dict], dict]:
    split = normalize_split_name(split)
    all_rows: List[Dict[str, Any]] = []
    per_source_counts = {}

    for d in input_dirs:
        p = resolve_candidates_file(d, split)
        rows = read_jsonl(p)
        per_source_counts[str(d)] = len(rows)
        for r in rows:
            rr = dict(r)
            rr["_source_dir"] = str(d)
            all_rows.append(rr)

    by_material: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        mk = str(r.get("material_key", r.get("material_id", r.get("sample_id", ""))))
        by_material.setdefault(mk, []).append(r)

    merged_rows: List[Dict[str, Any]] = []
    n_duplicates_removed = 0

    for mk, rows in by_material.items():
        uniq: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
        for r in rows:
            key = (mk, precursor_key(r.get("precursor_set", [])))
            if key in uniq:
                keep = uniq[key]
                if score_value(r) > score_value(keep):
                    r["_merged_from"] = sorted(set((keep.get("_merged_from") or [keep.get("_source_dir")]) + [r.get("_source_dir")]))
                    uniq[key] = r
                else:
                    keep["_merged_from"] = sorted(set((keep.get("_merged_from") or [keep.get("_source_dir")]) + [r.get("_source_dir")]))
                n_duplicates_removed += 1
            else:
                rr = dict(r)
                rr["_merged_from"] = [r.get("_source_dir")]
                uniq[key] = rr

        uniq_rows = list(uniq.values())
        uniq_rows.sort(
            key=lambda x: (
                -score_value(x),
                -len(x.get("precursor_set", []) or []),
                int(x.get("precursor_rank", 10**9)) if str(x.get("precursor_rank", "")).isdigit() else 10**9,
                str(x.get("_source_dir", "")),
            )
        )

        for new_rank, r in enumerate(uniq_rows, start=1):
            out = dict(r)
            out["material_key"] = mk
            out["precursor_rank"] = int(new_rank)
            out["stage2_model"] = f"{str(out.get('stage2_model', 'stage2'))}|merged"
            out["merged_from"] = sorted(set(str(x) for x in out.get("_merged_from", []) if x))
            out.pop("_source_dir", None)
            out.pop("_merged_from", None)
            merged_rows.append(out)

    summary = {
        "split": split,
        "n_input_rows_total": len(all_rows),
        "n_output_rows": len(merged_rows),
        "n_duplicates_removed": n_duplicates_removed,
        "n_materials": len(by_material),
        "per_source_counts": per_source_counts,
    }
    return merged_rows, summary


def main():
    parser = argparse.ArgumentParser(description="Merge strong stage2 candidate dirs by union + dedup.")
    parser.add_argument("--input_dirs", type=str, required=True, help="Comma-separated stage2 candidate dirs")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", type=str, default="val,test")
    args = parser.parse_args()

    input_dirs = [Path(x.strip()).expanduser().resolve() for x in args.input_dirs.split(",") if x.strip()]
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    splits = [normalize_split_name(x.strip()) for x in args.splits.split(",") if x.strip()]

    overall = {
        "mode": "merged_strong_stage2_candidate_union",
        "input_dirs": [str(x) for x in input_dirs],
        "splits": splits,
        "split_stats": {},
    }

    for split in splits:
        rows, info = merge_split(input_dirs=input_dirs, split=split)
        out_path = output_dir / f"{split}_candidates.jsonl"
        write_jsonl(out_path, rows)
        overall["split_stats"][split] = {**info, "output_path": str(out_path)}
        print(f"[{split}] input_total={info['n_input_rows_total']} output={info['n_output_rows']} dup_removed={info['n_duplicates_removed']}")

    schema = {
        "mode": "merged_strong_stage2_candidate_union",
        "required_fields": [
            "sample_id","material_id","material_key","precursor_rank","precursor_set","n_precursors",
            "stage2_score","stage2_model"
        ],
        "notes": [
            "Union of multiple strong stage2 candidate dirs.",
            "Dedup key = (material_key, sorted precursor_set).",
            "precursor_rank is reassigned within each material after merge.",
        ],
    }
    write_json(output_dir / "candidate_schema.json", schema)
    write_json(output_dir / "candidate_summary.json", overall)
    print(f"Saved candidate_schema.json -> {output_dir / 'candidate_schema.json'}")
    print(f"Saved candidate_summary.json -> {output_dir / 'candidate_summary.json'}")


if __name__ == "__main__":
    main()
