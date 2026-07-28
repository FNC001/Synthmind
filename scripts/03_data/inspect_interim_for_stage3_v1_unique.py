#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_interim_for_stage3_v1_unique.py

Scan /Users/wyc/MP_exp_doi/data/interim (or any given root) and produce:
1) directory inventory
2) dataset-level summaries for npz/json/csv/tsv
3) stage3-focused candidate analysis for mixed-type condition generation

Outputs:
- interim_inventory.json
- interim_inventory.md

Usage:
python inspect_interim_for_stage3_v1_unique.py \
  --root /Users/wyc/MP_exp_doi/data/interim \
  --output_dir /Users/wyc/MP_exp_doi/data/interim/_inspection_stage3
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


TEXT_EXTS = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".md"}
ARRAY_EXTS = {".npz", ".npy"}


def safe_json_load(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def summarize_json(path: Path) -> Dict[str, Any]:
    obj = safe_json_load(path)
    out: Dict[str, Any] = {
        "type": "json",
        "path": str(path),
        "exists": path.exists(),
    }
    if obj is None:
        out["read_error"] = True
        return out

    out["top_level_type"] = type(obj).__name__
    if isinstance(obj, dict):
        out["top_level_keys"] = list(obj.keys())[:50]
        if "continuous_schema" in obj and isinstance(obj["continuous_schema"], dict):
            out["continuous_cols"] = list(obj["continuous_schema"].keys())
        if "discrete_schema" in obj and isinstance(obj["discrete_schema"], dict):
            out["discrete_cols"] = list(obj["discrete_schema"].keys())
        if "continuous_cols" in obj and isinstance(obj["continuous_cols"], list):
            out["continuous_cols_declared"] = obj["continuous_cols"]
        if "discrete_cols" in obj and isinstance(obj["discrete_cols"], list):
            out["discrete_cols_declared"] = obj["discrete_cols"]
        if "used_precursor_cols" in obj:
            out["used_precursor_cols"] = obj["used_precursor_cols"]
    elif isinstance(obj, list):
        out["n_items"] = len(obj)
        if obj:
            out["first_item_type"] = type(obj[0]).__name__
            if isinstance(obj[0], dict):
                out["first_item_keys"] = list(obj[0].keys())[:50]
    return out


def summarize_npz(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "type": "npz",
        "path": str(path),
        "exists": path.exists(),
    }
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as e:
        out["read_error"] = str(e)
        return out

    keys = list(arr.files)
    out["keys"] = keys
    shapes = {}
    dtypes = {}
    preview = {}
    for k in keys:
        try:
            a = arr[k]
            shapes[k] = list(a.shape)
            dtypes[k] = str(a.dtype)
            if a.ndim == 1 and a.size > 0:
                preview[k] = {
                    "n_unique_head": int(len(np.unique(a[: min(len(a), 200)]))),
                    "sample_head": [str(x) for x in a[: min(len(a), 5)]],
                }
        except Exception as e:
            preview[k] = {"error": str(e)}
    out["shapes"] = shapes
    out["dtypes"] = dtypes
    out["preview"] = preview
    return out


def try_read_csv_head(path: Path, delimiter: str = ",", max_rows: int = 2000) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": str(path), "type": "csv_like"}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i + 1 >= max_rows:
                    break
    except Exception as e:
        out["read_error"] = str(e)
        return out

    if not rows:
        out["n_rows_scanned"] = 0
        out["columns"] = []
        return out

    cols = list(rows[0].keys())
    out["n_rows_scanned"] = len(rows)
    out["columns"] = cols

    col_stats = {}
    for c in cols:
        vals = [r.get(c) for r in rows]
        nonempty = [v for v in vals if v not in (None, "", "nan", "NaN", "None")]
        uniq = len(set(nonempty[: min(len(nonempty), 500)]))
        numeric_count = 0
        for v in nonempty[: min(len(nonempty), 500)]:
            try:
                float(v)
                numeric_count += 1
            except Exception:
                pass
        col_stats[c] = {
            "nonempty_ratio": round(len(nonempty) / max(len(vals), 1), 4),
            "unique_head": uniq,
            "numeric_ratio_head": round(numeric_count / max(min(len(nonempty), 500), 1), 4),
            "sample": nonempty[:3],
        }
    out["column_stats"] = col_stats
    return out


def summarize_tabular(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".tsv":
        return try_read_csv_head(path, delimiter="\t")
    return try_read_csv_head(path, delimiter=",")


def walk_inventory(root: Path) -> Dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file()]
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    ext_counter = Counter(p.suffix.lower() for p in files)

    interesting = []
    for p in files:
        suffix = p.suffix.lower()
        if suffix in ARRAY_EXTS or suffix in TEXT_EXTS or p.name in {"schema.json", "condition_schema.json"}:
            interesting.append(p)

    summaries = []
    for p in sorted(interesting):
        suffix = p.suffix.lower()
        name = p.name.lower()
        if suffix == ".npz":
            summaries.append(summarize_npz(p))
        elif suffix == ".json" or name in {"schema.json", "condition_schema.json"}:
            summaries.append(summarize_json(p))
        elif suffix in {".csv", ".tsv"}:
            summaries.append(summarize_tabular(p))
        else:
            summaries.append({
                "type": "file",
                "path": str(p),
                "suffix": suffix,
                "size_bytes": p.stat().st_size,
            })

    return {
        "root": str(root),
        "n_dirs": len(dirs),
        "n_files": len(files),
        "file_type_counts": dict(ext_counter),
        "summaries": summaries,
    }


def detect_stage3_candidates(inv: Dict[str, Any]) -> Dict[str, Any]:
    summaries = inv["summaries"]
    stage3_hits = []
    mixed_type_candidates = defaultdict(list)

    for s in summaries:
        p = s.get("path", "")
        pl = p.lower()

        if "stage3" in pl or "condition" in pl:
            stage3_hits.append(s)

        if s.get("type") == "json":
            cont = s.get("continuous_cols") or s.get("continuous_cols_declared") or []
            disc = s.get("discrete_cols") or s.get("discrete_cols_declared") or []
            if cont or disc:
                mixed_type_candidates["schema_files"].append({
                    "path": p,
                    "continuous_cols": cont,
                    "discrete_cols": disc,
                })

        if s.get("type") == "npz":
            keys = set(s.get("keys", []))
            if {"x", "y_set", "y_cond_continuous", "y_cond_continuous_mask"} <= keys:
                mixed_type_candidates["continuous_npz_datasets"].append({
                    "path": p,
                    "keys": s.get("keys", []),
                    "shapes": s.get("shapes", {}),
                })
            discrete_like = [
                k for k in keys
                if "discrete" in k or "categor" in k or "class" in k or "label" in k
            ]
            if discrete_like:
                mixed_type_candidates["possible_discrete_npz_keys"].append({
                    "path": p,
                    "keys": discrete_like,
                    "all_keys": s.get("keys", []),
                })

        if s.get("type") == "csv_like":
            cols = s.get("columns", [])
            lower_cols = [c.lower() for c in cols]

            for c in cols:
                cl = c.lower()
                if any(tok in cl for tok in ["atmos", "synth", "protocol", "heating", "cooling", "gas", "furnace"]):
                    mixed_type_candidates["possible_discrete_csv_columns"].append({
                        "path": p,
                        "column": c,
                    })
                if any(tok in cl for tok in ["temp", "time", "hour", "minute"]):
                    mixed_type_candidates["possible_continuous_csv_columns"].append({
                        "path": p,
                        "column": c,
                    })

            if any(tok in " ".join(lower_cols) for tok in ["temperature", "time", "atmosphere", "synthesis"]):
                mixed_type_candidates["mixed_like_csv_files"].append({
                    "path": p,
                    "columns": cols,
                })

    return {
        "stage3_related_files": stage3_hits,
        "mixed_type_candidates": dict(mixed_type_candidates),
    }


def build_recommendation(inv: Dict[str, Any], stage3: Dict[str, Any]) -> Dict[str, Any]:
    rec = {
        "current_status": [],
        "recommended_next_steps": [],
        "suggested_mixed_type_schema_template": {
            "continuous_cols": ["temperature_c", "time_h"],
            "discrete_cols": ["atmosphere", "synthesis_type", "heating_protocol"],
            "optional_discrete_cols": ["cooling_protocol", "container_type"],
        },
    }

    has_stage3_npz = len(stage3["mixed_type_candidates"].get("continuous_npz_datasets", [])) > 0
    has_discrete_npz = len(stage3["mixed_type_candidates"].get("possible_discrete_npz_keys", [])) > 0
    has_schema = len(stage3["mixed_type_candidates"].get("schema_files", [])) > 0

    if has_stage3_npz:
        rec["current_status"].append("Found continuous-only stage3-style NPZ datasets with x / y_set / y_cond_continuous / y_cond_continuous_mask.")
    else:
        rec["current_status"].append("Did not confidently find stage3 continuous NPZ datasets; inspect stage3-related files manually.")

    if has_discrete_npz:
        rec["current_status"].append("Found possible discrete keys in NPZ files; mixed-type extension may already be partially prepared.")
    else:
        rec["current_status"].append("Did not find obvious discrete condition arrays in NPZ files.")

    if has_schema:
        rec["current_status"].append("Found schema-like JSON files describing continuous/discrete columns.")
    else:
        rec["current_status"].append("Did not find a complete mixed-type schema declaration yet.")

    rec["recommended_next_steps"].extend([
        "Pick a single canonical stage3 source table (csv/jsonl/parquet) that contains both continuous and discrete process fields.",
        "Define mixed-type output schema explicitly: continuous_cols + discrete_cols + masks/encoders.",
        "Build a new stage3 mixed dataset export with train/val/test npz containing both continuous and discrete targets.",
        "Start with a TabDDPM-style baseline after the mixed dataset export is stable.",
        "Then implement TabNAT-style mixed conditional generation, followed by TabSyn as latent-diffusion comparison.",
    ])
    return rec


def render_markdown(inv: Dict[str, Any], stage3: Dict[str, Any], rec: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Interim inventory report")
    lines.append("")
    lines.append(f"- Root: `{inv['root']}`")
    lines.append(f"- Dirs: {inv['n_dirs']}")
    lines.append(f"- Files: {inv['n_files']}")
    lines.append("")
    lines.append("## File type counts")
    for k, v in sorted(inv["file_type_counts"].items()):
        lines.append(f"- `{k or '[no_ext]'}`: {v}")
    lines.append("")
    lines.append("## Stage3 / mixed-type candidates")
    lines.append(f"- Stage3-related files detected: {len(stage3['stage3_related_files'])}")
    for item in stage3["stage3_related_files"][:30]:
        lines.append(f"  - `{item.get('path')}`")
    lines.append("")
    lines.append("### Mixed-type candidate summary")
    for key, items in stage3["mixed_type_candidates"].items():
        lines.append(f"- **{key}**: {len(items)}")
        for it in items[:10]:
            lines.append(f"  - `{it}`")
    lines.append("")
    lines.append("## Recommendation")
    for x in rec["current_status"]:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("### Next steps")
    for i, x in enumerate(rec["recommended_next_steps"], start=1):
        lines.append(f"{i}. {x}")
    lines.append("")
    lines.append("### Suggested mixed-type schema template")
    lines.append("```json")
    lines.append(json.dumps(rec["suggested_mixed_type_schema_template"], ensure_ascii=False, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/Users/wyc/MP_exp_doi/data/interim")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    output_dir = Path(args.output_dir) if args.output_dir else (root / "_inspection_stage3")
    output_dir.mkdir(parents=True, exist_ok=True)

    inv = walk_inventory(root)
    stage3 = detect_stage3_candidates(inv)
    rec = build_recommendation(inv, stage3)

    full = {
        "inventory": inv,
        "stage3_analysis": stage3,
        "recommendation": rec,
    }

    json_path = output_dir / "interim_inventory.json"
    md_path = output_dir / "interim_inventory.md"

    write_json(json_path, full)
    md_path.write_text(render_markdown(inv, stage3, rec), encoding="utf-8")

    print(json.dumps({
        "root": str(root),
        "output_dir": str(output_dir),
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
