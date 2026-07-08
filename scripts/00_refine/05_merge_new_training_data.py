#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CANONICAL_KEYS = [
    "id",
    "synth_uid",
    "source_dataset",
    "record_index",
    "parent_formula",
    "material_id",
    "mp_formula",
    "poscar_path",
    "summary_json_path",
    "doi_json_path",
    "provenance_json_path",
    "match_score",
    "match_reason",
    "match_level",
    "synth_formula",
    "title",
    "synthesis_text",
    "dois",
    "precursors",
    "steps",
    "temperatures_c",
    "times_h",
    "max_temperature_c",
    "total_time_h",
    "atmosphere",
    "all_atmospheres",
    "solvent",
    "all_solvents",
    "ph",
    "raw_synthesis_record",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=True)


def normalize_text(value: Any, max_len: int = 800) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    return text[:max_len]


def normalize_formula(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value)).strip()


def precursor_names(row: Dict[str, Any]) -> Tuple[str, ...]:
    out: List[str] = []
    precursors = row.get("precursors") or []
    if isinstance(precursors, list):
        for item in precursors:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = item
            if name is not None and str(name).strip():
                out.append(str(name).strip())
    return tuple(sorted(out))


def doi_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    dois = row.get("dois") or []
    if isinstance(dois, str):
        dois = [dois]
    if not isinstance(dois, list):
        dois = []
    return tuple(sorted(str(x).strip().lower() for x in dois if str(x).strip()))


def content_fingerprint(row: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        doi_key(row),
        normalize_formula(row.get("synth_formula") or row.get("parent_formula")),
        precursor_names(row),
        normalize_text(row.get("synthesis_text")),
    )


def canonicalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {key: row.get(key) for key in CANONICAL_KEYS}
    extras = {k: v for k, v in row.items() if k not in out}
    if extras:
        out["_extra_fields"] = extras
    return out


def rows_by_id(rows: Sequence[Dict[str, Any]]) -> set[str]:
    return {str(row.get("id")) for row in rows if row.get("id") is not None}


def make_new_id(row: Dict[str, Any], suffix: str) -> str:
    base = row.get("id") or row.get("synth_uid") or "newdata_row"
    return f"{base}__dedup{suffix}"


def append_unique(
    base_rows: List[Dict[str, Any]],
    new_rows: Sequence[Dict[str, Any]],
    known_ids: set[str],
    known_fingerprints: set[Tuple[Any, ...]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    appended: List[Dict[str, Any]] = []
    skipped_duplicate_id = 0
    skipped_duplicate_content = 0
    renamed_duplicate_ids = 0

    for row in new_rows:
        row = canonicalize_row(row)
        row_id = str(row.get("id")) if row.get("id") is not None else ""
        fp = content_fingerprint(row)

        if row_id and row_id in known_ids:
            if fp in known_fingerprints:
                skipped_duplicate_id += 1
                continue
            renamed_duplicate_ids += 1
            row["id"] = make_new_id(row, str(renamed_duplicate_ids))
            row_id = str(row["id"])

        if fp in known_fingerprints:
            skipped_duplicate_content += 1
            continue

        appended.append(row)
        if row_id:
            known_ids.add(row_id)
        known_fingerprints.add(fp)

    merged = base_rows + appended
    stats = {
        "base_rows": len(base_rows),
        "candidate_new_rows": len(new_rows),
        "appended_new_rows": len(appended),
        "skipped_duplicate_id": skipped_duplicate_id,
        "skipped_duplicate_content": skipped_duplicate_content,
        "renamed_duplicate_ids": renamed_duplicate_ids,
        "merged_rows": len(merged),
    }
    return merged, stats


def count_levels(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("match_level")) for row in rows).most_common())


def count_sources(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(row.get("source_dataset")) for row in rows).most_common())


def has_structure_match(row: Dict[str, Any]) -> bool:
    return bool(row.get("material_id"))


def has_poscar(row: Dict[str, Any]) -> bool:
    return bool(row.get("poscar_path"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge new aligned SynPred training data with existing raw JSONL files."
    )
    ap.add_argument("--old_exact", default="data/raw/strict_exact_only.jsonl")
    ap.add_argument("--old_parent", default="data/raw/strict_parent_aug.jsonl")
    ap.add_argument(
        "--new_full",
        default="newdata/direct_aligned_json_20260608/codex_final_database_aligned_full.jsonl",
    )
    ap.add_argument("--out_dir", default="data/raw/merged_20260609")
    args = ap.parse_args()

    old_exact_path = Path(args.old_exact)
    old_parent_path = Path(args.old_parent)
    new_full_path = Path(args.new_full)
    out_dir = Path(args.out_dir)

    old_exact = [canonicalize_row(r) for r in read_jsonl(old_exact_path)]
    old_parent = [canonicalize_row(r) for r in read_jsonl(old_parent_path)]
    new_full = [canonicalize_row(r) for r in read_jsonl(new_full_path)]

    new_exact = [r for r in new_full if r.get("match_level") == "exact" and has_structure_match(r)]
    new_parent_like = [
        r
        for r in new_full
        if r.get("match_level") in {"composition", "reduced_composition"} and has_structure_match(r)
    ]
    new_matched = [r for r in new_full if has_structure_match(r)]
    new_unmatched = [r for r in new_full if not has_structure_match(r)]

    global_ids = rows_by_id(old_exact) | rows_by_id(old_parent)
    global_fps = {content_fingerprint(r) for r in old_exact + old_parent}

    merged_exact, exact_stats = append_unique(
        old_exact,
        new_exact,
        known_ids=set(global_ids),
        known_fingerprints=set(global_fps),
    )

    ids_after_exact = rows_by_id(merged_exact) | rows_by_id(old_parent)
    fps_after_exact = {content_fingerprint(r) for r in merged_exact + old_parent}
    merged_parent, parent_stats = append_unique(
        old_parent,
        new_parent_like,
        known_ids=ids_after_exact,
        known_fingerprints=fps_after_exact,
    )

    matched_train_ready = merged_exact + merged_parent
    all_aligned_archive = old_exact + old_parent + new_full

    output_paths = {
        "strict_exact_only_merged": out_dir / "strict_exact_only.jsonl",
        "strict_parent_aug_merged": out_dir / "strict_parent_aug.jsonl",
        "strict_matched_train_ready": out_dir / "strict_matched_train_ready.jsonl",
        "new_matched_only": out_dir / "new_matched_only.jsonl",
        "new_unmatched_archive": out_dir / "new_unmatched_archive.jsonl",
        "all_aligned_archive": out_dir / "all_aligned_archive.jsonl",
        "summary": out_dir / "merge_summary.json",
        "report": out_dir / "MERGE_REPORT.md",
    }

    written = {
        name: write_jsonl(path, rows)
        for name, path, rows in [
            ("strict_exact_only_merged", output_paths["strict_exact_only_merged"], merged_exact),
            ("strict_parent_aug_merged", output_paths["strict_parent_aug_merged"], merged_parent),
            ("strict_matched_train_ready", output_paths["strict_matched_train_ready"], matched_train_ready),
            ("new_matched_only", output_paths["new_matched_only"], new_matched),
            ("new_unmatched_archive", output_paths["new_unmatched_archive"], new_unmatched),
            ("all_aligned_archive", output_paths["all_aligned_archive"], all_aligned_archive),
        ]
    }

    summary = {
        "inputs": {
            "old_exact": str(old_exact_path),
            "old_parent": str(old_parent_path),
            "new_full": str(new_full_path),
            "old_exact_rows": len(old_exact),
            "old_parent_rows": len(old_parent),
            "new_full_rows": len(new_full),
        },
        "new_data": {
            "match_level_counts": count_levels(new_full),
            "source_dataset_counts": count_sources(new_full),
            "matched_rows_with_material_id": len(new_matched),
            "unmatched_rows_without_material_id": len(new_unmatched),
            "matched_rows_with_poscar_path": sum(1 for r in new_matched if has_poscar(r)),
            "matched_rows_without_poscar_path": sum(1 for r in new_matched if not has_poscar(r)),
            "new_exact_candidates": len(new_exact),
            "new_parent_like_candidates": len(new_parent_like),
        },
        "merge": {
            "exact": exact_stats,
            "parent_like": parent_stats,
            "strict_matched_train_ready_rows": len(matched_train_ready),
            "all_aligned_archive_rows": len(all_aligned_archive),
        },
        "outputs": {k: str(v) for k, v in output_paths.items()},
        "written_rows": written,
        "notes": [
            "Original raw JSONL files are not overwritten.",
            "New exact rows are appended to strict_exact_only.jsonl.",
            "New composition/reduced_composition rows are appended to strict_parent_aug.jsonl as parent-like matched records.",
            "New unmatched rows are archived separately and excluded from structure-matched training inputs.",
            "New matched rows currently have material_id but no poscar_path, so downstream graph/POSCAR feature stages may need MP archive path recovery before using them for graph-based training.",
        ],
    }
    write_json(output_paths["summary"], summary)

    report = [
        "# New Training Data Merge Report",
        "",
        "## Inputs",
        f"- old exact: `{old_exact_path}` ({len(old_exact)} rows)",
        f"- old parent: `{old_parent_path}` ({len(old_parent)} rows)",
        f"- new aligned: `{new_full_path}` ({len(new_full)} rows)",
        "",
        "## New Data Split",
        f"- exact matched: {len(new_exact)}",
        f"- composition/reduced-composition matched: {len(new_parent_like)}",
        f"- unmatched archive-only: {len(new_unmatched)}",
        f"- matched rows with `poscar_path`: {summary['new_data']['matched_rows_with_poscar_path']}",
        "",
        "## Merged Outputs",
        f"- strict exact: `{output_paths['strict_exact_only_merged']}` ({written['strict_exact_only_merged']} rows)",
        f"- strict parent-like: `{output_paths['strict_parent_aug_merged']}` ({written['strict_parent_aug_merged']} rows)",
        f"- matched train-ready bundle: `{output_paths['strict_matched_train_ready']}` ({written['strict_matched_train_ready']} rows)",
        f"- full archive: `{output_paths['all_aligned_archive']}` ({written['all_aligned_archive']} rows)",
        f"- summary JSON: `{output_paths['summary']}`",
        "",
        "## Caveat",
        "The new matched rows have `material_id` but no `poscar_path`. They are schema-aligned and suitable for text/condition/precursor refinement, but graph/POSCAR stages need structure path recovery before they can contribute real graph embeddings.",
        "",
    ]
    output_paths["report"].write_text("\n".join(report), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
