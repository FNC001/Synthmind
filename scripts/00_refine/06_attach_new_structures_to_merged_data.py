#!/usr/bin/env python3
"""
Attach verified POSCAR paths for the newly merged training records.

This script keeps the original raw JSONL files untouched. It copies the
validated new POSCAR files into the existing raw structure archive without
overwriting files, then writes a new merged raw-data directory whose new matched
records have poscar_path populated.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pymatgen.core import Composition, Structure


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = DEFAULT_PROJECT_ROOT / "data/raw/merged_20260609"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "data/raw/merged_20260609_with_structures"
DEFAULT_CIF_DIR = Path(
    "/Users/lihonglin/Desktop/Syn_DP/paper_1_29579_mp_structures_standard_merged/conventional_cif"
)
DEFAULT_POSCAR_DIR = Path(
    "/Users/lihonglin/Desktop/Syn_DP/paper_1_29579_mp_structures_standard_merged/conventional_poscar"
)
DEFAULT_TARGET_POSCAR_DIR = DEFAULT_PROJECT_ROOT / "data/raw/mp_full_archive_export/poscar"


JSONL_FILES = [
    "strict_exact_only.jsonl",
    "strict_parent_aug.jsonl",
    "strict_matched_train_ready.jsonl",
    "new_matched_only.jsonl",
    "new_unmatched_archive.jsonl",
    "all_aligned_archive.jsonl",
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
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def material_id_from_cif(path: Path) -> Optional[str]:
    match = re.match(r"(.+)_conventional\.cif$", path.name)
    return match.group(1) if match else None


def material_id_from_poscar(path: Path) -> Optional[str]:
    match = re.match(r"POSCAR_(.+)_conventional$", path.name)
    return match.group(1) if match else None


def index_files(directory: Path, parser) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)
    for path in directory.iterdir():
        if not path.is_file():
            continue
        mid = parser(path)
        if not mid:
            continue
        if mid in out:
            duplicates[mid].append(str(path))
        else:
            out[mid] = path
    if duplicates:
        sample = {k: v for k, v in list(duplicates.items())[:5]}
        raise RuntimeError(f"Duplicate material_id files in {directory}: {sample}")
    return out


def reduced_formula(formula: Optional[str]) -> Optional[Composition]:
    if not formula:
        return None
    return Composition(str(formula)).reduced_composition


def validate_structures(
    rows: List[Dict[str, Any]],
    poscar_by_mid: Dict[str, Path],
    cif_by_mid: Dict[str, Path],
) -> Dict[str, Any]:
    mids = sorted({str(r.get("material_id")) for r in rows if r.get("material_id")})
    struct_comp_by_mid: Dict[str, Composition] = {}
    counts: Counter[str] = Counter()
    samples: Dict[str, List[Dict[str, Any]]] = {
        "missing": [],
        "parse_errors": [],
        "cif_poscar_mismatches": [],
        "formula_mismatches": [],
    }

    for mid in mids:
        poscar = poscar_by_mid.get(mid)
        cif = cif_by_mid.get(mid)
        if not poscar or not cif:
            counts["missing_structure_file"] += 1
            if len(samples["missing"]) < 20:
                samples["missing"].append({"material_id": mid, "has_poscar": bool(poscar), "has_cif": bool(cif)})
            continue

        try:
            poscar_struct = Structure.from_file(str(poscar))
            poscar_comp = poscar_struct.composition.reduced_composition
            struct_comp_by_mid[mid] = poscar_comp
            counts["poscar_parsed"] += 1
        except Exception as exc:  # pragma: no cover - data dependent
            counts["poscar_parse_error"] += 1
            if len(samples["parse_errors"]) < 20:
                samples["parse_errors"].append({"material_id": mid, "poscar": str(poscar), "error": repr(exc)})
            continue

        try:
            cif_struct = Structure.from_file(str(cif))
            cif_comp = cif_struct.composition.reduced_composition
            if cif_comp == poscar_comp:
                counts["cif_poscar_comp_match"] += 1
            else:
                counts["cif_poscar_comp_mismatch"] += 1
                if len(samples["cif_poscar_mismatches"]) < 20:
                    samples["cif_poscar_mismatches"].append(
                        {
                            "material_id": mid,
                            "poscar_formula": poscar_comp.reduced_formula,
                            "cif_formula": cif_comp.reduced_formula,
                        }
                    )
        except Exception as exc:  # pragma: no cover - data dependent
            counts["cif_parse_error"] += 1
            if len(samples["parse_errors"]) < 20:
                samples["parse_errors"].append({"material_id": mid, "cif": str(cif), "error": repr(exc)})

    for row in rows:
        mid = row.get("material_id")
        comp = struct_comp_by_mid.get(str(mid))
        if comp is None:
            counts["row_missing_valid_poscar"] += 1
            continue
        for key in ("mp_formula", "parent_formula"):
            try:
                expected = reduced_formula(row.get(key))
            except Exception as exc:
                counts[f"{key}_parse_error"] += 1
                if len(samples["formula_mismatches"]) < 20:
                    samples["formula_mismatches"].append(
                        {"id": row.get("id"), "material_id": mid, "key": key, "error": repr(exc)}
                    )
                continue
            if expected is None:
                counts[f"{key}_missing"] += 1
            elif expected == comp:
                counts[f"{key}_matches_poscar"] += 1
            else:
                counts[f"{key}_mismatches_poscar"] += 1
                if len(samples["formula_mismatches"]) < 20:
                    samples["formula_mismatches"].append(
                        {
                            "id": row.get("id"),
                            "material_id": mid,
                            "key": key,
                            "record_formula": row.get(key),
                            "poscar_formula": comp.reduced_formula,
                        }
                    )

    return {"counts": dict(counts), "samples": samples}


def load_existing_poscar_names(input_dir: Path) -> set[str]:
    names: set[str] = set()
    for filename in ("strict_exact_only.jsonl", "strict_parent_aug.jsonl", "strict_matched_train_ready.jsonl"):
        path = input_dir / filename
        if not path.exists():
            continue
        for row in read_jsonl(path):
            poscar_path = row.get("poscar_path")
            if poscar_path:
                names.add(Path(str(poscar_path)).name)
    return names


def resolve_existing_path(path_value: Any, project_root: Path) -> Optional[Path]:
    if not path_value:
        return None
    raw = str(path_value)
    path = Path(raw)
    candidates = [
        path,
        project_root / raw,
        project_root / "data" / raw,
        project_root / "data/raw" / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach verified new POSCAR paths to merged raw data.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cif_dir", type=Path, default=DEFAULT_CIF_DIR)
    parser.add_argument("--poscar_dir", type=Path, default=DEFAULT_POSCAR_DIR)
    parser.add_argument("--target_poscar_dir", type=Path, default=DEFAULT_TARGET_POSCAR_DIR)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    new_rows = read_jsonl(args.input_dir / "new_matched_only.jsonl")
    poscar_by_mid = index_files(args.poscar_dir, material_id_from_poscar)
    cif_by_mid = index_files(args.cif_dir, material_id_from_cif)

    validation = validate_structures(new_rows, poscar_by_mid, cif_by_mid)
    formula_mismatches = validation["counts"].get("mp_formula_mismatches_poscar", 0) + validation["counts"].get(
        "parent_formula_mismatches_poscar", 0
    )
    if formula_mismatches:
        raise RuntimeError(f"Formula/POSCAR validation failed: {formula_mismatches} mismatches")

    unique_mids = sorted({str(r.get("material_id")) for r in new_rows if r.get("material_id")})
    missing_poscars = [mid for mid in unique_mids if mid not in poscar_by_mid]
    if missing_poscars:
        raise RuntimeError(f"Missing POSCAR files for {len(missing_poscars)} material_ids; sample={missing_poscars[:10]}")

    existing_names = load_existing_poscar_names(args.input_dir)
    source_name_conflicts = [poscar_by_mid[mid].name for mid in unique_mids if poscar_by_mid[mid].name in existing_names]
    target_name_conflicts = []
    for mid in unique_mids:
        src = poscar_by_mid[mid]
        dst = args.target_poscar_dir / src.name
        if dst.exists() and dst.resolve() != src.resolve() and not filecmp.cmp(src, dst, shallow=False):
            target_name_conflicts.append(src.name)
    if source_name_conflicts or target_name_conflicts:
        raise RuntimeError(
            "POSCAR filename conflict detected: "
            f"existing_path_conflicts={source_name_conflicts[:10]}, target_file_conflicts={target_name_conflicts[:10]}"
        )

    poscar_path_by_mid: Dict[str, str] = {}
    copied = 0
    if not args.dry_run:
        args.target_poscar_dir.mkdir(parents=True, exist_ok=True)
    for mid in unique_mids:
        src = poscar_by_mid[mid]
        dst = args.target_poscar_dir / src.name
        poscar_path_by_mid[mid] = str(dst)
        if not args.dry_run and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    written_rows: Dict[str, int] = {}
    updated_rows: Dict[str, int] = {}
    normalized_rows: Dict[str, int] = {}
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in JSONL_FILES:
        in_path = args.input_dir / filename
        if not in_path.exists():
            continue
        rows = read_jsonl(in_path)
        n_updated = 0
        n_normalized = 0
        for row in rows:
            resolved = resolve_existing_path(row.get("poscar_path"), DEFAULT_PROJECT_ROOT)
            if resolved is not None and row.get("poscar_path") != str(resolved):
                row["poscar_path"] = str(resolved)
                n_normalized += 1

            mid = row.get("material_id")
            if mid and not row.get("poscar_path") and str(mid) in poscar_path_by_mid:
                row["poscar_path"] = poscar_path_by_mid[str(mid)]
                reason = str(row.get("match_reason") or "")
                if "has_poscar" not in reason:
                    row["match_reason"] = (reason + "|has_poscar").strip("|")
                n_updated += 1
        updated_rows[filename] = n_updated
        normalized_rows[filename] = n_normalized
        if not args.dry_run:
            written_rows[filename] = write_jsonl(args.output_dir / filename, rows)

    for extra in ("merge_summary.json", "MERGE_REPORT.md"):
        src = args.input_dir / extra
        if src.exists() and not args.dry_run:
            shutil.copy2(src, args.output_dir / extra)

    summary = {
        "inputs": {
            "input_dir": str(args.input_dir),
            "cif_dir": str(args.cif_dir),
            "poscar_dir": str(args.poscar_dir),
            "target_poscar_dir": str(args.target_poscar_dir),
            "new_matched_rows": len(new_rows),
            "unique_new_material_ids": len(unique_mids),
        },
        "file_indexes": {
            "cif_files": len(cif_by_mid),
            "poscar_files": len(poscar_by_mid),
        },
        "validation": validation,
        "conflicts": {
            "source_basename_conflicts_with_existing_poscar_paths": len(source_name_conflicts),
            "target_file_conflicts": len(target_name_conflicts),
        },
        "copy": {
            "copied_poscar_files": copied,
            "referenced_poscar_files": len(poscar_path_by_mid),
            "dry_run": args.dry_run,
        },
        "updated_rows": updated_rows,
        "normalized_existing_poscar_rows": normalized_rows,
        "written_rows": written_rows,
        "output_dir": str(args.output_dir),
    }
    if not args.dry_run:
        write_json(args.output_dir / "structure_attach_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
