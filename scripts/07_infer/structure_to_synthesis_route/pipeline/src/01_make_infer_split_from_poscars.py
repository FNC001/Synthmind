#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

POSCAR_NAMES = {"POSCAR", "CONTCAR"}
POSCAR_SUFFIXES = {".vasp", ".poscar", ".contcar"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_.")
    return s or "structure"


def find_poscars(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.name in POSCAR_NAMES or p.suffix.lower() in POSCAR_SUFFIXES:
            files.append(p)
    return files


def try_parse_poscar_formula_and_sites(path: Path) -> Tuple[Optional[str], Optional[int]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None, None

    if len(lines) < 8:
        return None, None

    formula = None
    n_sites = None

    try:
        tokens5 = lines[5].split()
        tokens6 = lines[6].split()
        if tokens5 and tokens6 and all(re.fullmatch(r"[A-Za-z][A-Za-z]?", t) for t in tokens5):
            if all(re.fullmatch(r"\d+", t) for t in tokens6):
                elems = tokens5
                counts = [int(x) for x in tokens6]
                n_sites = sum(counts)
                formula = "".join(f"{el}{cnt if cnt != 1 else ''}" for el, cnt in zip(elems, counts))
                return formula, n_sites
    except Exception:
        pass

    try:
        tokens5 = lines[5].split()
        if tokens5 and all(re.fullmatch(r"\d+", t) for t in tokens5):
            counts = [int(x) for x in tokens5]
            n_sites = sum(counts)
            return None, n_sites
    except Exception:
        pass

    return None, None


def build_sample_id(idx: int, src: Path, formula_guess: Optional[str]) -> str:
    if formula_guess:
        tail = sanitize_name(formula_guess)
    else:
        base = src.parent.name if src.name in POSCAR_NAMES else src.stem
        tail = sanitize_name(base)
    return f"infer_{idx:05d}__{tail}"


def stage_poscars(poscar_paths: Iterable[Path], staged_dir: Path) -> List[Dict[str, object]]:
    ensure_dir(staged_dir)
    rows: List[Dict[str, object]] = []

    for idx, src in enumerate(poscar_paths, start=1):
        formula_guess, n_sites_guess = try_parse_poscar_formula_and_sites(src)
        sample_id = build_sample_id(idx, src, formula_guess)

        sample_dir = staged_dir / sample_id
        ensure_dir(sample_dir)
        dst = sample_dir / "POSCAR"
        shutil.copyfile(src, dst)

        row: Dict[str, object] = {
            "sample_id": sample_id,
            "material_id": sample_id,
            "split": "infer",
            "poscar_path": str(dst),
            "source_path": str(src.resolve()),
            "formula_guess": formula_guess,
            "n_sites_guess": n_sites_guess,
        }
        rows.append(row)

    return rows


def write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "sample_id",
        "material_id",
        "split",
        "formula_guess",
        "n_sites_guess",
        "source_path",
        "poscar_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_summary_json(path: Path, rows: List[Dict[str, object]], poscar_dir: Path, staged_dir: Path) -> None:
    formulas = Counter(str(r["formula_guess"]) for r in rows if r.get("formula_guess"))
    n_sites = [int(r["n_sites_guess"]) for r in rows if r.get("n_sites_guess") is not None]

    summary = {
        "input_poscar_dir": str(poscar_dir),
        "staged_poscar_dir": str(staged_dir),
        "n_structures": len(rows),
        "n_with_formula_guess": sum(1 for r in rows if r.get("formula_guess")),
        "n_with_site_count_guess": sum(1 for r in rows if r.get("n_sites_guess") is not None),
        "top_formula_guesses": formulas.most_common(20),
        "site_count_stats": {
            "min": min(n_sites) if n_sites else None,
            "max": max(n_sites) if n_sites else None,
            "mean": (sum(n_sites) / len(n_sites)) if n_sites else None,
        },
        "artifacts": {
            "infer_jsonl": str(path.parent / "infer.jsonl"),
            "manifest_csv": str(path.parent / "manifest.csv"),
            "staged_poscars": str(staged_dir),
        },
    }

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Make inference split files from a directory of POSCARs.")
    parser.add_argument("--poscar_dir", type=str, required=True, help="Directory containing POSCAR/CONTCAR/.vasp files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for infer.jsonl and manifest.csv")
    args = parser.parse_args()

    poscar_dir = Path(args.poscar_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    staged_dir = output_dir / "staged_poscars"

    poscar_paths = find_poscars(poscar_dir)
    if not poscar_paths:
        raise FileNotFoundError(f"No POSCAR/CONTCAR/.vasp files found under: {poscar_dir}")

    rows = stage_poscars(poscar_paths, staged_dir=staged_dir)

    infer_jsonl = output_dir / "infer.jsonl"
    manifest_csv = output_dir / "manifest.csv"
    summary_json = output_dir / "summary.json"

    write_jsonl(infer_jsonl, rows)
    write_manifest_csv(manifest_csv, rows)
    write_summary_json(summary_json, rows, poscar_dir=poscar_dir, staged_dir=staged_dir)

    print(f"[DONE] Found {len(rows)} structures")
    print(f"[OUT] infer_jsonl  -> {infer_jsonl}")
    print(f"[OUT] manifest_csv -> {manifest_csv}")
    print(f"[OUT] summary_json -> {summary_json}")
    print(f"[OUT] staged_poscars -> {staged_dir}")


if __name__ == "__main__":
    main()
