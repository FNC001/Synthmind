#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def safe_read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:
            return None


def normalize_elements(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return ";".join(sorted(str(v) for v in x if str(v).strip()))
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return ""
        if "-" in s and ";" not in s and "," not in s:
            return ";".join(sorted([v for v in s.split("-") if v]))
        if "," in s:
            return ";".join(sorted([v.strip() for v in s.split(",") if v.strip()]))
        if ";" in s:
            return ";".join(sorted([v.strip() for v in s.split(";") if v.strip()]))
        return s
    return str(x)


def infer_family_from_elements(elements):
    elems = set([e for e in str(elements).split(";") if e])
    if not elems:
        return "unknown"
    if "O" in elems and "P" in elems:
        return "phosphate_or_oxide"
    if "O" in elems and "S" in elems:
        return "sulfate_or_oxide"
    if "O" in elems and "C" in elems:
        return "carbonate_or_oxide"
    if "O" in elems:
        return "oxide"
    if any(x in elems for x in ["F", "Cl", "Br", "I"]):
        return "halide"
    if "N" in elems:
        return "nitride"
    if "S" in elems:
        return "sulfide"
    if "Se" in elems:
        return "selenide"
    if "P" in elems:
        return "phosphide_or_phosphate_like"
    return "non_oxide"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_json", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_dir = root / "data/raw/mp_full_archive_export/provenance_json"

    rows = []

    # 1. provenance_json: strongest source for mp metadata
    json_files = sorted(json_dir.glob("mp-*.json")) if json_dir.exists() else []
    if args.max_json and args.max_json > 0:
        json_files = json_files[: args.max_json]

    for p in json_files:
        obj = safe_read_json(p)
        if not isinstance(obj, dict):
            continue

        mp_id = obj.get("material_id") or p.stem
        formula = (
            obj.get("formula_pretty")
            or obj.get("pretty_formula")
            or obj.get("formula")
            or obj.get("formula_anonymous")
            or ""
        )

        elements = normalize_elements(obj.get("elements") or obj.get("chemsys"))

        composition = obj.get("composition") or {}
        if not elements and isinstance(composition, dict):
            elements = ";".join(sorted(composition.keys()))

        chemsys = obj.get("chemsys") or elements.replace(";", "-")

        sym = obj.get("symmetry") or {}
        crystal_system = ""
        spacegroup_symbol = ""
        spacegroup_number = ""

        if isinstance(sym, dict):
            crystal_system = sym.get("crystal_system", "")
            spacegroup_symbol = sym.get("symbol", "")
            spacegroup_number = sym.get("number", "")

        rows.append({
            "mp_id": str(mp_id),
            "formula": str(formula),
            "elements": elements,
            "chemsys": str(chemsys),
            "n_elements": len([e for e in elements.split(";") if e]),
            "mp_family": infer_family_from_elements(elements),
            "crystal_system": crystal_system,
            "spacegroup_symbol": spacegroup_symbol,
            "spacegroup_number": spacegroup_number,
            "metadata_source": str(p),
            "metadata_source_type": "mp_provenance_json",
        })

    # 2. benchmark manifests: useful direct formula/elements table
    for mf in [
        root / "data/infer/benchmark_10_manifest.csv",
        root / "data/infer/benchmark_30_manifest.csv",
    ]:
        if not mf.exists():
            continue
        try:
            df = pd.read_csv(mf)
        except Exception:
            continue

        for _, r in df.iterrows():
            mp_id = str(r.get("mp_id", "")).strip()
            if not mp_id:
                continue
            formula = str(r.get("formula", "")).strip()
            elements = normalize_elements(r.get("elements", ""))
            rows.append({
                "mp_id": mp_id,
                "formula": formula,
                "elements": elements,
                "chemsys": elements.replace(";", "-"),
                "n_elements": len([e for e in elements.split(";") if e]),
                "mp_family": infer_family_from_elements(elements),
                "crystal_system": "",
                "spacegroup_symbol": "",
                "spacegroup_number": "",
                "metadata_source": str(mf),
                "metadata_source_type": "benchmark_manifest",
            })

    meta = pd.DataFrame(rows)

    if meta.empty:
        summary = {
            "status": "blocked",
            "reason": "no mp metadata rows were extracted",
            "output_dir": str(out),
        }
        (out / "v32_mp_metadata_table_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise SystemExit("[BLOCKED] no mp metadata rows extracted")

    # Prefer manifest rows when duplicated, otherwise provenance json.
    priority = {
        "benchmark_manifest": 0,
        "mp_provenance_json": 1,
    }
    meta["source_priority"] = meta["metadata_source_type"].map(priority).fillna(9)
    meta = meta.sort_values(["mp_id", "source_priority"]).drop_duplicates("mp_id", keep="first")
    meta = meta.sort_values("mp_id").reset_index(drop=True)

    out_csv = out / "v32_mp_metadata_table.csv"
    out_preview = out / "v32_mp_metadata_table_preview.md"
    out_summary = out / "v32_mp_metadata_table_summary.json"

    meta.to_csv(out_csv, index=False)
    meta.head(80).to_markdown(out_preview, index=False)

    summary = {
        "status": "pass",
        "n_metadata_rows": int(len(meta)),
        "n_unique_mp_ids": int(meta["mp_id"].nunique()),
        "n_with_formula": int((meta["formula"].fillna("").astype(str).str.len() > 0).sum()),
        "n_with_elements": int((meta["elements"].fillna("").astype(str).str.len() > 0).sum()),
        "family_counts": meta["mp_family"].value_counts().to_dict(),
        "source_type_counts": meta["metadata_source_type"].value_counts().to_dict(),
        "output_csv": str(out_csv),
        "interpretation": "V32 builds an mp_id -> formula/elements/family metadata table for metadata-aware Stage3 alignment.",
    }

    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_preview)
    print("[SAVE]", out_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
