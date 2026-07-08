#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import pandas as pd


KEYWORDS = [
    "stage3",
    "condition",
    "candidate",
    "candidates",
    "mdn",
    "flow",
]


def safe_read_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_files", type=int, default=20000)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version = "benchmark_100_v34_real_stage3_expansion_from_v33_manifest"
    root = project_root / "outputs" / version

    manifest_p = root / "input_from_v33" / "v33_stage3_expansion_request_manifest.csv"
    manifest = pd.read_csv(manifest_p)

    requested_mp_ids = set(manifest["mp_id"].astype(str).tolist())

    search_roots = [
        project_root / "outputs",
        project_root / "data" / "interim" / "generative",
        project_root / "data" / "interim",
    ]

    candidate_files = []
    for sr in search_roots:
        if not sr.exists():
            continue
        for p in sr.rglob("*.csv"):
            s = str(p).lower()
            if all(k not in s for k in KEYWORDS):
                continue
            candidate_files.append(p)
            if len(candidate_files) >= args.max_files:
                break

    rows = []
    matched_rows = []

    possible_id_cols = [
        "mp_id",
        "material_id",
        "material_key",
        "case_id",
        "target_formula",
        "source_sample_id",
        "sample_id",
    ]

    possible_temp_cols = [
        "candidate_temperature_c",
        "temperature_c",
        "v28_temperature_c",
    ]

    possible_time_cols = [
        "candidate_time_h",
        "time_h",
        "v28_time_h",
    ]

    for p in candidate_files:
        df = safe_read_csv(p)
        if df is None or df.empty:
            rows.append({
                "path": str(p),
                "readable": False,
                "n_rows": 0,
                "matched_requested_mp_ids": "",
                "n_matched_rows": 0,
                "has_temperature": False,
                "has_time": False,
                "candidate_file_class": "unreadable_or_empty",
            })
            continue

        cols = list(df.columns)
        id_cols = [c for c in possible_id_cols if c in cols]
        temp_cols = [c for c in possible_temp_cols if c in cols]
        time_cols = [c for c in possible_time_cols if c in cols]

        matched_ids = set()
        n_match = 0

        for c in id_cols:
            vals = df[c].astype(str)
            for mp_id in requested_mp_ids:
                mask = vals.str.contains(mp_id, regex=False, na=False)
                if mask.any():
                    matched_ids.add(mp_id)
                    n_match += int(mask.sum())

        has_temp = len(temp_cols) > 0
        has_time = len(time_cols) > 0

        if matched_ids and has_temp and has_time:
            cls = "possible_existing_real_stage3_export"
        elif matched_ids:
            cls = "mp_id_match_but_missing_condition_columns"
        elif has_temp and has_time:
            cls = "condition_table_no_requested_mp_match"
        else:
            cls = "not_relevant"

        rows.append({
            "path": str(p),
            "readable": True,
            "n_rows": int(len(df)),
            "columns": json.dumps(cols, ensure_ascii=False),
            "id_columns": ";".join(id_cols),
            "temperature_columns": ";".join(temp_cols),
            "time_columns": ";".join(time_cols),
            "matched_requested_mp_ids": ";".join(sorted(matched_ids)),
            "n_matched_rows": int(n_match),
            "has_temperature": has_temp,
            "has_time": has_time,
            "candidate_file_class": cls,
        })

        if matched_ids and has_temp and has_time:
            for c in id_cols:
                vals = df[c].astype(str)
                mask_all = pd.Series(False, index=df.index)
                for mp_id in matched_ids:
                    mask_all = mask_all | vals.str.contains(mp_id, regex=False, na=False)
                sub = df.loc[mask_all].copy()
                if not sub.empty:
                    sub["matched_from_file"] = str(p)
                    sub["matched_id_column"] = c
                    matched_rows.append(sub)

    inventory = pd.DataFrame(rows)
    inventory = inventory.sort_values(
        ["candidate_file_class", "n_matched_rows", "n_rows"],
        ascending=[True, False, False],
    )

    out_inventory = out_dir / "v34_existing_stage3_export_discovery_inventory.csv"
    out_inventory_md = out_dir / "v34_existing_stage3_export_discovery_inventory.md"
    out_summary = out_dir / "v34_existing_stage3_export_discovery_summary.json"
    out_matches = out_dir / "v34_discovered_existing_stage3_rows_for_requested_targets.csv"

    inventory.to_csv(out_inventory, index=False)
    inventory.head(100).to_markdown(out_inventory_md, index=False)

    if matched_rows:
        matched = pd.concat(matched_rows, ignore_index=True, sort=False)
        matched.to_csv(out_matches, index=False)
        n_discovered_rows = int(len(matched))
    else:
        n_discovered_rows = 0

    possible = inventory[inventory["candidate_file_class"] == "possible_existing_real_stage3_export"]
    matched_ids_all = set()
    for x in possible["matched_requested_mp_ids"].dropna().astype(str):
        for t in x.split(";"):
            if t:
                matched_ids_all.add(t)

    missing = sorted(requested_mp_ids - matched_ids_all)

    summary = {
        "status": "pass",
        "n_requested_mp_targets": len(requested_mp_ids),
        "n_scanned_candidate_files": len(candidate_files),
        "n_possible_existing_real_stage3_export_files": int(len(possible)),
        "n_requested_targets_found_in_existing_exports": len(matched_ids_all),
        "requested_targets_found_in_existing_exports": sorted(matched_ids_all),
        "n_requested_targets_still_missing": len(missing),
        "requested_targets_still_missing": missing,
        "n_discovered_stage3_rows": n_discovered_rows,
        "output_inventory_csv": str(out_inventory),
        "output_discovered_rows_csv": str(out_matches) if matched_rows else None,
        "interpretation": (
            "V34 searches existing project outputs for real Stage3-style candidate tables "
            "containing requested V33 MP targets and temperature/time condition columns."
        ),
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", out_inventory)
    print("[SAVE]", out_inventory_md)
    print("[SAVE]", out_summary)
    if matched_rows:
        print("[SAVE]", out_matches)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
