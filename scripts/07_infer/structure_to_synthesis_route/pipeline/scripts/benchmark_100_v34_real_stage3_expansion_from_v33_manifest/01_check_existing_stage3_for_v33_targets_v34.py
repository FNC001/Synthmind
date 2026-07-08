#!/usr/bin/env python
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    project_root = Path(args.project_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    version = "benchmark_100_v34_real_stage3_expansion_from_v33_manifest"
    root = project_root / "outputs" / version

    manifest_p = root / "input_from_v33" / "v33_stage3_expansion_request_manifest.csv"
    stage3_p = root / "input_from_v33" / "v32_stage3_candidates_with_mp_metadata.csv"

    manifest = pd.read_csv(manifest_p)
    stage3 = pd.read_csv(stage3_p)

    id_col = "mp_id" if "mp_id" in stage3.columns else "case_id"
    requested = manifest["mp_id"].astype(str).unique().tolist()
    existing_ids = set(stage3[id_col].astype(str).unique())

    rows = []
    for _, r in manifest.iterrows():
        mp_id = str(r["mp_id"])
        has_existing = mp_id in existing_ids
        n_existing = int((stage3[id_col].astype(str) == mp_id).sum()) if has_existing else 0
        rows.append({
            "request_id": r.get("request_id", ""),
            "external_case_id": r.get("external_case_id", ""),
            "target_formula": r.get("target_formula", ""),
            "mp_id": mp_id,
            "mp_formula": r.get("mp_formula", ""),
            "mp_family": r.get("mp_family", ""),
            "has_existing_stage3_candidates": has_existing,
            "n_existing_stage3_candidate_rows": n_existing,
            "recommended_v34_action": "reuse_existing_stage3_candidates" if has_existing else "generate_or_export_new_real_stage3_candidates",
        })

    df = pd.DataFrame(rows)
    out_csv = out_dir / "v34_existing_stage3_target_check.csv"
    out_md = out_dir / "v34_existing_stage3_target_check.md"
    out_json = out_dir / "v34_existing_stage3_target_check_summary.json"

    df.to_csv(out_csv, index=False)
    out_md.write_text(df.to_markdown(index=False), encoding="utf-8")

    summary = {
        "status": "pass",
        "n_requested_targets": len(requested),
        "n_targets_with_existing_stage3": int(df["has_existing_stage3_candidates"].sum()),
        "n_targets_need_generation_or_export": int((~df["has_existing_stage3_candidates"]).sum()),
        "targets_need_generation_or_export": df.loc[~df["has_existing_stage3_candidates"], "mp_id"].tolist(),
        "output_csv": str(out_csv),
        "interpretation": "V34 checks whether V33 requested MP targets already exist in the current real Stage3 candidate library.",
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", out_csv)
    print("[SAVE]", out_md)
    print("[SAVE]", out_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
