#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import pandas as pd


def save_md(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_markdown(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    root = Path(args.project_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    v33 = root / "outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping"
    target_p = v33 / "expanded_stage3_targets_v33/v33_expanded_stage3_target_list.csv"

    if not target_p.exists():
        raise FileNotFoundError(f"Missing V33 target list: {target_p}")

    target = pd.read_csv(target_p)

    req = target[target["stage3_expansion_needed"] == True].copy()

    # Keep a clean request manifest for future MDN/Flow Stage3 export/generation.
    manifest = pd.DataFrame({
        "request_id": [f"v33_stage3_expand_{i:03d}" for i in range(1, len(req) + 1)],
        "external_case_id": req["external_case_id"],
        "target_formula": req["external_formula"],
        "target_elements": req["external_elements"],
        "target_family": req["external_family"],
        "material_id": req["direct_mp_id"],
        "mp_id": req["direct_mp_id"],
        "mp_formula": req["mp_formula"],
        "mp_elements": req["mp_elements"],
        "mp_family": req["mp_family"],
        "source_reason": req["reason_for_expansion"],
        "requested_stage3_models": "mdn;flow",
        "requested_candidate_type": "real_stage3_condition_candidates",
        "recommended_action": req["recommended_action"],
        "status": "pending_real_stage3_generation_or_export",
    })

    out_csv = out / "v33_stage3_expansion_request_manifest.csv"
    out_md = out / "v33_stage3_expansion_request_manifest.md"
    out_json = out / "v33_stage3_expansion_request_manifest_summary.json"

    manifest.to_csv(out_csv, index=False)
    save_md(manifest, out_md)

    summary = {
        "status": "pass",
        "n_stage3_expansion_requests": int(len(manifest)),
        "n_unique_mp_targets": int(manifest["mp_id"].nunique()),
        "requested_models": ["mdn", "flow"],
        "request_status": "pending_real_stage3_generation_or_export",
        "mp_targets": manifest["mp_id"].tolist(),
        "output_csv": str(out_csv),
        "interpretation": (
            "V33 request manifest converts formula-exact gap-case MP targets "
            "into a clean input table for future real Stage3 MDN/Flow condition-candidate export or generation."
        ),
    }

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[SAVE] {out_csv}")
    print(f"[SAVE] {out_md}")
    print(f"[SAVE] {out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
