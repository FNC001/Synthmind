#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PATCH_ACTIONS = {
    "merge_alias",
    "fix_formula_parse",
    "hydrate_normalize",
    "phase_suffix_remove",
    "oxidation_state_normalize",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage2 v4 precursor alias patch table from audit.")
    ap.add_argument("--audit_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_confidence", type=float, default=0.80)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv)
    patch = audit[
        audit["suggested_action"].isin(PATCH_ACTIONS)
        & (audit["confidence"].astype(float) >= float(args.min_confidence))
        & (audit["raw_label"].astype(str) != audit["suggested_canonical_label"].astype(str))
    ].copy()
    patch = patch.rename(columns={
        "raw_label": "raw_label",
        "suggested_canonical_label": "patched_label",
        "suggested_action": "patch_type",
    })
    patch = patch[["raw_label", "patched_label", "patch_type", "confidence", "reason"]]
    patch = patch.drop_duplicates("raw_label").sort_values(["patch_type", "raw_label"])
    patch.to_csv(out_dir / "precursor_alias_patch.csv", index=False)
    summary = {
        "n_patches": int(len(patch)),
        "patch_type_counts": patch["patch_type"].value_counts().to_dict(),
        "artifacts": {"patch_csv": str((out_dir / "precursor_alias_patch.csv").resolve())},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
