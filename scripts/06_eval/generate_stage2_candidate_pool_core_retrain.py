#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Compatibility wrapper for core-retrain Stage2 candidate generation.")
    ap.add_argument("--dataset_dir", default="data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only")
    ap.add_argument("--ontology_csv", default="data/interim/ontology/precursor_ontology_v3_20260610/precursor_ontology.csv")
    ap.add_argument("--template_csv", default="data/interim/templates/stage2_core_method_templates_20260610/method_precursor_templates.csv")
    ap.add_argument("--base_candidate_csv", required=True, help="All-method v5 candidate CSV for the requested split.")
    ap.add_argument("--patch_csv", default="data/interim/ontology/precursor_alias_patch_v5_20260610/precursor_alias_patch_v5.csv")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage2_candidate_pool_core_retrain_20260610")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--top_n", type=int, default=1000)
    args, extra = ap.parse_known_args()

    target = Path(__file__).resolve().with_name("generate_stage2_candidate_pool_core_methods.py")
    sys.argv = [
        str(target),
        "--dataset_dir", args.dataset_dir,
        "--ontology_csv", args.ontology_csv,
        "--template_csv", args.template_csv,
        "--base_candidate_csv", args.base_candidate_csv,
        "--patch_csv", args.patch_csv,
        "--output_dir", args.output_dir,
        "--split", args.split,
        "--top_n", str(args.top_n),
        *extra,
    ]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
