#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Compatibility wrapper for core-retrain Stage2 score calibration.")
    ap.add_argument("--val_csv", default="outputs/evaluation/stage2_candidate_pool_core_retrain_20260610/val_core_candidate_sets.csv")
    ap.add_argument("--test_csv", default="outputs/evaluation/stage2_candidate_pool_core_retrain_20260610/test_core_candidate_sets.csv")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage2_score_calibration_core_retrain_20260610")
    ap.add_argument("--dataset_dir", default="data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only")
    ap.add_argument("--patch_csv", default="data/interim/ontology/precursor_alias_patch_v5_20260610/precursor_alias_patch_v5.csv")
    ap.add_argument("--n_trials", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--per_method", action="store_true", default=True)
    args, extra = ap.parse_known_args()

    target = Path(__file__).resolve().with_name("calibrate_stage2_candidate_scores_core_methods.py")
    sys.argv = [
        str(target),
        "--val_csv", args.val_csv,
        "--test_csv", args.test_csv,
        "--output_dir", args.output_dir,
        "--dataset_dir", args.dataset_dir,
        "--patch_csv", args.patch_csv,
        "--n_trials", str(args.n_trials),
        "--seed", str(args.seed),
        "--per_method",
        *extra,
    ]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
