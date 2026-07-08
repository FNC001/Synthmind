#!/usr/bin/env python3
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build core-method precursor templates from the core train split.")
    ap.add_argument("--dataset_dir", default="data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only")
    ap.add_argument("--ontology_csv", default="data/interim/ontology/precursor_ontology_v3_20260610/precursor_ontology.csv")
    ap.add_argument("--output_dir", default="data/interim/templates/stage2_core_method_templates_20260610")
    args = ap.parse_args()

    target = Path(__file__).resolve().with_name("51_build_method_precursor_templates_v5.py")
    sys.argv = [
        str(target),
        "--dataset_dir", args.dataset_dir,
        "--ontology_csv", args.ontology_csv,
        "--output_dir", args.output_dir,
    ]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
