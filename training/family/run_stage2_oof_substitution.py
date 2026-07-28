#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def merge(fold_root: Path, candidate_name: str, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "training/family/merge_stage2_oof_candidates.py"),
            "--fold_root",
            str(fold_root),
            "--candidate_name",
            candidate_name,
            "--output_jsonl",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate formula-group OOF periodic-substitution candidate experts."
    )
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--neighbors", type=int, default=5000)
    parser.add_argument("--candidate_limit", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fold_root = Path(args.fold_root).resolve()
    folds = sorted(fold_root.glob("fold_*"), key=lambda path: int(path.name.split("_")[-1]))
    specs = [
        {
            "name": "raw",
            "candidate_name": "substitution_raw_candidates.jsonl",
            "extra": ["--neighbor_features", "raw"],
        },
        {
            "name": "periodic",
            "candidate_name": "substitution_periodic_candidates.jsonl",
            "extra": ["--neighbor_features", "periodic_group", "--periodic_exact_scale", "0.0"],
        },
        {
            "name": "allgroups_exact1p0",
            "candidate_name": "substitution_allgroups_exact1p0_candidates.jsonl",
            "extra": [
                "--neighbor_features",
                "periodic_group",
                "--periodic_exact_scale",
                "1.0",
                "--substitute_all_groups",
            ],
        },
    ]
    evaluator = PROJECT_ROOT / "training/family/evaluate_stage2_family_substitution.py"
    report = {"protocol": "formula_group_oof_periodic_substitution", "config": vars(args), "folds": {}}
    for fold_dir in folds:
        fold = int(fold_dir.name.split("_")[-1])
        report["folds"][str(fold)] = {}
        for spec in specs:
            candidates = fold_dir / str(spec["candidate_name"])
            metrics = fold_dir / f"substitution_{spec['name']}_metrics.json"
            if args.force or not candidates.exists():
                command = [
                    sys.executable,
                    str(evaluator),
                    "--input_dir",
                    str(fold_dir),
                    "--split",
                    "val",
                    "--output_json",
                    str(metrics),
                    "--output_candidates_jsonl",
                    str(candidates),
                    "--neighbors",
                    str(args.neighbors),
                    "--candidate_limit",
                    str(args.candidate_limit),
                    "--device",
                    str(args.device),
                    *[str(value) for value in spec["extra"]],
                ]
                print(json.dumps({"event": "start", "fold": fold, "expert": spec["name"]}), flush=True)
                run_logged(command, fold_dir / f"substitution_{spec['name']}.log")
            current = json.loads(metrics.read_text(encoding="utf-8"))
            report["folds"][str(fold)][str(spec["name"])] = current["metrics"]
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "fold": fold,
                        "expert": spec["name"],
                        "exact_hit@10": current["metrics"]["exact_hit@10"],
                    }
                ),
                flush=True,
            )
    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        output = Path(f"{prefix}_{spec['name']}_train_candidates.jsonl")
        merge(fold_root, str(spec["candidate_name"]), output)
        report[f"{spec['name']}_output"] = str(output)
    report_path = Path(f"{prefix}_metrics.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
