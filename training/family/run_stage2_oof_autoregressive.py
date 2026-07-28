#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train fixed-epoch formula-group OOF autoregressive experts and merge candidates."
    )
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--run_name", default="autoregressive_oof_fixed")
    parser.add_argument("--candidate_name", default="autoregressive_oof_candidates.jsonl")
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=38)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beam_per_length", type=int, default=100)
    parser.add_argument("--branch_factor", type=int, default=64)
    parser.add_argument("--length_normalization", type=float, default=0.0)
    parser.add_argument("--length_prior_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7960)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fold_root = Path(args.fold_root).resolve()
    fold_dirs = sorted(fold_root.glob("fold_*"), key=lambda path: int(path.name.split("_")[-1]))
    if not fold_dirs:
        raise RuntimeError(f"no fold directories under {fold_root}")
    python = sys.executable
    train_script = PROJECT_ROOT / "training/family/train_stage2_autoregressive_set.py"
    evaluate_script = PROJECT_ROOT / "training/family/evaluate_stage2_autoregressive_cardinality_beam.py"
    completed = []
    for ordinal, fold_dir in enumerate(fold_dirs):
        fold = int(fold_dir.name.split("_")[-1])
        run_dir = fold_dir / args.run_name
        candidates = fold_dir / args.candidate_name
        metrics = run_dir / "cardinality_beam_val.json"
        checkpoint = run_dir / "best_autoregressive_set.pt"
        if args.force or not checkpoint.exists():
            train_command = [
                python,
                str(train_script),
                "--input_dir",
                str(fold_dir),
                "--run_dir",
                str(run_dir),
                "--hidden",
                str(args.hidden),
                "--blocks",
                str(args.blocks),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--eval_every",
                str(args.epochs),
                "--patience",
                str(args.epochs + 1),
                "--selection_metric",
                "last",
                "--seed",
                str(args.seed + fold * 101),
                "--device",
                str(args.device),
            ]
            print(json.dumps({"event": "train_start", "fold": fold}), flush=True)
            run_logged(train_command, run_dir / "train.log")
        if args.force or not candidates.exists():
            evaluate_command = [
                python,
                str(evaluate_script),
                "--input_dir",
                str(fold_dir),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "val",
                "--beam_per_length",
                str(args.beam_per_length),
                "--branch_factor",
                str(args.branch_factor),
                "--batch_size",
                "2",
                "--fixed_length_normalization",
                str(args.length_normalization),
                "--fixed_length_prior_weight",
                str(args.length_prior_weight),
                "--output_json",
                str(metrics),
                "--output_candidates_jsonl",
                str(candidates),
                "--device",
                str(args.device),
            ]
            print(json.dumps({"event": "decode_start", "fold": fold}), flush=True)
            run_logged(evaluate_command, run_dir / "decode.log")
        fold_metrics = json.loads(metrics.read_text(encoding="utf-8"))
        completed.append(
            {
                "fold": fold,
                "rows": int(json.loads((fold_root / "manifest.json").read_text())["folds"][str(fold)]["query_rows"]),
                "exact_hit@10": float(fold_metrics["best"]["exact_hit@10"]),
                "oracle": float(fold_metrics["oracle_candidate_recall"]),
            }
        )
        print(json.dumps({"event": "fold_complete", **completed[-1]}), flush=True)

    merge_command = [
        python,
        str(PROJECT_ROOT / "training/family/merge_stage2_oof_candidates.py"),
        "--fold_root",
        str(fold_root),
        "--candidate_name",
        str(args.candidate_name),
        "--output_jsonl",
        str(Path(args.output_jsonl).resolve()),
    ]
    subprocess.run(merge_command, cwd=PROJECT_ROOT, check=True)
    report = {
        "protocol": "formula_group_oof_fixed_epoch_autoregressive_candidate_generation",
        "config": vars(args),
        "folds": completed,
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
    }
    report_path = Path(args.output_jsonl).resolve().with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
