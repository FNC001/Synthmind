#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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


def ensure_test_aliases(fold_dir: Path) -> list[Path]:
    """Provide the legacy trainer's unused test inputs without copying fold data."""
    created: list[Path] = []
    for suffix in ("npz", "csv"):
        source = fold_dir / f"val.{suffix}" if suffix == "npz" else fold_dir / "val_meta.csv"
        target = fold_dir / f"test.{suffix}" if suffix == "npz" else fold_dir / "test_meta.csv"
        if not target.exists():
            os.symlink(source.name, target)
            created.append(target)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train fixed-epoch formula-group OOF GFlowNet experts and merge beam candidates."
    )
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--run_name", default="gflownet_film_oof_fixed31")
    parser.add_argument("--candidate_name", default="gflownet_beam100_oof_candidates.jsonl")
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--x_mlp_hidden_dims", default="1536,768")
    parser.add_argument("--family_feature_count", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=31)
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--decode_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fold_root = Path(args.fold_root).resolve()
    fold_dirs = sorted(fold_root.glob("fold_*"), key=lambda path: int(path.name.split("_")[-1]))
    if not fold_dirs:
        raise RuntimeError(f"no fold directories under {fold_root}")
    python = sys.executable
    train_script = PROJECT_ROOT / "training/precursor/train_gflownet.py"
    evaluate_script = PROJECT_ROOT / "training/family/evaluate_stage2_beam.py"
    completed = []
    manifest = json.loads((fold_root / "manifest.json").read_text(encoding="utf-8"))

    for fold_dir in fold_dirs:
        fold = int(fold_dir.name.split("_")[-1])
        run_dir = fold_dir / args.run_name
        checkpoint = run_dir / "best_model.pt"
        candidates = fold_dir / args.candidate_name
        metrics = run_dir / "beam_val.json"
        aliases = ensure_test_aliases(fold_dir)
        try:
            if args.force or not checkpoint.exists():
                command = [
                    python, str(train_script),
                    "--input_dir", str(fold_dir),
                    "--run_dir", str(run_dir),
                    "--device", str(args.device),
                    "--hidden_dim", str(args.hidden_dim),
                    "--x_mlp_hidden_dims", str(args.x_mlp_hidden_dims),
                    "--family_feature_count", str(args.family_feature_count),
                    "--dropout", str(args.dropout),
                    "--batch_size", str(args.batch_size),
                    "--epochs", str(args.epochs),
                    "--patience", str(args.epochs + 1),
                    "--lr", str(args.lr),
                    "--seed", str(args.seed + fold * 101),
                    "--warmup_epochs", "8",
                    "--rl_weight", "0.1",
                    "--exact_bonus", "1.0",
                    "--length_penalty", "0.02",
                    "--checkpoint_selection", "last",
                    "--no_rerank",
                ]
                print(json.dumps({"event": "train_start", "fold": fold}), flush=True)
                run_logged(command, run_dir / "train.log")
            if args.force or not candidates.exists():
                command = [
                    python, str(evaluate_script),
                    "--input_dir", str(fold_dir),
                    "--checkpoint", str(checkpoint),
                    "--split", "val",
                    "--beam_width", str(args.beam_width),
                    "--batch_size", str(args.decode_batch_size),
                    "--device", str(args.device),
                    "--output_json", str(metrics),
                    "--output_candidates_jsonl", str(candidates),
                ]
                print(json.dumps({"event": "decode_start", "fold": fold}), flush=True)
                run_logged(command, run_dir / "decode.log")
        finally:
            for alias in aliases:
                alias.unlink(missing_ok=True)

        fold_metrics = json.loads(metrics.read_text(encoding="utf-8"))["metrics"]
        completed.append({
            "fold": fold,
            "rows": int(manifest["folds"][str(fold)]["query_rows"]),
            "exact_hit@10": float(fold_metrics["exact_hit@10"]),
            "oracle": float(fold_metrics["exact_hit@100"]),
        })
        print(json.dumps({"event": "fold_complete", **completed[-1]}), flush=True)

    subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "training/family/merge_stage2_oof_candidates.py"),
            "--fold_root", str(fold_root),
            "--candidate_name", str(args.candidate_name),
            "--output_jsonl", str(Path(args.output_jsonl).resolve()),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    report = {
        "protocol": "formula_group_oof_fixed_epoch_gflownet_beam_candidate_generation",
        "config": vars(args),
        "folds": completed,
        "output_jsonl": str(Path(args.output_jsonl).resolve()),
    }
    report_path = Path(args.output_jsonl).resolve().with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
