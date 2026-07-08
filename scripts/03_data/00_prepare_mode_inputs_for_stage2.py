#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def count_rows(path: Path) -> int:
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)
    return -1


def copy_as(src: Path, dst: Path) -> Dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "n_rows": count_rows(dst),
    }


def build_standard_mode_views(
    source_root: Path,
    output_root: Path,
    dataset_name: str,
) -> Dict[str, Any]:
    """
    source_root example:
      /.../data/interim/training_modes/stage2_hybrid_cgcnn

    Expected structure under source_root:
      relaxed_only/train/stage2_train_hybrid.csv
      relaxed_only/val/stage2_val_hybrid.csv
      relaxed_only/test/stage2_test_hybrid.csv

      gold_only/train/stage2_gold_train_holdout_hybrid.csv
      gold_only/val/stage2_val_hybrid.csv
      gold_only/test/stage2_test_hybrid.csv

      curriculum/phase1_train/stage2_train_hybrid.csv
      curriculum/phase2_train/stage2_gold_train_holdout_hybrid.csv
      curriculum/val/stage2_val_hybrid.csv
      curriculum/test/stage2_test_hybrid.csv
    """
    out_root = output_root / dataset_name
    ensure_dir(out_root)

    summary: Dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(out_root),
        "modes": {},
    }

    # -------------------------
    # relaxed_only
    # -------------------------
    relaxed_out = out_root / "relaxed_only"
    ensure_dir(relaxed_out)

    summary["modes"]["relaxed_only"] = {
        "train": copy_as(
            source_root / "relaxed_only" / "train" / "stage2_train_hybrid.csv",
            relaxed_out / "stage2_train_hybrid.csv",
        ),
        "val": copy_as(
            source_root / "relaxed_only" / "val" / "stage2_val_hybrid.csv",
            relaxed_out / "stage2_val_hybrid.csv",
        ),
        "test": copy_as(
            source_root / "relaxed_only" / "test" / "stage2_test_hybrid.csv",
            relaxed_out / "stage2_test_hybrid.csv",
        ),
        "gold_train_holdout": copy_as(
            source_root / "gold_only" / "train" / "stage2_gold_train_holdout_hybrid.csv",
            relaxed_out / "stage2_gold_train_holdout_hybrid.csv",
        ),
    }

    # -------------------------
    # gold_only
    # -------------------------
    gold_out = out_root / "gold_only"
    ensure_dir(gold_out)

    summary["modes"]["gold_only"] = {
        "train": copy_as(
            source_root / "gold_only" / "train" / "stage2_gold_train_holdout_hybrid.csv",
            gold_out / "stage2_train_hybrid.csv",
        ),
        "val": copy_as(
            source_root / "gold_only" / "val" / "stage2_val_hybrid.csv",
            gold_out / "stage2_val_hybrid.csv",
        ),
        "test": copy_as(
            source_root / "gold_only" / "test" / "stage2_test_hybrid.csv",
            gold_out / "stage2_test_hybrid.csv",
        ),
        "gold_train_holdout": copy_as(
            source_root / "gold_only" / "train" / "stage2_gold_train_holdout_hybrid.csv",
            gold_out / "stage2_gold_train_holdout_hybrid.csv",
        ),
    }

    # -------------------------
    # curriculum phase1
    # -------------------------
    phase1_out = out_root / "curriculum" / "phase1"
    ensure_dir(phase1_out)

    summary["modes"]["curriculum_phase1"] = {
        "train": copy_as(
            source_root / "curriculum" / "phase1_train" / "stage2_train_hybrid.csv",
            phase1_out / "stage2_train_hybrid.csv",
        ),
        "val": copy_as(
            source_root / "curriculum" / "val" / "stage2_val_hybrid.csv",
            phase1_out / "stage2_val_hybrid.csv",
        ),
        "test": copy_as(
            source_root / "curriculum" / "test" / "stage2_test_hybrid.csv",
            phase1_out / "stage2_test_hybrid.csv",
        ),
        "gold_train_holdout": copy_as(
            source_root / "curriculum" / "phase2_train" / "stage2_gold_train_holdout_hybrid.csv",
            phase1_out / "stage2_gold_train_holdout_hybrid.csv",
        ),
    }

    # -------------------------
    # curriculum phase2
    # -------------------------
    phase2_out = out_root / "curriculum" / "phase2"
    ensure_dir(phase2_out)

    summary["modes"]["curriculum_phase2"] = {
        "train": copy_as(
            source_root / "curriculum" / "phase2_train" / "stage2_gold_train_holdout_hybrid.csv",
            phase2_out / "stage2_train_hybrid.csv",
        ),
        "val": copy_as(
            source_root / "curriculum" / "val" / "stage2_val_hybrid.csv",
            phase2_out / "stage2_val_hybrid.csv",
        ),
        "test": copy_as(
            source_root / "curriculum" / "test" / "stage2_test_hybrid.csv",
            phase2_out / "stage2_test_hybrid.csv",
        ),
        "gold_train_holdout": copy_as(
            source_root / "curriculum" / "phase2_train" / "stage2_gold_train_holdout_hybrid.csv",
            phase2_out / "stage2_gold_train_holdout_hybrid.csv",
        ),
    }

    write_json(out_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical mode-specific stage2 input directories for 03_data builders.")
    parser.add_argument(
        "--source_root",
        type=str,
        default="/Users/wyc/SynPred/data/interim/training_modes/stage2_hybrid_cgcnn",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/Users/wyc/SynPred/data/interim/model_inputs/stage2_cvae_modes",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="stage2_hybrid_cgcnn",
    )
    args = parser.parse_args()

    summary = build_standard_mode_views(
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        dataset_name=args.dataset_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
