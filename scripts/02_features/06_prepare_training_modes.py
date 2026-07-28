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
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            # 减去表头
            return max(sum(1 for _ in f) - 1, 0)
    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    else:
        return -1


def copy_file(src: Path, dst: Path) -> Dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "src": str(src),
        "dst": str(dst),
        "n_rows": count_rows(dst),
    }


def prepare_modes(
    source_dir: Path,
    output_root: Path,
    train_file: str,
    val_file: str,
    test_file: str,
    gold_train_holdout_file: str,
    dataset_name: str,
) -> Dict[str, Any]:
    train_src = source_dir / train_file
    val_src = source_dir / val_file
    test_src = source_dir / test_file
    gold_src = source_dir / gold_train_holdout_file

    dataset_root = output_root / dataset_name
    ensure_dir(dataset_root)

    summary: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "source_dir": str(source_dir),
        "files": {
            "train": str(train_src),
            "val": str(val_src),
            "test": str(test_src),
            "gold_train_holdout": str(gold_src),
        },
        "modes": {},
    }

    # --------------------------------------------------
    # relaxed_only
    # --------------------------------------------------
    relaxed_dir = dataset_root / "relaxed_only"
    ensure_dir(relaxed_dir)

    summary["modes"]["relaxed_only"] = {
        "train": copy_file(train_src, relaxed_dir / "train" / train_src.name),
        "val": copy_file(val_src, relaxed_dir / "val" / val_src.name),
        "test": copy_file(test_src, relaxed_dir / "test" / test_src.name),
    }

    # --------------------------------------------------
    # gold_only
    # --------------------------------------------------
    gold_dir = dataset_root / "gold_only"
    ensure_dir(gold_dir)

    summary["modes"]["gold_only"] = {
        "train": copy_file(gold_src, gold_dir / "train" / gold_src.name),
        "val": copy_file(val_src, gold_dir / "val" / val_src.name),
        "test": copy_file(test_src, gold_dir / "test" / test_src.name),
    }

    # --------------------------------------------------
    # curriculum
    # --------------------------------------------------
    curriculum_dir = dataset_root / "curriculum"
    ensure_dir(curriculum_dir)

    summary["modes"]["curriculum"] = {
        "phase1_train": copy_file(train_src, curriculum_dir / "phase1_train" / train_src.name),
        "phase2_train": copy_file(gold_src, curriculum_dir / "phase2_train" / gold_src.name),
        "val": copy_file(val_src, curriculum_dir / "val" / val_src.name),
        "test": copy_file(test_src, curriculum_dir / "test" / test_src.name),
    }

    write_json(dataset_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare gold_only / relaxed_only / curriculum training modes from existing train/val/test/gold_train_holdout files."
    )
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--gold_train_holdout_file", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)

    summary = prepare_modes(
        source_dir=source_dir,
        output_root=output_root,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        gold_train_holdout_file=args.gold_train_holdout_file,
        dataset_name=args.dataset_name,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
