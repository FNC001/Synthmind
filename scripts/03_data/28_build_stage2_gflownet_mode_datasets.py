#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build mode-specific stage2 GFlowNet datasets (e.g. relaxed_only / gold_only)
by filtering an existing base bundle with split CSVs from either:

1) nested training_modes layout:
   <mode_input_root>/<train_mode>/train/*.csv
   <mode_input_root>/<train_mode>/val/*.csv
   <mode_input_root>/<train_mode>/test/*.csv

2) flat canonical layout:
   <mode_input_root>/<train_mode>/stage2_train_hybrid.csv
   <mode_input_root>/<train_mode>/stage2_val_hybrid.csv
   <mode_input_root>/<train_mode>/stage2_test_hybrid.csv

Default paths are adapted for the current SynPred project layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path("/Users/wyc/SynPred")
DEFAULT_BASE_INPUT_DIR = DEFAULT_PROJECT_ROOT / "data" / "interim" / "generative" / "stage2_gflownet_dataset" / "hybrid"
DEFAULT_MODE_INPUT_ROOT = DEFAULT_PROJECT_ROOT / "data" / "interim" / "model_inputs" / "stage2_cvae_modes" / "stage2_hybrid_cgcnn_chgnet"


JOIN_KEYS = [
    "id",
    "row_id",
    "sample_id",
    "material_id",
    "entry_id",
    "reaction_id",
    "synth_uid",
    "record_index",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_json(src: Path, dst: Path) -> None:
    write_json(dst, load_json(src))


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def save_npz_dict(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def _first_existing(candidates: Iterable[Path], what: str) -> Path:
    candidates = list(candidates)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 {what}，候选路径为：\n" + "\n".join(str(x) for x in candidates))


def _find_single_csv(split_dir: Path) -> Path:
    cands = sorted(split_dir.glob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"{split_dir} 下没有找到 split CSV")
    if len(cands) > 1:
        print(f"[Warn] {split_dir} 下找到多个 CSV，默认使用: {cands[0].name}")
    return cands[0]


def _resolve_mode_dir(root: Path, train_mode: str) -> Path:
    mode_map = {
        "relaxed_only": [root / "relaxed_only"],
        "gold_only": [root / "gold_only"],
        "curriculum": [root / "curriculum"],
        "curriculum_phase1": [root / "curriculum_phase1", root / "curriculum"],
        "curriculum_phase2": [root / "curriculum_phase2", root / "curriculum"],
    }
    if train_mode not in mode_map:
        raise ValueError(f"不支持的 train_mode: {train_mode}")
    return _first_existing(mode_map[train_mode], f"train_mode={train_mode} 对应的 mode 目录")


def _resolve_flat_csv(mode_dir: Path, split_name: str) -> Optional[Path]:
    candidates: List[Path] = []
    if split_name == "train":
        candidates = [
            mode_dir / "stage2_train_hybrid.csv",
            mode_dir / "stage2_train_ml.csv",
            mode_dir / "stage2_gold_train_holdout_hybrid.csv",
            mode_dir / "stage2_gold_train_holdout_ml.csv",
        ]
    elif split_name == "val":
        candidates = [
            mode_dir / "stage2_val_hybrid.csv",
            mode_dir / "stage2_val_ml.csv",
        ]
    elif split_name == "test":
        candidates = [
            mode_dir / "stage2_test_hybrid.csv",
            mode_dir / "stage2_test_ml.csv",
        ]
    for p in candidates:
        if p.exists():
            return p
    return None


def resolve_mode_split_csvs(mode_root: Path, train_mode: str) -> Dict[str, Path]:
    mode_dir = _resolve_mode_dir(mode_root, train_mode)

    # 1) nested layout
    if train_mode == "curriculum_phase1":
        phase1_train = mode_dir / "phase1_train"
        if phase1_train.exists():
            return {
                "train": _find_single_csv(phase1_train),
                "val": _find_single_csv(mode_dir / "val"),
                "test": _find_single_csv(mode_dir / "test"),
            }

    if train_mode == "curriculum_phase2":
        phase2_train = mode_dir / "phase2_train"
        if phase2_train.exists():
            return {
                "train": _find_single_csv(phase2_train),
                "val": _find_single_csv(mode_dir / "val"),
                "test": _find_single_csv(mode_dir / "test"),
            }

    nested_train = mode_dir / "train"
    nested_val = mode_dir / "val"
    nested_test = mode_dir / "test"
    if nested_train.exists() and nested_val.exists() and nested_test.exists():
        return {
            "train": _find_single_csv(nested_train),
            "val": _find_single_csv(nested_val),
            "test": _find_single_csv(nested_test),
        }

    # 2) flat canonical layout
    train_csv = _resolve_flat_csv(mode_dir, "train")
    val_csv = _resolve_flat_csv(mode_dir, "val")
    test_csv = _resolve_flat_csv(mode_dir, "test")
    if train_csv and val_csv and test_csv:
        return {
            "train": train_csv,
            "val": val_csv,
            "test": test_csv,
        }

    raise FileNotFoundError(
        "未能解析 mode split CSV。\n"
        f"mode_dir = {mode_dir}\n"
        "既没有找到嵌套目录 train/ val/ test，也没有找到平铺文件 "
        "stage2_train_*.csv / stage2_val_*.csv / stage2_test_*.csv"
    )


def find_join_key(base_meta: pd.DataFrame, split_df: pd.DataFrame) -> str:
    for key in JOIN_KEYS:
        if key in base_meta.columns and key in split_df.columns:
            base_nonempty = (
                base_meta[key].notna()
                & (base_meta[key].astype(str).str.strip() != "")
                & (base_meta[key].astype(str).str.lower() != "nan")
            )
            split_nonempty = (
                split_df[key].notna()
                & (split_df[key].astype(str).str.strip() != "")
                & (split_df[key].astype(str).str.lower() != "nan")
            )
            if base_nonempty.any() and split_nonempty.any():
                return key
    raise ValueError(
        "无法在 base meta 和 split CSV 中找到共同 join key。"
        f"\nbase columns: {list(base_meta.columns)}"
        f"\nsplit columns: {list(split_df.columns)}"
    )


def filter_one_split(
    split_name: str,
    base_npz: Dict[str, np.ndarray],
    base_meta: pd.DataFrame,
    split_csv: Path,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame, Dict[str, Any]]:
    split_df = pd.read_csv(split_csv)
    join_key = find_join_key(base_meta, split_df)

    base_key = base_meta[join_key].astype(str).str.strip()
    split_key = split_df[join_key].astype(str).str.strip()

    keep_ids = set(split_key.tolist())
    mask = base_key.isin(keep_ids).to_numpy()

    filtered_meta = base_meta.loc[mask].reset_index(drop=True)

    filtered_npz: Dict[str, np.ndarray] = {}
    n_rows = len(base_meta)
    for name, arr in base_npz.items():
        if hasattr(arr, "shape") and len(arr.shape) >= 1 and arr.shape[0] == n_rows:
            filtered_npz[name] = arr[mask]
        else:
            filtered_npz[name] = arr

    stats = {
        "split_name": split_name,
        "split_csv": str(split_csv),
        "join_key": join_key,
        "rows_before_filter": int(len(base_meta)),
        "rows_in_split_csv": int(len(split_df)),
        "rows_after_filter": int(len(filtered_meta)),
        "filtered_out_rows": int(len(base_meta) - len(filtered_meta)),
    }
    return filtered_npz, filtered_meta, stats


def infer_mode_dir_name(train_mode: str) -> str:
    if train_mode in {"relaxed_only", "gold_only", "curriculum_phase1", "curriculum_phase2", "curriculum"}:
        return train_mode
    return train_mode


def build_mode_dataset(
    base_input_dir: Path,
    mode_input_root: Path,
    train_mode: str,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = base_input_dir / infer_mode_dir_name(train_mode)

    ensure_dir(output_dir)

    required_base = [
        base_input_dir / "train.npz",
        base_input_dir / "val.npz",
        base_input_dir / "test.npz",
        base_input_dir / "train_meta.csv",
        base_input_dir / "val_meta.csv",
        base_input_dir / "test_meta.csv",
        base_input_dir / "action_vocab.json",
        base_input_dir / "precursor_names.json",
        base_input_dir / "summary.json",
    ]
    missing_base = [str(p) for p in required_base if not p.exists()]
    if missing_base:
        raise FileNotFoundError(
            "base_input_dir 缺少必需文件：\n" + "\n".join(missing_base) +
            f"\n\n当前 base_input_dir = {base_input_dir}"
        )

    split_csvs = resolve_mode_split_csvs(mode_input_root, train_mode)

    base_npz = {
        "train": load_npz_dict(base_input_dir / "train.npz"),
        "val": load_npz_dict(base_input_dir / "val.npz"),
        "test": load_npz_dict(base_input_dir / "test.npz"),
    }
    base_meta = {
        "train": pd.read_csv(base_input_dir / "train_meta.csv"),
        "val": pd.read_csv(base_input_dir / "val_meta.csv"),
        "test": pd.read_csv(base_input_dir / "test_meta.csv"),
    }

    split_stats: Dict[str, Any] = {}
    counts: Dict[str, int] = {}

    for split in ["train", "val", "test"]:
        filtered_npz, filtered_meta, stats = filter_one_split(
            split_name=split,
            base_npz=base_npz[split],
            base_meta=base_meta[split],
            split_csv=split_csvs[split],
        )

        save_npz_dict(output_dir / f"{split}.npz", filtered_npz)
        filtered_meta.to_csv(output_dir / f"{split}_meta.csv", index=False)

        split_stats[split] = stats
        counts[split] = int(len(filtered_meta))

    for name in ["action_vocab.json", "action_to_id.json", "precursor_names.json"]:
        src = base_input_dir / name
        if src.exists():
            copy_json(src, output_dir / name)

    summary = load_json(base_input_dir / "summary.json")
    schema = summary.get("schema", summary)
    summary_out = {
        "base_input_dir": str(base_input_dir),
        "mode_input_root": str(mode_input_root),
        "train_mode": train_mode,
        "output_dir": str(output_dir),
        "schema": schema,
        "counts": counts,
        "split_stats": split_stats,
    }
    write_json(output_dir / "summary.json", summary_out)
    write_json(output_dir / "build_summary.json", summary_out)

    if counts["train"] == 0:
        raise ValueError(f"构建后的 train split 为空：{output_dir}")

    return summary_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mode-specific stage2 GFlowNet dataset bundles.")
    parser.add_argument(
        "--base_input_dir",
        type=str,
        default=str(DEFAULT_BASE_INPUT_DIR),
        help=f"base GFlowNet bundle dir (default: {DEFAULT_BASE_INPUT_DIR})",
    )
    parser.add_argument(
        "--mode_input_root",
        type=str,
        default=str(DEFAULT_MODE_INPUT_ROOT),
        help=f"mode root, supports nested training_modes or flat canonical mode dirs (default: {DEFAULT_MODE_INPUT_ROOT})",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="gold_only",
        choices=["relaxed_only", "gold_only", "curriculum", "curriculum_phase1", "curriculum_phase2"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="optional explicit output dir; default is base_input_dir/train_mode",
    )
    args = parser.parse_args()

    base_input_dir = Path(args.base_input_dir).expanduser().resolve()
    mode_input_root = Path(args.mode_input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir.strip() else None

    summary = build_mode_dataset(
        base_input_dir=base_input_dir,
        mode_input_root=mode_input_root,
        train_mode=args.train_mode,
        output_dir=output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
