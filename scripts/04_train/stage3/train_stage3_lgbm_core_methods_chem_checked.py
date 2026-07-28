#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_npz(input_npz: Path, output_npz: Path, mask: np.ndarray) -> Dict[str, Any]:
    arr = np.load(input_npz, allow_pickle=True)
    payload = {}
    summary = {}
    n = int(mask.sum())
    for key in arr.files:
        value = arr[key]
        if value.shape[:1] == mask.shape:
            payload[key] = value[mask]
            summary[key] = list(payload[key].shape)
        else:
            payload[key] = value
            summary[key] = list(value.shape)
    np.savez_compressed(output_npz, **payload)
    summary["n_rows"] = n
    return summary


def make_filtered_input(input_dir: Path, output_dir: Path, methods: set[str]) -> Dict[str, Any]:
    ensure_dir(output_dir)
    schema = load_json(input_dir / "schema.json")
    write_json(output_dir / "schema.json", schema)
    for name in ["stage3_chem_checked_dataset_report.md", "summary.json"]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
    summaries = {}
    for split in ["train", "val", "test"]:
        meta = pd.read_csv(input_dir / f"{split}_meta.csv")
        mask = meta["reaction_method"].astype(str).isin(methods).to_numpy()
        meta_f = meta.loc[mask].reset_index(drop=True)
        meta_f.to_csv(output_dir / f"{split}_meta.csv", index=False)
        # Keep the richer CSV too for auditability.
        full = pd.read_csv(input_dir / f"{split}.csv")
        full.loc[mask].reset_index(drop=True).to_csv(output_dir / f"{split}.csv", index=False)
        try:
            full.loc[mask].reset_index(drop=True).to_parquet(output_dir / f"{split}.parquet", index=False)
        except Exception:
            pass
        summaries[split] = {
            "rows": int(mask.sum()),
            "method_counts": meta_f["reaction_method"].value_counts().to_dict(),
            "npz": filter_npz(input_dir / f"{split}.npz", output_dir / f"{split}.npz", mask),
        }
    write_json(output_dir / "filter_summary.json", {
        "source_input_dir": str(input_dir),
        "methods": sorted(methods),
        "splits": summaries,
    })
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare chemistry-checked Stage3 input and train LightGBM method experts.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--refined_dir", default="data/interim/refined/structdesc_refined_route_unified_20260609_units_normalized")
    ap.add_argument("--train_scope", choices=["core", "all"], default="core")
    ap.add_argument("--filtered_input_dir", default="")
    ap.add_argument("--num_boost_round", type=int, default=350)
    ap.add_argument("--early_stopping_rounds", type=int, default=35)
    ap.add_argument("--min_expert_train", type=int, default=300)
    ap.add_argument("--min_expert_val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if args.train_scope == "core":
        filtered = Path(args.filtered_input_dir).resolve() if args.filtered_input_dir else Path(
            "data/interim/generative/stage3_condition_dataset_chem_checked/core_methods_v5_20260610"
        ).resolve()
        make_filtered_input(input_dir, filtered, CORE_METHODS)
        train_input = filtered
    else:
        train_input = input_dir

    trainer = Path(__file__).resolve().with_name("train_stage3_lgbm_method_experts.py")
    cmd: List[str] = [
        "/Users/lihonglin/miniconda3/envs/py311/bin/python",
        str(trainer),
        "--input_dir", str(train_input),
        "--refined_dir", str(Path(args.refined_dir).resolve()),
        "--run_dir", str(Path(args.run_dir).resolve()),
        "--num_boost_round", str(args.num_boost_round),
        "--early_stopping_rounds", str(args.early_stopping_rounds),
        "--min_expert_train", str(args.min_expert_train),
        "--min_expert_val", str(args.min_expert_val),
        "--global_val_fallback",
        "--seed", str(args.seed),
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
