#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_x(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for key in ["x", "features", "X"]:
        if key in pack:
            return np.asarray(pack[key], dtype=np.float32)
    raise KeyError(f"Missing feature array in keys={list(pack)}")


def get_set_len(pack: Dict[str, np.ndarray]) -> np.ndarray:
    if "set_len" in pack:
        return np.asarray(pack["set_len"], dtype=int)
    if "y_multi_hot" in pack:
        return np.asarray(pack["y_multi_hot"].sum(axis=1), dtype=int)
    raise KeyError(f"Missing set_len/y_multi_hot in keys={list(pack)}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae_size": float(np.mean(np.abs(y_pred.astype(float) - y_true.astype(float)))),
        "within_1": float(np.mean(np.abs(y_pred.astype(float) - y_true.astype(float)) <= 1.0)),
        "mean_true_size": float(np.mean(y_true)),
        "mean_pred_size": float(np.mean(y_pred)),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a Stage2 precursor-set size predictor.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--max_depth", type=int, default=18)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    tr = load_npz(input_dir / "train.npz")
    va = load_npz(input_dir / "val.npz")
    te = load_npz(input_dir / "test.npz")

    x_train, y_train = get_x(tr), get_set_len(tr)
    x_val, y_val = get_x(va), get_set_len(va)
    x_test, y_test = get_x(te), get_set_len(te)

    model = ExtraTreesClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth) if args.max_depth > 0 else None,
        min_samples_leaf=int(args.min_samples_leaf),
        class_weight="balanced",
        random_state=int(args.seed),
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)

    pack = {
        "model": model,
        "config": vars(args),
        "classes": [int(x) for x in model.classes_.tolist()],
        "input_dir": str(input_dir),
    }
    joblib.dump(pack, run_dir / "set_size_predictor.joblib")

    summary = {
        "model": "stage2_set_size_extratrees",
        "config": vars(args),
        "data": {
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "x_dim": int(x_train.shape[1]),
            "train_size_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        },
        "val_metrics": metrics(y_val, val_pred),
        "test_metrics": metrics(y_test, test_pred),
        "artifacts": {
            "model": str(run_dir / "set_size_predictor.joblib"),
            "metrics": str(run_dir / "metrics.json"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
