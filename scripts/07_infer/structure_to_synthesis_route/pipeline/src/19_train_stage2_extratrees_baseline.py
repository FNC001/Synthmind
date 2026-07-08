#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import jaccard_score, f1_score


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int32)

    out = {}
    try:
        out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    except Exception:
        out["micro_f1"] = float("nan")

    try:
        out["samples_f1"] = float(f1_score(y_true, y_pred, average="samples", zero_division=0))
    except Exception:
        out["samples_f1"] = float("nan")

    try:
        out["samples_jaccard"] = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    except Exception:
        out["samples_jaccard"] = float("nan")

    exact = (y_true == y_pred).all(axis=1).mean()
    out["exact_match"] = float(exact)

    return out


def predict_proba_matrix(model: MultiOutputClassifier, x: np.ndarray) -> np.ndarray:
    probs = []
    for est in model.estimators_:
        p = est.predict_proba(x)
        if isinstance(p, list):
            p = p[0]
        if p.shape[1] == 1:
            # 该 label 训练集中只有单类，若类为 1 则概率全 1，否则全 0
            cls = int(est.classes_[0])
            prob1 = np.ones((x.shape[0],), dtype=np.float32) if cls == 1 else np.zeros((x.shape[0],), dtype=np.float32)
        else:
            idx = list(est.classes_).index(1) if 1 in est.classes_ else None
            prob1 = p[:, idx] if idx is not None else np.zeros((x.shape[0],), dtype=np.float32)
        probs.append(prob1.astype(np.float32))
    return np.stack(probs, axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train stable Stage2 ExtraTrees multi-label baseline.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--x_key", default="x")
    ap.add_argument("--y_key", default="y_multi_hot")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=28)
    ap.add_argument("--min_samples_leaf", type=int, default=1)
    ap.add_argument("--max_features", default="sqrt")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ensure_dir(run_dir)

    train = load_npz(input_dir / "train.npz")
    val = load_npz(input_dir / "val.npz")
    test = load_npz(input_dir / "test.npz")

    x_train = train[args.x_key].astype(np.float32)
    y_train = train[args.y_key].astype(np.int32)
    x_val = val[args.x_key].astype(np.float32)
    y_val = val[args.y_key].astype(np.int32)
    x_test = test[args.x_key].astype(np.float32)
    y_test = test[args.y_key].astype(np.int32)

    precursor_names_path = input_dir / "precursor_names.json"
    precursor_names = read_json(precursor_names_path) if precursor_names_path.exists() else []

    base = ExtraTreesClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=None if str(args.max_depth).lower() == "none" else int(args.max_depth),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features=args.max_features,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        class_weight="balanced",
    )

    model = MultiOutputClassifier(base, n_jobs=1)

    print(f"[Info] x_train={x_train.shape}, y_train={y_train.shape}")
    print(f"[Info] fitting ExtraTrees multi-label model...")
    model.fit(x_train, y_train)

    print("[Info] predicting validation/test probabilities...")
    val_prob = predict_proba_matrix(model, x_val)
    test_prob = predict_proba_matrix(model, x_test)

    metrics = {
        "config": vars(args),
        "data": {
            "input_dir": str(input_dir),
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_test": int(x_test.shape[0]),
            "x_dim": int(x_train.shape[1]),
            "y_dim": int(y_train.shape[1]),
            "n_precursor_names": int(len(precursor_names)),
        },
        "val": compute_metrics(y_val, val_prob, threshold=float(args.threshold)),
        "test": compute_metrics(y_test, test_prob, threshold=float(args.threshold)),
    }

    model_path = run_dir / "stage2_extratrees_multilabel.joblib"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"

    payload = {
        "model": model,
        "x_key": args.x_key,
        "y_key": args.y_key,
        "precursor_names": precursor_names,
        "input_dir": str(input_dir),
        "threshold": float(args.threshold),
        "config": vars(args),
    }

    joblib.dump(payload, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] model  ->", model_path)
    print("[DONE] metrics ->", metrics_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
