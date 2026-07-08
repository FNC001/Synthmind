#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_labels(s: Any) -> List[str]:
    try:
        obj = json.loads(str(s))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def load_split(family_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(family_dir / f"{split}_family_labels.csv")
    x = np.load(family_dir / f"{split}_features.npz", allow_pickle=True)["x"].astype(np.float32)
    return df, x


def build_X(df: pd.DataFrame, sample_x: np.ndarray, element_vocab: Sequence[str], method_vocab: Sequence[str]) -> np.ndarray:
    x_rows = sample_x[df["x_row_index"].to_numpy(dtype=int)]
    elem_idx = {e: i for i, e in enumerate(element_vocab)}
    method_idx = {m: i for i, m in enumerate(method_vocab)}
    elem_one = np.zeros((len(df), len(element_vocab)), dtype=np.float32)
    meth_one = np.zeros((len(df), len(method_vocab)), dtype=np.float32)
    for r, elem in enumerate(df["target_element"].astype(str)):
        j = elem_idx.get(elem)
        if j is not None:
            elem_one[r, j] = 1.0
    for r, meth in enumerate(df["reaction_method"].fillna("other").astype(str)):
        j = method_idx.get(meth)
        if j is not None:
            meth_one[r, j] = 1.0
    return np.concatenate([x_rows, elem_one, meth_one], axis=1).astype(np.float32)


def binarize(df: pd.DataFrame, families: Sequence[str]) -> np.ndarray:
    idx = {f: i for i, f in enumerate(families)}
    y = np.zeros((len(df), len(families)), dtype=np.int8)
    for i, labs in enumerate(df["element_family_labels"].apply(parse_labels)):
        for lab in labs:
            j = idx.get(lab)
            if j is not None:
                y[i, j] = 1
    return y


def predict_matrix(models: Dict[str, Any], X: np.ndarray, families: Sequence[str]) -> np.ndarray:
    out = np.zeros((X.shape[0], len(families)), dtype=np.float32)
    for j, fam in enumerate(families):
        model = models.get(fam)
        if model is None:
            continue
        out[:, j] = np.asarray(model.predict_proba(X)[:, 1], dtype=np.float32)
    return out


def metrics(y_true: np.ndarray, prob: np.ndarray, families: Sequence[str], df: pd.DataFrame) -> Dict[str, Any]:
    pred_top1 = np.zeros_like(y_true)
    pred_top1[np.arange(len(y_true)), np.argmax(prob, axis=1)] = 1
    top3_idx = np.argsort(-prob, axis=1)[:, :3]
    pred_top3 = np.zeros_like(y_true)
    for i in range(len(y_true)):
        pred_top3[i, top3_idx[i]] = 1
    true_nonempty = y_true.sum(axis=1) > 0
    exact_top1 = np.all(pred_top1 == y_true, axis=1)
    any_top1 = (pred_top1 & y_true).sum(axis=1) > 0
    any_top3 = (pred_top3 & y_true).sum(axis=1) > 0
    pred_thr = (prob >= 0.35).astype(np.int8)
    empty = pred_thr.sum(axis=1) == 0
    pred_thr[empty, np.argmax(prob[empty], axis=1)] = 1
    inter = (pred_thr & y_true).sum(axis=1).astype(float)
    prec = inter / np.clip(pred_thr.sum(axis=1), 1, None)
    rec = inter / np.clip(y_true.sum(axis=1), 1, None)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    method = {}
    for m, idx in df.groupby("reaction_method").groups.items():
        arr = np.asarray(list(idx), dtype=int)
        method[str(m)] = {
            "n": int(len(arr)),
            "top1_recall": float(any_top1[arr].mean()),
            "top3_recall": float(any_top3[arr].mean()),
            "threshold_f1": float(f1[arr].mean()),
        }
    return {
        "n_rows": int(len(y_true)),
        "family_top1_exact": float(exact_top1[true_nonempty].mean()) if np.any(true_nonempty) else 0.0,
        "family_top1_recall": float(any_top1[true_nonempty].mean()) if np.any(true_nonempty) else 0.0,
        "family_top3_recall": float(any_top3[true_nonempty].mean()) if np.any(true_nonempty) else 0.0,
        "threshold_family_f1": float(f1[true_nonempty].mean()) if np.any(true_nonempty) else 0.0,
        "method_metrics": method,
    }


def save_predictions(path: Path, df: pd.DataFrame, prob: np.ndarray, families: Sequence[str]) -> None:
    out = df.copy()
    for j, fam in enumerate(families):
        out[f"prob_family__{fam}"] = prob[:, j]
    out["pred_top1_family"] = [families[int(j)] for j in np.argmax(prob, axis=1)]
    out["pred_top3_families"] = [
        json.dumps([families[int(j)] for j in row], ensure_ascii=False)
        for row in np.argsort(-prob, axis=1)[:, :3]
    ]
    out.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage2 element-family predictor v3.")
    ap.add_argument("--family_dir", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_estimators", type=int, default=400)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    family_dir = Path(args.family_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    families = [f for f in load_json(family_dir / "family_vocab.json") if f != "unknown"]
    elements = load_json(family_dir / "element_vocab.json")
    train_df, train_x = load_split(family_dir, "train")
    val_df, val_x = load_split(family_dir, "val")
    test_df, test_x = load_split(family_dir, "test")
    methods = sorted(set(train_df["reaction_method"].fillna("other").astype(str)) | set(val_df["reaction_method"].fillna("other").astype(str)) | set(test_df["reaction_method"].fillna("other").astype(str)))
    X_train = build_X(train_df, train_x, elements, methods)
    X_val = build_X(val_df, val_x, elements, methods)
    X_test = build_X(test_df, test_x, elements, methods)
    y_train = binarize(train_df, families)
    y_val = binarize(val_df, families)
    y_test = binarize(test_df, families)

    models: Dict[str, Any] = {}
    for j, fam in enumerate(families):
        if y_train[:, j].sum() == 0:
            continue
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=int(args.seed) + j,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_train,
            y_train[:, j],
            eval_set=[(X_val, y_val[:, j])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        models[fam] = model
        print(f"[Model] {fam} positives={int(y_train[:, j].sum())}", flush=True)

    val_prob = predict_matrix(models, X_val, families)
    test_prob = predict_matrix(models, X_test, families)
    save_predictions(run_dir / "val_family_predictions.csv", val_df, val_prob, families)
    save_predictions(run_dir / "test_family_predictions.csv", test_df, test_prob, families)
    pack = {"models": models, "families": families, "elements": elements, "methods": methods, "config": vars(args)}
    joblib.dump(pack, run_dir / "family_predictor_lgbm_v3.joblib")
    summary = {
        "config": vars(args),
        "data": {
            "n_train_rows": int(len(train_df)),
            "n_val_rows": int(len(val_df)),
            "n_test_rows": int(len(test_df)),
            "n_families": int(len(families)),
            "n_features": int(X_train.shape[1]),
        },
        "metrics": {
            "val": metrics(y_val, val_prob, families, val_df),
            "test": metrics(y_test, test_prob, families, test_df),
        },
        "artifacts": {
            "model": str((run_dir / "family_predictor_lgbm_v3.joblib").resolve()),
            "test_predictions": str((run_dir / "test_family_predictions.csv").resolve()),
            "metrics": str((run_dir / "metrics.json").resolve()),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
