#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


MISSING = "<UNK_OR_MISSING>"
CORE_METHODS = {"solid_state", "solution", "melt_arc"}


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    pq = data_dir / f"{split}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(data_dir / f"{split}.csv")


def select_numeric_cols(df: pd.DataFrame) -> List[str]:
    cols = [
        c for c in df.columns
        if c.startswith("feat_")
        or c.startswith("precursor_family_frac__")
        or c in {
            "precursor_confidence_score", "precursor_f1_to_true", "precursor_jaccard_to_true",
            "precursor_top1_score", "precursor_top20_uncertainty", "contains_open_generated_precursor",
            "contains_repair_precursor", "precursor_set_size",
        }
    ]
    return cols


def make_features(frames: Sequence[pd.DataFrame], numeric_cols: Sequence[str] | None = None, feature_cols: Sequence[str] | None = None) -> tuple[List[pd.DataFrame], List[str], List[str]]:
    if numeric_cols is None:
        numeric_cols = select_numeric_cols(frames[0])
    cat_cols = [c for c in ["reaction_method", "synthesis_type", "precursor_input_mode"] if c in frames[0].columns]
    outs = []
    for df in frames:
        nums = pd.DataFrame(index=df.index)
        for c in numeric_cols:
            nums[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) if c in df.columns else 0.0
        cats = pd.get_dummies(df[cat_cols].astype(str), dtype=np.float32) if cat_cols else pd.DataFrame(index=df.index)
        outs.append(pd.concat([nums.reset_index(drop=True), cats.reset_index(drop=True)], axis=1))
    if feature_cols is None:
        feature_cols = sorted(set().union(*(set(x.columns) for x in outs)))
    fixed = []
    for x in outs:
        for c in feature_cols:
            if c not in x.columns:
                x[c] = 0.0
        fixed.append(x[list(feature_cols)])
    return fixed, list(numeric_cols), list(feature_cols)


def fit_binary(x: pd.DataFrame, y: np.ndarray, seed: int):
    if len(np.unique(y)) < 2 or lgb is None:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x, y)
        return model
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=240, learning_rate=0.04, num_leaves=31,
        min_child_samples=20, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=seed, n_jobs=4, verbose=-1,
    )
    model.fit(x, y)
    return model


def fit_multiclass(x: pd.DataFrame, labels: pd.Series, seed: int):
    enc = LabelEncoder().fit(labels.astype(str))
    y = enc.transform(labels.astype(str))
    if len(enc.classes_) < 2 or lgb is None:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x, y)
    else:
        model = lgb.LGBMClassifier(
            objective="multiclass", n_estimators=260, learning_rate=0.04, num_leaves=31,
            min_child_samples=15, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=seed, n_jobs=4, verbose=-1,
        )
        model.fit(x, y)
    return model, enc


def safe_proba(model: Any, x: pd.DataFrame, n_classes: int | None = None) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
    else:
        pred = model.predict(x)
        n = n_classes or int(np.max(pred) + 1)
        p = np.zeros((len(pred), n), dtype=float)
        p[np.arange(len(pred)), pred.astype(int)] = 1.0
    if n_classes is not None and p.shape[1] < n_classes:
        full = np.zeros((len(x), n_classes), dtype=float)
        full[:, : p.shape[1]] = p
        p = full
    return p


def topk_acc(labels: pd.Series, proba: np.ndarray, enc: LabelEncoder, k: int) -> float:
    if len(labels) == 0:
        return 0.0
    y = enc.transform(labels.astype(str))
    if len(enc.classes_) == 1:
        return float(np.mean(y == 0))
    return float(top_k_accuracy_score(y, proba, k=min(k, len(enc.classes_)), labels=np.arange(len(enc.classes_))))


def build_prior(train: pd.DataFrame, field: str, known_col: str) -> Dict[str, Any]:
    prior: Dict[str, Any] = {"global": {}, "by_method": {}, "by_method_synthesis": {}}
    known = train[pd.to_numeric(train[known_col], errors="coerce").fillna(0).astype(int) == 1]
    if len(known):
        prior["global"] = {str(k): int(v) for k, v in known[field].astype(str).value_counts().items()}
        for method, g in known.groupby("reaction_method", sort=False):
            prior["by_method"][str(method)] = {str(k): int(v) for k, v in g[field].astype(str).value_counts().items()}
        if "synthesis_type" in known.columns:
            for key, g in known.groupby(["reaction_method", "synthesis_type"], sort=False):
                prior["by_method_synthesis"]["|".join(map(str, key))] = {str(k): int(v) for k, v in g[field].astype(str).value_counts().items()}
    return prior


def eval_split(df: pd.DataFrame, x: pd.DataFrame, pack: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}
    for name, field, known_col, model_k, model_c, enc_k in [
        ("atmosphere", "atmosphere_target_class", "atmosphere_known_mask", "atmosphere_known", "atmosphere_class", "atmosphere_encoder"),
        ("solvent", "solvent_target_class", "solvent_known_mask", "solvent_known", "solvent_class", "solvent_encoder"),
    ]:
        y_known = pd.to_numeric(df[known_col], errors="coerce").fillna(0).astype(int).to_numpy()
        pred_known = pack[model_k].predict(x).astype(int)
        out[f"{name}_known_mask_accuracy"] = float(accuracy_score(y_known, pred_known))
        known_mask = y_known == 1
        if known_mask.any():
            proba = safe_proba(pack[model_c], x.loc[known_mask], len(pack[enc_k].classes_))
            labels = df.loc[known_mask, field].astype(str)
            out[f"{name}_known_label_top1_accuracy"] = topk_acc(labels, proba, pack[enc_k], 1)
            out[f"{name}_known_label_top3_accuracy"] = topk_acc(labels, proba, pack[enc_k], 3)
            pred_label = pack[enc_k].inverse_transform(np.argmax(proba, axis=1))
            out[f"{name}_strict_contribution_known_correct_rate"] = float(np.mean(pred_label.astype(str) == labels.to_numpy(str)))
        else:
            out[f"{name}_known_label_top1_accuracy"] = 0.0
            out[f"{name}_known_label_top3_accuracy"] = 0.0
            out[f"{name}_strict_contribution_known_correct_rate"] = 0.0
    by_method = {}
    for method, g in df.groupby("reaction_method", sort=False):
        idx = g.index
        xm = x.loc[idx]
        y = pd.to_numeric(g["atmosphere_known_mask"], errors="coerce").fillna(0).astype(int).to_numpy()
        pred = pack["atmosphere_known"].predict(xm).astype(int)
        by_method[str(method)] = {"atmosphere_known_mask_accuracy": float(accuracy_score(y, pred)), "rows": int(len(g))}
    out["by_method"] = by_method
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage3 v4 atmosphere/solvent missing-aware models.")
    ap.add_argument("--dataset_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v4_20260612")
    ap.add_argument("--run_dir", default="runs/stage3/atmosphere_solvent_missing_v4_20260612")
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    data_dir = Path(args.dataset_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = [read_split(data_dir, s) for s in ["train", "val", "test"]]
    xs, numeric_cols, feature_cols = make_features([train, val, test])
    x_train, x_val, x_test = xs

    atm_known = pd.to_numeric(train["atmosphere_known_mask"], errors="coerce").fillna(0).astype(int).to_numpy()
    solv_known = pd.to_numeric(train["solvent_known_mask"], errors="coerce").fillna(0).astype(int).to_numpy()
    model_atm_known = fit_binary(x_train, atm_known, args.seed)
    model_solv_known = fit_binary(x_train, solv_known, args.seed + 1)
    atm_mask = atm_known == 1
    solv_mask = solv_known == 1
    model_atm_class, enc_atm = fit_multiclass(x_train.loc[atm_mask], train.loc[atm_mask, "atmosphere_target_class"], args.seed + 2)
    model_solv_class, enc_solv = fit_multiclass(x_train.loc[solv_mask], train.loc[solv_mask, "solvent_target_class"], args.seed + 3)
    prior_atm = build_prior(train, "atmosphere_target_class", "atmosphere_known_mask")
    prior_solv = build_prior(train, "solvent_target_class", "solvent_known_mask")

    pack = {
        "atmosphere_known": model_atm_known,
        "atmosphere_class": model_atm_class,
        "atmosphere_encoder": enc_atm,
        "atmosphere_prior": prior_atm,
        "solvent_known": model_solv_known,
        "solvent_class": model_solv_class,
        "solvent_encoder": enc_solv,
        "solvent_prior": prior_solv,
        "numeric_cols": numeric_cols,
        "feature_cols": feature_cols,
        "config": vars(args),
    }
    joblib.dump(model_atm_known, run_dir / "model_atmosphere_known.pkl")
    joblib.dump(model_atm_class, run_dir / "model_atmosphere_class.pkl")
    joblib.dump(prior_atm, run_dir / "model_atmosphere_prior.pkl")
    joblib.dump(model_solv_known, run_dir / "model_solvent_known.pkl")
    joblib.dump(model_solv_class, run_dir / "model_solvent_class.pkl")
    joblib.dump(prior_solv, run_dir / "model_solvent_prior.pkl")
    joblib.dump(pack, run_dir / "model_pack.joblib")

    metrics = {
        "train": eval_split(train, x_train, pack),
        "val": eval_split(val, x_val, pack),
        "test": eval_split(test, x_test, pack),
    }
    write_json(run_dir / "metrics.json", metrics)
    report = ["# Stage3 Atmosphere/Solvent Missing-Aware v4", "", "```json", json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2), "```"]
    (run_dir / "missing_aware_label_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
