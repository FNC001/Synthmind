#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit


EXCLUDE_KEYWORDS = [
    "score",
    "prob",
    "rank",
    "wins",
    "losses",
    "win_rate",
    "mean_prob",
    "local_index",
    "stage35_v21",
    "stage35_v3",
    "stage35_v31",
    "stage35_v32",
    "stage35_v33",
    "stage35_v42",
    "route_warning_adjusted_score",
    "route_warning_score",
]


def should_exclude_feature(col: str) -> bool:
    c = col.lower()
    return any(k.lower() in c for k in EXCLUDE_KEYWORDS)


def metrics_dict(y_true, y_pred, y_prob):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "support_positive": int((np.asarray(y_true) == 1).sum()),
        "support_negative": int((np.asarray(y_true) == 0).sum()),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None
    try:
        out["average_precision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["average_precision"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="stage35_v43_template_pairwise_chemonly_extratrees")
    ap.add_argument("--test_size", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    all_diff_cols = [c for c in df.columns if c.startswith("diff__")]
    excluded = [c for c in all_diff_cols if should_exclude_feature(c)]
    feature_cols = [c for c in all_diff_cols if c not in excluded]

    if not feature_cols:
        raise SystemExit("[ERROR] No feature columns left after exclusion.")

    X_pos = df[feature_cols].fillna(0.0).astype(float)
    y_pos = np.ones(len(X_pos), dtype=int)

    X_neg = -X_pos
    y_neg = np.zeros(len(X_neg), dtype=int)

    X = pd.concat([X_pos, X_neg], ignore_index=True)
    y = np.concatenate([y_pos, y_neg], axis=0)
    groups = pd.concat([df["target_group"], df["target_group"]], ignore_index=True).astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    clf = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_prob = clf.predict_proba(X_train)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]
    train_pred = (train_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)

    model_path = output_dir / f"{args.model_name}.joblib"
    feature_cols_path = output_dir / "stage35_v43_template_pairwise_chemonly_feature_cols.json"
    excluded_cols_path = output_dir / "stage35_v43_template_pairwise_chemonly_excluded_cols.json"
    summary_path = output_dir / "stage35_v43_template_pairwise_chemonly_training_summary.json"
    report_path = output_dir / "stage35_v43_template_pairwise_chemonly_classification_report.txt"
    importance_csv = output_dir / "stage35_v43_template_pairwise_chemonly_feature_importance.csv"

    joblib.dump(clf, model_path)
    feature_cols_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    excluded_cols_path.write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(classification_report(y_test, test_pred, digits=4), encoding="utf-8")

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(importance_csv, index=False)

    summary = {
        "input_csv": str(input_csv.resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_original_pairs": int(len(df)),
        "n_symmetric_samples": int(len(X)),
        "n_all_diff_features": int(len(all_diff_cols)),
        "n_features_used": int(len(feature_cols)),
        "n_excluded_features": int(len(excluded)),
        "excluded_feature_keywords": EXCLUDE_KEYWORDS,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "split_strategy": "GroupShuffleSplit by target_group",
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "model": {
            "type": "ExtraTreesClassifier",
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "class_weight": "balanced",
        },
        "train_metrics": metrics_dict(y_train, train_pred, train_prob),
        "test_metrics": metrics_dict(y_test, test_pred, test_prob),
        "model_path": str(model_path),
        "feature_cols_path": str(feature_cols_path),
        "excluded_cols_path": str(excluded_cols_path),
        "classification_report": str(report_path),
        "feature_importance_csv": str(importance_csv),
        "note": "Stage35 v4.3 template-aware chemonly pairwise ranker. Existing score/rank/prob/win/loss features are excluded.",
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", model_path)
    print("[SAVE]", feature_cols_path)
    print("[SAVE]", excluded_cols_path)
    print("[SAVE]", summary_path)
    print("[SAVE]", report_path)
    print("[SAVE]", importance_csv)
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(imp.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
