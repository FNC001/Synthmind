#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# project bootstrap so this script can run directly from scripts/04_train/stage3
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
PARENT_DIR = THIS_DIR.parent
PROJECT_ROOT_CANDIDATE = THIS_DIR.parents[3] if len(THIS_DIR.parents) >= 4 else THIS_DIR
for extra in [THIS_DIR, PARENT_DIR, PARENT_DIR / "common", PROJECT_ROOT_CANDIDATE]:
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

# -----------------------------
# shared imports with fallback
# -----------------------------
try:
    from common.common_io import ensure_dir, load_json, resolve_input_paths, write_json
except Exception:
    from common_io import ensure_dir, load_json, resolve_input_paths, write_json

try:
    from common.common_metrics import evaluate_mixed_conditions
except Exception:
    from common_metrics import evaluate_mixed_conditions

TRAIN_MODE_CHOICES = [
    "relaxed_only",
    "gold_only",
    "curriculum",
    "curriculum_phase1",
    "curriculum_phase2",
]

DEFAULT_PROJECT_ROOT = Path("/Users/wyc/SynPred")
DEFAULT_STAGE3_INPUT_DIR = DEFAULT_PROJECT_ROOT / "data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"
DEFAULT_STAGE3_MODE_ROOT = DEFAULT_PROJECT_ROOT / "data/interim/generative/stage3_condition_dataset"
DEFAULT_STAGE3_BASELINE_RUN_DIR = DEFAULT_PROJECT_ROOT / "runs/stage3/stage3_baseline_commonized_v1"

STAGE3_REQUIRED = {
    "train_npz": ["train.npz"],
    "val_npz": ["val.npz"],
    "test_npz": ["test.npz"],
    "schema": ["schema.json", "condition_schema.json", "summary.json"],
}
STAGE3_OPTIONAL = {
    "train_meta": ["train_meta.csv", "train.csv", "stage3_train.csv"],
    "val_meta": ["val_meta.csv", "val.csv", "stage3_val.csv"],
    "test_meta": ["test_meta.csv", "test.csv", "stage3_test.csv"],
}


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def _pick_first(pack: Mapping[str, np.ndarray], candidates: Sequence[str], what: str, required: bool = True) -> Optional[np.ndarray]:
    for k in candidates:
        if k in pack:
            return pack[k]
    if required:
        raise KeyError(f"Missing {what}; available keys={list(pack.keys())}")
    return None


def _load_stage3_split(npz_path: Path) -> Dict[str, Any]:
    pack = _load_npz(npz_path)
    x = _pick_first(pack, ["x", "features", "X"], "x")
    y_set = _pick_first(pack, ["y_set", "precursor_set", "stage2_set"], "y_set", required=False)
    y_disc = _pick_first(pack, ["y_cond_discrete", "y_disc", "disc_targets"], "y_cond_discrete", required=False)
    y_cont = _pick_first(pack, ["y_cond_continuous", "y_cont", "cont_targets"], "y_cond_continuous", required=False)
    y_cont_mask = _pick_first(pack, ["y_cond_continuous_mask", "y_cont_mask", "cont_mask"], "y_cond_continuous_mask", required=False)
    sample_id = _pick_first(pack, ["sample_id", "row_id", "id"], "sample_id", required=False)

    x = np.asarray(x, dtype=np.float32)
    if y_set is not None:
        y_set = np.asarray(y_set, dtype=np.float32)
    if y_disc is not None:
        y_disc = np.asarray(y_disc)
        if y_disc.ndim == 1:
            y_disc = y_disc.reshape(-1, 1)
    if y_cont is not None:
        y_cont = np.asarray(y_cont, dtype=np.float32)
        if y_cont.ndim == 1:
            y_cont = y_cont.reshape(-1, 1)
    if y_cont_mask is None and y_cont is not None:
        y_cont_mask = np.ones_like(y_cont, dtype=np.float32)
    elif y_cont_mask is not None:
        y_cont_mask = np.asarray(y_cont_mask, dtype=np.float32)
        if y_cont_mask.ndim == 1:
            y_cont_mask = y_cont_mask.reshape(-1, 1)
    if sample_id is None:
        sample_id = np.asarray([str(i) for i in range(x.shape[0])], dtype=object)
    else:
        sample_id = np.asarray(sample_id, dtype=object)

    return {
        "x": x,
        "y_set": y_set,
        "y_disc": y_disc,
        "y_cont": y_cont,
        "y_cont_mask": y_cont_mask,
        "sample_id": sample_id,
        "pack_keys": list(pack.keys()),
    }


def _load_schema(schema_path: Path) -> Dict[str, Any]:
    obj = load_json(schema_path)
    return obj.get("schema", obj)


def _get_cont_names(schema: Mapping[str, Any], n_cont: int) -> List[str]:
    if "continuous_cols" in schema and isinstance(schema["continuous_cols"], list):
        cols = [str(x) for x in schema["continuous_cols"]]
        if len(cols) >= n_cont:
            return cols[:n_cont]
    if "continuous_schema" in schema and isinstance(schema["continuous_schema"], dict):
        cols = list(schema["continuous_schema"].keys())
        if len(cols) >= n_cont:
            return [str(x) for x in cols[:n_cont]]
    return [f"cont_{i}" for i in range(n_cont)]


def _get_disc_names(schema: Mapping[str, Any], n_disc: int) -> List[str]:
    if "discrete_cols" in schema and isinstance(schema["discrete_cols"], list):
        cols = [str(x) for x in schema["discrete_cols"]]
        if len(cols) >= n_disc:
            return cols[:n_disc]
    if "discrete_schema" in schema and isinstance(schema["discrete_schema"], dict):
        cols = list(schema["discrete_schema"].keys())
        if len(cols) >= n_disc:
            return [str(x) for x in cols[:n_disc]]
    return [f"disc_{i}" for i in range(n_disc)]


def build_features(x: np.ndarray, y_set: Optional[np.ndarray], use_y_set: bool) -> np.ndarray:
    if use_y_set and y_set is not None:
        return np.concatenate([x.astype(np.float32), y_set.astype(np.float32)], axis=1)
    return x.astype(np.float32)


class Stage3BaselineModel:
    def __init__(
        self,
        standardize: bool = False,
        ridge_alpha: float = 1.0,
        logreg_C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.standardize = bool(standardize)
        self.ridge_alpha = float(ridge_alpha)
        self.logreg_C = float(logreg_C)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.scaler: Optional[StandardScaler] = None
        self.cont_models: List[Optional[Ridge]] = []
        self.disc_models: List[Any] = []
        self.cont_names: List[str] = []
        self.disc_names: List[str] = []

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if not self.standardize:
            return X.astype(np.float32)
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X).astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    def fit(
        self,
        X: np.ndarray,
        y_cont: Optional[np.ndarray],
        y_cont_mask: Optional[np.ndarray],
        cont_names: Sequence[str],
        y_disc: Optional[np.ndarray],
        disc_names: Sequence[str],
    ) -> None:
        Xf = self._transform(X, fit=True)
        self.cont_models = []
        self.disc_models = []
        self.cont_names = list(cont_names)
        self.disc_names = list(disc_names)

        if y_cont is not None:
            mask = np.ones_like(y_cont, dtype=np.float32) if y_cont_mask is None else y_cont_mask.astype(np.float32)
            for j in range(y_cont.shape[1]):
                valid = mask[:, j] > 0.5
                if np.sum(valid) == 0:
                    self.cont_models.append(None)
                    continue
                m = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
                m.fit(Xf[valid], y_cont[valid, j])
                self.cont_models.append(m)

        if y_disc is not None:
            for j in range(y_disc.shape[1]):
                yj = y_disc[:, j]
                uniq = np.unique(yj)
                if uniq.shape[0] <= 1:
                    m = DummyClassifier(strategy="most_frequent")
                    m.fit(Xf, yj)
                else:
                    m = LogisticRegression(
                        C=self.logreg_C,
                        max_iter=self.max_iter,
                        random_state=self.random_state,
                        solver="lbfgs",
                    )
                    m.fit(Xf, yj)
                self.disc_models.append(m)

    def predict(self, X: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        Xf = self._transform(X, fit=False)
        out: Dict[str, Optional[np.ndarray]] = {"y_cont_pred": None, "y_disc_pred": None, "y_disc_prob": None}

        if self.cont_models:
            preds: List[np.ndarray] = []
            for m in self.cont_models:
                if m is None:
                    preds.append(np.zeros((Xf.shape[0],), dtype=np.float32))
                else:
                    preds.append(np.asarray(m.predict(Xf), dtype=np.float32))
            out["y_cont_pred"] = np.stack(preds, axis=1).astype(np.float32)

        if self.disc_models:
            disc_preds: List[np.ndarray] = []
            disc_probs: List[Any] = []
            for m in self.disc_models:
                pred = np.asarray(m.predict(Xf))
                disc_preds.append(pred)
                if hasattr(m, "predict_proba"):
                    disc_probs.append(np.asarray(m.predict_proba(Xf), dtype=np.float32))
                else:
                    disc_probs.append(None)
            out["y_disc_pred"] = np.stack(disc_preds, axis=1)
            out["y_disc_prob"] = disc_probs
        return out


def evaluate_disc_logloss(y_true: Optional[np.ndarray], disc_prob: Optional[List[Any]]) -> Dict[str, float]:
    if y_true is None or disc_prob is None:
        return {}
    metrics: Dict[str, float] = {}
    vals: List[float] = []
    for j in range(y_true.shape[1]):
        probs = disc_prob[j]
        if probs is None:
            metrics[f"disc{j}_logloss"] = float("nan")
            continue
        try:
            val = float(log_loss(y_true[:, j], probs, labels=list(range(probs.shape[1]))))
        except Exception:
            val = float("nan")
        metrics[f"disc{j}_logloss"] = val
        if not np.isnan(val):
            vals.append(val)
    metrics["disc_logloss_mean"] = float(np.mean(vals)) if vals else float("nan")
    return metrics


def save_predictions_csv(
    path: Path,
    meta_df: pd.DataFrame,
    sample_ids: np.ndarray,
    cont_names: Sequence[str],
    disc_names: Sequence[str],
    y_cont_true: Optional[np.ndarray],
    y_cont_pred: Optional[np.ndarray],
    y_cont_mask: Optional[np.ndarray],
    y_disc_true: Optional[np.ndarray],
    y_disc_pred: Optional[np.ndarray],
) -> None:
    out = meta_df.copy() if len(meta_df) == len(sample_ids) else pd.DataFrame({"sample_id": sample_ids.astype(str)})
    if "sample_id" not in out.columns:
        out["sample_id"] = sample_ids.astype(str)

    if y_cont_true is not None and y_cont_pred is not None:
        for j, name in enumerate(cont_names):
            out[f"true_{name}"] = y_cont_true[:, j]
            out[f"pred_{name}"] = y_cont_pred[:, j]
            if y_cont_mask is not None:
                out[f"mask_{name}"] = y_cont_mask[:, j]

    if y_disc_true is not None and y_disc_pred is not None:
        for j, name in enumerate(disc_names):
            out[f"true_{name}"] = y_disc_true[:, j]
            out[f"pred_{name}"] = y_disc_pred[:, j]

    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified stage3 baseline predictor (linear/logistic baseline).")
    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--mode_input_root", type=str, default="")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_STAGE3_INPUT_DIR))
    parser.add_argument("--train_mode", type=str, default="gold_only", choices=TRAIN_MODE_CHOICES)
    parser.add_argument("--run_dir", type=str, default=str(DEFAULT_STAGE3_BASELINE_RUN_DIR))
    parser.add_argument("--use_y_set", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--logreg_C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input_dir and not args.mode_input_root:
        args.input_dir = str(DEFAULT_STAGE3_INPUT_DIR)

    run_dir = Path(args.run_dir).expanduser().resolve()
    ensure_dir(run_dir)

    print(f"[Info] input_dir(default-ready) = {args.input_dir}")
    print(f"[Info] run_dir(default-ready)   = {run_dir}")

    resolved = resolve_input_paths(args, required=STAGE3_REQUIRED, optional=STAGE3_OPTIONAL)
    input_dir = Path(resolved.resolved_input_dir)

    print(f"[Info] resolved_mode = {resolved.resolved_mode}")
    print(f"[Info] resolved_root = {resolved.resolved_root}")
    print(f"[Info] resolved_input_dir = {resolved.resolved_input_dir}")

    train_split = _load_stage3_split(Path(resolved.files["train_npz"]))
    val_split = _load_stage3_split(Path(resolved.files["val_npz"]))
    test_split = _load_stage3_split(Path(resolved.files["test_npz"]))
    schema = _load_schema(Path(resolved.files["schema"]))

    train_meta = pd.read_csv(resolved.files["train_meta"]) if "train_meta" in resolved.files else pd.DataFrame({"sample_id": train_split["sample_id"].astype(str)})
    val_meta = pd.read_csv(resolved.files["val_meta"]) if "val_meta" in resolved.files else pd.DataFrame({"sample_id": val_split["sample_id"].astype(str)})
    test_meta = pd.read_csv(resolved.files["test_meta"]) if "test_meta" in resolved.files else pd.DataFrame({"sample_id": test_split["sample_id"].astype(str)})

    X_train = build_features(train_split["x"], train_split["y_set"], args.use_y_set)
    X_val = build_features(val_split["x"], val_split["y_set"], args.use_y_set)
    X_test = build_features(test_split["x"], test_split["y_set"], args.use_y_set)

    n_cont = 0 if train_split["y_cont"] is None else int(train_split["y_cont"].shape[1])
    n_disc = 0 if train_split["y_disc"] is None else int(train_split["y_disc"].shape[1])
    cont_names = _get_cont_names(schema, n_cont)
    disc_names = _get_disc_names(schema, n_disc)

    model = Stage3BaselineModel(
        standardize=args.standardize,
        ridge_alpha=args.ridge_alpha,
        logreg_C=args.logreg_C,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    model.fit(
        X=X_train,
        y_cont=train_split["y_cont"],
        y_cont_mask=train_split["y_cont_mask"],
        cont_names=cont_names,
        y_disc=train_split["y_disc"],
        disc_names=disc_names,
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate_mixed_conditions(
        y_cont_true=val_split["y_cont"],
        y_cont_pred=val_pred["y_cont_pred"],
        y_cont_mask=val_split["y_cont_mask"],
        cont_target_names=cont_names,
        y_disc_true=val_split["y_disc"],
        y_disc_pred=val_pred["y_disc_pred"],
        disc_target_names=disc_names,
        prefix="val",
    )
    test_metrics = evaluate_mixed_conditions(
        y_cont_true=test_split["y_cont"],
        y_cont_pred=test_pred["y_cont_pred"],
        y_cont_mask=test_split["y_cont_mask"],
        cont_target_names=cont_names,
        y_disc_true=test_split["y_disc"],
        y_disc_pred=test_pred["y_disc_pred"],
        disc_target_names=disc_names,
        prefix="test",
    )
    if val_split["y_disc"] is not None:
        val_metrics.update({f"val_{k}": v for k, v in evaluate_disc_logloss(val_split["y_disc"], val_pred["y_disc_prob"]).items()})
        test_metrics.update({f"test_{k}": v for k, v in evaluate_disc_logloss(test_split["y_disc"], test_pred["y_disc_prob"]).items()})

    save_predictions_csv(
        run_dir / "val_predictions.csv",
        meta_df=val_meta,
        sample_ids=val_split["sample_id"],
        cont_names=cont_names,
        disc_names=disc_names,
        y_cont_true=val_split["y_cont"],
        y_cont_pred=val_pred["y_cont_pred"],
        y_cont_mask=val_split["y_cont_mask"],
        y_disc_true=val_split["y_disc"],
        y_disc_pred=val_pred["y_disc_pred"],
    )
    save_predictions_csv(
        run_dir / "test_predictions.csv",
        meta_df=test_meta,
        sample_ids=test_split["sample_id"],
        cont_names=cont_names,
        disc_names=disc_names,
        y_cont_true=test_split["y_cont"],
        y_cont_pred=test_pred["y_cont_pred"],
        y_cont_mask=test_split["y_cont_mask"],
        y_disc_true=test_split["y_disc"],
        y_disc_pred=test_pred["y_disc_pred"],
    )

    with open(run_dir / "best_model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "config": vars(args),
                "resolved": {
                    "resolved_mode": resolved.resolved_mode,
                    "resolved_root": resolved.resolved_root,
                    "resolved_input_dir": resolved.resolved_input_dir,
                    "resolved_files": resolved.files,
                },
                "cont_names": cont_names,
                "disc_names": disc_names,
            },
            f,
        )

    train_log = [{
        "stage": "fit_once_baseline",
        "n_train": int(X_train.shape[0]),
        "input_dim": int(X_train.shape[1]),
        "n_cont": int(n_cont),
        "n_disc": int(n_disc),
    }]
    write_json(run_dir / "train_log.json", to_builtin(train_log))

    summary = {
        "config": vars(args),
        "resolved_mode": resolved.resolved_mode,
        "resolved_root": resolved.resolved_root,
        "resolved_input_dir": resolved.resolved_input_dir,
        "resolved_files": resolved.files,
        "data": {
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "input_dim": int(X_train.shape[1]),
            "use_y_set": bool(args.use_y_set),
            "n_cont": int(n_cont),
            "n_disc": int(n_disc),
            "train_keys": train_split["pack_keys"],
            "val_keys": val_split["pack_keys"],
            "test_keys": test_split["pack_keys"],
            "default_project_root": str(DEFAULT_PROJECT_ROOT),
            "default_input_dir": str(DEFAULT_STAGE3_INPUT_DIR),
            "default_run_dir": str(DEFAULT_STAGE3_BASELINE_RUN_DIR),
            "resolved_input_exists": input_dir.exists(),
        },
        "val_metrics": to_builtin(val_metrics),
        "test_metrics": to_builtin(test_metrics),
    }
    write_json(run_dir / "metrics.json", to_builtin(summary))
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
