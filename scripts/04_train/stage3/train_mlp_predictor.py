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
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# bootstrap: make sibling/common imports work when run directly
# ---------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_STAGE_ROOT = _THIS_DIR.parent
_COMMON_DIR = _STAGE_ROOT / "common"
_PROJECT_ROOT_CANDIDATES = [
    _THIS_FILE.parents[3] if len(_THIS_FILE.parents) >= 4 else None,
    Path("/Users/wyc/SynPred"),
]

for _p in (_THIS_DIR, _STAGE_ROOT, _COMMON_DIR, *[p for p in _PROJECT_ROOT_CANDIDATES if p is not None]):
    _sp = str(_p)
    if _p and _sp not in sys.path:
        sys.path.insert(0, _sp)


# -----------------------------
# shared imports with fallback
# -----------------------------
try:
    from common.common_io import ensure_dir, load_json, resolve_input_paths, write_json
except Exception:
    try:
        from common_io import ensure_dir, load_json, resolve_input_paths, write_json
    except Exception:
        def ensure_dir(path: Path) -> None:
            path.mkdir(parents=True, exist_ok=True)

        def load_json(path: Path) -> Any:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        def write_json(path: Path, obj: Any) -> None:
            ensure_dir(path.parent)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        resolve_input_paths = None  # type: ignore

try:
    from common.common_metrics import evaluate_mixed_conditions
except Exception:
    try:
        from common_metrics import evaluate_mixed_conditions
    except Exception:
        def evaluate_mixed_conditions(
            y_cont_true: Optional[np.ndarray],
            y_cont_pred: Optional[np.ndarray],
            y_cont_mask: Optional[np.ndarray],
            cont_target_names: Sequence[str],
            y_disc_true: Optional[np.ndarray],
            y_disc_pred: Optional[np.ndarray],
            disc_target_names: Sequence[str],
            prefix: str = "",
        ) -> Dict[str, float]:
            out: Dict[str, float] = {}

            if y_cont_true is not None and y_cont_pred is not None:
                yt = np.asarray(y_cont_true, dtype=np.float32)
                yp = np.asarray(y_cont_pred, dtype=np.float32)
                mask = np.ones_like(yt, dtype=np.float32) if y_cont_mask is None else np.asarray(y_cont_mask, dtype=np.float32)

                maes = []
                rmses = []
                for j, name in enumerate(cont_target_names):
                    valid = mask[:, j] > 0.5
                    if not np.any(valid):
                        continue
                    diff = yp[valid, j] - yt[valid, j]
                    mae = float(np.mean(np.abs(diff)))
                    rmse = float(np.sqrt(np.mean(np.square(diff))))
                    out[f"{name}_mae"] = mae
                    out[f"{name}_rmse"] = rmse
                    maes.append(mae)
                    rmses.append(rmse)
                if maes:
                    out["mae_mean"] = float(np.mean(maes))
                if rmses:
                    out["rmse_mean"] = float(np.mean(rmses))

            if y_disc_true is not None and y_disc_pred is not None:
                from sklearn.metrics import accuracy_score, f1_score

                yt = np.asarray(y_disc_true)
                yp = np.asarray(y_disc_pred)
                accs = []
                f1s = []
                for j, name in enumerate(disc_target_names):
                    acc = float(accuracy_score(yt[:, j], yp[:, j]))
                    f1 = float(f1_score(yt[:, j], yp[:, j], average="macro", zero_division=0))
                    out[f"{name}_accuracy"] = acc
                    out[f"{name}_macro_f1"] = f1
                    accs.append(acc)
                    f1s.append(f1)
                if accs:
                    out["disc_accuracy_mean"] = float(np.mean(accs))
                if f1s:
                    out["disc_macro_f1_mean"] = float(np.mean(f1s))

            if prefix:
                return {f"{prefix}_{k}": v for k, v in out.items()}
            return out


TRAIN_MODE_CHOICES = [
    "relaxed_only",
    "gold_only",
    "curriculum",
    "curriculum_phase1",
    "curriculum_phase2",
]

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

DEFAULT_PROJECT_ROOT = "/Users/wyc/SynPred"
DEFAULT_INPUT_DIR = "/Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"
DEFAULT_RUN_DIR = "/Users/wyc/SynPred/runs/stage3/mlp_predictor_commonized_v1"


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


def _sanitize_float_array(arr: Optional[np.ndarray], mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        out = np.where(m > 0.5, out, 0.0)
    return out.astype(np.float32)


def _load_stage3_split(npz_path: Path) -> Dict[str, Any]:
    pack = _load_npz(npz_path)
    x = _pick_first(pack, ["x", "features", "X"], "x")
    y_set = _pick_first(pack, ["y_set", "precursor_set", "stage2_set"], "y_set", required=False)
    y_disc = _pick_first(pack, ["y_cond_discrete", "y_disc", "disc_targets"], "y_cond_discrete", required=False)
    y_disc_mask = _pick_first(pack, ["y_cond_discrete_mask", "y_disc_mask", "disc_mask"], "y_cond_discrete_mask", required=False)
    y_cont = _pick_first(pack, ["y_cond_continuous", "y_cont", "cont_targets"], "y_cond_continuous", required=False)
    y_cont_mask = _pick_first(pack, ["y_cond_continuous_mask", "y_cont_mask", "cont_mask"], "y_cond_continuous_mask", required=False)
    sample_id = _pick_first(pack, ["sample_id", "row_id", "id"], "sample_id", required=False)

    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if y_set is not None:
        y_set = np.asarray(y_set, dtype=np.float32)
        y_set = np.nan_to_num(y_set, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if y_disc is not None:
        y_disc = np.asarray(y_disc)
        if y_disc.ndim == 1:
            y_disc = y_disc.reshape(-1, 1)
    if y_disc_mask is None and y_disc is not None:
        y_disc_mask = np.ones_like(y_disc, dtype=np.float32)
    elif y_disc_mask is not None:
        y_disc_mask = np.asarray(y_disc_mask, dtype=np.float32)
        if y_disc_mask.ndim == 1:
            y_disc_mask = y_disc_mask.reshape(-1, 1)

    if y_cont is not None:
        y_cont = np.asarray(y_cont, dtype=np.float32)
        if y_cont.ndim == 1:
            y_cont = y_cont.reshape(-1, 1)
    if y_cont_mask is None and y_cont is not None:
        y_cont_mask = np.isfinite(y_cont).astype(np.float32)
    elif y_cont_mask is not None:
        y_cont_mask = np.asarray(y_cont_mask, dtype=np.float32)
        if y_cont_mask.ndim == 1:
            y_cont_mask = y_cont_mask.reshape(-1, 1)
    if y_cont is not None:
        y_cont = _sanitize_float_array(y_cont, y_cont_mask)

    if sample_id is None:
        sample_id = np.asarray([str(i) for i in range(x.shape[0])], dtype=object)
    else:
        sample_id = np.asarray(sample_id, dtype=object)

    return {
        "x": x,
        "y_set": y_set,
        "y_disc": y_disc,
        "y_disc_mask": y_disc_mask,
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


def _select_active_cont_heads(y_cont: Optional[np.ndarray], y_cont_mask: Optional[np.ndarray], cont_names: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[int], List[Dict[str, Any]]]:
    if y_cont is None:
        return None, None, [], [], []
    mask = np.ones_like(y_cont, dtype=np.float32) if y_cont_mask is None else np.asarray(y_cont_mask, dtype=np.float32)
    active_idx: List[int] = []
    dropped: List[Dict[str, Any]] = []
    for j in range(y_cont.shape[1]):
        valid_n = int(np.sum(mask[:, j] > 0.5))
        if valid_n > 0:
            active_idx.append(j)
        else:
            dropped.append({"head": cont_names[j] if j < len(cont_names) else f"cont_{j}", "reason": "no_valid_train_samples"})
    if not active_idx:
        return None, None, [], [], dropped
    return y_cont[:, active_idx], mask[:, active_idx], [cont_names[j] for j in active_idx], active_idx, dropped


def _select_active_disc_heads(y_disc: Optional[np.ndarray], y_disc_mask: Optional[np.ndarray], disc_names: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], List[int], List[Dict[str, Any]]]:
    if y_disc is None:
        return None, None, [], [], []
    mask = np.ones_like(y_disc, dtype=np.float32) if y_disc_mask is None else np.asarray(y_disc_mask, dtype=np.float32)
    active_idx: List[int] = []
    dropped: List[Dict[str, Any]] = []
    for j in range(y_disc.shape[1]):
        valid = mask[:, j] > 0.5
        valid_n = int(np.sum(valid))
        head_name = disc_names[j] if j < len(disc_names) else f"disc_{j}"
        if valid_n == 0:
            dropped.append({"head": head_name, "reason": "no_valid_train_samples"})
            continue
        uniq = np.unique(y_disc[valid, j])
        if uniq.shape[0] <= 1:
            dropped.append({"head": head_name, "reason": "single_class_train", "class_value": int(uniq[0]) if uniq.shape[0] == 1 else None})
            continue
        active_idx.append(j)
    if not active_idx:
        return None, None, [], [], dropped
    return y_disc[:, active_idx], mask[:, active_idx], [disc_names[j] for j in active_idx], active_idx, dropped


class Stage3MLPModel:
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (512, 256),
        activation: str = "relu",
        alpha: float = 1e-4,
        batch_size: int = 256,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 15,
        standardize: bool = True,
        random_state: int = 42,
    ):
        self.hidden_dims = tuple(hidden_dims)
        self.activation = str(activation)
        self.alpha = float(alpha)
        self.batch_size = int(batch_size)
        self.learning_rate_init = float(learning_rate_init)
        self.max_iter = int(max_iter)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.standardize = bool(standardize)
        self.random_state = int(random_state)

        self.scaler: Optional[StandardScaler] = None
        self.cont_models: List[Optional[MLPRegressor]] = []
        self.disc_models: List[Optional[MLPClassifier]] = []
        self.cont_names: List[str] = []
        self.disc_names: List[str] = []

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if not self.standardize:
            return X
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
        y_disc_mask: Optional[np.ndarray],
        disc_names: Sequence[str],
    ) -> Dict[str, Any]:
        Xf = self._transform(X, fit=True)
        self.cont_models = []
        self.disc_models = []
        self.cont_names = list(cont_names)
        self.disc_names = list(disc_names)

        fit_report: Dict[str, Any] = {
            "requested_early_stopping": self.early_stopping,
            "effective_early_stopping": self.early_stopping,
            "disabled_early_stopping_for": [],
            "dropped_or_skipped_heads": [],
        }

        if y_cont is not None:
            mask = np.ones_like(y_cont, dtype=np.float32) if y_cont_mask is None else y_cont_mask.astype(np.float32)
            for j in range(y_cont.shape[1]):
                valid = mask[:, j] > 0.5
                if np.sum(valid) == 0:
                    self.cont_models.append(None)
                    fit_report["dropped_or_skipped_heads"].append({"head": self.cont_names[j], "reason": "no_valid_train_samples"})
                    continue
                model = MLPRegressor(
                    hidden_layer_sizes=self.hidden_dims,
                    activation=self.activation,
                    solver="adam",
                    alpha=self.alpha,
                    batch_size=self.batch_size,
                    learning_rate="adaptive",
                    learning_rate_init=self.learning_rate_init,
                    max_iter=self.max_iter,
                    early_stopping=self.early_stopping,
                    validation_fraction=self.validation_fraction,
                    n_iter_no_change=self.n_iter_no_change,
                    random_state=self.random_state,
                )
                model.fit(Xf[valid], y_cont[valid, j])
                self.cont_models.append(model)

        if y_disc is not None:
            mask = np.ones_like(y_disc, dtype=np.float32) if y_disc_mask is None else y_disc_mask.astype(np.float32)
            sparse_info = []
            for j in range(y_disc.shape[1]):
                valid = mask[:, j] > 0.5
                if np.sum(valid) == 0:
                    self.disc_models.append(None)
                    fit_report["dropped_or_skipped_heads"].append({"head": self.disc_names[j], "reason": "no_valid_train_samples"})
                    continue
                yj = y_disc[valid, j]
                uniq = np.unique(yj)
                if uniq.shape[0] <= 1:
                    self.disc_models.append(None)
                    fit_report["dropped_or_skipped_heads"].append({"head": self.disc_names[j], "reason": "single_class_train", "class_value": int(uniq[0]) if uniq.shape[0] == 1 else None})
                    continue

                effective_early_stopping = self.early_stopping
                class_counts = {int(k): int(v) for k, v in zip(*np.unique(yj, return_counts=True))}
                min_class_count = min(class_counts.values()) if class_counts else 0
                if self.early_stopping and min_class_count < 2:
                    effective_early_stopping = False
                    fit_report["effective_early_stopping"] = False
                    fit_report["disabled_early_stopping_for"].append(self.disc_names[j])
                    sparse_info.append({"head": self.disc_names[j], "min_class_count": int(min_class_count)})

                model = MLPClassifier(
                    hidden_layer_sizes=self.hidden_dims,
                    activation=self.activation,
                    solver="adam",
                    alpha=self.alpha,
                    batch_size=self.batch_size,
                    learning_rate="adaptive",
                    learning_rate_init=self.learning_rate_init,
                    max_iter=self.max_iter,
                    early_stopping=effective_early_stopping,
                    validation_fraction=self.validation_fraction,
                    n_iter_no_change=self.n_iter_no_change,
                    random_state=self.random_state,
                )
                model.fit(Xf[valid], yj)
                self.disc_models.append(model)
            fit_report["sparse_discrete_heads"] = sparse_info

        return fit_report

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
                if m is None:
                    disc_preds.append(np.zeros((Xf.shape[0],), dtype=np.int64))
                    disc_probs.append(None)
                else:
                    disc_preds.append(np.asarray(m.predict(Xf)))
                    disc_probs.append(np.asarray(m.predict_proba(Xf), dtype=np.float32))
            out["y_disc_pred"] = np.stack(disc_preds, axis=1)
            out["y_disc_prob"] = disc_probs
        return out


def evaluate_disc_logloss(y_true: Optional[np.ndarray], disc_prob: Optional[List[Any]], disc_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    if y_true is None or disc_prob is None:
        return {}
    metrics: Dict[str, float] = {}
    vals: List[float] = []
    mask = np.ones_like(y_true, dtype=np.float32) if disc_mask is None else np.asarray(disc_mask, dtype=np.float32)

    for j in range(y_true.shape[1]):
        probs = disc_prob[j]
        valid = mask[:, j] > 0.5
        if probs is None or not np.any(valid):
            metrics[f"disc{j}_logloss"] = float("nan")
            continue
        try:
            yj = y_true[valid, j]
            pj = probs[valid]
            val = float(log_loss(yj, pj, labels=list(range(pj.shape[1]))))
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
    y_disc_mask: Optional[np.ndarray],
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
            if y_disc_mask is not None:
                out[f"mask_{name}"] = y_disc_mask[:, j]

    out.to_csv(path, index=False)


def parse_hidden_dims(text: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    return tuple(vals) if vals else (512, 256)


def _safe_read_meta(resolved_files: Mapping[str, str], key: str, sample_ids: np.ndarray) -> pd.DataFrame:
    if key in resolved_files:
        return pd.read_csv(resolved_files[key])
    return pd.DataFrame({"sample_id": sample_ids.astype(str)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified stage3 MLP predictor (NPZ mixed dataset version, commonized).")
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--mode_input_root", type=str, default="")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--train_mode", type=str, default="gold_only", choices=TRAIN_MODE_CHOICES)
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)
    parser.add_argument("--use_y_set", action="store_true")
    parser.add_argument("--hidden_dims", type=str, default="512,256")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate_init", type=float, default=1e-3)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--n_iter_no_change", type=int, default=15)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input_dir and not args.mode_input_root:
        args.input_dir = DEFAULT_INPUT_DIR
    if not args.run_dir:
        args.run_dir = DEFAULT_RUN_DIR
    if not args.project_root:
        args.project_root = DEFAULT_PROJECT_ROOT

    print(f"[Info] input_dir(default-ready) = {args.input_dir}")
    print(f"[Info] run_dir(default-ready)   = {args.run_dir}")

    resolved = resolve_input_paths(args, required=STAGE3_REQUIRED, optional=STAGE3_OPTIONAL)
    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    print(f"[Info] resolved_mode = {resolved.resolved_mode}")
    print(f"[Info] resolved_root = {resolved.resolved_root}")
    print(f"[Info] resolved_input_dir = {resolved.resolved_input_dir}")

    train_split = _load_stage3_split(Path(resolved.files["train_npz"]))
    val_split = _load_stage3_split(Path(resolved.files["val_npz"]))
    test_split = _load_stage3_split(Path(resolved.files["test_npz"]))
    schema = _load_schema(Path(resolved.files["schema"]))

    train_meta = _safe_read_meta(resolved.files, "train_meta", train_split["sample_id"])
    val_meta = _safe_read_meta(resolved.files, "val_meta", val_split["sample_id"])
    test_meta = _safe_read_meta(resolved.files, "test_meta", test_split["sample_id"])

    X_train = build_features(train_split["x"], train_split["y_set"], args.use_y_set)
    X_val = build_features(val_split["x"], val_split["y_set"], args.use_y_set)
    X_test = build_features(test_split["x"], test_split["y_set"], args.use_y_set)

    raw_n_cont = 0 if train_split["y_cont"] is None else int(train_split["y_cont"].shape[1])
    raw_n_disc = 0 if train_split["y_disc"] is None else int(train_split["y_disc"].shape[1])
    cont_names_raw = _get_cont_names(schema, raw_n_cont)
    disc_names_raw = _get_disc_names(schema, raw_n_disc)

    y_cont_train, y_cont_mask_train, cont_names, active_cont_idx, dropped_cont = _select_active_cont_heads(
        train_split["y_cont"], train_split["y_cont_mask"], cont_names_raw
    )
    y_cont_val = None if val_split["y_cont"] is None or not active_cont_idx else val_split["y_cont"][:, active_cont_idx]
    y_cont_test = None if test_split["y_cont"] is None or not active_cont_idx else test_split["y_cont"][:, active_cont_idx]
    y_cont_mask_val = None if val_split["y_cont_mask"] is None or not active_cont_idx else val_split["y_cont_mask"][:, active_cont_idx]
    y_cont_mask_test = None if test_split["y_cont_mask"] is None or not active_cont_idx else test_split["y_cont_mask"][:, active_cont_idx]

    y_disc_train, y_disc_mask_train, disc_names, active_disc_idx, dropped_disc = _select_active_disc_heads(
        train_split["y_disc"], train_split["y_disc_mask"], disc_names_raw
    )
    y_disc_val = None if val_split["y_disc"] is None or not active_disc_idx else val_split["y_disc"][:, active_disc_idx]
    y_disc_test = None if test_split["y_disc"] is None or not active_disc_idx else test_split["y_disc"][:, active_disc_idx]
    y_disc_mask_val = None if val_split["y_disc_mask"] is None or not active_disc_idx else val_split["y_disc_mask"][:, active_disc_idx]
    y_disc_mask_test = None if test_split["y_disc_mask"] is None or not active_disc_idx else test_split["y_disc_mask"][:, active_disc_idx]

    n_cont = 0 if y_cont_train is None else int(y_cont_train.shape[1])
    n_disc = 0 if y_disc_train is None else int(y_disc_train.shape[1])

    model = Stage3MLPModel(
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        activation=args.activation,
        alpha=args.alpha,
        batch_size=args.batch_size,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=args.early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        standardize=args.standardize,
        random_state=args.seed,
    )
    fit_report = model.fit(
        X=X_train,
        y_cont=y_cont_train,
        y_cont_mask=y_cont_mask_train,
        cont_names=cont_names,
        y_disc=y_disc_train,
        y_disc_mask=y_disc_mask_train,
        disc_names=disc_names,
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate_mixed_conditions(
        y_cont_true=y_cont_val,
        y_cont_pred=val_pred["y_cont_pred"],
        y_cont_mask=y_cont_mask_val,
        cont_target_names=cont_names,
        y_disc_true=y_disc_val,
        y_disc_pred=val_pred["y_disc_pred"],
        disc_target_names=disc_names,
        prefix="val",
    )
    test_metrics = evaluate_mixed_conditions(
        y_cont_true=y_cont_test,
        y_cont_pred=test_pred["y_cont_pred"],
        y_cont_mask=y_cont_mask_test,
        cont_target_names=cont_names,
        y_disc_true=y_disc_test,
        y_disc_pred=test_pred["y_disc_pred"],
        disc_target_names=disc_names,
        prefix="test",
    )
    if y_disc_val is not None:
        val_metrics.update({f"val_{k}": v for k, v in evaluate_disc_logloss(y_disc_val, val_pred["y_disc_prob"], y_disc_mask_val).items()})
        test_metrics.update({f"test_{k}": v for k, v in evaluate_disc_logloss(y_disc_test, test_pred["y_disc_prob"], y_disc_mask_test).items()})

    save_predictions_csv(
        run_dir / "val_predictions.csv",
        meta_df=val_meta,
        sample_ids=val_split["sample_id"],
        cont_names=cont_names,
        disc_names=disc_names,
        y_cont_true=y_cont_val,
        y_cont_pred=val_pred["y_cont_pred"],
        y_cont_mask=y_cont_mask_val,
        y_disc_true=y_disc_val,
        y_disc_pred=val_pred["y_disc_pred"],
        y_disc_mask=y_disc_mask_val,
    )
    save_predictions_csv(
        run_dir / "test_predictions.csv",
        meta_df=test_meta,
        sample_ids=test_split["sample_id"],
        cont_names=cont_names,
        disc_names=disc_names,
        y_cont_true=y_cont_test,
        y_cont_pred=test_pred["y_cont_pred"],
        y_cont_mask=y_cont_mask_test,
        y_disc_true=y_disc_test,
        y_disc_pred=test_pred["y_disc_pred"],
        y_disc_mask=y_disc_mask_test,
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
                "fit_report": fit_report,
                "active_cont_indices": active_cont_idx,
                "active_disc_indices": active_disc_idx,
            },
            f,
        )

    train_log = [{
        "stage": "fit_once_mlp",
        "n_train": int(X_train.shape[0]),
        "input_dim": int(X_train.shape[1]),
        "raw_n_cont": int(raw_n_cont),
        "raw_n_disc": int(raw_n_disc),
        "n_cont": int(n_cont),
        "n_disc": int(n_disc),
        "hidden_dims": list(parse_hidden_dims(args.hidden_dims)),
        "dropped_cont_heads": dropped_cont,
        "dropped_disc_heads": dropped_disc,
        **to_builtin(fit_report),
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
            "raw_n_cont": int(raw_n_cont),
            "raw_n_disc": int(raw_n_disc),
            "n_cont": int(n_cont),
            "n_disc": int(n_disc),
            "active_cont_indices": active_cont_idx,
            "active_disc_indices": active_disc_idx,
            "dropped_cont_heads": dropped_cont,
            "dropped_disc_heads": dropped_disc,
            "train_keys": train_split["pack_keys"],
            "val_keys": val_split["pack_keys"],
            "test_keys": test_split["pack_keys"],
            "default_project_root": DEFAULT_PROJECT_ROOT,
            "default_input_dir": DEFAULT_INPUT_DIR,
            "default_run_dir": DEFAULT_RUN_DIR,
            "resolved_input_exists": bool(Path(resolved.resolved_input_dir).exists()),
        },
        "training": to_builtin(fit_report),
        "val_metrics": to_builtin(val_metrics),
        "test_metrics": to_builtin(test_metrics),
    }
    write_json(run_dir / "metrics.json", to_builtin(summary))
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
