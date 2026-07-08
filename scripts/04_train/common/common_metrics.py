from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def prefix_metrics(metrics: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


# =========================
# stage2: multi-label / set prediction
# =========================
def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "samples_jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "mean_true_labels": float(np.mean(y_true.sum(axis=1))),
        "mean_pred_labels": float(np.mean(y_pred.sum(axis=1))),
    }



def reward_from_sets_numpy(
    pred_y: np.ndarray,
    true_y: np.ndarray,
    exact_bonus: float = 0.25,
    length_penalty: float = 0.02,
) -> np.ndarray:
    pred_y = np.asarray(pred_y).astype(np.float32)
    true_y = np.asarray(true_y).astype(np.float32)
    inter = (pred_y * true_y).sum(axis=1)
    pred_cnt = pred_y.sum(axis=1)
    true_cnt = true_y.sum(axis=1)
    f1 = (2.0 * inter) / np.clip(pred_cnt + true_cnt, 1.0, None)
    exact = (pred_y == true_y).all(axis=1).astype(np.float32)
    len_gap = np.abs(pred_cnt - true_cnt)
    reward = f1 + exact_bonus * exact - length_penalty * len_gap
    return np.clip(reward, 1e-4, None)



def exact_hit_at_k(grouped_rows: Sequence[Sequence[Mapping[str, Any]]], k: int) -> float:
    hits: List[float] = []
    for rows in grouped_rows:
        cur = list(rows[:k])
        hits.append(float(any(int(r.get("exact_match", 0)) == 1 for r in cur)))
    return float(np.mean(hits)) if hits else math.nan



def oracle_reward_mean(grouped_rows: Sequence[Sequence[Mapping[str, Any]]]) -> float:
    vals: List[float] = []
    for rows in grouped_rows:
        if not rows:
            vals.append(0.0)
        else:
            vals.append(max(float(r.get("reward", 0.0)) for r in rows))
    return float(np.mean(vals)) if vals else math.nan



def grouped_top1_predictions(grouped_rows: Sequence[Sequence[Mapping[str, Any]]], fallback_dim: int) -> np.ndarray:
    zero = np.zeros(fallback_dim, dtype=np.int32)
    preds: List[np.ndarray] = []
    for rows in grouped_rows:
        if not rows:
            preds.append(zero.copy())
        else:
            preds.append(np.asarray(rows[0]["cand_vec"]).astype(np.int32))
    return np.vstack(preds).astype(np.int32)



def evaluate_candidate_groups(
    grouped_rows: Sequence[Sequence[Mapping[str, Any]]],
    y_true: np.ndarray,
    fallback_dim: int,
    prefix: str,
    topk_values: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = grouped_top1_predictions(grouped_rows, fallback_dim=fallback_dim)
    out = prefix_metrics(evaluate_multilabel(y_true, y_pred), prefix)
    out[f"{prefix}_exact_hit@1"] = exact_hit_at_k(grouped_rows, 1)
    out[f"{prefix}_oracle_reward_mean"] = oracle_reward_mean(grouped_rows)
    out[f"{prefix}_mean_candidates"] = float(np.mean([len(rows) for rows in grouped_rows])) if grouped_rows else math.nan
    for k in topk_values:
        out[f"{prefix}_exact_hit@{int(k)}"] = exact_hit_at_k(grouped_rows, int(k))
    return out


# =========================
# stage3: regression / mixed condition prediction
# =========================
def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    diff = y_pred - y_true
    mae_per_dim = np.mean(np.abs(diff), axis=0)
    rmse_per_dim = np.sqrt(np.mean(np.square(diff), axis=0))

    metrics: Dict[str, float] = {
        "mae_mean": float(np.mean(mae_per_dim)),
        "rmse_mean": float(np.mean(rmse_per_dim)),
    }
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        metrics[f"dim{j}_mae"] = float(mean_absolute_error(yt, yp))
        metrics[f"dim{j}_rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
        try:
            metrics[f"dim{j}_r2"] = float(r2_score(yt, yp))
        except Exception:
            metrics[f"dim{j}_r2"] = math.nan
    if prefix:
        return prefix_metrics(metrics, prefix)
    return metrics



def evaluate_regression_with_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    prefix: str = "",
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if mask is None:
        mask = np.ones_like(y_true, dtype=np.float32)
    else:
        mask = np.asarray(mask, dtype=np.float32)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        mask = mask.reshape(-1, 1)

    names = list(target_names) if target_names is not None else [f"dim{j}" for j in range(y_true.shape[1])]
    metrics: Dict[str, float] = {}
    maes: List[float] = []
    rmses: List[float] = []

    for j in range(y_true.shape[1]):
        valid = mask[:, j] > 0.5
        name = names[j] if j < len(names) else f"dim{j}"
        if not np.any(valid):
            metrics[f"{name}_mae"] = math.nan
            metrics[f"{name}_rmse"] = math.nan
            metrics[f"{name}_r2"] = math.nan
            continue
        yt = y_true[valid, j]
        yp = y_pred[valid, j]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        maes.append(mae)
        rmses.append(rmse)
        metrics[f"{name}_mae"] = mae
        metrics[f"{name}_rmse"] = rmse
        try:
            metrics[f"{name}_r2"] = float(r2_score(yt, yp))
        except Exception:
            metrics[f"{name}_r2"] = math.nan

    metrics["mae_mean"] = float(np.mean(maes)) if maes else math.nan
    metrics["rmse_mean"] = float(np.mean(rmses)) if rmses else math.nan
    if prefix:
        return prefix_metrics(metrics, prefix)
    return metrics



def evaluate_multiclass(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if prefix:
        return prefix_metrics(metrics, prefix)
    return metrics



def evaluate_mixed_conditions(
    y_cont_true: Optional[np.ndarray] = None,
    y_cont_pred: Optional[np.ndarray] = None,
    y_cont_mask: Optional[np.ndarray] = None,
    cont_target_names: Optional[Sequence[str]] = None,
    y_disc_true: Optional[np.ndarray] = None,
    y_disc_pred: Optional[np.ndarray] = None,
    disc_target_names: Optional[Sequence[str]] = None,
    prefix: str = "",
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    if y_cont_true is not None and y_cont_pred is not None:
        metrics.update(
            evaluate_regression_with_mask(
                y_true=y_cont_true,
                y_pred=y_cont_pred,
                mask=y_cont_mask,
                target_names=cont_target_names,
            )
        )

    if y_disc_true is not None and y_disc_pred is not None:
        y_disc_true = np.asarray(y_disc_true)
        y_disc_pred = np.asarray(y_disc_pred)
        if y_disc_true.ndim == 1:
            metrics.update({f"disc_{k}": v for k, v in evaluate_multiclass(y_disc_true, y_disc_pred).items()})
        else:
            names = list(disc_target_names) if disc_target_names is not None else [f"disc{j}" for j in range(y_disc_true.shape[1])]
            accs: List[float] = []
            f1s: List[float] = []
            for j in range(y_disc_true.shape[1]):
                cur = evaluate_multiclass(y_disc_true[:, j], y_disc_pred[:, j])
                name = names[j] if j < len(names) else f"disc{j}"
                metrics[f"{name}_accuracy"] = cur["accuracy"]
                metrics[f"{name}_macro_f1"] = cur["macro_f1"]
                accs.append(cur["accuracy"])
                f1s.append(cur["macro_f1"])
            metrics["disc_accuracy_mean"] = float(np.mean(accs)) if accs else math.nan
            metrics["disc_macro_f1_mean"] = float(np.mean(f1s)) if f1s else math.nan

    if prefix:
        return prefix_metrics(metrics, prefix)
    return metrics


# =========================
# compare helpers
# =========================
def build_compare_record(
    *,
    task: str,
    model_family: str,
    model_name: str,
    input_mode: str,
    train_mode: str,
    run_dir: str,
    rerank_enabled: bool = False,
    extra: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "task": task,
        "model_family": model_family,
        "model_name": model_name,
        "input_mode": input_mode,
        "train_mode": train_mode,
        "run_dir": run_dir,
        "rerank_enabled": bool(rerank_enabled),
    }
    if extra:
        row.update(dict(extra))
    if metrics:
        row.update(dict(metrics))
    return row
