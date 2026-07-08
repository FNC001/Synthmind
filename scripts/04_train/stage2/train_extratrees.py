#!/usr/bin/env python3
from __future__ import annotations


# ============================================================
# Constant-label warning handling
# Added by pipeline maintenance patch.
#
# In Stage2 precursor multilabel prediction, many precursor labels
# can be all-zero in the current training split. sklearn's
# OneVsRestClassifier warns for every such label:
#   "Label not xxx is present in all training examples."
#
# This is expected for sparse precursor spaces and does not mean
# the run failed. We suppress only this specific warning pattern.
# ============================================================
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Label not .* is present in all training examples\.",
    category=UserWarning,
    module=r"sklearn\.multiclass"
)

"""
Commonized stage2 ExtraTrees training script for SynPred.

Features:
- compatible with common stage2 dataset layouts
- supports NPZ keys: x/features/X and y_multi_hot/y/labels/targets
- automatic validation threshold search
- exports metrics, predictions, model, candidates, and candidate metrics
- candidate format aligned with stage2 AR/CVAE/baseline_linear scripts
"""


import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODE_INPUT_ROOT = "/Users/wyc/SynPred/data/interim/generative/stage2_setpred_dataset/hybrid"
DEFAULT_RUN_DIR = "/Users/wyc/SynPred/runs/stage2/tree_ensemble_commonized_v1"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


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


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    exact_bonus: float,
    length_penalty: float,
) -> np.ndarray:
    pred_y = pred_y.astype(np.float32)
    true_y = true_y.astype(np.float32)
    inter = (pred_y * true_y).sum(axis=1)
    pred_cnt = pred_y.sum(axis=1)
    true_cnt = true_y.sum(axis=1)
    f1 = (2.0 * inter) / np.clip(pred_cnt + true_cnt, 1.0, None)
    exact = (pred_y == true_y).all(axis=1).astype(np.float32)
    len_gap = np.abs(pred_cnt - true_cnt)
    reward = f1 + float(exact_bonus) * exact - float(length_penalty) * len_gap
    return np.clip(reward, 1e-4, None)


def _first_existing(candidates: List[Path], what: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"未找到 {what}，候选路径为：\n" + "\n".join(str(x) for x in candidates)
    )


def resolve_input_paths(
    args: argparse.Namespace,
    required: Sequence[str],
    optional: Optional[Sequence[str]] = None,
):
    optional = list(optional or [])

    input_dir_str = str(getattr(args, "input_dir", "")).strip()
    mode_root_str = str(getattr(args, "mode_input_root", "")).strip()

    if input_dir_str:
        input_dir = Path(input_dir_str).expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"--input_dir 不存在: {input_dir}")
        resolved_mode = "legacy_input_dir"
        resolved_root = str(input_dir.parent)

    else:
        root = Path(mode_root_str).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--mode_input_root 不存在: {root}")

        train_mode = getattr(args, "train_mode", "gold_only")
        mode_candidates = [
            root / train_mode,
            root / "gold_only",
            root / "relaxed_only",
        ]
        input_dir = _first_existing(mode_candidates, f"train_mode={train_mode} 对应的数据目录")
        resolved_mode = train_mode
        resolved_root = str(root)

    missing = []
    files: Dict[str, str] = {}
    for name in list(required) + list(optional):
        p = input_dir / name
        if p.exists():
            files[name] = str(p)
        elif name in required:
            missing.append(str(p))

    if missing:
        raise FileNotFoundError("输入目录缺少必需文件：\n" + "\n".join(missing))

    class _Resolved:
        def __init__(self, files, input_dir, resolved_mode, resolved_root):
            self.files = files
            self.resolved_input_dir = str(input_dir)
            self.resolved_mode = resolved_mode
            self.resolved_root = resolved_root

    return _Resolved(files, input_dir, resolved_mode, resolved_root)


REQUIRED_FILES = [
    "train.npz",
    "val.npz",
    "test.npz",
    "train_meta.csv",
    "val_meta.csv",
    "test_meta.csv",
    "summary.json",
]

OPTIONAL_FILES = [
    "precursor_names.json",
    "label_cols.json",
    "label_names.json",
    "schema.json",
]


def _pick_first(pack: Dict[str, np.ndarray], names: Sequence[str], kind: str) -> np.ndarray:
    for n in names:
        if n in pack:
            return pack[n]
    raise KeyError(f"在 NPZ 中找不到 {kind}，候选键：{list(names)}，实际键：{list(pack.keys())}")


def _load_summary_schema(summary_path: str, files: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    summary = load_json(summary_path)
    schema = summary.get("schema", summary) if isinstance(summary, dict) else {}
    if "schema.json" in files:
        try:
            schema = load_json(files["schema.json"])
        except Exception:
            pass
    return summary, schema


def _read_json_if_exists(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    return load_json(str(p))


def _resolve_label_names(files: Dict[str, str], schema: Dict[str, Any], y_dim: int) -> List[str]:
    candidates: List[Optional[Any]] = [
        _read_json_if_exists(files.get("precursor_names.json")),
        _read_json_if_exists(files.get("label_cols.json")),
        _read_json_if_exists(files.get("label_names.json")),
    ]

    for key in [
        "precursor_names",
        "label_names",
        "label_cols",
        "target_names",
        "y_names",
    ]:
        if key in schema:
            candidates.append(schema[key])

    for key in [
        "precursor_names_path",
        "label_names_path",
        "label_cols_path",
        "target_names_path",
    ]:
        if key in schema:
            candidates.append(_read_json_if_exists(schema[key]))

    for item in candidates:
        if item is None:
            continue

        if isinstance(item, dict):
            if "names" in item and isinstance(item["names"], list):
                item = item["names"]
            elif "labels" in item and isinstance(item["labels"], list):
                item = item["labels"]

        if isinstance(item, list) and len(item) == y_dim:
            return [str(x) for x in item]

    return [f"precursor_{i}" for i in range(y_dim)]


def _as_2d_float(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"特征数组必须是二维，当前 shape={a.shape}")
    return a.astype(np.float32)


def _as_2d_int_binary(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"标签数组必须是二维 multi-hot，当前 shape={a.shape}")
    if a.dtype.kind in {"b", "i", "u", "f"}:
        return (a > 0).astype(np.int32)
    raise ValueError(f"标签数组类型不支持：{a.dtype}")


def load_stage2_split_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    pack = load_npz(npz_path)
    x = _pick_first(pack, ["x", "features", "X"], kind="features")
    y = _pick_first(pack, ["y_multi_hot", "y", "labels", "targets"], kind="labels")
    return _as_2d_float(x), _as_2d_int_binary(y)


def _label_statistics(y_train: np.ndarray) -> Dict[str, Any]:
    label_pos = y_train.sum(axis=0)
    constant_zero = np.where(label_pos == 0)[0].tolist()
    constant_one = np.where(label_pos == y_train.shape[0])[0].tolist()
    row_cnt = y_train.sum(axis=1).astype(np.float32)
    return {
        "mean_true_labels": float(np.mean(row_cnt)),
        "median_true_labels": float(np.median(row_cnt)),
        "max_true_labels": float(np.max(row_cnt)),
        "min_true_labels": float(np.min(row_cnt)),
        "constant_zero_labels": int(len(constant_zero)),
        "constant_one_labels": int(len(constant_one)),
    }


def build_model(args: argparse.Namespace) -> Pipeline:
    base = ExtraTreesClassifier(
        n_estimators=int(args.n_estimators),
        max_depth=None if int(args.max_depth) <= 0 else int(args.max_depth),
        min_samples_split=int(args.min_samples_split),
        min_samples_leaf=int(args.min_samples_leaf),
        max_features=args.max_features,
        bootstrap=bool(args.bootstrap),
        class_weight="balanced" if bool(args.class_weight_balanced) else None,
        random_state=int(args.seed),
        n_jobs=int(args.n_jobs),
        verbose=0,
    )
    clf = OneVsRestClassifier(base, n_jobs=int(args.ovr_n_jobs))

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if bool(args.standardize):
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def safe_predict_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    """
    Return score matrix with shape [n_samples, n_labels].
    For ExtraTrees OVR, predict_proba is preferred.
    """
    clf = model.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(x)

        if isinstance(scores, list):
            cols = []
            for s in scores:
                s = np.asarray(s)
                if s.ndim == 2 and s.shape[1] >= 2:
                    cols.append(s[:, 1])
                else:
                    cols.append(s.reshape(-1))
            return np.stack(cols, axis=1).astype(np.float32)

        scores = np.asarray(scores)
        if scores.ndim == 3:
            return scores[:, :, 1].astype(np.float32)
        if scores.ndim == 2:
            return scores.astype(np.float32)

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(x)
        return np.asarray(scores, dtype=np.float32)

    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float32)


def apply_threshold(
    scores: np.ndarray,
    threshold: float,
    force_non_empty: bool = True,
) -> np.ndarray:
    pred = (scores >= float(threshold)).astype(np.int32)

    if force_non_empty:
        empty = pred.sum(axis=1) == 0
        if np.any(empty):
            argmax = np.argmax(scores[empty], axis=1)
            pred[empty] = 0
            pred[np.where(empty)[0], argmax] = 1

    return pred


def search_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_name: str,
    threshold_grid: Sequence[float],
    force_non_empty: bool = True,
) -> Tuple[float, Dict[str, float]]:
    best_thr = float(threshold_grid[0])
    best_metrics: Optional[Dict[str, float]] = None
    best_metric = -1e18

    for thr in threshold_grid:
        pred = apply_threshold(scores, threshold=float(thr), force_non_empty=force_non_empty)
        metrics = evaluate_multilabel(y_true.astype(int), pred.astype(int))
        cur = float(metrics.get(metric_name, float("-inf")))

        if cur > best_metric:
            best_metric = cur
            best_thr = float(thr)
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("threshold search failed")

    return best_thr, best_metrics


def multihot_to_label_lists(y: np.ndarray, label_names: List[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for i in range(y.shape[0]):
        idx = np.where(y[i] > 0)[0].tolist()
        out.append([label_names[j] for j in idx if 0 <= j < len(label_names)])
    return out


def save_prediction_csv(
    path: Path,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    label_names: List[str],
) -> None:
    out = meta_df.copy()
    out["sample_index"] = np.arange(len(out), dtype=int)
    out["true_labels"] = [
        json.dumps(x, ensure_ascii=False)
        for x in multihot_to_label_lists(y_true, label_names)
    ]
    out["pred_labels"] = [
        json.dumps(x, ensure_ascii=False)
        for x in multihot_to_label_lists(y_pred, label_names)
    ]
    out["n_true_labels"] = y_true.sum(axis=1)
    out["n_pred_labels"] = y_pred.sum(axis=1)
    out["score_max"] = scores.max(axis=1)
    out["score_mean"] = scores.mean(axis=1)
    out.to_csv(path, index=False)


def _make_candidate_base_row(meta_df: pd.DataFrame, sample_idx: int) -> Dict[str, Any]:
    if sample_idx < len(meta_df):
        base = meta_df.iloc[sample_idx].to_dict()
    else:
        base = {}

    base["sample_index"] = int(sample_idx)

    if "candidate_group_id" not in base:
        if "sample_id" in base:
            base["candidate_group_id"] = str(base["sample_id"])
        elif "material_id" in base:
            base["candidate_group_id"] = str(base["material_id"])
        elif "target_id" in base:
            base["candidate_group_id"] = str(base["target_id"])
        elif "mp_id" in base:
            base["candidate_group_id"] = str(base["mp_id"])
        else:
            base["candidate_group_id"] = f"sample_{sample_idx}"

    return base


def _candidate_score(scores_row: np.ndarray, cand_vec: np.ndarray) -> float:
    idx = np.where(cand_vec > 0)[0]
    if len(idx) == 0:
        return float("-inf")
    # mean positive score is more stable across different cardinalities than raw sum
    return float(np.mean(scores_row[idx]))


def build_candidate_rows_for_split(
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold_pred: np.ndarray,
    label_names: List[str],
    candidate_cardinalities: Sequence[int],
    exact_bonus: float,
    length_penalty: float,
    topn: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_samples, n_labels = scores.shape

    cardinalities = sorted(set(int(k) for k in candidate_cardinalities if int(k) > 0))
    if not cardinalities:
        cardinalities = [1, 2, 3, 4, 5]

    for i in range(n_samples):
        base = _make_candidate_base_row(meta_df, i)
        sample_scores = scores[i]
        true_i = y_true[i].astype(np.int32)

        cand_map: Dict[Tuple[int, ...], Dict[str, Any]] = {}

        def _add_candidate(cand_vec: np.ndarray, source: str) -> None:
            cand_vec = cand_vec.astype(np.int32)
            key = tuple(np.where(cand_vec > 0)[0].tolist())
            if not key:
                return

            score_val = _candidate_score(sample_scores, cand_vec)
            oracle_reward = float(
                reward_from_sets_numpy(
                    cand_vec[None, :],
                    true_i[None, :],
                    exact_bonus=exact_bonus,
                    length_penalty=length_penalty,
                )[0]
            )
            exact_match = int(np.all(cand_vec == true_i))

            item = {
                "source": source,
                "score": score_val,
                "oracle_reward": oracle_reward,
                "exact_match": exact_match,
                "cand_len": int(cand_vec.sum()),
                "cand_vec": cand_vec,
                "y_true": true_i,
            }

            cur = cand_map.get(key)
            if cur is None or item["score"] > cur["score"]:
                cand_map[key] = item

        # Candidate 1: threshold prediction, aligned with pred_val/pred_test.
        _add_candidate(threshold_pred[i].astype(np.int32), "threshold")

        # Candidates 2+: fixed cardinality top-k sets.
        order = np.argsort(-sample_scores)
        for k in cardinalities:
            kk = min(int(k), n_labels)
            cand = np.zeros(n_labels, dtype=np.int32)
            cand[order[:kk]] = 1
            _add_candidate(cand, f"top{k}")

        ranked = list(cand_map.values())

        # Rank threshold candidate first so top1 matches pred_val/pred_test.
        # Other candidates are ranked by model score.
        def _rank_key(z: Dict[str, Any]) -> Tuple[int, float]:
            is_threshold = 1 if str(z.get("source", "")) == "threshold" else 0
            return (is_threshold, float(z.get("score", 0.0)))

        ranked.sort(key=_rank_key, reverse=True)

        for rank, cand_row in enumerate(ranked[:topn], start=1):
            item = dict(base)
            item["rank"] = int(rank)
            item["source"] = cand_row["source"]
            item["score"] = float(cand_row["score"])
            item["oracle_reward"] = float(cand_row["oracle_reward"])
            item["exact_match"] = int(cand_row["exact_match"])
            item["cand_len"] = int(cand_row["cand_len"])
            item["pred_labels"] = json.dumps(
                multihot_to_label_lists(cand_row["cand_vec"][None, :], label_names)[0],
                ensure_ascii=False,
            )
            item["true_labels"] = json.dumps(
                multihot_to_label_lists(cand_row["y_true"][None, :], label_names)[0],
                ensure_ascii=False,
            )
            rows.append(item)

    return rows


def save_candidates_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def _candidate_topk_metrics_from_csv(
    candidates_csv: Path,
    topks: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    if not candidates_csv.exists():
        return {"error": f"candidate file not found: {str(candidates_csv)}"}

    df = pd.read_csv(candidates_csv)
    if df.empty:
        return {"error": f"empty candidate file: {str(candidates_csv)}"}

    if "sample_index" in df.columns:
        group_col = "sample_index"
    elif "candidate_group_id" in df.columns:
        group_col = "candidate_group_id"
    elif "sample_id" in df.columns:
        group_col = "sample_id"
    else:
        group_col = None

    metrics: Dict[str, Any] = {
        "file": str(candidates_csv),
        "n_rows": int(len(df)),
    }

    if group_col is None:
        metrics["error"] = "No sample_index, candidate_group_id, or sample_id column found."
        return metrics

    def _loads_set(x: Any) -> set:
        if pd.isna(x):
            return set()
        try:
            obj = json.loads(x)
            if isinstance(obj, list):
                return set(str(v) for v in obj)
        except Exception:
            pass
        return set()

    def _f1(pred: set, true: set) -> float:
        if len(pred) == 0 and len(true) == 0:
            return 1.0
        denom = len(pred) + len(true)
        if denom == 0:
            return 0.0
        return 2.0 * len(pred & true) / denom

    def _jaccard(pred: set, true: set) -> float:
        union = pred | true
        if not union:
            return 1.0
        return len(pred & true) / len(union)

    grouped = df.sort_values([group_col, "rank"]).groupby(group_col, sort=False)

    metrics["group_col"] = group_col
    metrics["n_samples"] = int(len(grouped))
    metrics["mean_candidates_per_sample"] = float(grouped.size().mean())

    unique_counts: List[int] = []
    exact_by_k: Dict[int, List[float]] = {k: [] for k in topks}
    best_f1_by_k: Dict[int, List[float]] = {k: [] for k in topks}
    best_jacc_by_k: Dict[int, List[float]] = {k: [] for k in topks}

    for _, g in grouped:
        pred_sets = [_loads_set(x) for x in g["pred_labels"].tolist()]
        true_set = _loads_set(g["true_labels"].iloc[0])

        unique_counts.append(len({tuple(sorted(s)) for s in pred_sets}))

        for k in topks:
            cur = pred_sets[:k]
            exact_by_k[k].append(float(any(s == true_set for s in cur)))
            best_f1_by_k[k].append(max((_f1(s, true_set) for s in cur), default=0.0))
            best_jacc_by_k[k].append(max((_jaccard(s, true_set) for s in cur), default=0.0))

    metrics["mean_unique_candidates_per_sample"] = (
        float(np.mean(unique_counts)) if unique_counts else 0.0
    )

    for k in topks:
        metrics[f"top{k}_exact_match"] = float(np.mean(exact_by_k[k])) if exact_by_k[k] else 0.0
        metrics[f"top{k}_best_f1"] = float(np.mean(best_f1_by_k[k])) if best_f1_by_k[k] else 0.0
        metrics[f"top{k}_best_jaccard"] = float(np.mean(best_jacc_by_k[k])) if best_jacc_by_k[k] else 0.0

    return metrics


def _parse_float_list(s: str) -> List[float]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def _parse_int_list(s: str) -> List[int]:
    vals = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stage2 ExtraTrees multilabel baseline.")
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--input_mode", type=str, default="hybrid")
    parser.add_argument("--mode_input_root", type=str, default=DEFAULT_MODE_INPUT_ROOT)
    parser.add_argument(
        "--train_mode",
        type=str,
        default="gold_only",
        choices=["relaxed_only", "gold_only", "curriculum", "curriculum_phase1", "curriculum_phase2"],
    )
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR)

    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--max_depth", type=int, default=0)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--class_weight_balanced", action="store_true")

    parser.add_argument("--metric_name", type=str, default="samples_f1")
    parser.add_argument(
        "--threshold_grid",
        type=str,
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
    )
    parser.add_argument("--force_non_empty", action="store_true", default=True)
    parser.add_argument("--no_force_non_empty", action="store_true")

    parser.add_argument("--save_candidates", action="store_true", default=True)
    parser.add_argument("--no_save_candidates", action="store_true")
    parser.add_argument("--candidate_cardinalities", type=str, default="1,2,3,4,5")
    parser.add_argument("--save_topn_candidates", type=int, default=10)
    parser.add_argument("--topk_values", type=str, default="1,3,5,10")
    parser.add_argument("--exact_bonus", type=float, default=0.25)
    parser.add_argument("--length_penalty", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--ovr_n_jobs", type=int, default=1)

    args = parser.parse_args()

    if args.no_force_non_empty:
        args.force_non_empty = False
    if args.no_save_candidates:
        args.save_candidates = False

    np.random.seed(int(args.seed))

    resolved = resolve_input_paths(args, required=REQUIRED_FILES, optional=OPTIONAL_FILES)
    run_dir = Path(args.run_dir).expanduser().resolve()
    ensure_dir(run_dir)

    print(f"[Info] resolved_mode = {resolved.resolved_mode}")
    print(f"[Info] resolved_root = {resolved.resolved_root}")
    print(f"[Info] resolved_input_dir = {resolved.resolved_input_dir}")

    files = resolved.files
    _, schema = _load_summary_schema(files["summary.json"], files)

    x_train, y_train = load_stage2_split_npz(files["train.npz"])
    x_val, y_val = load_stage2_split_npz(files["val.npz"])
    x_test, y_test = load_stage2_split_npz(files["test.npz"])

    train_meta = pd.read_csv(files["train_meta.csv"])
    val_meta = pd.read_csv(files["val_meta.csv"])
    test_meta = pd.read_csv(files["test_meta.csv"])

    if x_train.shape[0] == 0:
        raise ValueError("训练集为空。")
    if y_train.shape[1] == 0:
        raise ValueError("标签维度为 0。")

    label_stats = _label_statistics(y_train)
    if label_stats["constant_zero_labels"] or label_stats["constant_one_labels"]:
        print(
            "[Warn] Detected constant labels in training set: "
            f"all_zero={label_stats['constant_zero_labels']}, "
            f"all_one={label_stats['constant_one_labels']}."
        )

    label_names = _resolve_label_names(files, schema, y_train.shape[1])
    threshold_grid = _parse_float_list(str(args.threshold_grid))
    if not threshold_grid:
        threshold_grid = [0.5]

    model = build_model(args)

    print("[Info] fitting ExtraTrees OVR model ...")
    model.fit(x_train, y_train)

    val_scores = safe_predict_scores(model, x_val)
    best_thr, val_metrics = search_best_threshold(
        y_true=y_val,
        scores=val_scores,
        metric_name=str(args.metric_name),
        threshold_grid=threshold_grid,
        force_non_empty=bool(args.force_non_empty),
    )
    val_pred = apply_threshold(
        val_scores,
        threshold=best_thr,
        force_non_empty=bool(args.force_non_empty),
    )

    test_scores = safe_predict_scores(model, x_test)
    test_pred = apply_threshold(
        test_scores,
        threshold=best_thr,
        force_non_empty=bool(args.force_non_empty),
    )
    test_metrics = evaluate_multilabel(y_test.astype(int), test_pred.astype(int))

    save_prediction_csv(
        run_dir / "pred_val.csv",
        val_meta,
        y_val.astype(int),
        val_pred.astype(int),
        val_scores,
        label_names,
    )
    save_prediction_csv(
        run_dir / "pred_test.csv",
        test_meta,
        y_test.astype(int),
        test_pred.astype(int),
        test_scores,
        label_names,
    )

    with open(run_dir / "best_model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "threshold": float(best_thr),
                "label_names": label_names,
                "config": vars(args),
                "resolved_input_dir": resolved.resolved_input_dir,
            },
            f,
        )

    train_log = {
        "status": "finished",
        "best_threshold": float(best_thr),
        "threshold_grid": threshold_grid,
        "metric_name": str(args.metric_name),
        "val_metric": float(val_metrics.get(str(args.metric_name), math.nan)),
    }
    write_json(run_dir / "train_log.json", train_log)

    candidate_metrics: Dict[str, Any] = {}

    if bool(args.save_candidates):
        candidate_cardinalities = _parse_int_list(str(args.candidate_cardinalities))
        topk_values = _parse_int_list(str(args.topk_values))

        print("[Info] building ExtraTrees candidate pools for val/test ...")

        val_candidate_rows = build_candidate_rows_for_split(
            meta_df=val_meta,
            y_true=y_val.astype(int),
            scores=val_scores,
            threshold_pred=val_pred.astype(int),
            label_names=label_names,
            candidate_cardinalities=candidate_cardinalities,
            exact_bonus=float(args.exact_bonus),
            length_penalty=float(args.length_penalty),
            topn=int(args.save_topn_candidates),
        )
        test_candidate_rows = build_candidate_rows_for_split(
            meta_df=test_meta,
            y_true=y_test.astype(int),
            scores=test_scores,
            threshold_pred=test_pred.astype(int),
            label_names=label_names,
            candidate_cardinalities=candidate_cardinalities,
            exact_bonus=float(args.exact_bonus),
            length_penalty=float(args.length_penalty),
            topn=int(args.save_topn_candidates),
        )

        val_candidates_path = run_dir / "val_candidates.csv"
        test_candidates_path = run_dir / "test_candidates.csv"

        save_candidates_csv(val_candidates_path, val_candidate_rows)
        save_candidates_csv(test_candidates_path, test_candidate_rows)

        candidate_metrics = {
            "val": _candidate_topk_metrics_from_csv(val_candidates_path, topks=topk_values),
            "test": _candidate_topk_metrics_from_csv(test_candidates_path, topks=topk_values),
            "ranking_note": (
                "Rank 1 is the threshold prediction corresponding to pred_val/pred_test. "
                "Top-k cardinality candidates are ranked afterward by model score. "
                "oracle_reward and exact_match are saved only for offline diagnostics."
            ),
        }

        write_json(run_dir / "candidate_metrics.json", candidate_metrics)

    summary = {
        "config": vars(args),
        "resolved_mode": resolved.resolved_mode,
        "resolved_root": resolved.resolved_root,
        "resolved_input_dir": resolved.resolved_input_dir,
        "resolved_files": files,
        "data": {
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_test": int(x_test.shape[0]),
            "x_dim": int(x_train.shape[1]),
            "y_dim": int(y_train.shape[1]),
            **label_stats,
        },
        "training": {
            "best_threshold": float(best_thr),
            "threshold_grid": threshold_grid,
            "best_val_metric": float(val_metrics.get(str(args.metric_name), math.nan)),
            "metric_name": str(args.metric_name),
            "force_non_empty": bool(args.force_non_empty),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "candidate_metrics": candidate_metrics,
    }

    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
