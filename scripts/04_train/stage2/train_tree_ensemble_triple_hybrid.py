#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, jaccard_score


# -----------------------------------------------------------------------------
# robust local utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def attach_sample_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_index" not in out.columns:
        out["sample_index"] = np.arange(len(out))
    return out


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


REQUIRED_FILES = {
    "train_npz": "train.npz",
    "val_npz": "val.npz",
    "test_npz": "test.npz",
    "train_meta_csv": "train_meta.csv",
    "val_meta_csv": "val_meta.csv",
    "test_meta_csv": "test_meta.csv",
    "summary": ["summary.json", "schema.json"],
}
OPTIONAL_FILES = {
    "precursor_names": "precursor_names.json",
    "label_cols": "label_cols.json",
    "label_names": "label_names.json",
    "schema": "schema.json",
}


def _first_existing(candidates: Sequence[Path], what: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 {what}，候选路径为：\n" + "\n".join(str(x) for x in candidates))


def _default_project_root() -> Path:
    return Path("/Users/wyc/SynPred").expanduser().resolve()


def _candidate_input_dirs(train_mode: str) -> List[Path]:
    root = _default_project_root()
    return [
        root / "data/interim/generative/stage2_setpred_dataset/triple_hybrid" / train_mode,
        root / "data/interim/generative/stage2_setpred_dataset/hybrid" / train_mode,
        root / "data/interim/model_inputs/stage2_cvae_modes/stage2_hybrid_cgcnn_chgnet" / train_mode,
    ]


def resolve_input_paths(args: argparse.Namespace, required: Dict[str, Any], optional: Dict[str, Any]):
    train_mode = getattr(args, "train_mode", "gold_only")

    candidate_dirs: List[Path] = []

    input_dir = str(getattr(args, "input_dir", "") or "").strip()
    if input_dir:
        candidate_dirs.append(Path(input_dir).expanduser().resolve())

    mode_input_root = str(getattr(args, "mode_input_root", "") or "").strip()
    if mode_input_root:
        root = Path(mode_input_root).expanduser().resolve()
        candidate_dirs.extend([
            root / train_mode,
            root,
        ])

    candidate_dirs.extend(_candidate_input_dirs(train_mode))

    seen = set()
    uniq_candidates = []
    for p in candidate_dirs:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq_candidates.append(p)

    resolved_input_dir = None
    missing_debug: List[str] = []
    for cand in uniq_candidates:
        if not cand.exists():
            missing_debug.append(f"[目录不存在] {cand}")
            continue

        ok = True
        for _, req in required.items():
            req_names = req if isinstance(req, list) else [req]
            if not any((cand / name).exists() for name in req_names):
                ok = False
                missing_debug.append(f"[缺少必需文件] {cand} :: {req_names}")
                break
        if ok:
            resolved_input_dir = cand
            break

    if resolved_input_dir is None:
        raise FileNotFoundError(
            "未找到可用的数据目录。\n已尝试如下候选：\n" + "\n".join(missing_debug if missing_debug else [str(x) for x in uniq_candidates])
        )

    files: Dict[str, str] = {}
    for key, req in required.items():
        req_names = req if isinstance(req, list) else [req]
        p = _first_existing([resolved_input_dir / name for name in req_names], f"{key} 文件")
        files[key] = str(p)

    for key, opt in optional.items():
        opt_names = opt if isinstance(opt, list) else [opt]
        for name in opt_names:
            p = resolved_input_dir / name
            if p.exists():
                files[key] = str(p)
                break

    class _Resolved:
        def __init__(self, files, resolved_input_dir, resolved_mode, resolved_root):
            self.files = files
            self.resolved_input_dir = str(resolved_input_dir)
            self.resolved_mode = resolved_mode
            self.resolved_root = resolved_root

    return _Resolved(
        files=files,
        resolved_input_dir=resolved_input_dir,
        resolved_mode=("legacy_input_dir" if input_dir else train_mode),
        resolved_root=str(resolved_input_dir.parent),
    )


def _extract_x(pack: Dict[str, Any]) -> np.ndarray:
    for k in ["x", "X", "features"]:
        if k in pack:
            return np.asarray(pack[k], dtype=np.float32)
    raise KeyError(f"NPZ 中未找到特征键，已有键：{list(pack.keys())}")


def _extract_y(pack: Dict[str, Any]) -> np.ndarray:
    for k in ["y_multi_hot", "y", "labels", "targets"]:
        if k in pack:
            return np.asarray(pack[k]).astype(np.int32)
    raise KeyError(f"NPZ 中未找到标签键，已有键：{list(pack.keys())}")


def _load_label_names(files: Dict[str, str], summary_obj: Dict[str, Any], n_labels: int) -> List[str]:
    for k in ["precursor_names", "label_cols", "label_names"]:
        if k in files:
            obj = load_json(files[k])
            if isinstance(obj, list) and len(obj) == n_labels:
                return [str(x) for x in obj]

    schema = summary_obj.get("schema", summary_obj) if isinstance(summary_obj, dict) else {}
    if isinstance(schema, dict):
        for key in ["precursor_names", "label_cols", "label_names"]:
            val = schema.get(key)
            if isinstance(val, list) and len(val) == n_labels:
                return [str(x) for x in val]
            if isinstance(val, str):
                p = Path(val).expanduser()
                if p.exists():
                    obj = load_json(p)
                    if isinstance(obj, list) and len(obj) == n_labels:
                        return [str(x) for x in obj]

    return [f"label_{i}" for i in range(n_labels)]


def _multihot_to_label_lists(y: np.ndarray, label_names: Sequence[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for row in y:
        idx = np.where(np.asarray(row) > 0)[0].tolist()
        out.append([label_names[i] for i in idx])
    return out


def _search_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best_metrics = evaluate_multilabel(y_true, (y_prob >= 0.5).astype(np.int32))
    best_score = best_metrics["samples_f1"]
    for thr in np.linspace(0.1, 0.9, 17):
        y_pred = (y_prob >= thr).astype(np.int32)
        metrics = evaluate_multilabel(y_true, y_pred)
        if metrics["samples_f1"] > best_score:
            best_score = metrics["samples_f1"]
            best_thr = float(thr)
            best_metrics = metrics
    return best_thr, best_metrics


def _predict_proba_ovr(model: OneVsRestClassifier, x: np.ndarray) -> np.ndarray:
    probs: List[np.ndarray] = []
    for est in model.estimators_:
        if hasattr(est, "predict_proba"):
            p = est.predict_proba(x)
            if p.ndim == 2 and p.shape[1] >= 2:
                probs.append(p[:, 1])
            else:
                probs.append(np.asarray(p).reshape(-1))
        else:
            pred = est.predict(x)
            probs.append(np.asarray(pred, dtype=np.float32).reshape(-1))
    return np.stack(probs, axis=1).astype(np.float32)


def _force_non_empty(y_prob: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = y_pred.copy()
    empty = np.where(out.sum(axis=1) == 0)[0]
    if len(empty) == 0:
        return out
    best_idx = np.argmax(y_prob[empty], axis=1)
    out[empty, best_idx] = 1
    return out


def _save_predictions(
    path: Path,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
) -> None:
    out = attach_sample_index(meta_df)
    out["true_labels"] = [json.dumps(x, ensure_ascii=False) for x in _multihot_to_label_lists(y_true, label_names)]
    out["pred_labels"] = [json.dumps(x, ensure_ascii=False) for x in _multihot_to_label_lists(y_pred, label_names)]
    out["n_true_labels"] = y_true.sum(axis=1)
    out["n_pred_labels"] = y_pred.sum(axis=1)
    out["max_prob"] = y_prob.max(axis=1)
    out.to_csv(path, index=False)


def build_model(args: argparse.Namespace) -> OneVsRestClassifier:
    class_weight = "balanced" if args.class_weight_balanced else None
    base = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        class_weight=class_weight,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    return OneVsRestClassifier(base, n_jobs=args.ovr_n_jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Commonized triple-hybrid ExtraTrees stage2 trainer.")
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--input_mode", type=str, default="triple_hybrid")
    parser.add_argument("--mode_input_root", type=str, default="")
    parser.add_argument("--train_mode", type=str, default="gold_only",
                        choices=["relaxed_only", "gold_only", "curriculum", "curriculum_phase1", "curriculum_phase2"])
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/generative/stage2_setpred_dataset/triple_hybrid/gold_only",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="/Users/wyc/SynPred/runs/stage2/tree_ensemble_triple_hybrid_commonized_v1",
    )

    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--criterion", type=str, default="gini", choices=["gini", "entropy", "log_loss"])
    parser.add_argument("--max_depth", type=int, default=0)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--class_weight_balanced", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--ovr_n_jobs", type=int, default=1)
    parser.add_argument("--force_non_empty", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    resolved = resolve_input_paths(args, required=REQUIRED_FILES, optional=OPTIONAL_FILES)
    print(f"[Info] resolved_mode = {resolved.resolved_mode}")
    print(f"[Info] resolved_root = {resolved.resolved_root}")
    print(f"[Info] resolved_input_dir = {resolved.resolved_input_dir}")

    run_dir = Path(args.run_dir).expanduser().resolve()
    ensure_dir(run_dir)

    files = resolved.files
    train_pack = load_npz(files["train_npz"])
    val_pack = load_npz(files["val_npz"])
    test_pack = load_npz(files["test_npz"])
    train_meta = pd.read_csv(files["train_meta_csv"])
    val_meta = pd.read_csv(files["val_meta_csv"])
    test_meta = pd.read_csv(files["test_meta_csv"])

    summary_obj = load_json(files["summary"])
    x_train, y_train = _extract_x(train_pack), _extract_y(train_pack)
    x_val, y_val = _extract_x(val_pack), _extract_y(val_pack)
    x_test, y_test = _extract_x(test_pack), _extract_y(test_pack)

    label_names = _load_label_names(files, summary_obj, y_train.shape[1])

    model = build_model(args)
    model.fit(x_train, y_train)

    val_prob = _predict_proba_ovr(model, x_val)
    threshold, val_metrics = _search_best_threshold(y_val, val_prob)
    val_pred = (val_prob >= threshold).astype(np.int32)
    if args.force_non_empty:
        val_pred = _force_non_empty(val_prob, val_pred)
        val_metrics = evaluate_multilabel(y_val, val_pred)

    test_prob = _predict_proba_ovr(model, x_test)
    test_pred = (test_prob >= threshold).astype(np.int32)
    if args.force_non_empty:
        test_pred = _force_non_empty(test_prob, test_pred)
    test_metrics = evaluate_multilabel(y_test, test_pred)

    _save_predictions(run_dir / "pred_val.csv", val_meta, y_val, val_prob, val_pred, label_names)
    _save_predictions(run_dir / "pred_test.csv", test_meta, y_test, test_prob, test_pred, label_names)

    with open(run_dir / "best_model.pkl", "wb") as f:
        pickle.dump({"model": model, "threshold": threshold, "label_names": label_names, "config": vars(args)}, f)

    train_log = [{
        "stage": "fit_complete",
        "threshold": float(threshold),
        "val_samples_f1": float(val_metrics["samples_f1"]),
        "test_samples_f1": float(test_metrics["samples_f1"]),
    }]
    write_json(run_dir / "train_log.json", train_log)

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
            "n_labels": int(y_train.shape[1]),
            "input_mode": args.input_mode,
        },
        "training": {
            "best_threshold": float(threshold),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
