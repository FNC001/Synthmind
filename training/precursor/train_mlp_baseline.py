#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader, Dataset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_hidden_dims(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def choose_device(device_arg: str) -> torch.device:
    if str(device_arg) != "auto":
        return torch.device(str(device_arg))
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def extract_x(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["x", "features", "X"]:
        if k in pack:
            x = np.asarray(pack[k], dtype=np.float32)
            if x.ndim != 2:
                raise ValueError(f"{k} must be 2D, got shape={x.shape}")
            return x
    raise KeyError(f"Cannot find x/features/X in npz keys={list(pack.keys())}")


def extract_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["y_multi_hot", "y", "labels", "targets"]:
        if k in pack:
            y = np.asarray(pack[k])
            if y.ndim != 2:
                raise ValueError(f"{k} must be 2D multi-hot, got shape={y.shape}")
            return (y > 0).astype(np.float32)
    raise KeyError(f"Cannot find y_multi_hot/y/labels/targets in npz keys={list(pack.keys())}")


class Stage2MultiLabelDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "samples_jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "mean_true_labels": float(np.mean(y_true.sum(axis=1))),
        "mean_pred_labels": float(np.mean(y_pred.sum(axis=1))),
    }


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    probs = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.sigmoid(logits)
        ys.append(y.numpy().astype(np.int32))
        probs.append(p.cpu().numpy().astype(np.float32))
    return np.vstack(ys), np.vstack(probs)


def threshold_predict(probs: np.ndarray, threshold: float) -> np.ndarray:
    y = (probs >= float(threshold)).astype(np.int32)
    empty = y.sum(axis=1) == 0
    if np.any(empty):
        best = np.argmax(probs[empty], axis=1)
        y[empty, best] = 1
    return y


def topk_from_true_count_predict(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Diagnostic upper-aligned baseline:
    use the true set size as k, then select top-k labels.
    This is not deployable without knowing true length, but useful for
    separating ranking quality from cardinality prediction.
    """
    n, d = probs.shape
    out = np.zeros((n, d), dtype=np.int32)
    ks = y_true.sum(axis=1).astype(int)
    ks = np.clip(ks, 1, d)
    for i, k in enumerate(ks):
        idx = np.argpartition(-probs[i], kth=k - 1)[:k]
        out[i, idx] = 1
    return out


def multihot_to_label_lists(y: np.ndarray, names: List[str]) -> List[List[str]]:
    out = []
    for i in range(y.shape[0]):
        idx = np.where(y[i] > 0)[0].tolist()
        out.append([names[j] for j in idx if 0 <= j < len(names)])
    return out


def save_prediction_csv(
    path: Path,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    precursor_names: List[str],
) -> None:
    out = meta_df.copy()
    out["sample_index"] = np.arange(len(out), dtype=int)
    out["true_labels"] = [
        json.dumps(x, ensure_ascii=False)
        for x in multihot_to_label_lists(y_true, precursor_names)
    ]
    out["pred_labels"] = [
        json.dumps(x, ensure_ascii=False)
        for x in multihot_to_label_lists(y_pred, precursor_names)
    ]
    out["n_true_labels"] = y_true.sum(axis=1)
    out["n_pred_labels"] = y_pred.sum(axis=1)
    out["top_score"] = probs.max(axis=1)
    out.to_csv(path, index=False)


def label_statistics(y_train: np.ndarray) -> Dict[str, Any]:
    label_pos = y_train.sum(axis=0)
    row_cnt = y_train.sum(axis=1)
    return {
        "mean_true_labels": float(np.mean(row_cnt)),
        "median_true_labels": float(np.median(row_cnt)),
        "max_true_labels": float(np.max(row_cnt)),
        "min_true_labels": float(np.min(row_cnt)),
        "constant_zero_labels": int(np.sum(label_pos == 0)),
        "constant_one_labels": int(np.sum(label_pos == y_train.shape[0])),
    }


def build_pos_weight(y_train: np.ndarray, max_pos_weight: float) -> torch.Tensor:
    pos = y_train.sum(axis=0).astype(np.float32)
    neg = y_train.shape[0] - pos
    w = neg / np.clip(pos, 1.0, None)
    w[pos <= 0] = 1.0
    w = np.clip(w, 1.0, float(max_pos_weight))
    return torch.tensor(w, dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage2 multilabel MLP baseline for precursor-set prediction.")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--hidden_dims", type=str, default="512,256")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--metric_name", type=str, default="samples_f1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_pos_weight", action="store_true", default=True)
    p.add_argument("--no_pos_weight", action="store_true")
    p.add_argument("--max_pos_weight", type=float, default=20.0)
    p.add_argument("--topk_from_true_count", action="store_true")
    args = p.parse_args()

    if args.no_pos_weight:
        args.use_pos_weight = False

    set_seed(args.seed)

    input_dir = Path(args.input_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    ensure_dir(run_dir)

    train_pack = load_npz(input_dir / "train.npz")
    val_pack = load_npz(input_dir / "val.npz")
    test_pack = load_npz(input_dir / "test.npz")

    train_meta = pd.read_csv(input_dir / "train_meta.csv") if (input_dir / "train_meta.csv").exists() else pd.DataFrame()
    val_meta = pd.read_csv(input_dir / "val_meta.csv") if (input_dir / "val_meta.csv").exists() else pd.DataFrame()
    test_meta = pd.read_csv(input_dir / "test_meta.csv") if (input_dir / "test_meta.csv").exists() else pd.DataFrame()

    x_train, y_train = extract_x(train_pack), extract_y(train_pack)
    x_val, y_val = extract_x(val_pack), extract_y(val_pack)
    x_test, y_test = extract_x(test_pack), extract_y(test_pack)

    precursor_names_path = input_dir / "precursor_names.json"
    if precursor_names_path.exists():
        precursor_names = [str(x) for x in load_json(precursor_names_path)]
    else:
        precursor_names = [f"precursor_{i}" for i in range(y_train.shape[1])]
    if len(precursor_names) < y_train.shape[1]:
        precursor_names += [f"precursor_{i}" for i in range(len(precursor_names), y_train.shape[1])]
    precursor_names = precursor_names[: y_train.shape[1]]

    label_stats = label_statistics(y_train)
    print(f"[Info] x_dim={x_train.shape[1]} n_labels={y_train.shape[1]}")
    print(f"[Info] label_stats={label_stats}")

    device = choose_device(args.device)
    print(f"[Info] device={device}")

    train_loader = DataLoader(Stage2MultiLabelDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Stage2MultiLabelDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Stage2MultiLabelDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = MLP(
        input_dim=x_train.shape[1],
        output_dim=y_train.shape[1],
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    if args.use_pos_weight:
        pos_weight = build_pos_weight(y_train, args.max_pos_weight).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metric = -1e18
    best_epoch = -1
    bad = 0
    logs: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.item()))

        yv, pv = predict_probs(model, val_loader, device)
        pred_v = threshold_predict(pv, args.threshold)
        val_metrics = evaluate_from_binary(yv, pred_v)

        if args.topk_from_true_count:
            pred_v_topk = topk_from_true_count_predict(pv, yv)
            val_metrics_topk = evaluate_from_binary(yv, pred_v_topk)
        else:
            val_metrics_topk = {}

        cur = float(val_metrics.get(args.metric_name, val_metrics["samples_f1"]))

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "val_metrics_threshold": val_metrics,
            "val_metrics_topk_from_true_count": val_metrics_topk,
        }
        logs.append(rec)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={rec['train_loss']:.4f} "
            f"val_{args.metric_name}={cur:.4f} "
            f"best={max(best_metric, cur):.4f}"
        )

        if cur > best_metric:
            best_metric = cur
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "x_dim": int(x_train.shape[1]),
                    "n_labels": int(y_train.shape[1]),
                    "precursor_names": precursor_names,
                    "best_epoch": int(best_epoch),
                    "best_val_metric": float(best_metric),
                },
                run_dir / "best_model.pt",
            )
        else:
            bad += 1

        if bad >= args.patience:
            print(f"[Early Stop] patience reached at epoch {epoch}")
            break

    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    yv, pv = predict_probs(model, val_loader, device)
    yt, pt = predict_probs(model, test_loader, device)

    val_pred = threshold_predict(pv, args.threshold)
    test_pred = threshold_predict(pt, args.threshold)

    val_metrics = evaluate_from_binary(yv, val_pred)
    test_metrics = evaluate_from_binary(yt, test_pred)

    if args.topk_from_true_count:
        val_topk_pred = topk_from_true_count_predict(pv, yv)
        test_topk_pred = topk_from_true_count_predict(pt, yt)
        val_topk_metrics = evaluate_from_binary(yv, val_topk_pred)
        test_topk_metrics = evaluate_from_binary(yt, test_topk_pred)
    else:
        val_topk_metrics = {}
        test_topk_metrics = {}

    if len(val_meta) == len(yv):
        save_prediction_csv(run_dir / "pred_val.csv", val_meta, yv, val_pred, pv, precursor_names)
    if len(test_meta) == len(yt):
        save_prediction_csv(run_dir / "pred_test.csv", test_meta, yt, test_pred, pt, precursor_names)

    write_json(run_dir / "train_log.json", logs)

    summary = {
        "model": "stage2_multilabel_mlp_baseline",
        "config": vars(args),
        "data": {
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_test": int(x_test.shape[0]),
            "x_dim": int(x_train.shape[1]),
            "n_labels": int(y_train.shape[1]),
            **label_stats,
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_metric": float(best_metric),
            "metric_name": str(args.metric_name),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_topk_from_true_count_metrics": val_topk_metrics,
        "test_topk_from_true_count_metrics": test_topk_metrics,
        "notes": {
            "threshold_metrics": "Deployable threshold-based multilabel prediction.",
            "topk_from_true_count": "Diagnostic only. Uses true label count as k, not available at real inference.",
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
