#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


@dataclass
class GraphItem:
    item_id: str
    material_id: Optional[str]
    formula: Optional[str]
    doi: Optional[str]
    split_group: Optional[str]
    atomic_numbers: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_dist: torch.Tensor
    y: torch.Tensor


class CGCNNStage2Dataset(Dataset):
    def __init__(self, pkl_path: Path):
        raw = load_pickle(pkl_path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected list in {pkl_path}, got {type(raw)}")

        self.items: List[GraphItem] = []
        for row in raw:
            self.items.append(
                GraphItem(
                    item_id=str(row.get("id")),
                    material_id=row.get("material_id"),
                    formula=row.get("formula"),
                    doi=row.get("doi"),
                    split_group=row.get("split_group"),
                    atomic_numbers=torch.as_tensor(row["atomic_numbers"], dtype=torch.long),
                    edge_src=torch.as_tensor(row["edge_src"], dtype=torch.long),
                    edge_dst=torch.as_tensor(row["edge_dst"], dtype=torch.long),
                    edge_dist=torch.as_tensor(row["edge_dist"], dtype=torch.float32),
                    y=torch.as_tensor(row["y"], dtype=torch.float32),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> GraphItem:
        return self.items[idx]


def collate_graph_items(batch: List[GraphItem]) -> Dict[str, Any]:
    node_offset = 0
    atomic_numbers_all = []
    edge_src_all = []
    edge_dst_all = []
    edge_dist_all = []
    targets = []
    graph_node_slices: List[Tuple[int, int]] = []

    for item in batch:
        n = int(item.atomic_numbers.numel())
        atomic_numbers_all.append(item.atomic_numbers)
        edge_src_all.append(item.edge_src + node_offset)
        edge_dst_all.append(item.edge_dst + node_offset)
        edge_dist_all.append(item.edge_dist)
        targets.append(item.y)
        graph_node_slices.append((node_offset, node_offset + n))
        node_offset += n

    return {
        "atomic_numbers": torch.cat(atomic_numbers_all, dim=0),
        "edge_src": torch.cat(edge_src_all, dim=0),
        "edge_dst": torch.cat(edge_dst_all, dim=0),
        "edge_dist": torch.cat(edge_dist_all, dim=0),
        "targets": torch.stack(targets, dim=0),
        "graph_node_slices": graph_node_slices,
    }


class CGCNNConv(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        src_x = x[edge_src]
        dst_x = x[edge_dst]
        ed = edge_dist.unsqueeze(-1)
        msg_in = torch.cat([src_x, dst_x, ed], dim=-1)
        msgs = self.msg_mlp(msg_in)

        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_dst, msgs)

        out = self.upd(torch.cat([x, agg], dim=-1))
        out = self.norm(x + out)
        return out


class CGCNNStage2(nn.Module):
    def __init__(
        self,
        n_labels: int,
        hidden_dim: int = 128,
        n_conv_layers: int = 4,
        max_z: int = 120,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.hidden_dim = hidden_dim
        self.n_conv_layers = n_conv_layers
        self.max_z = max_z
        self.dropout = dropout

        self.atom_emb = nn.Embedding(max_z + 1, hidden_dim)
        self.convs = nn.ModuleList([CGCNNConv(hidden_dim) for _ in range(n_conv_layers)])
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, n_labels)

    def _encode_nodes(
        self,
        atomic_numbers: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> torch.Tensor:
        x = self.atom_emb(atomic_numbers.clamp(min=0, max=self.max_z))
        for conv in self.convs:
            x = conv(x, edge_src, edge_dst, edge_dist)
        return x

    def _pool_graphs(
        self,
        node_repr: torch.Tensor,
        graph_node_slices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        pooled = []
        for start, end in graph_node_slices:
            h = node_repr[start:end]
            mean_pool = h.mean(dim=0)
            max_pool = h.max(dim=0).values
            pooled.append(torch.cat([mean_pool, max_pool], dim=0))
        return torch.stack(pooled, dim=0)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_dist: torch.Tensor,
        graph_node_slices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        node_repr = self._encode_nodes(atomic_numbers, edge_src, edge_dst, edge_dist)
        graph_repr = self._pool_graphs(node_repr, graph_node_slices)
        emb = self.readout_mlp(graph_repr)
        logits = self.head(emb)
        return logits

    def extract_embedding(
        self,
        atomic_numbers: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_dist: torch.Tensor,
        graph_node_slices: Optional[List[Tuple[int, int]]] = None,
        **_: Any,
    ) -> torch.Tensor:
        if graph_node_slices is None:
            graph_node_slices = [(0, int(atomic_numbers.numel()))]
        node_repr = self._encode_nodes(atomic_numbers, edge_src, edge_dst, edge_dist)
        graph_repr = self._pool_graphs(node_repr, graph_node_slices)
        emb = self.readout_mlp(graph_repr)
        if emb.shape[0] == 1:
            return emb[0]
        return emb


@torch.no_grad()
def multilabel_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = (((1 - preds) * targets)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    set_match = preds.eq(targets).all(dim=1).float().mean().item()
    inter = (preds * targets).sum(dim=1)
    union = ((preds + targets) > 0).float().sum(dim=1)
    jaccard = torch.where(union > 0, inter / union, torch.ones_like(union))
    avg_jaccard = jaccard.mean().item()

    return {
        "micro_f1": float(micro_f1),
        "set_match": float(set_match),
        "avg_jaccard": float(avg_jaccard),
    }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["atomic_numbers"] = out["atomic_numbers"].to(device)
    out["edge_src"] = out["edge_src"].to(device)
    out["edge_dst"] = out["edge_dst"].to(device)
    out["edge_dist"] = out["edge_dist"].to(device)
    out["targets"] = out["targets"].to(device)
    return out


def run_epoch(
    model: CGCNNStage2,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    loss_fn: nn.Module,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(
            atomic_numbers=batch["atomic_numbers"],
            edge_src=batch["edge_src"],
            edge_dst=batch["edge_dst"],
            edge_dist=batch["edge_dist"],
            graph_node_slices=batch["graph_node_slices"],
        )
        loss = loss_fn(logits, batch["targets"])

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * batch["targets"].size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(batch["targets"].detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = multilabel_metrics_from_logits(logits_cat, targets_cat)

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a self-contained CGCNN-like stage2 precursor predictor.")
    parser.add_argument("--cache_dir", type=str, default="/Users/wyc/SynPred/data/interim/graph_cache/cgcnn_stage2")
    parser.add_argument("--run_dir", type=str, default="/Users/wyc/SynPred/runs/graph_models/cgcnn_stage2_multilabel")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_conv_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    seed_everything(args.seed)

    cache_dir = Path(args.cache_dir)
    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    split_paths = {
        "train": cache_dir / "train.pkl",
        "val": cache_dir / "val.pkl",
        "test": cache_dir / "test.pkl",
        "gold_train_holdout": cache_dir / "gold_train_holdout.pkl",
    }
    for k, p in split_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing split cache: {k} -> {p}")

    datasets = {k: CGCNNStage2Dataset(v) for k, v in split_paths.items()}
    n_labels = int(datasets["train"][0].y.numel())

    loaders = {
        k: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(k == "train"),
            num_workers=args.num_workers,
            collate_fn=collate_graph_items,
        )
        for k, ds in datasets.items()
    }

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model = CGCNNStage2(
        n_labels=n_labels,
        hidden_dim=args.hidden_dim,
        n_conv_layers=args.n_conv_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = math.inf
    best_epoch = -1
    history = []

    ckpt_path = run_dir / "best_cgcnn_stage2.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], optimizer, device, loss_fn)
        val_metrics = run_epoch(model, loaders["val"], None, device, loss_fn)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['micro_f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['micro_f1']:.4f} "
            f"val_jaccard={val_metrics['avg_jaccard']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": {
                        "n_labels": n_labels,
                        "hidden_dim": args.hidden_dim,
                        "n_conv_layers": args.n_conv_layers,
                        "dropout": args.dropout,
                    },
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    final_eval = {
        split: run_epoch(model, loaders[split], None, device, loss_fn)
        for split in ["train", "val", "test", "gold_train_holdout"]
    }

    summary = {
        "config": vars(args),
        "device": str(device),
        "n_labels": n_labels,
        "dataset_sizes": {k: len(v) for k, v in datasets.items()},
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_eval": final_eval,
        "checkpoint": str(ckpt_path),
    }

    write_json(run_dir / "train_history.json", history)
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
