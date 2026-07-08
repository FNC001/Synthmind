#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# -----------------------------
# SynPred local bootstrap
# -----------------------------
import sys
from pathlib import Path as _PathForBootstrap

_CURRENT_FILE = _PathForBootstrap(__file__).resolve()
_PROJECT_CANDIDATES = [
    _CURRENT_FILE.parents[3],  # /Users/wyc/SynPred
    _PathForBootstrap("/Users/wyc/SynPred"),
]
for _proj in _PROJECT_CANDIDATES:
    if _proj.exists():
        _paths = [
            str(_proj),
            str(_proj / "scripts/04_train"),
            str(_proj / "scripts/04_train/common"),
            str(_proj / "scripts"),
        ]
        for _p in _paths:
            if _p not in sys.path:
                sys.path.insert(0, _p)

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from common.common_io import ensure_dir, load_json, parse_hidden_dims, resolve_input_paths, write_json
except Exception:
    from common_io import ensure_dir, load_json, parse_hidden_dims, resolve_input_paths, write_json

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
DEFAULT_RUN_ROOT = Path("/Users/wyc/SynPred/runs/stage3/train_condition_residual_mdn_mixed")

STAGE3_MIXED_REQUIRED = {
    "train_npz": ["train.npz"],
    "val_npz": ["val.npz"],
    "test_npz": ["test.npz"],
    "schema": ["schema.json", "condition_schema.json"],
}
STAGE3_MIXED_OPTIONAL = {
    "train_meta": ["train_meta.csv", "train.csv", "stage3_train.csv"],
    "val_meta": ["val_meta.csv", "val.csv", "stage3_val.csv"],
    "test_meta": ["test_meta.csv", "test.csv", "stage3_test.csv"],
}


class Stage3BaselineModel:
    def __init__(self):
        self.standardize = False
        self.scaler = None
        self.cont_models = []
        self.disc_models = []
        self.cont_names = []
        self.disc_names = []

    def _transform(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if not getattr(self, "standardize", False) or getattr(self, "scaler", None) is None:
            return X.astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    def predict(self, X: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        Xf = self._transform(X, fit=False)
        out: Dict[str, Optional[np.ndarray]] = {"y_cont_pred": None, "y_disc_pred": None, "y_disc_prob": None}
        if getattr(self, "cont_models", None):
            preds: List[np.ndarray] = []
            for m in self.cont_models:
                if m is None:
                    preds.append(np.zeros((Xf.shape[0],), dtype=np.float32))
                else:
                    preds.append(np.asarray(m.predict(Xf), dtype=np.float32))
            out["y_cont_pred"] = np.stack(preds, axis=1).astype(np.float32)
        if getattr(self, "disc_models", None):
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


def build_features_np(x: np.ndarray, y_set: Optional[np.ndarray], use_y_set: bool) -> np.ndarray:
    if use_y_set and y_set is not None:
        return np.concatenate([x.astype(np.float32), y_set.astype(np.float32)], axis=1)
    return x.astype(np.float32)


class PickleBaselineAdapter(nn.Module):
    def __init__(self, payload: Mapping[str, Any], device: torch.device):
        super().__init__()
        self.payload = payload
        self.model = payload["model"]
        cfg = payload.get("config", {})
        self.use_y_set = bool(cfg.get("use_y_set", False))
        self.device = device

    def forward(self, x: torch.Tensor, y_set: torch.Tensor) -> Dict[str, Any]:
        x_np = x.detach().cpu().numpy().astype(np.float32)
        y_set_np = y_set.detach().cpu().numpy().astype(np.float32) if y_set is not None else None
        feat = build_features_np(x_np, y_set_np, self.use_y_set)
        pred = self.model.predict(feat)
        cont = pred.get("y_cont_pred")
        if cont is None:
            cont = np.zeros((x_np.shape[0], 0), dtype=np.float32)
        cont_t = torch.tensor(cont, dtype=torch.float32, device=self.device)
        return {"cont_pred": cont_t, "disc_logits": []}


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_state_dict(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for k in ["model_state_dict", "model_state", "state_dict"]:
        if k in payload:
            return payload[k]
    return payload


class Stage3MixedDataset(Dataset):
    def __init__(self, npz_path: Path):
        arr = np.load(npz_path, allow_pickle=True)
        required = ["x", "y_set", "y_cond_discrete", "y_cond_continuous", "y_cond_continuous_mask"]
        missing = [k for k in required if k not in arr]
        if missing:
            raise KeyError(f"Missing keys in {npz_path}: {missing}. Available keys={list(arr.files)}")
        self.x = torch.tensor(arr["x"], dtype=torch.float32)
        self.y_set = torch.tensor(arr["y_set"], dtype=torch.float32)
        self.y_disc = torch.tensor(arr["y_cond_discrete"], dtype=torch.long)
        self.y_cont = torch.tensor(arr["y_cond_continuous"], dtype=torch.float32)
        self.y_cont_mask = torch.tensor(arr["y_cond_continuous_mask"], dtype=torch.float32)
        self.sample_id = np.asarray(arr["sample_id"], dtype=object) if "sample_id" in arr.files else np.asarray([str(i) for i in range(self.x.shape[0])], dtype=object)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": self.x[idx],
            "y_set": self.y_set[idx],
            "y_disc": self.y_disc[idx],
            "y_cont": self.y_cont[idx],
            "y_cont_mask": self.y_cont_mask[idx],
            "sample_id": str(self.sample_id[idx]),
        }


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0, use_layernorm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if use_layernorm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Stage3ConditionPredictor(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_set_dim: int,
        hidden_dims: Sequence[int],
        disc_class_sizes: Sequence[int],
        n_cont: int,
        dropout: float = 0.1,
        use_layernorm: bool = False,
        set_proj_dim: int = 256,
        fuse_mode: str = "concat",
    ):
        super().__init__()
        self.disc_class_sizes = list(disc_class_sizes)
        self.n_cont = int(n_cont)
        self.fuse_mode = str(fuse_mode)
        if self.fuse_mode not in {"concat", "add"}:
            raise ValueError("fuse_mode must be one of: concat, add")
        self.set_encoder = MLP([y_set_dim, set_proj_dim], dropout=0.0, use_layernorm=False)
        if self.fuse_mode == "concat":
            trunk_in_dim = x_dim + set_proj_dim
        else:
            self.x_proj = nn.Identity() if x_dim == set_proj_dim else MLP([x_dim, set_proj_dim], dropout=0.0)
            trunk_in_dim = set_proj_dim
        trunk_dims = [trunk_in_dim] + list(hidden_dims)
        if len(trunk_dims) < 2:
            trunk_dims = [trunk_in_dim, trunk_in_dim]
        self.trunk = MLP(trunk_dims, dropout=dropout, use_layernorm=use_layernorm)
        self.disc_heads = nn.ModuleList([nn.Linear(trunk_dims[-1], k) for k in self.disc_class_sizes])
        self.cont_head = nn.Linear(trunk_dims[-1], self.n_cont)

    def forward(self, x: torch.Tensor, y_set: torch.Tensor) -> Dict[str, Any]:
        set_repr = self.set_encoder(y_set)
        if self.fuse_mode == "concat":
            fused = torch.cat([x, set_repr], dim=1)
        else:
            fused = self.x_proj(x) + set_repr
        h = self.trunk(fused)
        return {
            "disc_logits": [head(h) for head in self.disc_heads],
            "cont_pred": self.cont_head(h),
        }


class MDNHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_mixtures: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.n_mixtures = int(n_mixtures)
        self.pi = nn.Linear(in_dim, self.n_mixtures)
        self.mu = nn.Linear(in_dim, self.n_mixtures * self.out_dim)
        self.log_sigma = nn.Linear(in_dim, self.n_mixtures * self.out_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = h.shape[0]
        pi_logits = self.pi(h)
        mu = self.mu(h).view(b, self.n_mixtures, self.out_dim)
        log_sigma = self.log_sigma(h).view(b, self.n_mixtures, self.out_dim).clamp(min=-7.0, max=5.0)
        return pi_logits, mu, log_sigma

    @staticmethod
    def masked_nll(
        pi_logits: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        head_weights: torch.Tensor,
    ) -> torch.Tensor:
        valid_rows = mask.sum(dim=1) > 0
        if not torch.any(valid_rows):
            return pi_logits.new_tensor(0.0)
        target = target[valid_rows]
        mask = mask[valid_rows]
        pi_logits = pi_logits[valid_rows]
        mu = mu[valid_rows]
        log_sigma = log_sigma[valid_rows]

        target = target.unsqueeze(1)
        mask = mask.unsqueeze(1)
        head_weights = head_weights.view(1, 1, -1)
        inv_sigma = torch.exp(-log_sigma)
        z = (target - mu) * inv_sigma
        log_prob_dim = -0.5 * (z ** 2 + 2.0 * log_sigma + math.log(2.0 * math.pi))
        log_prob_dim = log_prob_dim * mask * head_weights
        valid_dim_count = (mask * head_weights).sum(dim=-1).clamp_min(1.0)
        log_prob_comp = log_prob_dim.sum(dim=-1) / valid_dim_count
        log_pi = F.log_softmax(pi_logits, dim=-1)
        log_mix = torch.logsumexp(log_pi + log_prob_comp, dim=-1)
        return -log_mix.mean()

    @staticmethod
    def predictive_mean(pi_logits: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        w = F.softmax(pi_logits, dim=-1).unsqueeze(-1)
        return (w * mu).sum(dim=1)

    @staticmethod
    def sample(pi_logits: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, n_samples: int) -> torch.Tensor:
        probs = F.softmax(pi_logits, dim=-1)
        b, _, _ = mu.shape
        cat = torch.distributions.Categorical(probs=probs)
        idx = cat.sample((int(n_samples),))
        sigma = torch.exp(log_sigma)
        samples = []
        for k in range(int(n_samples)):
            idx_k = idx[k]
            batch_ids = torch.arange(b, device=mu.device)
            mu_k = mu[batch_ids, idx_k, :]
            sigma_k = sigma[batch_ids, idx_k, :]
            eps = torch.randn_like(mu_k)
            samples.append(mu_k + sigma_k * eps)
        return torch.stack(samples, dim=0)


class ResidualConditionMDNMixed(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_set_dim: int,
        hidden_dims: Sequence[int],
        disc_class_sizes: Sequence[int],
        y_cont_dim: int,
        n_mixtures: int,
        dropout: float = 0.1,
        use_layernorm: bool = False,
        set_proj_dim: int = 256,
        fuse_mode: str = "concat",
    ):
        super().__init__()
        self.disc_class_sizes = list(disc_class_sizes)
        self.y_cont_dim = int(y_cont_dim)
        self.fuse_mode = str(fuse_mode)
        if self.fuse_mode not in {"concat", "add"}:
            raise ValueError("fuse_mode must be one of: concat, add")

        self.set_encoder = MLP([y_set_dim, set_proj_dim], dropout=0.0, use_layernorm=False)
        if self.fuse_mode == "concat":
            base_dim = x_dim + set_proj_dim
        else:
            self.x_proj = nn.Identity() if x_dim == set_proj_dim else MLP([x_dim, set_proj_dim], dropout=0.0)
            base_dim = set_proj_dim

        trunk_hidden = list(hidden_dims)
        if not trunk_hidden:
            trunk_hidden = [max(base_dim, 128)]
        self.trunk = MLP([base_dim] + trunk_hidden, dropout=dropout, use_layernorm=use_layernorm)
        out_dim = trunk_hidden[-1]
        self.disc_heads = nn.ModuleList([nn.Linear(out_dim, k) for k in self.disc_class_sizes])
        self.mdn = MDNHead(out_dim, self.y_cont_dim, n_mixtures)

    def encode(self, x: torch.Tensor, y_set: torch.Tensor) -> torch.Tensor:
        set_repr = self.set_encoder(y_set)
        if self.fuse_mode == "concat":
            fused = torch.cat([x, set_repr], dim=1)
        else:
            fused = self.x_proj(x) + set_repr
        return self.trunk(fused)

    def forward(self, x: torch.Tensor, y_set: torch.Tensor) -> Dict[str, Any]:
        h = self.encode(x, y_set)
        pi_logits, mu, log_sigma = self.mdn(h)
        return {
            "features": h,
            "resid_disc_logits": [head(h) for head in self.disc_heads],
            "pi_logits": pi_logits,
            "mu": mu,
            "log_sigma": log_sigma,
        }


class EarlyStopper:
    def __init__(self, patience: int, minimize: bool = True, min_delta: float = 1e-8):
        self.patience = int(patience)
        self.minimize = bool(minimize)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> Tuple[bool, bool]:
        improved = False
        if self.best is None:
            improved = True
        else:
            improved = value < (self.best - self.min_delta) if self.minimize else value > (self.best + self.min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def load_schema(schema_path: Path) -> Dict[str, Any]:
    schema = load_json(schema_path)
    if "continuous_cols" in schema or "discrete_cols" in schema:
        return schema
    cond = {
        "continuous_cols": list(schema.get("continuous_schema", {}).keys()),
        "discrete_cols": list(schema.get("discrete_schema", {}).keys()),
        "continuous_schema": schema.get("continuous_schema", {}),
        "discrete_schema": schema.get("discrete_schema", {}),
    }
    if "feature_cols" in schema:
        cond["feature_cols"] = schema["feature_cols"]
    if "sample_id_col" in schema:
        cond["sample_id_col"] = schema["sample_id_col"]
    return cond


def extract_cont_stats(schema: Mapping[str, Any], cont_names: Sequence[str], train_ds: Stage3MixedDataset) -> List[Dict[str, float]]:
    stats = []
    schema_map = schema.get("continuous_schema", {}) if isinstance(schema, Mapping) else {}
    train_np = train_ds.y_cont.numpy()
    train_mask = train_ds.y_cont_mask.numpy()
    for i, name in enumerate(cont_names):
        if name in schema_map:
            st = schema_map[name]
            mean = float(st.get("mean", 0.0))
            std = float(st.get("std", 1.0))
            if abs(std) < 1e-12:
                std = 1.0
        else:
            valid = train_mask[:, i] > 0.5
            if np.any(valid):
                mean = float(train_np[valid, i].mean())
                std = float(train_np[valid, i].std())
            else:
                mean, std = 0.0, 1.0
            if abs(std) < 1e-12:
                std = 1.0
        stats.append({"name": name, "mean": mean, "std": std})
    return stats


def inverse_transform_np(values: np.ndarray, stats: List[Dict[str, float]]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for i, st in enumerate(stats):
        out[:, i] = out[:, i] * float(st["std"]) + float(st["mean"])
    return out


def detect_targets_are_standardized(train_ds: Stage3MixedDataset, cont_stats: List[Dict[str, float]]) -> bool:
    """
    Conservative heuristic.
    Returns True only when the observed targets look strongly standardized:
    - per-head mean close to 0
    - per-head std roughly around 1
    This avoids misclassifying raw targets that simply have smaller scale than
    schema mean/std-transformed values.
    """
    y = train_ds.y_cont.numpy().astype(np.float32)
    mask = train_ds.y_cont_mask.numpy().astype(np.float32)

    mean_close_flags = []
    std_close_flags = []

    for j in range(y.shape[1]):
        valid = mask[:, j] > 0.5
        if not np.any(valid):
            continue
        col = y[valid, j]
        col_mean = float(np.mean(col))
        col_std = float(np.std(col))
        mean_close_flags.append(abs(col_mean) < 3.0)
        std_close_flags.append(0.2 < col_std < 5.0)

    if not mean_close_flags:
        return False
    return bool(all(mean_close_flags) and all(std_close_flags))


def build_class_weight_tensor(y_disc: np.ndarray, disc_class_sizes: Sequence[int], device: torch.device) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if y_disc.size == 0:
        return out
    for j, n_classes in enumerate(disc_class_sizes):
        col = y_disc[:, j]
        counts = np.bincount(np.clip(col, 0, n_classes - 1), minlength=n_classes).astype(np.float32)
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / counts
        weights = weights / weights.mean()
        out.append(torch.tensor(weights, dtype=torch.float32, device=device))
    return out


def multihead_classification_loss(
    logits_list: Sequence[torch.Tensor],
    target: torch.Tensor,
    class_weight_tensors: Optional[Sequence[torch.Tensor]],
    head_weights: Sequence[float],
) -> torch.Tensor:
    if not logits_list:
        return target.new_tensor(0.0, dtype=torch.float32)
    losses = []
    for j, logits in enumerate(logits_list):
        weight = None if class_weight_tensors is None else class_weight_tensors[j]
        loss_j = F.cross_entropy(logits, target[:, j], weight=weight)
        losses.append(loss_j * float(head_weights[j]))
    return torch.stack(losses).sum() / max(1, len(losses))


def collect_train_raw_minmax(train_ds: Stage3MixedDataset, cont_stats: List[Dict[str, float]], targets_are_standardized: bool) -> Tuple[np.ndarray, np.ndarray]:
    if targets_are_standardized:
        raw = inverse_transform_np(train_ds.y_cont.numpy(), cont_stats)
    else:
        raw = train_ds.y_cont.numpy().astype(np.float32).copy()
    mask = train_ds.y_cont_mask.numpy() > 0.5
    mins, maxs = [], []
    for j in range(raw.shape[1]):
        valid = mask[:, j]
        if np.any(valid):
            mins.append(float(raw[valid, j].min()))
            maxs.append(float(raw[valid, j].max()))
        else:
            mins.append(-np.inf)
            maxs.append(np.inf)
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def clip_continuous_to_train_range(pred_raw: np.ndarray, train_min_raw: np.ndarray, train_max_raw: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(pred_raw, train_min_raw[None, :]), train_max_raw[None, :]).astype(np.float32)


@torch.no_grad()
def evaluate_split(
    model: ResidualConditionMDNMixed,
    baseline: Stage3ConditionPredictor,
    loader: DataLoader,
    device: torch.device,
    cont_col_names: Sequence[str],
    disc_col_names: Sequence[str],
    cont_stats: List[Dict[str, float]],
    n_gen_samples: int,
    clip_to_train_range: bool,
    train_min_raw: np.ndarray,
    train_max_raw: np.ndarray,
    targets_are_standardized: bool,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()
    baseline.eval()

    sample_ids: List[str] = []
    y_disc_true_all: List[np.ndarray] = []
    y_disc_pred_all: List[np.ndarray] = []
    y_cont_true_all: List[np.ndarray] = []
    y_cont_pred_all: List[np.ndarray] = []
    y_cont_mask_all: List[np.ndarray] = []
    oracle_preds: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y_set = batch["y_set"].to(device)
        y_disc = batch["y_disc"].to(device)
        y_cont = batch["y_cont"].to(device)
        y_mask = batch["y_cont_mask"].to(device)
        sample_ids.extend(batch["sample_id"])

        base_out = baseline(x, y_set)
        base_cont = base_out["cont_pred"]
        out = model(x, y_set)

        resid_mean = model.mdn.predictive_mean(out["pi_logits"], out["mu"])
        cont_pred = base_cont + resid_mean

        disc_pred = []
        for logits in out["resid_disc_logits"]:
            disc_pred.append(torch.argmax(logits, dim=1))
        disc_pred = torch.stack(disc_pred, dim=1) if disc_pred else torch.empty((x.shape[0], 0), dtype=torch.long, device=x.device)

        samples_resid = model.mdn.sample(out["pi_logits"], out["mu"], out["log_sigma"], n_gen_samples)
        samples_total = samples_resid + base_cont.unsqueeze(0)

        y_true_np = y_cont.cpu().numpy().astype(np.float32)
        y_mask_np = y_mask.cpu().numpy().astype(np.float32)
        samples_total_np = samples_total.cpu().numpy().astype(np.float32)

        if targets_are_standardized:
            true_raw = inverse_transform_np(y_true_np, cont_stats)
        else:
            true_raw = y_true_np.copy()

        best_raw = []
        for bi in range(samples_total_np.shape[1]):
            cand_norm = samples_total_np[:, bi, :]
            if targets_are_standardized:
                cand_raw = inverse_transform_np(cand_norm, cont_stats)
            else:
                cand_raw = cand_norm.copy()
            if clip_to_train_range:
                cand_raw = clip_continuous_to_train_range(cand_raw, train_min_raw, train_max_raw)
            valid = y_mask_np[bi] > 0.5
            if np.any(valid):
                errs = np.mean(np.abs(cand_raw[:, valid] - true_raw[bi:bi+1, valid]), axis=1)
                best_idx = int(np.argmin(errs))
            else:
                best_idx = 0
            best_raw.append(cand_raw[best_idx])
        oracle_raw = np.vstack(best_raw).astype(np.float32)

        cont_pred_np = cont_pred.cpu().numpy().astype(np.float32)
        if targets_are_standardized:
            cont_pred_raw = inverse_transform_np(cont_pred_np, cont_stats)
        else:
            cont_pred_raw = cont_pred_np.copy()
        if clip_to_train_range:
            cont_pred_raw = clip_continuous_to_train_range(cont_pred_raw, train_min_raw, train_max_raw)

        y_disc_true_all.append(y_disc.cpu().numpy().astype(np.int64))
        y_disc_pred_all.append(disc_pred.cpu().numpy().astype(np.int64))
        y_cont_true_all.append(true_raw)
        y_cont_pred_all.append(cont_pred_raw)
        y_cont_mask_all.append(y_mask_np)
        oracle_preds.append(oracle_raw)

    y_disc_true = np.vstack(y_disc_true_all) if y_disc_true_all else None
    y_disc_pred = np.vstack(y_disc_pred_all) if y_disc_pred_all else None
    y_cont_true = np.vstack(y_cont_true_all)
    y_cont_pred = np.vstack(y_cont_pred_all)
    y_cont_mask = np.vstack(y_cont_mask_all)
    y_oracle = np.vstack(oracle_preds)

    metrics = evaluate_mixed_conditions(
        y_cont_true=y_cont_true,
        y_cont_pred=y_cont_pred,
        y_cont_mask=y_cont_mask,
        cont_target_names=cont_col_names,
        y_disc_true=y_disc_true,
        y_disc_pred=y_disc_pred,
        disc_target_names=disc_col_names,
        prefix="top1",
    )
    oracle_metrics = evaluate_mixed_conditions(
        y_cont_true=y_cont_true,
        y_cont_pred=y_oracle,
        y_cont_mask=y_cont_mask,
        cont_target_names=cont_col_names,
        prefix="oracle_best_of_k",
    )
    metrics.update(oracle_metrics)

    monitor_key = "top1_continuous_mean_mae_raw"
    if monitor_key not in metrics:
        monitor_key = "top1_mae_mean" if "top1_mae_mean" in metrics else next(iter(metrics.keys()))
    metrics["monitor"] = float(metrics.get(monitor_key, math.inf))

    pred_df = pd.DataFrame({"sample_id": sample_ids})
    for j, name in enumerate(cont_col_names):
        pred_df[f"true_{name}"] = y_cont_true[:, j]
        pred_df[f"pred_{name}"] = y_cont_pred[:, j]
        pred_df[f"oracle_{name}"] = y_oracle[:, j]
        pred_df[f"mask_{name}"] = y_cont_mask[:, j]
    if y_disc_true is not None and y_disc_pred is not None:
        for j, name in enumerate(disc_col_names):
            pred_df[f"true_{name}"] = y_disc_true[:, j]
            pred_df[f"pred_{name}"] = y_disc_pred[:, j]

    return metrics, pred_df


def load_baseline_model(
    ckpt_path: Path,
    x_dim: int,
    y_set_dim: int,
    disc_class_sizes: Sequence[int],
    n_cont: int,
    args: argparse.Namespace,
    device: torch.device,
):
    try:
        payload = torch.load(ckpt_path, map_location=device)
        if isinstance(payload, Mapping):
            cfg = payload.get("model_config", {})
            hidden_dims = cfg.get("hidden_dims") or parse_hidden_dims(getattr(args, "baseline_hidden_dims", "512,256")) or parse_hidden_dims(args.hidden_dims)
            dropout = float(cfg.get("dropout", args.dropout))
            use_layernorm = bool(cfg.get("use_layernorm", args.use_layernorm))
            set_proj_dim = int(cfg.get("set_proj_dim", args.set_proj_dim))
            fuse_mode = str(cfg.get("fuse_mode", args.fuse_mode))
            n_cont = int(cfg.get("n_cont", n_cont))
            if "disc_class_sizes" in cfg:
                disc_class_sizes = list(cfg["disc_class_sizes"])
            model = Stage3ConditionPredictor(
                x_dim=int(cfg.get("x_dim", x_dim)),
                y_set_dim=int(cfg.get("y_set_dim", y_set_dim)),
                hidden_dims=hidden_dims,
                disc_class_sizes=disc_class_sizes,
                n_cont=n_cont,
                dropout=dropout,
                use_layernorm=use_layernorm,
                set_proj_dim=set_proj_dim,
                fuse_mode=fuse_mode,
            ).to(device)
            model.load_state_dict(pick_state_dict(payload), strict=False)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            return model
    except Exception:
        pass

    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(f"Unsupported baseline checkpoint format: {ckpt_path}")
    model = PickleBaselineAdapter(payload=payload, device=device).to(device)
    model.eval()
    return model


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train stage3 residual MDN mixed model with fixed raw/standardized target handling.")
    p.add_argument("--project_root", type=str, default="/Users/wyc/SynPred")
    p.add_argument("--mode_input_root", type=str, default="/Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset")
    p.add_argument("--train_mode", type=str, default="gold_only", choices=TRAIN_MODE_CHOICES)
    p.add_argument("--input_dir", type=str, default="/Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1")
    p.add_argument("--run_dir", type=str, default=str(DEFAULT_RUN_ROOT / "gold_only"))
    p.add_argument("--baseline_ckpt", type=str, default="/Users/wyc/SynPred/runs/stage3/stage3_baseline_commonized_v1/best_model.pkl")
    p.add_argument("--baseline_hidden_dims", type=str, default="512,256")
    p.add_argument("--hidden_dims", type=str, default="512,256")
    p.add_argument("--set_proj_dim", type=int, default=256)
    p.add_argument("--fuse_mode", type=str, default="concat", choices=["concat", "add"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_layernorm", action="store_true")
    p.add_argument("--n_mixtures", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--metric_name", type=str, default="top1_continuous_mean_mae_raw")
    p.add_argument("--n_gen_samples", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--discrete_head_weights", type=str, default="")
    p.add_argument("--continuous_head_weights", type=str, default="")
    p.add_argument("--clip_to_train_range", action="store_true")
    p.add_argument("--targets_are_standardized", type=str, default="auto", choices=["auto", "true", "false"])
    return p


def main() -> None:
    args = build_argparser().parse_args()

    print(f"[Info] input_dir(default-ready) = {args.input_dir}")
    print(f"[Info] run_dir(default-ready)   = {args.run_dir}")
    print(f"[Info] baseline(default-ready)  = {args.baseline_ckpt}")

    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    resolved = resolve_input_paths(args, required=STAGE3_MIXED_REQUIRED, optional=STAGE3_MIXED_OPTIONAL)
    files = resolved.files
    print(f"[Info] resolved_mode = {resolved.resolved_mode}")
    print(f"[Info] resolved_root = {resolved.resolved_root}")
    print(f"[Info] resolved_input_dir = {resolved.resolved_input_dir}")

    schema = load_schema(Path(files["schema"]))
    train_ds = Stage3MixedDataset(Path(files["train_npz"]))
    val_ds = Stage3MixedDataset(Path(files["val_npz"]))
    test_ds = Stage3MixedDataset(Path(files["test_npz"]))

    disc_col_names = list(schema.get("discrete_cols", []))
    cont_col_names = list(schema.get("continuous_cols", []))
    disc_schema = schema.get("discrete_schema", {})
    disc_class_sizes = [int(disc_schema[name].get("n_classes", int(train_ds.y_disc[:, i].max().item()) + 1)) for i, name in enumerate(disc_col_names)]
    if not cont_col_names:
        cont_col_names = [f"cont_{i}" for i in range(train_ds.y_cont.shape[1])]
    if not disc_col_names and train_ds.y_disc.shape[1] > 0:
        disc_col_names = [f"disc_{i}" for i in range(train_ds.y_disc.shape[1])]
        disc_class_sizes = [int(train_ds.y_disc[:, i].max().item()) + 1 for i in range(train_ds.y_disc.shape[1])]

    cont_stats = extract_cont_stats(schema, cont_col_names, train_ds)
    if str(args.targets_are_standardized).lower() == "true":
        targets_are_standardized = True
    elif str(args.targets_are_standardized).lower() == "false":
        targets_are_standardized = False
    else:
        targets_are_standardized = detect_targets_are_standardized(train_ds, cont_stats)
    print(f"[Info] targets_are_standardized = {targets_are_standardized}")

    train_min_raw, train_max_raw = collect_train_raw_minmax(train_ds, cont_stats, targets_are_standardized)

    device = choose_device(args.device)
    pin_memory = device.type == "cuda"
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    va_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    te_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    baseline = load_baseline_model(
        ckpt_path=Path(args.baseline_ckpt),
        x_dim=int(train_ds.x.shape[1]),
        y_set_dim=int(train_ds.y_set.shape[1]),
        disc_class_sizes=disc_class_sizes,
        n_cont=int(train_ds.y_cont.shape[1]),
        args=args,
        device=device,
    )

    model = ResidualConditionMDNMixed(
        x_dim=int(train_ds.x.shape[1]),
        y_set_dim=int(train_ds.y_set.shape[1]),
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        disc_class_sizes=disc_class_sizes,
        y_cont_dim=int(train_ds.y_cont.shape[1]),
        n_mixtures=args.n_mixtures,
        dropout=args.dropout,
        use_layernorm=args.use_layernorm,
        set_proj_dim=args.set_proj_dim,
        fuse_mode=args.fuse_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    disc_head_weights = [1.0] * len(disc_col_names)
    cont_head_weights = [1.0] * len(cont_col_names)
    if args.discrete_head_weights.strip():
        vals = [float(x) for x in args.discrete_head_weights.split(",") if x.strip()]
        if len(vals) != len(disc_col_names):
            raise ValueError("discrete_head_weights length must match number of discrete heads")
        disc_head_weights = vals
    if args.continuous_head_weights.strip():
        vals = [float(x) for x in args.continuous_head_weights.split(",") if x.strip()]
        if len(vals) != len(cont_col_names):
            raise ValueError("continuous_head_weights length must match number of continuous heads")
        cont_head_weights = vals
    class_weight_tensors = build_class_weight_tensor(train_ds.y_disc.numpy(), disc_class_sizes, device) if (args.use_class_weights and disc_col_names) else None
    hw_t = torch.tensor(cont_head_weights, device=device, dtype=torch.float32)

    full_train_rows = int(((train_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    full_val_rows = int(((val_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    full_test_rows = int(((test_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    print(
        f"[Info] device={device} | n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_ds)} | "
        f"full-mask rows train/val/test = {full_train_rows}/{full_val_rows}/{full_test_rows}"
    )

    early_stopper = EarlyStopper(patience=args.patience, minimize=True)
    best_score = None
    best_epoch = -1
    best_ckpt_path = run_dir / "best_stage3_residual_mdn_mixed.pt"
    train_log: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        for batch in tr_loader:
            x = batch["x"].to(device)
            y_set = batch["y_set"].to(device)
            y_disc = batch["y_disc"].to(device)
            y_cont = batch["y_cont"].to(device)
            y_mask = batch["y_cont_mask"].to(device)

            with torch.no_grad():
                base_out = baseline(x, y_set)
                base_cont = base_out["cont_pred"].detach()

            resid_cont_target = y_cont - base_cont
            out = model(x, y_set)
            loss_cont = model.mdn.masked_nll(out["pi_logits"], out["mu"], out["log_sigma"], resid_cont_target, y_mask, hw_t)
            loss_disc = multihead_classification_loss(out["resid_disc_logits"], y_disc, class_weight_tensors, disc_head_weights)
            loss = loss_cont + loss_disc

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics, _ = evaluate_split(
            model=model,
            baseline=baseline,
            loader=va_loader,
            device=device,
            cont_col_names=cont_col_names,
            disc_col_names=disc_col_names,
            cont_stats=cont_stats,
            n_gen_samples=args.n_gen_samples,
            clip_to_train_range=args.clip_to_train_range,
            train_min_raw=train_min_raw,
            train_max_raw=train_max_raw,
            targets_are_standardized=targets_are_standardized,
        )
        current = float(val_metrics.get(args.metric_name, val_metrics.get("monitor", math.inf)))
        improved, should_stop = early_stopper.step(current)
        if improved:
            best_score = current
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "x_dim": int(train_ds.x.shape[1]),
                    "y_set_dim": int(train_ds.y_set.shape[1]),
                    "hidden_dims": parse_hidden_dims(args.hidden_dims),
                    "disc_class_sizes": disc_class_sizes,
                    "y_cont_dim": int(train_ds.y_cont.shape[1]),
                    "n_mixtures": int(args.n_mixtures),
                    "dropout": float(args.dropout),
                    "use_layernorm": bool(args.use_layernorm),
                    "set_proj_dim": int(args.set_proj_dim),
                    "fuse_mode": str(args.fuse_mode),
                },
                "best_metric_value": float(current),
                "epoch": int(epoch),
                "cont_col_names": cont_col_names,
                "disc_col_names": disc_col_names,
                "cont_stats": cont_stats,
                "targets_are_standardized": bool(targets_are_standardized),
                "args": vars(args),
                "resolved": {
                    "resolved_mode": resolved.resolved_mode,
                    "resolved_root": resolved.resolved_root,
                    "resolved_input_dir": resolved.resolved_input_dir,
                    "files": files,
                },
            }, best_ckpt_path)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "val_metrics": val_metrics,
            "monitor": current,
            "best": float(best_score) if best_score is not None else None,
        }
        train_log.append(row)
        print(f"[Epoch {epoch:03d}] train_loss={row['train_loss']:.4f} val_{args.metric_name}={current:.4f} best={best_score:.4f}")
        if should_stop:
            print(f"[Early Stop] patience reached at epoch {epoch}")
            break

    if best_score is None:
        raise RuntimeError("Training failed: no checkpoint saved.")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_pred = evaluate_split(
        model, baseline, va_loader, device, cont_col_names, disc_col_names, cont_stats,
        args.n_gen_samples, args.clip_to_train_range, train_min_raw, train_max_raw, targets_are_standardized
    )
    test_metrics, test_pred = evaluate_split(
        model, baseline, te_loader, device, cont_col_names, disc_col_names, cont_stats,
        args.n_gen_samples, args.clip_to_train_range, train_min_raw, train_max_raw, targets_are_standardized
    )

    write_json(run_dir / "train_log.json", train_log)
    val_pred.to_csv(run_dir / "pred_val.csv", index=False)
    test_pred.to_csv(run_dir / "pred_test.csv", index=False)

    summary = {
        "config": vars(args),
        "resolved_mode": resolved.resolved_mode,
        "resolved_root": resolved.resolved_root,
        "resolved_input_dir": resolved.resolved_input_dir,
        "resolved_files": files,
        "targets_are_standardized": bool(targets_are_standardized),
        "data": {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "x_dim": int(train_ds.x.shape[1]),
            "y_set_dim": int(train_ds.y_set.shape[1]),
            "n_discrete_heads": int(train_ds.y_disc.shape[1]),
            "n_continuous_heads": int(train_ds.y_cont.shape[1]),
            "disc_col_names": disc_col_names,
            "cont_col_names": cont_col_names,
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_metric": float(best_score),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
