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
import copy
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


def _residual_clip_train(delta, clip_q: float = 0.5):
    import numpy as _np
    d = _np.asarray(delta, dtype=_np.float32)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    if clip_q is None or clip_q <= 0:
        lo = _np.min(d, axis=0)
        hi = _np.max(d, axis=0)
    else:
        lo = _np.percentile(d, clip_q, axis=0)
        hi = _np.percentile(d, 100.0 - clip_q, axis=0)
    d = _np.clip(d, lo, hi)
    mean = d.mean(axis=0).astype(_np.float32)
    std = d.std(axis=0).astype(_np.float32)
    std = _np.where(std < 1e-6, 1.0, std).astype(_np.float32)
    return d.astype(_np.float32), mean, std, lo.astype(_np.float32), hi.astype(_np.float32)


def _residual_standardize(delta, mean, std):
    import numpy as _np
    d = _np.asarray(delta, dtype=_np.float32)
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    m = _np.asarray(mean, dtype=_np.float32).reshape(1, -1)
    s = _np.asarray(std, dtype=_np.float32).reshape(1, -1)
    s = _np.where(s < 1e-6, 1.0, s)
    return ((d - m) / s).astype(_np.float32)


def _residual_destandardize(delta_z, mean, std):
    import numpy as _np
    z = _np.asarray(delta_z, dtype=_np.float32)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    m = _np.asarray(mean, dtype=_np.float32).reshape(1, -1)
    s = _np.asarray(std, dtype=_np.float32).reshape(1, -1)
    s = _np.where(s < 1e-6, 1.0, s)
    return (z * s + m).astype(_np.float32)


TRAIN_MODE_CHOICES = [
    "relaxed_only",
    "gold_only",
    "curriculum",
    "curriculum_phase1",
    "curriculum_phase2",
]

DEFAULT_PROJECT_ROOT = Path("/Users/wyc/SynPred")
DEFAULT_RUN_ROOT = Path("/Users/wyc/SynPred/runs/stage3/train_condition_residual_flow_mixed")

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


class BaselineCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "Stage3BaselineModel":
            return Stage3BaselineModel
        return super().find_class(module, name)


def load_pickle_baseline_payload(path: Path) -> Mapping[str, Any]:
    with open(path, "rb") as f:
        payload = BaselineCompatUnpickler(f).load()
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(f"Unsupported baseline checkpoint format: {path}")
    return payload


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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
        return {"disc_logits": [head(h) for head in self.disc_heads], "cont_pred": self.cont_head(h)}


class ResidualContextEncoder(nn.Module):
    def __init__(self, x_dim: int, y_set_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1, use_layernorm: bool = False, set_proj_dim: int = 256, fuse_mode: str = "concat"):
        super().__init__()
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
            trunk_dims = [trunk_in_dim, max(trunk_in_dim, 128)]
        self.trunk = MLP(trunk_dims, dropout=dropout, use_layernorm=use_layernorm)
        self.out_dim = trunk_dims[-1]

    def forward(self, x: torch.Tensor, y_set: torch.Tensor) -> torch.Tensor:
        set_repr = self.set_encoder(y_set)
        if self.fuse_mode == "concat":
            fused = torch.cat([x, set_repr], dim=1)
        else:
            fused = self.x_proj(x) + set_repr
        return self.trunk(fused)


class ConditionalDiagonalAffine(nn.Module):
    def __init__(self, dim: int, context_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.net = MLP([context_dim, hidden_dim, hidden_dim, 2 * dim], dropout=dropout, use_layernorm=False)

    def _st(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self.net(context).chunk(2, dim=-1)
        s = 2.0 * torch.tanh(0.5 * s)
        return s, t

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self._st(context)
        y = x * torch.exp(s) + t
        logdet = s.sum(dim=-1)
        return y, logdet

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s, t = self._st(context)
        x = (y - t) * torch.exp(-s)
        logdet = -s.sum(dim=-1)
        return x, logdet


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim: int, context_dim: int, mask: torch.Tensor, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("mask", mask.float().view(1, dim))
        self.net = MLP([dim + context_dim, hidden_dim, hidden_dim, 2 * dim], dropout=dropout, use_layernorm=False)

    def _st(self, x_masked: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x_masked, context], dim=-1)
        s, t = self.net(h).chunk(2, dim=-1)
        s = 2.0 * torch.tanh(0.5 * s)
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        return s, t

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        s, t = self._st(x_masked, context)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=-1)

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_masked = y * self.mask
        s, t = self._st(y_masked, context)
        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        return x, -s.sum(dim=-1)


class ConditionalRealNVP(nn.Module):
    def __init__(self, dim: int, context_dim: int, n_layers: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        layers: List[nn.Module] = []
        if self.dim <= 1:
            for _ in range(n_layers):
                layers.append(ConditionalDiagonalAffine(dim=self.dim, context_dim=context_dim, hidden_dim=hidden_dim, dropout=dropout))
        else:
            for i in range(n_layers):
                mask = torch.tensor([(j + i) % 2 for j in range(self.dim)], dtype=torch.float32)
                if mask.sum() == 0 or mask.sum() == self.dim:
                    mask = 1.0 - mask
                layers.append(ConditionalAffineCoupling(dim=self.dim, context_dim=context_dim, mask=mask, hidden_dim=hidden_dim, dropout=dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, ld = layer(z, context)
            logdet = logdet + ld
        return z, logdet

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, context)
            logdet = logdet + ld
        return x, logdet

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z, logdet = self.forward(x, context)
        log_base = -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)
        return log_base + logdet


class ResidualConditionFlowMixed(nn.Module):
    def __init__(self, x_dim: int, y_set_dim: int, disc_class_sizes: Sequence[int], y_cont_dim: int, hidden_dims: Sequence[int], flow_hidden_dim: int = 256, n_flow_layers: int = 6, dropout: float = 0.1, use_layernorm: bool = False, set_proj_dim: int = 256, fuse_mode: str = "concat"):
        super().__init__()
        self.disc_class_sizes = list(disc_class_sizes)
        self.y_cont_dim = int(y_cont_dim)
        self.context_encoder = ResidualContextEncoder(x_dim=x_dim, y_set_dim=y_set_dim, hidden_dims=hidden_dims, dropout=dropout, use_layernorm=use_layernorm, set_proj_dim=set_proj_dim, fuse_mode=fuse_mode)
        self.flow = ConditionalRealNVP(dim=y_cont_dim, context_dim=self.context_encoder.out_dim, n_layers=n_flow_layers, hidden_dim=flow_hidden_dim, dropout=dropout)
        self.resid_disc_heads = nn.ModuleList([nn.Linear(self.context_encoder.out_dim, k) for k in self.disc_class_sizes])

    def encode_context(self, x: torch.Tensor, y_set: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(x, y_set)

    def forward(self, x: torch.Tensor, y_set: torch.Tensor) -> Dict[str, Any]:
        context = self.encode_context(x, y_set)
        return {"context": context, "resid_disc_logits": [head(context) for head in self.resid_disc_heads]}

    def flow_nll(self, residual: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        full_rows = (mask > 0.5).all(dim=1)
        n_valid = int(full_rows.sum().item())
        if n_valid == 0:
            return residual.sum() * 0.0, 0
        logp = self.flow.log_prob(residual[full_rows], context[full_rows])
        return -logp.mean(), n_valid

    def top1_residual(self, context: torch.Tensor) -> torch.Tensor:
        z0 = torch.zeros(context.shape[0], self.y_cont_dim, device=context.device, dtype=context.dtype)
        x, _ = self.flow.inverse(z0, context)
        return x

    def sample_residual(self, context: torch.Tensor, n_samples: int) -> torch.Tensor:
        b = context.shape[0]
        # FIX: use true stochastic sampling instead of all-zero latent codes
        z = torch.randn(n_samples, b, self.y_cont_dim, device=context.device, dtype=context.dtype)
        z_flat = z.view(n_samples * b, self.y_cont_dim)
        ctx_flat = context.unsqueeze(0).expand(n_samples, b, context.shape[1]).reshape(n_samples * b, context.shape[1])
        x_flat, _ = self.flow.inverse(z_flat, ctx_flat)
        return x_flat.view(n_samples, b, self.y_cont_dim)


class EarlyStopper:
    def __init__(self, patience: int, minimize: bool = True, min_delta: float = 1e-8):
        self.patience = int(patience)
        self.minimize = bool(minimize)
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> Tuple[bool, bool]:
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
    return {
        "continuous_cols": list(schema.get("continuous_schema", {}).keys()),
        "discrete_cols": list(schema.get("discrete_schema", {}).keys()),
        "continuous_schema": schema.get("continuous_schema", {}),
        "discrete_schema": schema.get("discrete_schema", {}),
    }


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


def maybe_inverse_transform_np(
    values: np.ndarray,
    stats: List[Dict[str, float]],
    targets_are_standardized: bool,
) -> np.ndarray:
    out = values.copy().astype(np.float32)
    if not targets_are_standardized:
        return out
    for i, st in enumerate(stats):
        out[:, i] = out[:, i] * float(st["std"]) + float(st["mean"])
    return out


def detect_targets_are_standardized(train_ds: Stage3MixedDataset, cont_stats: List[Dict[str, float]]) -> bool:
    """
    Conservative heuristic:
    Stage3 dataset usually stores normalized continuous targets.
    Return True only when target columns look roughly standardized.
    """
    y = train_ds.y_cont.numpy().astype(np.float32)
    mask = train_ds.y_cont_mask.numpy().astype(np.float32)

    flags = []
    for j in range(y.shape[1]):
        valid = mask[:, j] > 0.5
        if not np.any(valid):
            continue
        col = y[valid, j]
        col_mean = float(np.mean(col))
        col_std = float(np.std(col))
        flags.append(abs(col_mean) < 3.0 and 0.2 < col_std < 5.0)

    return bool(flags and all(flags))


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


def multihead_classification_loss(logits_list: Sequence[torch.Tensor], target: torch.Tensor, class_weight_tensors: Optional[Sequence[torch.Tensor]], head_weights: Sequence[float]) -> torch.Tensor:
    if not logits_list:
        return target.new_tensor(0.0, dtype=torch.float32)
    losses = []
    for j, logits in enumerate(logits_list):
        weight = None if class_weight_tensors is None else class_weight_tensors[j]
        loss_j = F.cross_entropy(logits, target[:, j], weight=weight)
        losses.append(loss_j * float(head_weights[j]))
    return torch.stack(losses).sum() / max(1, len(losses))


def collect_train_raw_minmax(
    train_ds: Stage3MixedDataset,
    cont_stats: List[Dict[str, float]],
    targets_are_standardized: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = maybe_inverse_transform_np(train_ds.y_cont.numpy(), cont_stats, targets_are_standardized)
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


def load_baseline_model(ckpt_path: Path, x_dim: int, y_set_dim: int, disc_class_sizes: Sequence[int], n_cont: int, args: argparse.Namespace, device: torch.device):
    try:
        payload = torch.load(ckpt_path, map_location=device)
        if isinstance(payload, Mapping):
            cfg = payload.get("model_config", payload.get("config", {}))
            hidden_dims = cfg.get("hidden_dims") or parse_hidden_dims(getattr(args, "baseline_hidden_dims", "512,256")) or parse_hidden_dims(args.hidden_dims)
            model = Stage3ConditionPredictor(
                x_dim=int(cfg.get("x_dim", x_dim)),
                y_set_dim=int(cfg.get("y_set_dim", y_set_dim)),
                hidden_dims=hidden_dims,
                disc_class_sizes=list(cfg.get("disc_class_sizes", disc_class_sizes)),
                n_cont=int(cfg.get("n_cont", n_cont)),
                dropout=float(cfg.get("dropout", args.dropout)),
                use_layernorm=bool(cfg.get("use_layernorm", args.use_layernorm)),
                set_proj_dim=int(cfg.get("set_proj_dim", args.set_proj_dim)),
                fuse_mode=str(cfg.get("fuse_mode", args.fuse_mode)),
            ).to(device)
            model.load_state_dict(pick_state_dict(payload), strict=False)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            return model
    except Exception:
        pass

    payload = load_pickle_baseline_payload(ckpt_path)
    model = PickleBaselineAdapter(payload=payload, device=device).to(device)
    model.eval()
    return model

def build_baseline_from_checkpoint(
    baseline_ckpt: str,
    x_dim: int,
    y_set_dim: int,
    disc_class_sizes,
    n_cont: int,
    *args,
    hidden_dims="512,256",
    dropout: float = 0.1,
    use_layernorm: bool = False,
    set_proj_dim: int = 256,
    fuse_mode: str = "concat",
    device: str | torch.device = "cpu",
    **kwargs,
):
    """
    Compatibility wrapper for
    39_export_stage3_precursor_conditioned_candidates_mixture_flow.py

    Supports both call styles:
      1) build_baseline_from_checkpoint(..., device)
      2) build_baseline_from_checkpoint(..., hidden_dims=..., dropout=..., ..., device=...)
    """
    # 兼容 39 里把 device 当作第 6 个位置参数传进来的情况
    if len(args) >= 1:
        first = args[0]
        if isinstance(first, torch.device):
            device = first
        elif isinstance(first, str) and first in {"cpu", "cuda", "mps"}:
            device = first
        else:
            hidden_dims = first

    if isinstance(device, str):
        device = torch.device(device)

    class _Args:
        pass

    fake_args = _Args()

    if isinstance(hidden_dims, str):
        fake_args.hidden_dims = hidden_dims
        fake_args.baseline_hidden_dims = hidden_dims
    else:
        fake_args.hidden_dims = ",".join(str(x) for x in hidden_dims)
        fake_args.baseline_hidden_dims = fake_args.hidden_dims

    fake_args.dropout = float(dropout)
    fake_args.use_layernorm = bool(use_layernorm)
    fake_args.set_proj_dim = int(set_proj_dim)
    fake_args.fuse_mode = str(fuse_mode)

    model = load_baseline_model(
        ckpt_path=Path(baseline_ckpt).expanduser().resolve(),
        x_dim=int(x_dim),
        y_set_dim=int(y_set_dim),
        disc_class_sizes=list(disc_class_sizes),
        n_cont=int(n_cont),
        args=fake_args,
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def _raw_metric_aliases(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out = dict(metrics)
    if "mae_mean" in metrics:
        out[f"{prefix}_mean_mae_raw"] = float(metrics["mae_mean"])
    if "rmse_mean" in metrics:
        out[f"{prefix}_mean_rmse_raw"] = float(metrics["rmse_mean"])
    return out


@torch.no_grad()
def evaluate_split(
    baseline: Stage3ConditionPredictor,
    model: ResidualConditionFlowMixed,
    loader: DataLoader,
    device: torch.device,
    disc_col_names: Sequence[str],
    cont_col_names: Sequence[str],
    cont_stats: List[Dict[str, float]],
    disc_head_weights: Sequence[float],
    cont_head_weights: Sequence[float],
    class_weight_tensors: Optional[Sequence[torch.Tensor]],
    n_gen_samples: int,
    train_min_raw: Optional[np.ndarray],
    train_max_raw: Optional[np.ndarray],
    clip_to_train_range: bool,
    targets_are_standardized: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model.eval()
    baseline.eval()

    total_loss = 0.0
    total_count = 0
    total_valid_flow_rows = 0

    sample_ids_all: List[str] = []
    disc_true_all: List[np.ndarray] = []
    disc_base_all: List[np.ndarray] = []
    disc_top1_all: List[np.ndarray] = []
    disc_oracle_all: List[np.ndarray] = []
    cont_true_norm_all: List[np.ndarray] = []
    mask_all: List[np.ndarray] = []
    cont_base_norm_all: List[np.ndarray] = []
    cont_top1_norm_all: List[np.ndarray] = []
    cont_oracle_norm_all: List[np.ndarray] = []
    cont_oracle_raw_all: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        y_set = batch["y_set"].to(device)
        y_disc = batch["y_disc"].to(device)
        y_cont = batch["y_cont"].to(device)
        y_mask = batch["y_cont_mask"].to(device)
        batch_ids = [str(v) for v in batch["sample_id"]]

        base_out = baseline(x, y_set)
        base_disc_logits = [logits.detach() for logits in base_out["disc_logits"]]
        base_disc = torch.stack([torch.argmax(logits, dim=1) for logits in base_disc_logits], dim=1) if base_disc_logits else torch.zeros((x.shape[0], 0), dtype=torch.long, device=x.device)
        base_cont = base_out["cont_pred"].detach()

        resid_cont_target = y_cont - base_cont
        out = model(x, y_set)
        context = out["context"]

        final_disc_logits = [b + r for b, r in zip(base_disc_logits, out["resid_disc_logits"])]
        loss_disc = multihead_classification_loss(final_disc_logits, y_disc, class_weight_tensors, disc_head_weights) if final_disc_logits else x.new_tensor(0.0)
        loss_flow, n_valid = model.flow_nll(resid_cont_target, context, y_mask)
        total_valid_flow_rows += n_valid
        loss = loss_disc + loss_flow
        total_loss += float(loss.item()) * x.shape[0]
        total_count += x.shape[0]

        top1_residual = model.top1_residual(context)
        top1_cont = base_cont + top1_residual
        top1_disc = torch.stack([torch.argmax(logits, dim=1) for logits in final_disc_logits], dim=1) if final_disc_logits else torch.zeros((x.shape[0], 0), dtype=torch.long, device=x.device)

        y_cont_np = y_cont.detach().cpu().numpy().astype(np.float32)
        y_mask_np = y_mask.detach().cpu().numpy().astype(np.float32)
        cont_true_raw = maybe_inverse_transform_np(y_cont_np, cont_stats, targets_are_standardized)

        cont_top1_norm_np = top1_cont.detach().cpu().numpy().astype(np.float32)
        cont_top1_raw = maybe_inverse_transform_np(cont_top1_norm_np, cont_stats, targets_are_standardized)
        if clip_to_train_range and train_min_raw is not None and train_max_raw is not None:
            cont_top1_raw = clip_continuous_to_train_range(cont_top1_raw, train_min_raw, train_max_raw)

        if int(n_gen_samples) > 1:
            sample_residual = model.sample_residual(context, n_samples=int(n_gen_samples))
            cand_cont = base_cont.unsqueeze(0) + sample_residual
            cand_np = cand_cont.detach().cpu().numpy().astype(np.float32)
            oracle_rows_raw = []
            oracle_rows_norm = []
            for i in range(cand_np.shape[1]):
                cand_i_norm = cand_np[:, i, :]
                cand_i_raw = maybe_inverse_transform_np(cand_i_norm, cont_stats, targets_are_standardized)
                if clip_to_train_range and train_min_raw is not None and train_max_raw is not None:
                    cand_i_raw = clip_continuous_to_train_range(cand_i_raw, train_min_raw, train_max_raw)
                valid = y_mask_np[i] > 0.5
                if np.any(valid):
                    errs = np.mean(np.abs(cand_i_raw[:, valid] - cont_true_raw[i][None, valid]), axis=1)
                    best_idx = int(np.argmin(errs))
                else:
                    best_idx = 0
                oracle_rows_raw.append(cand_i_raw[best_idx])
                oracle_rows_norm.append(cand_i_norm[best_idx])
            cont_oracle_raw_batch = np.vstack(oracle_rows_raw).astype(np.float32)
            cont_oracle_norm_batch = np.vstack(oracle_rows_norm).astype(np.float32)
        else:
            cont_oracle_raw_batch = cont_top1_raw.copy()
            cont_oracle_norm_batch = cont_top1_norm_np.copy()

        sample_ids_all.extend(batch_ids)
        disc_true_all.append(y_disc.detach().cpu().numpy().astype(np.int64))
        disc_base_all.append(base_disc.detach().cpu().numpy().astype(np.int64))
        disc_top1_all.append(top1_disc.detach().cpu().numpy().astype(np.int64))
        disc_oracle_all.append(top1_disc.detach().cpu().numpy().astype(np.int64))
        cont_true_norm_all.append(y_cont_np)
        mask_all.append(y_mask_np)
        cont_base_norm_all.append(base_cont.detach().cpu().numpy().astype(np.float32))
        cont_top1_norm_all.append(cont_top1_norm_np)
        cont_oracle_norm_all.append(cont_oracle_norm_batch.astype(np.float32))
        cont_oracle_raw_all.append(cont_oracle_raw_batch.astype(np.float32))

    disc_true = np.vstack(disc_true_all) if disc_true_all else np.zeros((0, 0), dtype=np.int64)
    disc_base = np.vstack(disc_base_all) if disc_base_all else np.zeros((0, 0), dtype=np.int64)
    disc_top1 = np.vstack(disc_top1_all) if disc_top1_all else np.zeros((0, 0), dtype=np.int64)
    disc_oracle = np.vstack(disc_oracle_all) if disc_oracle_all else np.zeros((0, 0), dtype=np.int64)
    cont_true_norm = np.vstack(cont_true_norm_all).astype(np.float32)
    mask = np.vstack(mask_all).astype(np.float32)
    cont_base_norm = np.vstack(cont_base_norm_all).astype(np.float32)
    cont_top1_norm = np.vstack(cont_top1_norm_all).astype(np.float32)
    cont_oracle_norm = np.vstack(cont_oracle_norm_all).astype(np.float32)
    cont_oracle_raw = np.vstack(cont_oracle_raw_all).astype(np.float32) if cont_oracle_raw_all else None

    cont_true_raw = maybe_inverse_transform_np(cont_true_norm, cont_stats, targets_are_standardized)
    cont_base_raw = maybe_inverse_transform_np(cont_base_norm, cont_stats, targets_are_standardized)
    cont_top1_raw = maybe_inverse_transform_np(cont_top1_norm, cont_stats, targets_are_standardized)
    if cont_oracle_raw is None:
        cont_oracle_raw = maybe_inverse_transform_np(cont_oracle_norm, cont_stats, targets_are_standardized)
    if clip_to_train_range and train_min_raw is not None and train_max_raw is not None:
        cont_top1_raw = clip_continuous_to_train_range(cont_top1_raw, train_min_raw, train_max_raw)
        cont_oracle_raw = clip_continuous_to_train_range(cont_oracle_raw, train_min_raw, train_max_raw)

    loss_mean = total_loss / max(total_count, 1)
    metrics: Dict[str, Any] = {"loss": float(loss_mean), "n_valid_flow_rows": int(total_valid_flow_rows)}

    top1_cont = evaluate_mixed_conditions(
        y_cont_true=cont_true_raw,
        y_cont_pred=cont_top1_raw,
        y_cont_mask=mask,
        cont_target_names=cont_col_names,
    )
    # Robust discrete evaluation:
    # Some residual-flow variants only model continuous residuals and may return
    # no discrete logits. In that case disc_top1 has shape (N, 0), while
    # disc_col_names can still contain task names from schema. Do not evaluate
    # discrete metrics unless predictions have enough columns.
    has_top1_disc_pred = (
        bool(disc_col_names)
        and isinstance(disc_top1, np.ndarray)
        and disc_top1.ndim == 2
        and disc_top1.shape[1] >= len(disc_col_names)
    )
    top1_disc_m = evaluate_mixed_conditions(
        y_disc_true=disc_true,
        y_disc_pred=disc_top1,
        disc_target_names=disc_col_names,
    ) if has_top1_disc_pred else {}
    oracle_cont = evaluate_mixed_conditions(
        y_cont_true=cont_true_raw,
        y_cont_pred=cont_oracle_raw,
        y_cont_mask=mask,
        cont_target_names=cont_col_names,
    )
    base_cont = evaluate_mixed_conditions(
        y_cont_true=cont_true_raw,
        y_cont_pred=cont_base_raw,
        y_cont_mask=mask,
        cont_target_names=cont_col_names,
    )
    def _has_disc_pred(pred):
        return (
            bool(disc_col_names)
            and isinstance(pred, np.ndarray)
            and pred.ndim == 2
            and pred.shape[1] >= len(disc_col_names)
        )

    base_disc_m = evaluate_mixed_conditions(
        y_disc_true=disc_true,
        y_disc_pred=disc_base,
        disc_target_names=disc_col_names,
    ) if _has_disc_pred(disc_base) else {}

    metrics.update(_raw_metric_aliases(top1_cont, "top1_continuous"))
    metrics.update(_raw_metric_aliases(oracle_cont, "oracle_best_of_k_continuous"))
    metrics.update(_raw_metric_aliases(base_cont, "baseline_continuous"))
    metrics.update(top1_cont)
    if "disc_macro_f1_mean" in top1_disc_m:
        metrics["top1_discrete_mean_macro_f1"] = float(top1_disc_m["disc_macro_f1_mean"])
        metrics["top1_discrete_mean_accuracy"] = float(top1_disc_m.get("disc_accuracy_mean", math.nan))
    if "disc_macro_f1_mean" in base_disc_m:
        metrics["baseline_discrete_mean_macro_f1"] = float(base_disc_m["disc_macro_f1_mean"])
        metrics["baseline_discrete_mean_accuracy"] = float(base_disc_m.get("disc_accuracy_mean", math.nan))

    metrics["top1_continuous_per_head_raw"] = {
        name: {
            "mae": float(top1_cont.get(f"{name}_mae", math.nan)),
            "rmse": float(top1_cont.get(f"{name}_rmse", math.nan)),
        }
        for name in cont_col_names
    }
    metrics["oracle_best_of_k_continuous_per_head_raw"] = {
        name: {
            "mae": float(oracle_cont.get(f"{name}_mae", math.nan)),
            "rmse": float(oracle_cont.get(f"{name}_rmse", math.nan)),
        }
        for name in cont_col_names
    }
    if disc_col_names:
        metrics["top1_discrete_per_head"] = {
            name: {
                "accuracy": float(top1_disc_m.get(f"{name}_accuracy", math.nan)),
                "macro_f1": float(top1_disc_m.get(f"{name}_macro_f1", math.nan)),
            }
            for name in disc_col_names
        }

    arrays = {
        "sample_id": np.asarray(sample_ids_all, dtype=object),
        "disc_true": disc_true,
        "disc_base": disc_base,
        "disc_top1": disc_top1,
        "disc_oracle": disc_oracle,
        "cont_true_norm": cont_true_norm,
        "cont_true_raw": cont_true_raw,
        "mask": mask,
        "cont_base_norm": cont_base_norm,
        "cont_base_raw": cont_base_raw,
        "cont_top1_norm": cont_top1_norm,
        "cont_top1_raw": cont_top1_raw,
        "cont_oracle_norm": cont_oracle_norm,
        "cont_oracle_raw": cont_oracle_raw,
    }
    return metrics, arrays


def save_predictions(path: Path, arrays: Dict[str, Any], disc_col_names: Sequence[str], cont_col_names: Sequence[str]) -> None:
    df = pd.DataFrame({"sample_id": [str(x) for x in arrays["sample_id"].tolist()]})

    def _has_col(key: str, i: int) -> bool:
        arr = arrays.get(key, None)
        return isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] > i

    # Some baselines/adapters may not return discrete logits.
    # Save discrete columns only when the corresponding prediction arrays exist.
    for i, name in enumerate(disc_col_names):
        if _has_col("disc_true", i):
            df[f"true_{name}"] = arrays["disc_true"][:, i]
        if _has_col("disc_base", i):
            df[f"baseline_{name}"] = arrays["disc_base"][:, i]
        if _has_col("disc_top1", i):
            df[f"top1_{name}"] = arrays["disc_top1"][:, i]
        if _has_col("disc_oracle", i):
            df[f"oracle_best_of_k_{name}"] = arrays["disc_oracle"][:, i]

    for i, name in enumerate(cont_col_names):
        df[f"true_{name}_norm"] = arrays["cont_true_norm"][:, i]
        df[f"true_{name}_raw"] = arrays["cont_true_raw"][:, i]
        df[f"mask_{name}"] = arrays["mask"][:, i]
        df[f"baseline_{name}_norm"] = arrays["cont_base_norm"][:, i]
        df[f"baseline_{name}_raw"] = arrays["cont_base_raw"][:, i]
        df[f"top1_{name}_norm"] = arrays["cont_top1_norm"][:, i]
        df[f"top1_{name}_raw"] = arrays["cont_top1_raw"][:, i]
        df[f"oracle_best_of_k_{name}_norm"] = arrays["cont_oracle_norm"][:, i]
        df[f"oracle_best_of_k_{name}_raw"] = arrays["cont_oracle_raw"][:, i]

    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def choose_metric_value(metrics: Dict[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Unsupported metric_name: {metric_name}")
    return float(metrics[metric_name])


def is_minimize_metric(metric_name: str) -> bool:
    return ("mae" in metric_name) or ("rmse" in metric_name) or (metric_name == "loss")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train residual stage3 mixed flow with shared common_io/common_metrics.")
    p.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    p.add_argument("--input_mode", type=str, default="/Users/wyc/SynPred")
    p.add_argument("--input_dir", type=str, default="/Users/wyc/SynPred/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1")
    p.add_argument("--mode_input_root", type=str, default=str(DEFAULT_PROJECT_ROOT / "data/interim/generative/stage3_condition_dataset/hybrid_core"))
    p.add_argument("--train_mode", type=str, default="gold_only", choices=TRAIN_MODE_CHOICES)
    p.add_argument("--baseline_ckpt", type=str, default="/Users/wyc/SynPred/runs/stage3/stage3_baseline_commonized_v1/best_model.pkl")
    p.add_argument("--run_dir", type=str, default="/Users/wyc/SynPred/runs/stage3/train_condition_residual_flow_mixed/gold_only")
    p.add_argument("--baseline_hidden_dims", type=str, default="512,256")
    p.add_argument("--hidden_dims", type=str, default="512,256")
    p.add_argument("--set_proj_dim", type=int, default=256)
    p.add_argument("--flow_hidden_dim", type=int, default=256)
    p.add_argument("--n_flow_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use_layernorm", action="store_true")
    p.add_argument("--fuse_mode", type=str, default="concat", choices=["concat", "add"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--metric_name", type=str, default="top1_continuous_mean_mae_raw")
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--discrete_head_weights", type=str, default="")
    p.add_argument("--continuous_head_weights", type=str, default="")
    p.add_argument("--n_gen_samples", type=int, default=8)
    p.add_argument("--clip_to_train_range", action="store_true")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--targets_are_standardized", type=str, default="auto", choices=["auto", "true", "false"])
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    print(f"[Info] input_dir(default-ready) = {args.input_dir}")
    print(f"[Info] run_dir(default-ready)   = {args.run_dir}")
    print(f"[Info] baseline(default-ready)  = {args.baseline_ckpt}")

    resolved = resolve_input_paths(args, required=STAGE3_MIXED_REQUIRED, optional=STAGE3_MIXED_OPTIONAL)
    files = resolved.files

    run_dir = Path(args.run_dir) if str(args.run_dir).strip() else (DEFAULT_RUN_ROOT / resolved.resolved_mode)
    ensure_dir(run_dir)

    print(f"[Info] resolved_mode = {resolved.resolved_mode}")
    print(f"[Info] resolved_root = {resolved.resolved_root}")
    print(f"[Info] resolved_input_dir = {resolved.resolved_input_dir}")
    schema = load_schema(Path(files["schema"]))
    train_ds = Stage3MixedDataset(Path(files["train_npz"]))
    val_ds = Stage3MixedDataset(Path(files["val_npz"]))
    test_ds = Stage3MixedDataset(Path(files["test_npz"]))

    disc_col_names = list(schema.get("discrete_cols", []))
    cont_col_names = list(schema.get("continuous_cols", []))
    if not cont_col_names:
        cont_col_names = [f"cont_{i}" for i in range(train_ds.y_cont.shape[1])]
    if not disc_col_names and train_ds.y_disc.shape[1] > 0:
        disc_col_names = [f"disc_{i}" for i in range(train_ds.y_disc.shape[1])]

    disc_schema = schema.get("discrete_schema", {})
    disc_class_sizes = [int(disc_schema.get(name, {}).get("n_classes", int(train_ds.y_disc[:, i].max().item()) + 1)) for i, name in enumerate(disc_col_names)]

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

    model = ResidualConditionFlowMixed(
        x_dim=int(train_ds.x.shape[1]),
        y_set_dim=int(train_ds.y_set.shape[1]),
        disc_class_sizes=disc_class_sizes,
        y_cont_dim=int(train_ds.y_cont.shape[1]),
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        flow_hidden_dim=int(args.flow_hidden_dim),
        n_flow_layers=int(args.n_flow_layers),
        dropout=float(args.dropout),
        use_layernorm=bool(args.use_layernorm),
        set_proj_dim=int(args.set_proj_dim),
        fuse_mode=str(args.fuse_mode),
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

    class_weight_tensors = build_class_weight_tensor(train_ds.y_disc.numpy(), disc_class_sizes, device) if (args.use_class_weights and len(disc_col_names) > 0) else None

    early_stopper = EarlyStopper(patience=args.patience, minimize=is_minimize_metric(args.metric_name))
    best_epoch = -1
    best_score: Optional[float] = None
    train_log: List[Dict[str, Any]] = []

    full_train_rows = int(((train_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    full_val_rows = int(((val_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    full_test_rows = int(((test_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())

    print(
        f"[Info] device={device} | n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_ds)} | "
        f"full-mask rows train/val/test = {full_train_rows}/{full_val_rows}/{full_test_rows}"
    )

    best_ckpt_path = run_dir / "best_stage3_residual_flow_mixed.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        valid_rows_seen = 0
        for batch in tr_loader:
            x = batch["x"].to(device)
            y_set = batch["y_set"].to(device)
            y_disc = batch["y_disc"].to(device)
            y_cont = batch["y_cont"].to(device)
            y_mask = batch["y_cont_mask"].to(device)

            with torch.no_grad():
                base_out = baseline(x, y_set)
                base_disc_logits = [logits.detach() for logits in base_out["disc_logits"]]
                base_cont = base_out["cont_pred"].detach()

            resid_cont_target = y_cont - base_cont
            out = model(x, y_set)
            context = out["context"]

            final_disc_logits = [b + r for b, r in zip(base_disc_logits, out["resid_disc_logits"])]
            loss_disc = multihead_classification_loss(final_disc_logits, y_disc, class_weight_tensors, disc_head_weights) if final_disc_logits else x.new_tensor(0.0)
            loss_flow, n_valid = model.flow_nll(resid_cont_target, context, y_mask)
            loss = loss_disc + loss_flow

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))
            valid_rows_seen += int(n_valid)

        val_metrics, _ = evaluate_split(
            baseline=baseline,
            model=model,
            loader=va_loader,
            device=device,
            disc_col_names=disc_col_names,
            cont_col_names=cont_col_names,
            cont_stats=cont_stats,
            disc_head_weights=disc_head_weights,
            cont_head_weights=cont_head_weights,
            class_weight_tensors=class_weight_tensors,
            n_gen_samples=args.n_gen_samples,
            train_min_raw=train_min_raw,
            train_max_raw=train_max_raw,
            clip_to_train_range=args.clip_to_train_range,
            targets_are_standardized=targets_are_standardized,
        )
        score = choose_metric_value(val_metrics, args.metric_name)
        improved, should_stop = early_stopper.step(score)
        if improved:
            best_epoch = epoch
            best_score = score
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "config": vars(args),
                    "epoch": epoch,
                    "best_score": float(score),
                    "metric_name": args.metric_name,
                    "schema": {
                        "disc_col_names": disc_col_names,
                        "disc_class_sizes": disc_class_sizes,
                        "cont_col_names": cont_col_names,
                        "cont_stats": cont_stats,
                    },
                    "baseline_ckpt": str(args.baseline_ckpt),
                    "targets_are_standardized": bool(targets_are_standardized),
                },
                best_ckpt_path,
            )

        train_rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "train_flow_rows": int(valid_rows_seen),
            "val_metrics": val_metrics,
            "best_metric_so_far": float(early_stopper.best) if early_stopper.best is not None else math.nan,
        }
        train_log.append(train_rec)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_rec['train_loss']:.4f} "
            f"train_flow_rows={valid_rows_seen} "
            f"val_{args.metric_name}={float(val_metrics.get(args.metric_name, float('nan'))):.4f} "
            f"best_{args.metric_name}={float(early_stopper.best):.4f}"
        )
        if should_stop:
            print(f"[Early Stop] patience reached at epoch {epoch}")
            break

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    write_json(run_dir / "train_log.json", train_log)

    train_metrics, train_arrays = evaluate_split(baseline, model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory), device, disc_col_names, cont_col_names, cont_stats, disc_head_weights, cont_head_weights, class_weight_tensors, args.n_gen_samples, train_min_raw, train_max_raw, args.clip_to_train_range, targets_are_standardized)
    val_metrics, val_arrays = evaluate_split(baseline, model, va_loader, device, disc_col_names, cont_col_names, cont_stats, disc_head_weights, cont_head_weights, class_weight_tensors, args.n_gen_samples, train_min_raw, train_max_raw, args.clip_to_train_range, targets_are_standardized)
    test_metrics, test_arrays = evaluate_split(baseline, model, te_loader, device, disc_col_names, cont_col_names, cont_stats, disc_head_weights, cont_head_weights, class_weight_tensors, args.n_gen_samples, train_min_raw, train_max_raw, args.clip_to_train_range, targets_are_standardized)

    save_predictions(run_dir / "train_predictions.csv", train_arrays, disc_col_names, cont_col_names)
    save_predictions(run_dir / "val_predictions.csv", val_arrays, disc_col_names, cont_col_names)
    save_predictions(run_dir / "test_predictions.csv", test_arrays, disc_col_names, cont_col_names)

    config_snapshot = {
        **vars(args),
        "resolved_mode": resolved.resolved_mode,
        "resolved_root": resolved.resolved_root,
        "resolved_input_dir": resolved.resolved_input_dir,
        "resolved_schema_path": str(files["schema"]),
        "targets_are_standardized": targets_are_standardized,
    }
    write_json(run_dir / "config_snapshot.json", config_snapshot)

    summary = {
        "config": config_snapshot,
        "device": str(device),
        "schema_path": str(files["schema"]),
        "targets_are_standardized": targets_are_standardized,
        "data": {
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "n_test": int(len(test_ds)),
            "x_dim": int(train_ds.x.shape[1]),
            "y_set_dim": int(train_ds.y_set.shape[1]),
            "n_discrete_heads": int(train_ds.y_disc.shape[1]),
            "n_continuous_heads": int(train_ds.y_cont.shape[1]),
            "disc_col_names": disc_col_names,
            "cont_col_names": cont_col_names,
            "disc_class_sizes": disc_class_sizes,
            "full_mask_rows_train": int(full_train_rows),
            "full_mask_rows_val": int(full_val_rows),
            "full_mask_rows_test": int(full_test_rows),
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_score": float(best_score) if best_score is not None else math.nan,
            "metric_name": args.metric_name,
            "reloaded_best_checkpoint": True,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "checkpoint": str(best_ckpt_path),
            "train_log": str(run_dir / "train_log.json"),
            "metrics": str(run_dir / "metrics.json"),
            "config_snapshot": str(run_dir / "config_snapshot.json"),
            "train_predictions": str(run_dir / "train_predictions.csv"),
            "val_predictions": str(run_dir / "val_predictions.csv"),
            "test_predictions": str(run_dir / "test_predictions.csv"),
        },
    }
    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
