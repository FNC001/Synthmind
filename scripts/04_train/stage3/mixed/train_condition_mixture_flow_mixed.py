#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


TRAIN_MODE_CHOICES = [
    "relaxed_only",
    "gold_only",
    "curriculum",
    "curriculum_phase1",
    "curriculum_phase2",
]

DEFAULT_PROJECT_ROOT = Path("/Users/wyc/MP_exp_doi")
DEFAULT_RUN_ROOT = DEFAULT_PROJECT_ROOT / "runs/generative/stage3/condition_mixture_flow_mixed_v1"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_hidden_dims(s: str | Sequence[int]) -> List[int]:
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def choose_device(device_arg: str) -> torch.device:
    if str(device_arg) != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _first_existing(candidates: List[Path], what: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 {what}，候选路径为：\n" + "\n".join(str(x) for x in candidates))


def _resolve_mode_dir(root: Path, train_mode: str) -> Path:
    mode_map = {
        "relaxed_only": [root / "relaxed_only"],
        "gold_only": [root / "gold_only"],
        "curriculum": [root / "curriculum"],
        "curriculum_phase1": [root / "curriculum_phase1", root / "curriculum" / "phase1", root / "curriculum"],
        "curriculum_phase2": [root / "curriculum_phase2", root / "curriculum" / "phase2", root / "curriculum"],
    }
    if train_mode not in mode_map:
        raise ValueError(f"不支持的 train_mode: {train_mode}")
    return _first_existing(mode_map[train_mode], f"train_mode={train_mode} 对应的数据目录")


def _validate_bundle_dir(bundle_dir: Path) -> Dict[str, Path]:
    schema = bundle_dir / "schema.json"
    if not schema.exists():
        fallback = bundle_dir / "condition_schema.json"
        if fallback.exists():
            schema = fallback
    files = {
        "schema": schema,
        "train_npz": bundle_dir / "train.npz",
        "val_npz": bundle_dir / "val.npz",
        "test_npz": bundle_dir / "test.npz",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("stage3 mixture flow mixed bundle 缺少必需文件：\n" + "\n".join(missing))
    return files


def resolve_input_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    if str(args.input_dir).strip():
        bundle_dir = Path(args.input_dir).expanduser().resolve()
        if not bundle_dir.exists():
            raise FileNotFoundError(f"--input_dir 不存在: {bundle_dir}")
        files = _validate_bundle_dir(bundle_dir)
        return {
            "resolved_mode": "legacy_input_dir",
            "resolved_root": str(bundle_dir.parent),
            "resolved_input_dir": str(bundle_dir),
            "files": {k: str(v) for k, v in files.items()},
        }

    if not str(args.mode_input_root).strip():
        raise ValueError("必须提供 --input_dir 或 --mode_input_root")

    root = Path(args.mode_input_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--mode_input_root 不存在: {root}")

    if (root / "train.npz").exists() and ((root / "schema.json").exists() or (root / "condition_schema.json").exists()):
        files = _validate_bundle_dir(root)
        return {
            "resolved_mode": "direct_bundle_dir",
            "resolved_root": str(root.parent),
            "resolved_input_dir": str(root),
            "files": {k: str(v) for k, v in files.items()},
        }

    mode_dir = _resolve_mode_dir(root, args.train_mode)
    files = _validate_bundle_dir(mode_dir)
    return {
        "resolved_mode": args.train_mode,
        "resolved_root": str(root),
        "resolved_input_dir": str(mode_dir),
        "files": {k: str(v) for k, v in files.items()},
    }


def inverse_transform_np(values: np.ndarray, stats: List[Dict[str, float]]) -> np.ndarray:
    out = values.copy().astype(np.float32)
    for i, st in enumerate(stats):
        out[:, i] = out[:, i] * float(st["std"]) + float(st["mean"])
    return out


def clip_to_train_range_fn(pred: np.ndarray, train_min: np.ndarray, train_max: np.ndarray) -> np.ndarray:
    return np.clip(pred, train_min[None, :], train_max[None, :])


def pad_base_disc_logits(
    base_disc_logits,
    batch_size: int,
    disc_class_sizes,
    device,
    dtype,
):
    """
    baseline 有时不会返回完整数量的 discrete logits。
    这里按 disc_class_sizes 补齐缺失的头，缺哪个就补一个全零 logits。
    """
    out = list(base_disc_logits)
    if len(out) < len(disc_class_sizes):
        for cls_size in disc_class_sizes[len(out):]:
            out.append(
                torch.zeros(
                    batch_size,
                    int(cls_size),
                    device=device,
                    dtype=dtype,
                )
            )
    return out


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
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
        self.sample_id = np.asarray(arr["sample_id"]) if "sample_id" in arr.files else np.asarray([str(i) for i in range(self.x.shape[0])], dtype=object)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return {
            "x": self.x[idx],
            "y_set": self.y_set[idx],
            "y_disc": self.y_disc[idx],
            "y_cont": self.y_cont[idx],
            "y_cont_mask": self.y_cont_mask[idx],
            "sample_id": str(self.sample_id[idx]),
        }


# ---------------------------------------------------------------------
# Common modules
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Baseline compatibility
# ---------------------------------------------------------------------
class Stage3BaselineModel(nn.Module):
    """
    Compatibility class for torch baseline checkpoints.
    """
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


def build_features_np(x: np.ndarray, y_set: Optional[np.ndarray], use_y_set: bool) -> np.ndarray:
    if use_y_set and y_set is not None:
        return np.concatenate([x, y_set], axis=1)
    return x


class SklearnBaselineAdapter(nn.Module):
    """
    Adapter for pickle/joblib sklearn-style baseline payloads.
    Important:
    - do NOT register payload models as torch submodules
    - keep them as plain Python attributes
    """
    def __init__(
        self,
        payload: dict,
        disc_class_sizes: Sequence[int],
        y_cont_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self.disc_class_sizes = list(disc_class_sizes)
        self.y_cont_dim = int(y_cont_dim)
        self.device = device

        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "use_y_set", bool(payload.get("use_y_set", True)))

        disc_models = payload.get("disc_models", None)
        if disc_models is None:
            disc_models = payload.get("clf_models", None)
        if disc_models is None:
            disc_models = payload.get("discrete_models", None)
        if disc_models is None:
            disc_models = []

        cont_models = payload.get("cont_models", None)
        if cont_models is None:
            cont_models = payload.get("reg_models", None)
        if cont_models is None:
            cont_models = payload.get("continuous_models", None)
        if cont_models is None:
            cont_models = []

        multioutput_model = payload.get("model", None)

        # Legacy pickle may contain a broken torch module object.
        # We never call that object; only keep sklearn-style predictors with .predict().
        if multioutput_model is not None and hasattr(multioutput_model, "forward") and not hasattr(multioutput_model, "predict"):
            multioutput_model = None
            print("[Baseline Adapter] ignoring legacy torch payload['model']; using cont_models/disc_models or zero fallback.")

        object.__setattr__(self, "disc_models", disc_models)
        object.__setattr__(self, "cont_models", cont_models)
        object.__setattr__(self, "multioutput_model", multioutput_model)

    def _predict_disc_logits(self, feats: np.ndarray) -> list[torch.Tensor]:
        logits_list = []

        if self.disc_models:
            for i, n_cls in enumerate(self.disc_class_sizes):
                if i >= len(self.disc_models) or self.disc_models[i] is None:
                    logits = np.zeros((feats.shape[0], n_cls), dtype=np.float32)
                    logits_list.append(torch.tensor(logits, dtype=torch.float32, device=self.device))
                    continue

                model = self.disc_models[i]
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feats)
                    probs = np.asarray(probs, dtype=np.float32)

                    logits = np.full((feats.shape[0], n_cls), -20.0, dtype=np.float32)
                    classes = getattr(model, "classes_", np.arange(probs.shape[1]))

                    for j, cls in enumerate(classes):
                        cls = int(cls)
                        if 0 <= cls < n_cls:
                            logits[:, cls] = np.log(np.clip(probs[:, j], 1e-8, 1.0))
                else:
                    pred = np.asarray(model.predict(feats), dtype=np.int64)
                    logits = np.full((feats.shape[0], n_cls), -20.0, dtype=np.float32)
                    logits[np.arange(feats.shape[0]), np.clip(pred, 0, n_cls - 1)] = 0.0

                logits_list.append(torch.tensor(logits, dtype=torch.float32, device=self.device))

            return logits_list

        for n_cls in self.disc_class_sizes:
            logits = np.zeros((feats.shape[0], n_cls), dtype=np.float32)
            logits_list.append(torch.tensor(logits, dtype=torch.float32, device=self.device))
        return logits_list

    def _predict_cont(self, feats: np.ndarray) -> torch.Tensor:
        if self.cont_models:
            cols = []
            for i in range(self.y_cont_dim):
                if i >= len(self.cont_models) or self.cont_models[i] is None:
                    cols.append(np.zeros((feats.shape[0],), dtype=np.float32))
                    continue
                pred = np.asarray(self.cont_models[i].predict(feats), dtype=np.float32).reshape(-1)
                cols.append(pred)
            arr = np.stack(cols, axis=1)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        if self.multioutput_model is not None and hasattr(self.multioutput_model, "predict"):
            pred = np.asarray(self.multioutput_model.predict(feats), dtype=np.float32)
            if pred.ndim == 1:
                pred = pred[:, None]
            if pred.shape[1] >= self.y_cont_dim:
                pred = pred[:, -self.y_cont_dim:]
            else:
                pad = np.zeros((pred.shape[0], self.y_cont_dim - pred.shape[1]), dtype=np.float32)
                pred = np.concatenate([pred, pad], axis=1)
            return torch.tensor(pred, dtype=torch.float32, device=self.device)

        return torch.zeros((feats.shape[0], self.y_cont_dim), dtype=torch.float32, device=self.device)

    def forward(self, x: torch.Tensor, y_set: torch.Tensor) -> dict[str, Any]:
        x_np = x.detach().cpu().numpy()
        y_np = y_set.detach().cpu().numpy() if y_set is not None else None
        feats = build_features_np(x_np, y_np, self.use_y_set)

        disc_logits = self._predict_disc_logits(feats)
        cont_pred = self._predict_cont(feats)

        return {
            "disc_logits": disc_logits,
            "cont_pred": cont_pred,
        }



def load_pickle_baseline_payload(path: Path) -> dict:
    import __main__ as main_mod

    alias_map = {
        "Stage3BaselineModel": Stage3BaselineModel,
    }
    for name, cls in alias_map.items():
        if not hasattr(main_mod, name):
            setattr(main_mod, name, cls)

    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            if joblib is not None:
                return joblib.load(path)
            raise

def build_baseline_from_checkpoint(
    ckpt_path: Path | str,
    x_dim: int,
    y_set_dim: int,
    disc_class_sizes: Sequence[int],
    y_cont_dim: int,
    device: torch.device,
) -> nn.Module:
    ckpt_path = Path(ckpt_path)

    obj = None
    try:
        obj = torch.load(ckpt_path, map_location=device)
    except Exception:
        obj = None

    if isinstance(obj, nn.Module):
        model = obj.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    if isinstance(obj, dict) and "model_state_dict" in obj:
        # Prefer exact architecture stored in baseline checkpoint.
        model_cfg = obj.get("model_config", {}) or {}
        cfg = obj.get("config", {}) or obj.get("args", {}) or {}

        x_dim_ckpt = int(model_cfg.get("x_dim", x_dim))
        y_set_dim_ckpt = int(model_cfg.get("y_set_dim", y_set_dim))

        hidden_dims_ckpt = model_cfg.get("hidden_dims", cfg.get("hidden_dims", "512,256"))
        if isinstance(hidden_dims_ckpt, str):
            hidden_dims_ckpt = parse_hidden_dims(hidden_dims_ckpt)
        else:
            hidden_dims_ckpt = list(hidden_dims_ckpt)

        disc_class_sizes_ckpt = model_cfg.get("disc_class_sizes", disc_class_sizes)
        y_cont_dim_ckpt = int(model_cfg.get("y_cont_dim", y_cont_dim))
        dropout_ckpt = float(model_cfg.get("dropout", cfg.get("dropout", 0.1)))
        use_layernorm_ckpt = bool(model_cfg.get("use_layernorm", cfg.get("use_layernorm", False)))
        set_proj_dim_ckpt = int(model_cfg.get("set_proj_dim", cfg.get("set_proj_dim", 256)))
        fuse_mode_ckpt = str(model_cfg.get("fuse_mode", cfg.get("fuse_mode", "concat")))

        print(
            "[Info] baseline_ckpt architecture: "
            f"x_dim={x_dim_ckpt}, y_set_dim={y_set_dim_ckpt}, "
            f"set_proj_dim={set_proj_dim_ckpt}, fuse_mode={fuse_mode_ckpt}, "
            f"hidden_dims={hidden_dims_ckpt}"
        )

        model = Stage3BaselineModel(
            x_dim=x_dim_ckpt,
            y_set_dim=y_set_dim_ckpt,
            hidden_dims=hidden_dims_ckpt,
            disc_class_sizes=disc_class_sizes_ckpt,
            n_cont=y_cont_dim_ckpt,
            dropout=dropout_ckpt,
            use_layernorm=use_layernorm_ckpt,
            set_proj_dim=set_proj_dim_ckpt,
            fuse_mode=fuse_mode_ckpt,
        ).to(device)

        model.load_state_dict(obj["model_state_dict"], strict=False)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    payload = load_pickle_baseline_payload(ckpt_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported baseline payload type: {type(payload)}")

    model = SklearnBaselineAdapter(
        payload=payload,
        disc_class_sizes=disc_class_sizes,
        y_cont_dim=y_cont_dim,
        device=device,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------
# Mixture-flow compatible stage3 model
# ---------------------------------------------------------------------
class ResidualContextEncoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_set_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.1,
        use_layernorm: bool = False,
        set_proj_dim: int = 256,
        fuse_mode: str = "concat",
    ):
        super().__init__()
        self.fuse_mode = str(fuse_mode)
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


class MixtureResidualConditionFlowMixed(nn.Module):
    """
    Exporter-compatible reconstructed version:
    - encode_context(x, y_set)
    - resid_disc_heads
    - gating + components names compatible with old exporter/checkpoint expectations
    - sample_continuous(context, n_samples) -> residual samples [S,B,D]
    """
    def __init__(
        self,
        x_dim: int,
        y_set_dim: int,
        disc_class_sizes: Sequence[int],
        y_cont_dim: int,
        hidden_dims: Sequence[int],
        flow_hidden_dim: int = 128,
        n_flow_layers: int = 4,
        n_components: int = 3,
        gating_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_layernorm: bool = False,
        set_proj_dim: int = 256,
        fuse_mode: str = "concat",
    ):
        super().__init__()
        self.disc_class_sizes = list(disc_class_sizes)
        self.y_cont_dim = int(y_cont_dim)
        self.n_components = int(n_components)
        self.n_flow_layers = int(n_flow_layers)

        self.context_encoder = ResidualContextEncoder(
            x_dim=x_dim,
            y_set_dim=y_set_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_layernorm=use_layernorm,
            set_proj_dim=set_proj_dim,
            fuse_mode=fuse_mode,
        )
        ctx_dim = self.context_encoder.out_dim

        self.resid_disc_heads = nn.ModuleList([nn.Linear(ctx_dim, k) for k in self.disc_class_sizes])

        self.gating = MLP([ctx_dim, gating_hidden_dim, self.n_components], dropout=dropout, use_layernorm=use_layernorm)
        self.components = nn.ModuleList([
            MLP([ctx_dim, flow_hidden_dim, 2 * self.y_cont_dim], dropout=dropout, use_layernorm=use_layernorm)
            for _ in range(self.n_components)
        ])

    def encode_context(self, x: torch.Tensor, y_set: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(x, y_set)

    def _component_params(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gating_logits = self.gating(context)
        params = [comp(context) for comp in self.components]
        params = torch.stack(params, dim=1)
        means = params[..., :self.y_cont_dim]
        log_scales = params[..., self.y_cont_dim:].clamp(-5.0, 3.0)
        return gating_logits, means, log_scales

    def forward(self, x: torch.Tensor, y_set: torch.Tensor) -> Dict[str, Any]:
        context = self.encode_context(x, y_set)
        gating_logits, means, log_scales = self._component_params(context)
        top_idx = torch.argmax(gating_logits, dim=1)
        top1_residual = means[torch.arange(context.shape[0], device=context.device), top_idx]
        return {
            "context": context,
            "resid_disc_logits": [head(context) for head in self.resid_disc_heads],
            "gating_logits": gating_logits,
            "means": means,
            "log_scales": log_scales,
            "top1_residual": top1_residual,
        }

    def nll(self, residual: torch.Tensor, context: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
        full_rows = (mask > 0.5).all(dim=1)
        n_valid = int(full_rows.sum().item())
        if n_valid == 0:
            return residual.sum() * 0.0, 0
        r = residual[full_rows]
        gating_logits, means, log_scales = self._component_params(context[full_rows])
        scales = torch.exp(log_scales)

        diff = (r[:, None, :] - means) / torch.clamp(scales, min=1e-6)
        log_prob_comp = -0.5 * (diff.pow(2) + 2.0 * log_scales + math.log(2.0 * math.pi)).sum(dim=-1)
        log_mix = F.log_softmax(gating_logits, dim=-1)
        log_prob = torch.logsumexp(log_mix + log_prob_comp, dim=-1)
        return -log_prob.mean(), n_valid

    @torch.no_grad()
    def top1_residual(self, context: torch.Tensor) -> torch.Tensor:
        gating_logits, means, _ = self._component_params(context)
        top_idx = torch.argmax(gating_logits, dim=1)
        return means[torch.arange(context.shape[0], device=context.device), top_idx]

    @torch.no_grad()
    def sample_continuous(self, context: torch.Tensor, n_samples: int) -> torch.Tensor:
        b = context.shape[0]
        gating_logits, means, log_scales = self._component_params(context)
        probs = F.softmax(gating_logits, dim=-1)
        cat = torch.distributions.Categorical(probs=probs)
        comp_idx = cat.sample((n_samples,))

        means_rep = means.unsqueeze(0).expand(n_samples, b, self.n_components, self.y_cont_dim)
        log_scales_rep = log_scales.unsqueeze(0).expand(n_samples, b, self.n_components, self.y_cont_dim)

        gather_idx = comp_idx[..., None, None].expand(n_samples, b, 1, self.y_cont_dim)
        chosen_means = torch.gather(means_rep, 2, gather_idx).squeeze(2)
        chosen_log_scales = torch.gather(log_scales_rep, 2, gather_idx).squeeze(2)

        eps = torch.randn_like(chosen_means)
        return chosen_means + torch.exp(chosen_log_scales) * eps

    @torch.no_grad()
    def sample_residual(self, context: torch.Tensor, n_samples: int) -> torch.Tensor:
        return self.sample_continuous(context, n_samples)


# ---------------------------------------------------------------------
# Metrics / losses
# ---------------------------------------------------------------------
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
            if self.minimize:
                improved = value < (self.best - self.min_delta)
            else:
                improved = value > (self.best + self.min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def is_minimize_metric(metric_name: str) -> bool:
    m = str(metric_name).lower()
    return any(k in m for k in ["mae", "rmse", "mse", "loss", "nll"])


def choose_metric_value(metrics: Dict[str, Any], metric_name: str) -> float:
    if metric_name in metrics and np.isfinite(metrics[metric_name]):
        return float(metrics[metric_name])
    for v in metrics.values():
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    return float("inf")


def build_class_weight_tensor(y_disc: np.ndarray, disc_class_sizes: Sequence[int], device: torch.device) -> List[torch.Tensor]:
    weights = []
    for i, n_cls in enumerate(disc_class_sizes):
        counts = np.bincount(y_disc[:, i], minlength=n_cls).astype(np.float64)
        counts = np.clip(counts, 1.0, None)
        w = counts.sum() / counts
        w = w / w.mean()
        weights.append(torch.tensor(w.astype(np.float32), device=device))
    return weights


def multihead_classification_loss(logits_list, targets, class_weight_tensors, head_weights):
    if len(logits_list) == 0:
        return torch.tensor(0.0, device=targets.device)
    losses = []
    for i, logits in enumerate(logits_list):
        weight = class_weight_tensors[i] if class_weight_tensors is not None else None
        loss_i = F.cross_entropy(logits, targets[:, i], weight=weight, reduction="mean")
        losses.append(loss_i * float(head_weights[i]))
    return torch.stack(losses).sum() / max(float(sum(head_weights)), 1e-8)


def per_head_regression_metrics(y_true, y_pred, y_mask, col_names):
    out = {}
    for i, name in enumerate(col_names):
        m = y_mask[:, i] > 0.5
        if m.sum() == 0:
            out[name] = {"mae": math.nan, "rmse": math.nan, "n_valid": 0}
            continue
        yt = y_true[m, i]
        yp = y_pred[m, i]
        out[name] = {
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
            "n_valid": int(m.sum()),
        }
    return out


def per_head_classification_metrics(y_true, y_pred, col_names):
    out = {}
    for i, name in enumerate(col_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[name] = {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        }
    return out


def mean_over_metric_dict(d, key):
    vals = [v[key] for v in d.values() if np.isfinite(v.get(key, np.nan))]
    return float(np.mean(vals)) if vals else math.nan


def collect_train_raw_minmax(ds: Stage3MixedDataset, cont_stats):
    y_true_raw = inverse_transform_np(ds.y_cont.cpu().numpy(), cont_stats)
    y_mask = ds.y_cont_mask.cpu().numpy() > 0.5
    mins, maxs = [], []
    for j in range(y_true_raw.shape[1]):
        vals = y_true_raw[y_mask[:, j], j]
        if vals.size == 0:
            mins.append(-np.inf)
            maxs.append(np.inf)
        else:
            mins.append(float(vals.min()))
            maxs.append(float(vals.max()))
    return np.asarray(mins, dtype=np.float32), np.asarray(maxs, dtype=np.float32)


def save_predictions(path: Path, arrays: Dict[str, Any], disc_col_names, cont_col_names):
    df = pd.DataFrame({"sample_id": [str(x) for x in arrays["sample_id"].tolist()]})
    for i, name in enumerate(disc_col_names):
        df[f"true_{name}"] = arrays["disc_true"][:, i]
        df[f"baseline_{name}"] = arrays["disc_base"][:, i]
        df[f"top1_{name}"] = arrays["disc_top1"][:, i]
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


@torch.no_grad()
def evaluate_split(
    baseline,
    model,
    loader,
    device,
    disc_col_names,
    cont_col_names,
    cont_stats,
    disc_class_sizes,
    disc_head_weights,
    cont_head_weights,
    class_weight_tensors,
    n_gen_samples,
    train_min_raw,
    train_max_raw,
    clip_to_train_range,
):
    model.eval()
    baseline.eval()

    sample_ids_all = []
    disc_true_all, disc_base_all, disc_top1_all, disc_oracle_all = [], [], [], []
    cont_true_norm_all, cont_true_raw_all, mask_all = [], [], []
    cont_base_norm_all, cont_base_raw_all = [], []
    cont_top1_norm_all, cont_top1_raw_all = [], []
    cont_oracle_norm_all, cont_oracle_raw_all = [], []

    total_loss = 0.0
    total_rows = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_set = batch["y_set"].to(device)
        y_disc = batch["y_disc"].to(device)
        y_cont = batch["y_cont"].to(device)
        y_mask = batch["y_cont_mask"].to(device)
        batch_ids = [str(v) for v in batch["sample_id"]]

        base_out = baseline(x, y_set)
        base_cont = base_out["cont_pred"]
        base_disc_logits = pad_base_disc_logits(
            base_out.get("disc_logits", []),
            batch_size=x.size(0),
            disc_class_sizes=disc_class_sizes,
            device=x.device,
            dtype=x.dtype,
        )

        out = model(x, y_set)
        residual_target = y_cont - base_cont

        final_disc_logits = [b + r for b, r in zip(base_disc_logits, out["resid_disc_logits"])]
        loss_disc = multihead_classification_loss(final_disc_logits, y_disc, class_weight_tensors, disc_head_weights)
        loss_cont, n_valid = model.nll(residual_target, out["context"], y_mask)
        loss = loss_disc + loss_cont

        top1_disc = [torch.argmax(logits, dim=1) for logits in final_disc_logits]
        disc_base = [torch.argmax(logits, dim=1) for logits in base_disc_logits]

        top1_resid = model.top1_residual(out["context"])
        top1_cont = base_cont + top1_resid

        sample_resid = model.sample_continuous(out["context"], n_gen_samples)
        sample_cont = base_cont.unsqueeze(0) + sample_resid

        target = y_cont.unsqueeze(0).expand(n_gen_samples, -1, -1)
        mask = y_mask.unsqueeze(0).expand(n_gen_samples, -1, -1)
        sqerr = ((sample_cont - target) ** 2) * mask
        denom = torch.clamp(mask.sum(dim=2), min=1.0)
        per_sample_err = sqerr.sum(dim=2) / denom
        best_idx = torch.argmin(per_sample_err, dim=0)
        oracle_cont = sample_cont[best_idx, torch.arange(x.shape[0], device=x.device)]
        disc_oracle = top1_disc

        disc_true_all.append(y_disc.detach().cpu().numpy())
        disc_base_all.append(torch.stack(disc_base, dim=1).detach().cpu().numpy() if disc_base else np.zeros((x.shape[0], 0), dtype=np.int64))
        disc_top1_all.append(torch.stack(top1_disc, dim=1).detach().cpu().numpy() if top1_disc else np.zeros((x.shape[0], 0), dtype=np.int64))
        disc_oracle_all.append(torch.stack(disc_oracle, dim=1).detach().cpu().numpy() if disc_oracle else np.zeros((x.shape[0], 0), dtype=np.int64))

        cont_true_norm = y_cont.detach().cpu().numpy()
        cont_base_norm = base_cont.detach().cpu().numpy()
        cont_top1_norm = top1_cont.detach().cpu().numpy()
        cont_oracle_norm = oracle_cont.detach().cpu().numpy()
        y_mask_np = y_mask.detach().cpu().numpy()

        cont_true_raw = inverse_transform_np(cont_true_norm, cont_stats)
        cont_base_raw = inverse_transform_np(cont_base_norm, cont_stats)
        cont_top1_raw = inverse_transform_np(cont_top1_norm, cont_stats)
        cont_oracle_raw = inverse_transform_np(cont_oracle_norm, cont_stats)

        if clip_to_train_range and train_min_raw is not None and train_max_raw is not None:
            cont_base_raw = clip_to_train_range_fn(cont_base_raw, train_min_raw, train_max_raw)
            cont_top1_raw = clip_to_train_range_fn(cont_top1_raw, train_min_raw, train_max_raw)
            cont_oracle_raw = clip_to_train_range_fn(cont_oracle_raw, train_min_raw, train_max_raw)

        sample_ids_all.extend(batch_ids)
        cont_true_norm_all.append(cont_true_norm)
        cont_true_raw_all.append(cont_true_raw)
        mask_all.append(y_mask_np)
        cont_base_norm_all.append(cont_base_norm)
        cont_base_raw_all.append(cont_base_raw)
        cont_top1_norm_all.append(cont_top1_norm)
        cont_top1_raw_all.append(cont_top1_raw)
        cont_oracle_norm_all.append(cont_oracle_norm)
        cont_oracle_raw_all.append(cont_oracle_raw)

        total_loss += float(loss.item()) * max(n_valid, 1)
        total_rows += max(n_valid, 1)

    disc_true = np.concatenate(disc_true_all, axis=0) if disc_true_all else np.zeros((0, len(disc_col_names)), dtype=np.int64)
    disc_base = np.concatenate(disc_base_all, axis=0) if disc_base_all else np.zeros((0, len(disc_col_names)), dtype=np.int64)
    disc_top1 = np.concatenate(disc_top1_all, axis=0) if disc_top1_all else np.zeros((0, len(disc_col_names)), dtype=np.int64)
    disc_oracle = np.concatenate(disc_oracle_all, axis=0) if disc_oracle_all else np.zeros((0, len(disc_col_names)), dtype=np.int64)

    cont_true_norm = np.concatenate(cont_true_norm_all, axis=0)
    cont_true_raw = np.concatenate(cont_true_raw_all, axis=0)
    y_mask = np.concatenate(mask_all, axis=0)
    cont_base_norm = np.concatenate(cont_base_norm_all, axis=0)
    cont_base_raw = np.concatenate(cont_base_raw_all, axis=0)
    cont_top1_norm = np.concatenate(cont_top1_norm_all, axis=0)
    cont_top1_raw = np.concatenate(cont_top1_raw_all, axis=0)
    cont_oracle_norm = np.concatenate(cont_oracle_norm_all, axis=0)
    cont_oracle_raw = np.concatenate(cont_oracle_raw_all, axis=0)

    disc_base_metrics = per_head_classification_metrics(disc_true, disc_base, disc_col_names) if len(disc_col_names) > 0 else {}
    disc_top1_metrics = per_head_classification_metrics(disc_true, disc_top1, disc_col_names) if len(disc_col_names) > 0 else {}
    disc_oracle_metrics = per_head_classification_metrics(disc_true, disc_oracle, disc_col_names) if len(disc_col_names) > 0 else {}

    cont_base_raw_metrics = per_head_regression_metrics(cont_true_raw, cont_base_raw, y_mask, cont_col_names)
    cont_top1_raw_metrics = per_head_regression_metrics(cont_true_raw, cont_top1_raw, y_mask, cont_col_names)
    cont_oracle_raw_metrics = per_head_regression_metrics(cont_true_raw, cont_oracle_raw, y_mask, cont_col_names)

    metrics = {
        "loss": float(total_loss / max(total_rows, 1)),
        "baseline_discrete_joint_accuracy": float(np.mean(np.all(disc_true == disc_base, axis=1))) if len(disc_col_names) > 0 else math.nan,
        "baseline_discrete_mean_macro_f1": mean_over_metric_dict(disc_base_metrics, "macro_f1") if disc_base_metrics else math.nan,
        "baseline_discrete_mean_accuracy": mean_over_metric_dict(disc_base_metrics, "accuracy") if disc_base_metrics else math.nan,
        "baseline_discrete_per_head": disc_base_metrics,
        "top1_discrete_joint_accuracy": float(np.mean(np.all(disc_true == disc_top1, axis=1))) if len(disc_col_names) > 0 else math.nan,
        "top1_discrete_mean_macro_f1": mean_over_metric_dict(disc_top1_metrics, "macro_f1") if disc_top1_metrics else math.nan,
        "top1_discrete_mean_accuracy": mean_over_metric_dict(disc_top1_metrics, "accuracy") if disc_top1_metrics else math.nan,
        "top1_discrete_per_head": disc_top1_metrics,
        "oracle_best_of_k_discrete_joint_accuracy": float(np.mean(np.all(disc_true == disc_oracle, axis=1))) if len(disc_col_names) > 0 else math.nan,
        "oracle_best_of_k_discrete_mean_macro_f1": mean_over_metric_dict(disc_oracle_metrics, "macro_f1") if disc_oracle_metrics else math.nan,
        "oracle_best_of_k_discrete_mean_accuracy": mean_over_metric_dict(disc_oracle_metrics, "accuracy") if disc_oracle_metrics else math.nan,
        "oracle_best_of_k_discrete_per_head": disc_oracle_metrics,
        "baseline_continuous_mean_mae_raw": mean_over_metric_dict(cont_base_raw_metrics, "mae"),
        "top1_continuous_mean_mae_raw": mean_over_metric_dict(cont_top1_raw_metrics, "mae"),
        "oracle_best_of_k_continuous_mean_mae_raw": mean_over_metric_dict(cont_oracle_raw_metrics, "mae"),
        "baseline_continuous_per_head_raw": cont_base_raw_metrics,
        "top1_continuous_per_head_raw": cont_top1_raw_metrics,
        "oracle_best_of_k_continuous_per_head_raw": cont_oracle_raw_metrics,
    }

    arrays = {
        "sample_id": np.asarray(sample_ids_all, dtype=object),
        "disc_true": disc_true,
        "disc_base": disc_base,
        "disc_top1": disc_top1,
        "disc_oracle": disc_oracle,
        "cont_true_norm": cont_true_norm,
        "cont_true_raw": cont_true_raw,
        "mask": y_mask,
        "cont_base_norm": cont_base_norm,
        "cont_base_raw": cont_base_raw,
        "cont_top1_norm": cont_top1_norm,
        "cont_top1_raw": cont_top1_raw,
        "cont_oracle_norm": cont_oracle_norm,
        "cont_oracle_raw": cont_oracle_raw,
    }
    return metrics, arrays


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train stage3 mixture flow mixed residual model")
    parser.add_argument("--input_dir", type=str, default="")
    parser.add_argument("--mode_input_root", type=str, default="")
    parser.add_argument("--train_mode", type=str, default="relaxed_only", choices=TRAIN_MODE_CHOICES)
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--hidden_dims", type=str, default="512,256")
    parser.add_argument("--set_proj_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--n_flow_layers", type=int, default=4)
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument("--gating_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_layernorm", action="store_true")
    parser.add_argument("--fuse_mode", type=str, default="concat", choices=["concat", "add"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--metric_name", type=str, default="top1_continuous_mean_mae_raw")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--discrete_head_weights", type=str, default="")
    parser.add_argument("--continuous_head_weights", type=str, default="")
    parser.add_argument("--n_gen_samples", type=int, default=8)
    parser.add_argument("--clip_to_train_range", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = Path(args.run_dir).expanduser().resolve()
    ensure_dir(run_dir)

    resolved = resolve_input_bundle(args)
    files = {k: Path(v) for k, v in resolved["files"].items()}
    print(f"[Info] resolved_mode = {resolved['resolved_mode']}")
    print(f"[Info] resolved_root = {resolved['resolved_root']}")
    print(f"[Info] resolved_input_dir = {resolved['resolved_input_dir']}")

    schema = load_json(files["schema"])
    discrete_schema = schema.get("discrete_schema", {}) or {}
    continuous_schema = schema.get("continuous_schema", {}) or {}
    disc_col_names = list(discrete_schema.keys())
    cont_col_names = list(continuous_schema.keys())
    disc_class_sizes = [int(discrete_schema[c]["n_classes"]) for c in disc_col_names]
    cont_stats = [continuous_schema[c] for c in cont_col_names]
    if not cont_col_names:
        raise ValueError(f"No continuous heads found in {files['schema'].name}")

    train_ds = Stage3MixedDataset(files["train_npz"])
    val_ds = Stage3MixedDataset(files["val_npz"])
    test_ds = Stage3MixedDataset(files["test_npz"])

    tr_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    va_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    te_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    device = choose_device(args.device)

    baseline = build_baseline_from_checkpoint(
        ckpt_path=Path(args.baseline_ckpt),
        x_dim=int(train_ds.x.shape[1]),
        y_set_dim=int(train_ds.y_set.shape[1]),
        disc_class_sizes=disc_class_sizes,
        y_cont_dim=int(train_ds.y_cont.shape[1]),
        device=device,
    )

    model = MixtureResidualConditionFlowMixed(
        x_dim=int(train_ds.x.shape[1]),
        y_set_dim=int(train_ds.y_set.shape[1]),
        disc_class_sizes=disc_class_sizes,
        y_cont_dim=int(train_ds.y_cont.shape[1]),
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        flow_hidden_dim=int(args.flow_hidden_dim),
        n_flow_layers=int(args.n_flow_layers),
        n_components=int(args.n_components),
        gating_hidden_dim=int(args.gating_hidden_dim),
        dropout=float(args.dropout),
        use_layernorm=bool(args.use_layernorm),
        set_proj_dim=int(args.set_proj_dim),
        fuse_mode=str(args.fuse_mode),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
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

    class_weight_tensors = build_class_weight_tensor(train_ds.y_disc.numpy(), disc_class_sizes, device) if args.use_class_weights and disc_col_names else None
    train_min_raw, train_max_raw = collect_train_raw_minmax(train_ds, cont_stats)
    early_stopper = EarlyStopper(patience=int(args.patience), minimize=is_minimize_metric(args.metric_name))
    best_epoch = -1
    best_score = None
    train_log: List[Dict[str, Any]] = []

    full_train_rows = int(((train_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    full_val_rows = int(((val_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    full_test_rows = int(((test_ds.y_cont_mask > 0.5).all(dim=1)).sum().item())
    print(
        f"[Info] device={device} | n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_ds)} | "
        f"full-mask rows train/val/test = {full_train_rows}/{full_val_rows}/{full_test_rows}"
    )

    for epoch in range(1, int(args.epochs) + 1):
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
                base_cont = base_out["cont_pred"].detach()
                base_disc_logits = pad_base_disc_logits(
                    base_out.get("disc_logits", []),
                    batch_size=x.size(0),
                    disc_class_sizes=disc_class_sizes,
                    device=x.device,
                    dtype=x.dtype,
                )

            residual_target = y_cont - base_cont
            out = model(x, y_set)
            final_disc_logits = [b + r for b, r in zip(base_disc_logits, out["resid_disc_logits"])]
            loss_disc = multihead_classification_loss(final_disc_logits, y_disc, class_weight_tensors, disc_head_weights)
            loss_cont, n_valid = model.nll(residual_target, out["context"], y_mask)
            loss = loss_disc + loss_cont

            optimizer.zero_grad()
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
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
            disc_class_sizes=disc_class_sizes,
            disc_head_weights=disc_head_weights,
            cont_head_weights=cont_head_weights,
            class_weight_tensors=class_weight_tensors,
            n_gen_samples=int(args.n_gen_samples),
            train_min_raw=train_min_raw,
            train_max_raw=train_max_raw,
            clip_to_train_range=bool(args.clip_to_train_range),
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
                },
                run_dir / "best_stage3_condition_mixture_flow_mixed.pt",
            )

        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "train_flow_rows": int(valid_rows_seen),
            "val_metrics": val_metrics,
            "best_metric_so_far": float(early_stopper.best),
        }
        train_log.append(rec)

        print(
            f"[Epoch {epoch:03d}] train_loss={rec['train_loss']:.4f} "
            f"train_flow_rows={valid_rows_seen} "
            f"val_{args.metric_name}={float(val_metrics.get(args.metric_name, float('nan'))):.4f} "
            f"best_{args.metric_name}={float(early_stopper.best):.4f}"
        )

        if should_stop:
            print(f"[Early Stop] patience reached at epoch {epoch}")
            break

    ckpt = torch.load(run_dir / "best_stage3_condition_mixture_flow_mixed.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    write_json(run_dir / "train_log.json", train_log)

    train_metrics, train_arrays = evaluate_split(
        baseline, model, DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers)),
        device, disc_col_names, cont_col_names, cont_stats, disc_class_sizes,
        disc_head_weights, cont_head_weights, class_weight_tensors,
        int(args.n_gen_samples), train_min_raw, train_max_raw, bool(args.clip_to_train_range)
    )
    val_metrics, val_arrays = evaluate_split(
        baseline, model, va_loader,
        device, disc_col_names, cont_col_names, cont_stats, disc_class_sizes,
        disc_head_weights, cont_head_weights, class_weight_tensors,
        int(args.n_gen_samples), train_min_raw, train_max_raw, bool(args.clip_to_train_range)
    )
    test_metrics, test_arrays = evaluate_split(
        baseline, model, te_loader,
        device, disc_col_names, cont_col_names, cont_stats, disc_class_sizes,
        disc_head_weights, cont_head_weights, class_weight_tensors,
        int(args.n_gen_samples), train_min_raw, train_max_raw, bool(args.clip_to_train_range)
    )

    save_predictions(run_dir / "train_predictions.csv", train_arrays, disc_col_names, cont_col_names)
    save_predictions(run_dir / "val_predictions.csv", val_arrays, disc_col_names, cont_col_names)
    save_predictions(run_dir / "test_predictions.csv", test_arrays, disc_col_names, cont_col_names)

    config_snapshot = {
        **vars(args),
        "resolved_mode": resolved["resolved_mode"],
        "resolved_root": resolved["resolved_root"],
        "resolved_input_dir": resolved["resolved_input_dir"],
        "resolved_schema_path": str(files["schema"]),
    }
    write_json(run_dir / "config_snapshot.json", config_snapshot)
    summary = {
        "config": config_snapshot,
        "device": str(device),
        "schema_path": str(files["schema"]),
        "data": {
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "n_test": int(len(test_ds)),
            "x_dim": int(train_ds.x.shape[1]),
            "y_set_dim": int(train_ds.y_set.shape[1]),
            "n_discrete_heads": int(train_ds.y_disc.shape[1]),
            "n_continuous_heads": int(train_ds.y_cont.shape[1]),
            "full_train_rows": int(full_train_rows),
            "full_val_rows": int(full_val_rows),
            "full_test_rows": int(full_test_rows),
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score) if best_score is not None else math.nan,
    }
    write_json(run_dir / "metrics.json", summary)

    print(json.dumps({
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "val_top1_continuous_mean_mae_raw": val_metrics.get("top1_continuous_mean_mae_raw"),
        "test_top1_continuous_mean_mae_raw": test_metrics.get("top1_continuous_mean_mae_raw"),
        "val_top1_discrete_mean_macro_f1": val_metrics.get("top1_discrete_mean_macro_f1"),
        "test_top1_discrete_mean_macro_f1": test_metrics.get("top1_discrete_mean_macro_f1"),
        "artifacts": {
            "best_ckpt": str(run_dir / "best_stage3_condition_mixture_flow_mixed.pt"),
            "metrics": str(run_dir / "metrics.json"),
            "train_log": str(run_dir / "train_log.json"),
            "train_predictions": str(run_dir / "train_predictions.csv"),
            "val_predictions": str(run_dir / "val_predictions.csv"),
            "test_predictions": str(run_dir / "test_predictions.csv"),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
