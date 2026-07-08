#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader, Dataset, TensorDataset


STOP_TOKEN = "<stop>"

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = TRAIN_ROOT.parent.parent
for _p in [PROJECT_ROOT, TRAIN_ROOT, SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# =============================================================================
# IO / utils
# =============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


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


def parse_hidden_dims(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def parse_int_list(s: str) -> List[int]:
    vals: List[int] = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            vals.append(int(x))
    return vals


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Input resolution
# =============================================================================
def _first_existing(candidates: List[Path], what: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"未找到 {what}，候选路径为：\n" + "\n".join(str(x) for x in candidates)
    )


def _resolve_mode_dir(root: Path, train_mode: str) -> Path:
    mapping = {
        "curriculum_phase1": [
            root / "curriculum_phase1",
            root / "curriculum" / "phase1",
            root / "curriculum" / "phase1_train",
        ],
        "curriculum_phase2": [
            root / "curriculum_phase2",
            root / "curriculum" / "phase2",
            root / "curriculum" / "phase2_train",
        ],
        "curriculum": [root / "curriculum"],
        "relaxed_only": [root / "relaxed_only"],
        "gold_only": [root / "gold_only"],
    }
    candidates = mapping.get(train_mode, [root / train_mode])
    return _first_existing(candidates, f"train_mode={train_mode} 对应的数据目录")


def _validate_bundle_dir(bundle_dir: Path) -> Dict[str, str]:
    required = {
        "train_npz": bundle_dir / "train.npz",
        "val_npz": bundle_dir / "val.npz",
        "test_npz": bundle_dir / "test.npz",
        "train_meta": bundle_dir / "train_meta.csv",
        "val_meta": bundle_dir / "val_meta.csv",
        "test_meta": bundle_dir / "test_meta.csv",
        "action_vocab": bundle_dir / "action_vocab.json",
        "precursor_names": bundle_dir / "precursor_names.json",
        "summary": bundle_dir / "summary.json",
    }
    optional = {
        "action_to_id": bundle_dir / "action_to_id.json",
        "label_cols": bundle_dir / "label_cols.json",
        "label_names": bundle_dir / "label_names.json",
        "schema": bundle_dir / "schema.json",
    }

    missing = [str(v) for v in required.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("gflownet bundle 缺少必需文件：\n" + "\n".join(missing))

    out = {k: str(v) for k, v in required.items()}
    for k, v in optional.items():
        if v.exists():
            out[k] = str(v)
    return out


def resolve_input_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    input_dir_str = str(getattr(args, "input_dir", "")).strip()
    mode_root_str = str(getattr(args, "mode_input_root", "")).strip()

    if input_dir_str:
        bundle_dir = Path(input_dir_str).expanduser().resolve()
        if not bundle_dir.exists():
            raise FileNotFoundError(f"--input_dir 不存在: {bundle_dir}")
        return {
            "resolved_mode": "legacy_input_dir",
            "resolved_root": str(bundle_dir.parent),
            "resolved_input_dir": str(bundle_dir),
            "files": _validate_bundle_dir(bundle_dir),
        }

    if not mode_root_str:
        raise ValueError("必须提供 --input_dir 或 --mode_input_root")

    root = Path(mode_root_str).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--mode_input_root 不存在: {root}")

    train_mode = getattr(args, "train_mode", "gold_only")
    mode_dir = _resolve_mode_dir(root, train_mode)
    files = _validate_bundle_dir(mode_dir)
    return {
        "resolved_mode": train_mode,
        "resolved_root": str(root),
        "resolved_input_dir": str(mode_dir),
        "files": files,
    }


# =============================================================================
# Data helpers
# =============================================================================
def _extract_x(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["x", "features", "X"]:
        if k in pack:
            arr = np.asarray(pack[k], dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"特征键 {k} 必须是二维矩阵，当前 shape={arr.shape}")
            return arr
    raise KeyError(f"NPZ 中未找到 x/features/X，已有键：{list(pack.keys())}")


def _extract_y(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["y_multi_hot", "y", "labels", "targets"]:
        if k in pack:
            arr = np.asarray(pack[k])
            if arr.ndim != 2:
                raise ValueError(f"标签键 {k} 必须是二维 multi-hot，当前 shape={arr.shape}")
            return (arr > 0).astype(np.int32)
    raise KeyError(f"NPZ 中未找到 y_multi_hot/y/labels/targets，已有键：{list(pack.keys())}")


def _extract_traj_actions(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["traj_actions", "actions", "action_seq", "seq"]:
        if k in pack:
            arr = np.asarray(pack[k])
            if arr.ndim != 2:
                raise ValueError(f"轨迹动作键 {k} 必须是二维，当前 shape={arr.shape}")
            return arr.astype(np.int64)
    raise KeyError(f"NPZ 中未找到 traj_actions/actions/action_seq/seq，已有键：{list(pack.keys())}")


def _extract_traj_mask(pack: Dict[str, np.ndarray]) -> np.ndarray:
    for k in ["traj_mask", "action_mask", "mask", "seq_mask"]:
        if k in pack:
            arr = np.asarray(pack[k])
            if arr.ndim != 2:
                raise ValueError(f"轨迹 mask 键 {k} 必须是二维，当前 shape={arr.shape}")
            return arr.astype(np.float32)
    raise KeyError(f"NPZ 中未找到 traj_mask/action_mask/mask/seq_mask，已有键：{list(pack.keys())}")


def ensure_precursor_names(names: Sequence[Any], n_precursors: int) -> List[str]:
    out = [str(x) for x in list(names)]
    if len(out) < n_precursors:
        out += [f"precursor_{i}" for i in range(len(out), n_precursors)]
    return out[:n_precursors]


def label_statistics(y_train: np.ndarray) -> Dict[str, Any]:
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


class Stage2GFlowNetDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_multi_hot: np.ndarray,
        traj_actions: np.ndarray,
        traj_mask: np.ndarray,
    ):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_multi_hot = torch.tensor(y_multi_hot, dtype=torch.float32)
        self.traj_actions = torch.tensor(traj_actions, dtype=torch.long)
        self.traj_mask = torch.tensor(traj_mask, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.x[idx],
            self.y_multi_hot[idx],
            self.traj_actions[idx],
            self.traj_mask[idx],
        )


class CandidateRerankDataset(Dataset):
    def __init__(self, feats: np.ndarray, targets: np.ndarray):
        self.feats = torch.tensor(feats, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.feats.shape[0])

    def __getitem__(self, idx: int):
        return self.feats[idx], self.targets[idx]


# =============================================================================
# Element parsing / soft bias
# =============================================================================
ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)")

DEFAULT_STRUCTURAL_ELEMENTS = frozenset({
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Tl",
    "Si", "Ge", "Sn", "Pb",
    "B", "P", "As", "Sb", "Bi",
    "S", "Se", "Te",
    "Th", "U",
})


def parse_ignore_elements(s: str) -> Set[str]:
    return {x.strip() for x in str(s).split(",") if x.strip()}


def extract_elements_from_formula(
    formula: Any,
    ignore_elements: Optional[Set[str]] = None,
    structural_only: bool = True,
) -> Set[str]:
    ignore_elements = ignore_elements or set()
    text = "" if pd.isna(formula) else str(formula)
    elems = set(ELEMENT_PAT.findall(text))
    elems = {e for e in elems if e not in ignore_elements}
    if structural_only:
        elems = {e for e in elems if e in DEFAULT_STRUCTURAL_ELEMENTS}
    return elems


def build_precursor_element_index(
    precursor_names: Sequence[str],
    ignore_elements: Optional[Set[str]] = None,
) -> List[Set[str]]:
    return [
        extract_elements_from_formula(name, ignore_elements=ignore_elements, structural_only=True)
        for name in precursor_names
    ]


def infer_formula_list_from_meta(meta_df: pd.DataFrame) -> List[str]:
    for col in ["formula", "target_formula", "composition", "pretty_formula", "material_formula"]:
        if col in meta_df.columns:
            return [str(x) for x in meta_df[col].fillna("").tolist()]
    return [""] * len(meta_df)


def build_element_bias_for_batch(
    formulas: Sequence[str],
    precursor_elements: Sequence[Set[str]],
    n_precursors: int,
    stop_id: int,
    target_hit_bonus: float,
    extra_element_penalty: float,
    no_overlap_penalty: float,
    stop_bias: float,
    ignore_elements: Optional[Set[str]] = None,
) -> torch.Tensor:
    """
    Return [bsz, n_precursors + 1] additive bias.
    This is soft bias, not hard masking.
    """
    bsz = len(formulas)
    n_actions = n_precursors + 1
    bias = torch.zeros((bsz, n_actions), dtype=torch.float32)

    for i, formula in enumerate(formulas):
        target_elems = extract_elements_from_formula(
            formula,
            ignore_elements=ignore_elements,
            structural_only=True,
        )
        if not target_elems:
            bias[i, stop_id] += float(stop_bias)
            continue

        for j in range(n_precursors):
            pe = precursor_elements[j]
            if not pe:
                continue

            overlap = pe & target_elems
            extra = pe - target_elems

            if not overlap:
                bias[i, j] -= float(no_overlap_penalty)
            else:
                bias[i, j] += float(target_hit_bonus) * float(len(overlap))
                bias[i, j] -= float(extra_element_penalty) * float(len(extra))

        bias[i, stop_id] += float(stop_bias)

    return bias


# =============================================================================
# Models
# =============================================================================
class MLP(nn.Module):
    def __init__(self, dims: List[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GFlowNetPolicy(nn.Module):
    """
    GFlowNet-style forward-policy set constructor.

    State  = target feature x + current selected-set mask + current step id
    Action = choose one precursor token or STOP
    """

    def __init__(
        self,
        x_dim: int,
        n_precursors: int,
        hidden_dim: int,
        max_traj_len: int,
        x_mlp_hidden_dims: List[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.x_dim = int(x_dim)
        self.n_precursors = int(n_precursors)
        self.n_actions = int(n_precursors + 1)
        self.max_traj_len = int(max_traj_len)
        self.hidden_dim = int(hidden_dim)

        self.x_proj = MLP([x_dim] + x_mlp_hidden_dims + [hidden_dim], dropout=dropout)
        self.set_proj = MLP([n_precursors, hidden_dim, hidden_dim], dropout=dropout)
        self.step_emb = nn.Embedding(max_traj_len + 1, hidden_dim)
        self.policy_head = MLP([hidden_dim * 3, hidden_dim, self.n_actions], dropout=dropout)

    def forward_state(
        self,
        x: torch.Tensor,
        selected_mask: torch.Tensor,
        step_ids: torch.Tensor,
    ) -> torch.Tensor:
        x_ctx = self.x_proj(x)
        set_ctx = self.set_proj(selected_mask)
        step_ctx = self.step_emb(step_ids)
        state = torch.cat([x_ctx, set_ctx, step_ctx], dim=-1)
        logits = self.policy_head(state)
        return logits


class CandidateReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        dims = [input_dim] + hidden_dims + [1]
        self.net = MLP(dims, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =============================================================================
# Policy helpers
# =============================================================================
def masked_log_softmax(logits: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(invalid_mask, -1e9)
    return torch.log_softmax(logits, dim=-1)


def build_invalid_mask(
    selected_mask: torch.Tensor,
    stop_id: int,
    stopped: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, n_prec = selected_mask.shape
    invalid = torch.zeros((bsz, n_prec + 1), dtype=torch.bool, device=selected_mask.device)
    invalid[:, :n_prec] = selected_mask > 0.5

    if stopped is not None:
        invalid[stopped, :] = True
        invalid[stopped, stop_id] = False

    return invalid


def update_selected_non_inplace(
    selected: torch.Tensor,
    action: torch.Tensor,
    stop_id: int,
) -> torch.Tensor:
    add_mask = action != stop_id
    if not add_mask.any():
        return selected

    next_selected = selected.clone()
    rows = torch.nonzero(add_mask, as_tuple=False).squeeze(1)
    cols = action[rows]
    valid = (cols >= 0) & (cols < selected.shape[1])
    if valid.any():
        next_selected[rows[valid], cols[valid]] = 1.0
    return next_selected


def force_non_empty_selected_from_first_step(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    selected: torch.Tensor,
    stop_id: int,
    element_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    empty = selected.sum(dim=1) < 0.5
    if not empty.any():
        return selected

    bsz = x.shape[0]
    device = x.device
    init_selected = torch.zeros((bsz, model.n_precursors), dtype=torch.float32, device=device)
    step_ids = torch.zeros((bsz,), dtype=torch.long, device=device)

    logits = model.forward_state(x, init_selected, step_ids)
    if element_bias is not None:
        logits = logits + element_bias.to(device)
    logits[:, stop_id] = -1e9

    best_non_stop = torch.argmax(logits, dim=1)

    fixed = selected.clone()
    rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
    cols = best_non_stop[rows]
    valid = (cols >= 0) & (cols < model.n_precursors)
    if valid.any():
        fixed[rows[valid], cols[valid]] = 1.0
    return fixed


def teacher_forcing_loss(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    traj_actions: torch.Tensor,
    traj_mask: torch.Tensor,
    stop_id: int,
) -> torch.Tensor:
    device = x.device
    bsz, T = traj_actions.shape
    n_prec = model.n_precursors

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    losses: List[torch.Tensor] = []

    for t in range(T):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids)
        invalid = build_invalid_mask(selected, stop_id=stop_id)
        logp = masked_log_softmax(logits, invalid)

        tgt = traj_actions[:, t].clamp(min=0, max=model.n_actions - 1)
        nll = -logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        mask_t = traj_mask[:, t]
        losses.append(nll * mask_t)

        selected = update_selected_non_inplace(selected, tgt, stop_id)

    return torch.stack(losses, dim=1).sum() / traj_mask.sum().clamp_min(1.0)


def reward_from_sets(
    pred_y: torch.Tensor,
    true_y: torch.Tensor,
    exact_bonus: float = 0.25,
    length_penalty: float = 0.02,
) -> torch.Tensor:
    inter = (pred_y * true_y).sum(dim=1)
    pred_cnt = pred_y.sum(dim=1)
    true_cnt = true_y.sum(dim=1)

    f1 = (2.0 * inter) / (pred_cnt + true_cnt).clamp_min(1.0)
    exact = (pred_y == true_y).all(dim=1).float()
    len_gap = (pred_cnt - true_cnt).abs()

    reward = f1 + float(exact_bonus) * exact - float(length_penalty) * len_gap
    return reward.clamp_min(1e-4)


def reward_from_numpy(
    pred_y: np.ndarray,
    true_y: np.ndarray,
    exact_bonus: float = 0.25,
    length_penalty: float = 0.02,
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


def trajectory_logprob_for_actions(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    actions: torch.Tensor,
    stop_id: int,
    element_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute log P(actions | x) under current policy.
    Used to give greedy candidates a real trajectory log-probability.
    """
    device = x.device
    bsz, T = actions.shape
    n_prec = model.n_precursors

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    stopped = torch.zeros((bsz,), dtype=torch.bool, device=device)
    logprob_sum = torch.zeros((bsz,), dtype=torch.float32, device=device)

    if element_bias is not None:
        element_bias = element_bias.to(device)

    for t in range(T):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids)
        if element_bias is not None:
            logits = logits + element_bias

        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        logp = masked_log_softmax(logits, invalid)

        act = actions[:, t].clamp(min=0, max=model.n_actions - 1)
        chosen = logp.gather(1, act.unsqueeze(1)).squeeze(1)
        chosen = torch.where(stopped, torch.zeros_like(chosen), chosen)
        logprob_sum = logprob_sum + chosen

        selected = update_selected_non_inplace(selected, act, stop_id)
        stopped = stopped | (act == stop_id)

    return logprob_sum


# =============================================================================
# Decode functions
# =============================================================================
@torch.no_grad()
def greedy_decode(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    force_non_empty: bool = True,
    element_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    bsz = x.shape[0]
    n_prec = model.n_precursors

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    stopped = torch.zeros(bsz, dtype=torch.bool, device=device)
    actions = torch.full((bsz, max_traj_len), stop_id, dtype=torch.long, device=device)

    if element_bias is not None:
        element_bias = element_bias.to(device)

    for t in range(max_traj_len):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids)
        if element_bias is not None:
            logits = logits + element_bias

        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        logits = logits.masked_fill(invalid, -1e9)
        act = torch.argmax(logits, dim=1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        selected = update_selected_non_inplace(selected, act, stop_id)
        stopped = stopped | (act == stop_id)

    if force_non_empty:
        selected = force_non_empty_selected_from_first_step(
            model=model,
            x=x,
            selected=selected,
            stop_id=stop_id,
            element_bias=element_bias,
        )

    logprob_sum = trajectory_logprob_for_actions(
        model=model,
        x=x,
        actions=actions,
        stop_id=stop_id,
        element_bias=element_bias,
    )

    return actions, selected, logprob_sum


@torch.no_grad()
def sample_decode(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    temperature: float = 1.0,
    force_non_empty: bool = True,
    element_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    No-grad version used for evaluation and candidate collection.
    """
    device = x.device
    bsz = x.shape[0]
    n_prec = model.n_precursors
    temperature = max(float(temperature), 1e-6)

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    stopped = torch.zeros(bsz, dtype=torch.bool, device=device)
    actions = torch.full((bsz, max_traj_len), stop_id, dtype=torch.long, device=device)
    logprob_sum = torch.zeros(bsz, dtype=torch.float32, device=device)

    if element_bias is not None:
        element_bias = element_bias.to(device)

    for t in range(max_traj_len):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids) / temperature
        if element_bias is not None:
            logits = logits + element_bias

        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        logp = masked_log_softmax(logits, invalid)
        probs = torch.exp(logp).clamp_min(1e-12)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        act = torch.multinomial(probs, num_samples=1).squeeze(1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        chosen_logp = logp.gather(1, act.unsqueeze(1)).squeeze(1)
        chosen_logp = torch.where(stopped, torch.zeros_like(chosen_logp), chosen_logp)
        logprob_sum = logprob_sum + chosen_logp

        selected = update_selected_non_inplace(selected, act, stop_id)
        stopped = stopped | (act == stop_id)

    if force_non_empty:
        selected = force_non_empty_selected_from_first_step(
            model=model,
            x=x,
            selected=selected,
            stop_id=stop_id,
            element_bias=element_bias,
        )

    return actions, selected, logprob_sum


def sample_decode_train(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    temperature: float = 1.0,
    force_non_empty: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gradient-enabled sampling for weak REINFORCE.
    Important: do NOT decorate with @torch.no_grad().
    """
    device = x.device
    bsz = x.shape[0]
    n_prec = model.n_precursors
    temperature = max(float(temperature), 1e-6)

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    stopped = torch.zeros(bsz, dtype=torch.bool, device=device)
    actions = torch.full((bsz, max_traj_len), stop_id, dtype=torch.long, device=device)
    logprob_sum = torch.zeros(bsz, dtype=torch.float32, device=device)

    for t in range(max_traj_len):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids) / temperature
        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        logp = masked_log_softmax(logits, invalid)
        probs = torch.exp(logp).clamp_min(1e-12)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # Sampling is non-differentiable, but logprob_sum keeps gradient
        # through log P_theta(a_t | s_t), which is exactly REINFORCE.
        act = torch.multinomial(probs.detach(), num_samples=1).squeeze(1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        chosen_logp = logp.gather(1, act.unsqueeze(1)).squeeze(1)
        chosen_logp = torch.where(stopped, torch.zeros_like(chosen_logp), chosen_logp)
        logprob_sum = logprob_sum + chosen_logp

        selected = update_selected_non_inplace(selected, act, stop_id)
        stopped = stopped | (act == stop_id)

    if force_non_empty:
        # Force-fixing empty sets is a decode-time safety fallback.
        # It is not part of the REINFORCE trajectory probability, so selected
        # may be changed but logprob_sum is still for the sampled trajectory.
        selected = force_non_empty_selected_from_first_step(
            model=model,
            x=x,
            selected=selected,
            stop_id=stop_id,
            element_bias=None,
        )

    return actions, selected, logprob_sum


# =============================================================================
# Metrics / eval
# =============================================================================
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
def evaluate_loss(
    model: GFlowNetPolicy,
    loader: DataLoader,
    device: torch.device,
    stop_id: int,
) -> float:
    model.eval()
    losses: List[float] = []

    for x, _, traj_actions, traj_mask in loader:
        x = x.to(device)
        traj_actions = traj_actions.to(device)
        traj_mask = traj_mask.to(device)
        loss = teacher_forcing_loss(model, x, traj_actions, traj_mask, stop_id=stop_id)
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def predict_greedy_multihot(
    model: GFlowNetPolicy,
    loader: DataLoader,
    device: torch.device,
    max_traj_len: int,
    stop_id: int,
    force_non_empty: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []

    for x, y_multi_hot, _, _ in loader:
        x = x.to(device)
        _, pred_y, _ = greedy_decode(
            model=model,
            x=x,
            max_traj_len=max_traj_len,
            stop_id=stop_id,
            force_non_empty=force_non_empty,
            element_bias=None,
        )
        y_true_all.append(y_multi_hot.numpy().astype(int))
        y_pred_all.append(pred_y.cpu().numpy().astype(int))

    return np.vstack(y_true_all), np.vstack(y_pred_all)


def multihot_to_label_lists(y: np.ndarray, precursor_names: List[str]) -> List[List[str]]:
    out: List[List[str]] = []
    for i in range(y.shape[0]):
        idx = np.where(y[i] > 0)[0].tolist()
        out.append([precursor_names[j] for j in idx if 0 <= j < len(precursor_names)])
    return out


def _candidate_group_id_from_meta(meta_df: pd.DataFrame, i: int) -> str:
    if i >= len(meta_df):
        return f"sample_{i}"

    row = meta_df.iloc[i]
    for col in ["candidate_group_id", "sample_id", "material_id", "target_id", "mp_id", "task_id"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return f"sample_{i}"


def save_prediction_csv(
    path: Path,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    precursor_names: List[str],
) -> None:
    out = meta_df.copy()
    out["sample_index"] = np.arange(len(out), dtype=int)
    out["candidate_group_id"] = [
        _candidate_group_id_from_meta(meta_df, i) for i in range(len(out))
    ]
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
    out.to_csv(path, index=False)


# =============================================================================
# Bundle infer helpers
# =============================================================================
def _infer_stop_id(
    action_vocab: List[str],
    action_to_id: Optional[Dict[str, Any]],
    n_precursors: int,
) -> int:
    if action_to_id is not None and STOP_TOKEN in action_to_id:
        return int(action_to_id[STOP_TOKEN])
    if STOP_TOKEN in action_vocab:
        return int(action_vocab.index(STOP_TOKEN))
    if len(action_vocab) == n_precursors + 1:
        return int(len(action_vocab) - 1)
    return int(n_precursors)


def _infer_max_traj_len(
    schema: Dict[str, Any],
    traj_actions_train: np.ndarray,
    traj_mask_train: np.ndarray,
) -> int:
    if getattr(traj_actions_train, "ndim", 0) == 2 and traj_actions_train.shape[1] > 0:
        return int(traj_actions_train.shape[1])
    if getattr(traj_mask_train, "ndim", 0) == 2 and traj_mask_train.shape[1] > 0:
        return int(traj_mask_train.shape[1])
    for k in ["max_traj_len", "traj_len", "max_seq_len", "seq_len"]:
        if k in schema:
            return int(schema[k])
    raise ValueError("无法确定 max_traj_len，请检查 summary.json 或 traj_actions / traj_mask 形状。")


def _infer_n_precursors(
    schema: Dict[str, Any],
    y_train: np.ndarray,
    precursor_names: List[str],
    action_vocab: List[str],
) -> int:
    if getattr(y_train, "ndim", 0) == 2 and y_train.shape[1] > 0:
        return int(y_train.shape[1])
    if precursor_names:
        return int(len(precursor_names))
    if "n_precursors" in schema:
        return int(schema["n_precursors"])
    if len(action_vocab) > 0:
        return int(len(action_vocab) - 1)
    raise ValueError("无法确定 n_precursors。")


def choose_device(device_arg: str) -> torch.device:
    if str(device_arg) != "auto":
        return torch.device(str(device_arg))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Candidate pool / rerank
# =============================================================================
def build_candidate_feature_matrix(
    x_batch: np.ndarray,
    cand_batch: np.ndarray,
    logprob_batch: np.ndarray,
) -> np.ndarray:
    cand_len = cand_batch.sum(axis=1, keepdims=True).astype(np.float32)
    logprob_col = logprob_batch.reshape(-1, 1).astype(np.float32)
    return np.concatenate(
        [
            x_batch.astype(np.float32),
            cand_batch.astype(np.float32),
            cand_len,
            logprob_col,
        ],
        axis=1,
    )


@torch.no_grad()
def collect_candidate_pool(
    model: GFlowNetPolicy,
    loader: DataLoader,
    meta_df: pd.DataFrame,
    device: torch.device,
    max_traj_len: int,
    stop_id: int,
    num_samples: int,
    sample_temperatures: Sequence[float],
    exact_bonus: float,
    length_penalty: float,
    include_greedy: bool = True,
    force_non_empty: bool = True,
    element_bias_enabled: bool = False,
    precursor_elements: Optional[List[Set[str]]] = None,
    target_hit_bonus: float = 6.0,
    extra_element_penalty: float = 1.0,
    no_overlap_penalty: float = 6.0,
    stop_bias: float = -2.0,
    ignore_elements: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    model.eval()

    sample_temperatures = [float(t) for t in sample_temperatures if float(t) > 0]
    if not sample_temperatures:
        sample_temperatures = [1.0]

    rows: List[Dict[str, Any]] = []
    grouped: List[List[Dict[str, Any]]] = []
    y_true_all: List[np.ndarray] = []
    greedy_pred_all: List[np.ndarray] = []

    formula_list = infer_formula_list_from_meta(meta_df)

    global_idx = 0
    for x, y_true, _, _ in loader:
        x_np = x.numpy().astype(np.float32)
        y_np = y_true.numpy().astype(np.int32)
        x = x.to(device)
        bsz = x.shape[0]

        batch_formulas = formula_list[global_idx:global_idx + bsz]
        element_bias = None
        if element_bias_enabled and precursor_elements is not None:
            element_bias = build_element_bias_for_batch(
                formulas=batch_formulas,
                precursor_elements=precursor_elements,
                n_precursors=model.n_precursors,
                stop_id=stop_id,
                target_hit_bonus=target_hit_bonus,
                extra_element_penalty=extra_element_penalty,
                no_overlap_penalty=no_overlap_penalty,
                stop_bias=stop_bias,
                ignore_elements=ignore_elements,
            ).to(device)

        batch_groups: List[Dict[Tuple[int, ...], Dict[str, Any]]] = [dict() for _ in range(bsz)]

        def _merge_candidate(
            sample_i: int,
            cand_vec: np.ndarray,
            actions_vec: np.ndarray,
            logprob_val: float,
            source: str,
        ) -> None:
            key = tuple(np.where(cand_vec > 0)[0].tolist())
            if len(key) == 0:
                return

            cur = batch_groups[sample_i].get(key)
            reward_val = float(
                reward_from_numpy(
                    cand_vec[None, :],
                    y_np[sample_i:sample_i + 1],
                    exact_bonus=exact_bonus,
                    length_penalty=length_penalty,
                )[0]
            )
            exact = int(np.all(cand_vec == y_np[sample_i]))
            cand_len = int(cand_vec.sum())
            action_len = int(np.sum(actions_vec != stop_id))
            row = {
                "sample_index": int(global_idx + sample_i),
                "cand_key": json.dumps(list(key), ensure_ascii=False),
                "logprob": float(logprob_val),
                "source": source,
                "reward": reward_val,
                "exact_match": exact,
                "cand_len": cand_len,
                "action_len": action_len,
                "cand_vec": cand_vec.astype(np.int32),
                "y_true": y_np[sample_i].astype(np.int32),
                "x": x_np[sample_i].astype(np.float32),
            }
            if cur is None or row["logprob"] > cur["logprob"]:
                batch_groups[sample_i][key] = row

        if include_greedy:
            greedy_actions, greedy_pred, greedy_logprob = greedy_decode(
                model=model,
                x=x,
                max_traj_len=max_traj_len,
                stop_id=stop_id,
                force_non_empty=force_non_empty,
                element_bias=element_bias,
            )
            greedy_actions_np = greedy_actions.cpu().numpy().astype(np.int32)
            greedy_pred_np = greedy_pred.cpu().numpy().astype(np.int32)
            greedy_logprob_np = greedy_logprob.cpu().numpy().astype(np.float32)
            greedy_pred_all.append(greedy_pred_np)

            for i in range(bsz):
                _merge_candidate(
                    sample_i=i,
                    cand_vec=greedy_pred_np[i],
                    actions_vec=greedy_actions_np[i],
                    logprob_val=float(greedy_logprob_np[i]),
                    source="greedy",
                )
        else:
            greedy_pred_all.append(np.zeros_like(y_np))

        for sample_round in range(int(num_samples)):
            temp = sample_temperatures[sample_round % len(sample_temperatures)]
            sampled_actions, sampled_pred, logprob_sum = sample_decode(
                model=model,
                x=x,
                max_traj_len=max_traj_len,
                stop_id=stop_id,
                temperature=temp,
                force_non_empty=force_non_empty,
                element_bias=element_bias,
            )
            sampled_actions_np = sampled_actions.cpu().numpy().astype(np.int32)
            sampled_pred_np = sampled_pred.cpu().numpy().astype(np.int32)
            logprob_np = logprob_sum.cpu().numpy().astype(np.float32)

            for i in range(bsz):
                _merge_candidate(
                    sample_i=i,
                    cand_vec=sampled_pred_np[i],
                    actions_vec=sampled_actions_np[i],
                    logprob_val=float(logprob_np[i]),
                    source=f"sample_t{temp:g}",
                )

        for i in range(bsz):
            cand_rows = list(batch_groups[i].values())
            cand_rows.sort(key=lambda z: z["logprob"], reverse=True)
            grouped.append(cand_rows)
            y_true_all.append(y_np[i])

            for row in cand_rows:
                rows.append({
                    "sample_index": int(row["sample_index"]),
                    "source": row["source"],
                    "logprob": float(row["logprob"]),
                    "reward": float(row["reward"]),
                    "exact_match": int(row["exact_match"]),
                    "cand_len": int(row["cand_len"]),
                    "action_len": int(row["action_len"]),
                    "cand_key": row["cand_key"],
                    "cand_labels_idx": row["cand_key"],
                })

        global_idx += bsz

    y_true_arr = np.vstack(y_true_all).astype(np.int32)
    greedy_arr = np.vstack(greedy_pred_all).astype(np.int32)

    return {
        "rows_df": pd.DataFrame(rows),
        "grouped": grouped,
        "y_true": y_true_arr,
        "greedy_pred": greedy_arr,
    }


def flatten_candidate_groups_to_dataset(
    grouped: List[List[Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray]:
    feat_list: List[np.ndarray] = []
    target_list: List[float] = []

    for cand_rows in grouped:
        if not cand_rows:
            continue
        x_batch = np.stack([r["x"] for r in cand_rows], axis=0)
        cand_batch = np.stack([r["cand_vec"] for r in cand_rows], axis=0)
        logprob_batch = np.array([r["logprob"] for r in cand_rows], dtype=np.float32)
        feats = build_candidate_feature_matrix(x_batch, cand_batch, logprob_batch)
        feat_list.append(feats)
        target_list.extend(float(r["reward"]) for r in cand_rows)

    if not feat_list:
        raise ValueError("候选池为空，无法构造 reranker 数据集。")

    return np.vstack(feat_list).astype(np.float32), np.array(target_list, dtype=np.float32)


@torch.no_grad()
def predict_reranker_scores(
    model: CandidateReranker,
    feats: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    ds = TensorDataset(torch.tensor(feats, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out: List[np.ndarray] = []

    for (xb,) in loader:
        xb = xb.to(device)
        pred = model(xb)
        out.append(pred.cpu().numpy())

    return np.concatenate(out, axis=0).astype(np.float32)


def score_grouped_candidates(
    reranker: CandidateReranker,
    grouped: List[List[Dict[str, Any]]],
    device: torch.device,
    batch_size: int = 1024,
) -> List[List[Dict[str, Any]]]:
    scored: List[List[Dict[str, Any]]] = []

    for cand_rows in grouped:
        if not cand_rows:
            scored.append([])
            continue

        x_batch = np.stack([r["x"] for r in cand_rows], axis=0)
        cand_batch = np.stack([r["cand_vec"] for r in cand_rows], axis=0)
        logprob_batch = np.array([r["logprob"] for r in cand_rows], dtype=np.float32)
        feats = build_candidate_feature_matrix(x_batch, cand_batch, logprob_batch)
        scores = predict_reranker_scores(
            reranker,
            feats,
            device=device,
            batch_size=batch_size,
        )

        cur_rows: List[Dict[str, Any]] = []
        for row, score in zip(cand_rows, scores):
            new_row = dict(row)
            new_row["rerank_score"] = float(score)
            cur_rows.append(new_row)

        cur_rows.sort(key=lambda z: z["rerank_score"], reverse=True)
        scored.append(cur_rows)

    return scored


def grouped_top1_predictions(
    grouped: List[List[Dict[str, Any]]],
    fallback_dim: int,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    zero = np.zeros(fallback_dim, dtype=np.int32)

    for rows in grouped:
        if not rows:
            preds.append(zero.copy())
        else:
            preds.append(rows[0]["cand_vec"].astype(np.int32))

    return np.vstack(preds).astype(np.int32)


def compute_exact_hit_at_k(grouped: List[List[Dict[str, Any]]], k: int) -> float:
    hits: List[float] = []
    for rows in grouped:
        cur = rows[:int(k)]
        hits.append(float(any(int(r["exact_match"]) == 1 for r in cur)))
    return float(np.mean(hits)) if hits else math.nan


def compute_oracle_reward_mean(grouped: List[List[Dict[str, Any]]]) -> float:
    best_rewards: List[float] = []
    for rows in grouped:
        if not rows:
            best_rewards.append(0.0)
        else:
            best_rewards.append(max(float(r["reward"]) for r in rows))
    return float(np.mean(best_rewards)) if best_rewards else math.nan


def evaluate_candidate_groups(
    grouped: List[List[Dict[str, Any]]],
    y_true: np.ndarray,
    fallback_dim: int,
    prefix: str,
    topk_values: Iterable[int],
) -> Dict[str, Any]:
    y_pred = grouped_top1_predictions(grouped, fallback_dim=fallback_dim)
    metrics = evaluate_from_binary(y_true.astype(int), y_pred.astype(int))
    out: Dict[str, Any] = {f"{prefix}_{k}": v for k, v in metrics.items()}
    out[f"{prefix}_exact_hit@1"] = compute_exact_hit_at_k(grouped, k=1)
    out[f"{prefix}_oracle_reward_mean"] = compute_oracle_reward_mean(grouped)
    out[f"{prefix}_mean_candidates"] = (
        float(np.mean([len(rows) for rows in grouped])) if grouped else math.nan
    )

    for k in topk_values:
        out[f"{prefix}_exact_hit@{int(k)}"] = compute_exact_hit_at_k(grouped, k=int(k))

    return out


# =============================================================================
# Reranker train
# =============================================================================
def train_reranker(
    train_feats: np.ndarray,
    train_targets: np.ndarray,
    val_feats: np.ndarray,
    val_targets: np.ndarray,
    device: torch.device,
    hidden_dims: List[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
) -> Tuple[CandidateReranker, List[Dict[str, float]], Dict[str, Any]]:
    train_ds = CandidateRerankDataset(train_feats, train_targets)
    val_ds = CandidateRerankDataset(val_feats, val_targets)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = CandidateReranker(
        input_dim=train_feats.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    bad_epochs = 0
    logs: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        val_loss = float(np.mean(val_losses)) if val_losses else math.nan
        logs.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

        print(
            f"[Rerank Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"best={min(best_val, val_loss):.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "best_val_loss": float(val_loss),
                "input_dim": int(train_feats.shape[1]),
                "hidden_dims": hidden_dims,
                "dropout": float(dropout),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(patience):
            print(f"[Rerank Early Stop] patience reached at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Reranker 训练失败，未产生 best_state。")

    model.load_state_dict(best_state["model_state_dict"])
    return model, logs, best_state


# =============================================================================
# Save helpers
# =============================================================================
def save_candidate_csv(
    path: Path,
    meta_df: pd.DataFrame,
    grouped: List[List[Dict[str, Any]]],
    precursor_names: List[str],
    topn: int = 10,
) -> None:
    rows: List[Dict[str, Any]] = []

    for sample_idx, cand_rows in enumerate(grouped):
        base = (
            meta_df.iloc[sample_idx].to_dict()
            if sample_idx < len(meta_df)
            else {"sample_index": sample_idx}
        )
        base["sample_index"] = int(sample_idx)
        base["candidate_group_id"] = _candidate_group_id_from_meta(meta_df, sample_idx)

        for rank, row in enumerate(cand_rows[:int(topn)], start=1):
            item = dict(base)
            item["rank"] = int(rank)
            item["source"] = row.get("source", "")
            item["logprob"] = float(row.get("logprob", 0.0))
            item["reward"] = float(row.get("reward", 0.0))
            item["exact_match"] = int(row.get("exact_match", 0))
            item["cand_len"] = int(row.get("cand_len", 0))
            item["action_len"] = int(row.get("action_len", 0))
            item["rerank_score"] = float(row.get("rerank_score", np.nan))

            pred_labels = multihot_to_label_lists(
                row["cand_vec"][None, :].astype(int),
                precursor_names,
            )[0]
            true_labels = multihot_to_label_lists(
                row["y_true"][None, :].astype(int),
                precursor_names,
            )[0]
            item["pred_labels"] = json.dumps(pred_labels, ensure_ascii=False)
            item["true_labels"] = json.dumps(true_labels, ensure_ascii=False)
            rows.append(item)

    pd.DataFrame(rows).to_csv(path, index=False)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Stage2 GFlowNet-style precursor-set generator with candidate reranking."
    )

    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--input_mode", type=str, default="hybrid")
    parser.add_argument("--mode_input_root", type=str, default="")
    parser.add_argument(
        "--train_mode",
        type=str,
        default="gold_only",
        choices=["relaxed_only", "gold_only", "curriculum", "curriculum_phase1", "curriculum_phase2"],
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="/Users/wyc/SynPred/runs/stage2/gflownet_hybrid_rerank_v1",
    )
    parser.add_argument("--device", type=str, default="auto")

    # policy
    parser.add_argument(
        "--max_traj_len_override",
        type=int,
        default=0,
        help="Override inferred max_traj_len. Useful for curriculum finetuning to keep Phase1/Phase2 model shape consistent.",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--x_mlp_hidden_dims", type=str, default="512")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--metric_name", type=str, default="samples_f1")
    parser.add_argument("--seed", type=int, default=42)

    # weak RL
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--rl_weight", type=float, default=0.2)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--exact_bonus", type=float, default=0.25)
    parser.add_argument("--length_penalty", type=float, default=0.02)

    parser.add_argument("--force_non_empty", action="store_true", default=True)
    parser.add_argument("--no_force_non_empty", action="store_true")

    # candidate pool / rerank
    parser.add_argument("--rerank_enabled", action="store_true", default=True)
    parser.add_argument("--no_rerank", action="store_true")
    parser.add_argument("--rerank_num_samples_train", type=int, default=7)
    parser.add_argument("--rerank_num_samples_eval", type=int, default=15)
    parser.add_argument("--rerank_sample_temperatures", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--rerank_hidden_dims", type=str, default="512,256")
    parser.add_argument("--rerank_dropout", type=float, default=0.1)
    parser.add_argument("--rerank_lr", type=float, default=1e-3)
    parser.add_argument("--rerank_weight_decay", type=float, default=1e-5)
    parser.add_argument("--rerank_batch_size", type=int, default=512)
    parser.add_argument("--rerank_epochs", type=int, default=30)
    parser.add_argument("--rerank_patience", type=int, default=6)
    parser.add_argument("--save_topn_candidates", type=int, default=10)
    parser.add_argument("--topk_values", type=str, default="1,3,5,10")

    # finetune
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="",
        help="Path to pretrained model checkpoint for curriculum/finetune training.",
    )

    # optional element soft-bias decoding for candidate collection
    parser.add_argument("--element_bias_enabled", action="store_true", default=False)
    parser.add_argument("--target_hit_bonus", type=float, default=6.0)
    parser.add_argument("--extra_element_penalty", type=float, default=1.0)
    parser.add_argument("--no_overlap_penalty", type=float, default=6.0)
    parser.add_argument("--stop_bias", type=float, default=-2.0)
    parser.add_argument("--ignore_elements", type=str, default="H,O")

    args = parser.parse_args()

    if args.no_force_non_empty:
        args.force_non_empty = False
    if args.no_rerank:
        args.rerank_enabled = False

    set_seed(int(args.seed))

    resolved = resolve_input_bundle(args)
    run_dir = Path(args.run_dir).expanduser().resolve()
    ensure_dir(run_dir)

    print(f"[Info] resolved_mode = {resolved['resolved_mode']}")
    print(f"[Info] resolved_root = {resolved['resolved_root']}")
    print(f"[Info] resolved_input_dir = {resolved['resolved_input_dir']}")

    files = resolved["files"]

    train_pack = load_npz(files["train_npz"])
    val_pack = load_npz(files["val_npz"])
    test_pack = load_npz(files["test_npz"])

    train_meta = pd.read_csv(files["train_meta"])
    val_meta = pd.read_csv(files["val_meta"])
    test_meta = pd.read_csv(files["test_meta"])

    action_vocab = load_json(files["action_vocab"])
    action_to_id = load_json(files["action_to_id"]) if "action_to_id" in files else None
    precursor_names_raw = load_json(files["precursor_names"])
    summary_obj = load_json(files["summary"])
    schema = summary_obj.get("schema", summary_obj) if isinstance(summary_obj, dict) else {}

    x_train = _extract_x(train_pack)
    y_train = _extract_y(train_pack)
    traj_actions_train = _extract_traj_actions(train_pack)
    traj_mask_train = _extract_traj_mask(train_pack)

    x_val = _extract_x(val_pack)
    y_val = _extract_y(val_pack)
    traj_actions_val = _extract_traj_actions(val_pack)
    traj_mask_val = _extract_traj_mask(val_pack)

    x_test = _extract_x(test_pack)
    y_test = _extract_y(test_pack)
    traj_actions_test = _extract_traj_actions(test_pack)
    traj_mask_test = _extract_traj_mask(test_pack)

    if x_train.shape[0] == 0:
        raise ValueError(f"训练集为空: {files['train_npz']}")

    n_precursors = _infer_n_precursors(schema, y_train, precursor_names_raw, action_vocab)
    precursor_names = ensure_precursor_names(precursor_names_raw, n_precursors)
    stop_id = _infer_stop_id(action_vocab, action_to_id, n_precursors)

    max_traj_len = _infer_max_traj_len(schema, traj_actions_train, traj_mask_train)
    if int(args.max_traj_len_override) > 0:
        print(
            f"[Info] Override max_traj_len: "
            f"inferred={max_traj_len}, override={args.max_traj_len_override}"
        )
        max_traj_len = int(args.max_traj_len_override)

    label_stats = label_statistics(y_train)
    if label_stats["constant_zero_labels"] or label_stats["constant_one_labels"]:
        print(
            "[Warn] Detected constant labels in training set: "
            f"all_zero={label_stats['constant_zero_labels']}, "
            f"all_one={label_stats['constant_one_labels']}."
        )

    train_ds = Stage2GFlowNetDataset(x_train, y_train, traj_actions_train, traj_mask_train)
    val_ds = Stage2GFlowNetDataset(x_val, y_val, traj_actions_val, traj_mask_val)
    test_ds = Stage2GFlowNetDataset(x_test, y_test, traj_actions_test, traj_mask_test)

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False)

    device = choose_device(str(args.device))
    print(f"[Info] device = {device}")

    model = GFlowNetPolicy(
        x_dim=int(x_train.shape[1]),
        n_precursors=int(n_precursors),
        hidden_dim=int(args.hidden_dim),
        max_traj_len=int(max_traj_len),
        x_mlp_hidden_dims=parse_hidden_dims(args.x_mlp_hidden_dims),
        dropout=float(args.dropout),
    ).to(device)

    if args.pretrained_model and Path(args.pretrained_model).exists():
        print(f"[Info] Loading pretrained model from {args.pretrained_model}")
        ckpt = torch.load(args.pretrained_model, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [Warn] Missing keys: {missing[:10]}")
        if unexpected:
            print(f"  [Warn] Unexpected keys: {unexpected[:10]}")
        print("  [Info] Pretrained model loaded.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_metric = -1e18
    best_epoch = -1
    best_state: Optional[Dict[str, Any]] = None
    bad_epochs = 0
    train_log: List[Dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_sup_losses: List[float] = []
        epoch_rl_losses: List[float] = []
        epoch_total_losses: List[float] = []

        for x, y_multi_hot, traj_actions, traj_mask in train_loader:
            x = x.to(device)
            y_multi_hot = y_multi_hot.to(device)
            traj_actions = traj_actions.to(device)
            traj_mask = traj_mask.to(device)

            sup_loss = teacher_forcing_loss(
                model=model,
                x=x,
                traj_actions=traj_actions,
                traj_mask=traj_mask,
                stop_id=stop_id,
            )

            if epoch > int(args.warmup_epochs) and float(args.rl_weight) > 0:
                _, pred_y, logprob_sum = sample_decode_train(
                    model=model,
                    x=x,
                    max_traj_len=max_traj_len,
                    stop_id=stop_id,
                    temperature=float(args.sample_temperature),
                    force_non_empty=bool(args.force_non_empty),
                )

                # Reward is non-differentiable w.r.t. sampled discrete set.
                # REINFORCE requires gradient only through logprob_sum.
                reward = reward_from_sets(
                    pred_y=pred_y,
                    true_y=y_multi_hot,
                    exact_bonus=float(args.exact_bonus),
                    length_penalty=float(args.length_penalty),
                ).detach()

                baseline = reward.mean()
                advantage = reward - baseline
                rl_loss = -(advantage * logprob_sum).mean()
            else:
                rl_loss = torch.tensor(0.0, device=device)

            total_loss = sup_loss + float(args.rl_weight) * rl_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_sup_losses.append(float(sup_loss.item()))
            epoch_rl_losses.append(float(rl_loss.item()))
            epoch_total_losses.append(float(total_loss.item()))

        val_loss = evaluate_loss(model, val_loader, device, stop_id=stop_id)
        y_val_true, y_val_pred = predict_greedy_multihot(
            model=model,
            loader=val_loader,
            device=device,
            max_traj_len=max_traj_len,
            stop_id=stop_id,
            force_non_empty=bool(args.force_non_empty),
        )
        val_metrics = evaluate_from_binary(y_val_true.astype(int), y_val_pred.astype(int))
        cur_metric = float(val_metrics.get(str(args.metric_name), val_metrics["samples_f1"]))

        rec = {
            "epoch": int(epoch),
            "train_sup_loss": float(np.mean(epoch_sup_losses)) if epoch_sup_losses else math.nan,
            "train_rl_loss": float(np.mean(epoch_rl_losses)) if epoch_rl_losses else math.nan,
            "train_total_loss": float(np.mean(epoch_total_losses)) if epoch_total_losses else math.nan,
            "val_loss": float(val_loss),
            "val_metrics": val_metrics,
        }
        train_log.append(rec)

        print(
            f"[Epoch {epoch:03d}] "
            f"sup_loss={rec['train_sup_loss']:.4f} "
            f"rl_loss={rec['train_rl_loss']:.4f} "
            f"total_loss={rec['train_total_loss']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_{args.metric_name}={cur_metric:.4f} "
            f"best={max(best_metric, cur_metric):.4f}"
        )

        if cur_metric > best_metric:
            best_metric = cur_metric
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": int(epoch),
                "best_val_metric": float(cur_metric),
                "config": vars(args),
                "x_dim": int(x_train.shape[1]),
                "n_precursors": int(n_precursors),
                "max_traj_len": int(max_traj_len),
                "stop_id": int(stop_id),
                "action_vocab": action_vocab,
                "action_to_id": action_to_id,
                "precursor_names": precursor_names,
            }
            torch.save(best_state, run_dir / "best_model.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= int(args.patience):
            print(f"[Early Stop] patience reached at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce checkpoint.")

    model.load_state_dict(best_state["model_state_dict"])
    write_json(run_dir / "train_log.json", train_log)

    y_val_true, y_val_pred = predict_greedy_multihot(
        model=model,
        loader=val_loader,
        device=device,
        max_traj_len=max_traj_len,
        stop_id=stop_id,
        force_non_empty=bool(args.force_non_empty),
    )
    y_test_true, y_test_pred = predict_greedy_multihot(
        model=model,
        loader=test_loader,
        device=device,
        max_traj_len=max_traj_len,
        stop_id=stop_id,
        force_non_empty=bool(args.force_non_empty),
    )

    greedy_val_metrics = evaluate_from_binary(y_val_true.astype(int), y_val_pred.astype(int))
    greedy_test_metrics = evaluate_from_binary(y_test_true.astype(int), y_test_pred.astype(int))

    save_prediction_csv(
        run_dir / "pred_val_greedy.csv",
        val_meta,
        y_val_true.astype(int),
        y_val_pred.astype(int),
        precursor_names,
    )
    save_prediction_csv(
        run_dir / "pred_test_greedy.csv",
        test_meta,
        y_test_true.astype(int),
        y_test_pred.astype(int),
        precursor_names,
    )

    summary: Dict[str, Any] = {
        "config": vars(args),
        "resolved_mode": resolved["resolved_mode"],
        "resolved_root": resolved["resolved_root"],
        "resolved_input_dir": resolved["resolved_input_dir"],
        "resolved_files": files,
        "device": str(device),
        "data": {
            "n_train": int(x_train.shape[0]),
            "n_val": int(x_val.shape[0]),
            "n_test": int(x_test.shape[0]),
            "x_dim": int(x_train.shape[1]),
            "n_precursors": int(n_precursors),
            "action_vocab_size": int(len(action_vocab)),
            "stop_id": int(stop_id),
            "max_traj_len": int(max_traj_len),
            "traj_actions_len": int(traj_actions_train.shape[1]) if traj_actions_train.ndim == 2 else None,
            "traj_mask_len": int(traj_mask_train.shape[1]) if traj_mask_train.ndim == 2 else None,
            **label_stats,
        },
        "training": {
            "best_epoch": int(best_epoch),
            "best_val_metric": float(best_metric),
            "metric_name": str(args.metric_name),
            "force_non_empty": bool(args.force_non_empty),
            "early_stopping_metric": f"greedy_{args.metric_name}",
            "weak_rl_note": (
                "Weak REINFORCE is implemented with gradient-enabled sample_decode_train. "
                "This is not full Trajectory Balance; no backward policy or logZ is trained."
            ),
        },
        "greedy_val_metrics": greedy_val_metrics,
        "greedy_test_metrics": greedy_test_metrics,
    }

    topk_values = parse_int_list(str(args.topk_values))
    sample_temps = parse_float_list(str(args.rerank_sample_temperatures))
    if not topk_values:
        topk_values = [1, 3, 5, 10]
    if not sample_temps:
        sample_temps = [1.0]

    ignore_elements = parse_ignore_elements(args.ignore_elements)
    precursor_elements = None
    if bool(args.element_bias_enabled):
        precursor_elements = build_precursor_element_index(
            precursor_names,
            ignore_elements=ignore_elements,
        )
        print(
            "[Info] Element soft-bias enabled: "
            f"target_hit_bonus={args.target_hit_bonus}, "
            f"extra_element_penalty={args.extra_element_penalty}, "
            f"no_overlap_penalty={args.no_overlap_penalty}, "
            f"stop_bias={args.stop_bias}, "
            f"ignore_elements={sorted(ignore_elements)}"
        )

    if args.rerank_enabled:
        print("[Info] collecting candidate pools for train/val/test ...")

        train_pool = collect_candidate_pool(
            model=model,
            loader=train_eval_loader,
            meta_df=train_meta,
            device=device,
            max_traj_len=max_traj_len,
            stop_id=stop_id,
            num_samples=int(args.rerank_num_samples_train),
            sample_temperatures=sample_temps,
            exact_bonus=float(args.exact_bonus),
            length_penalty=float(args.length_penalty),
            include_greedy=True,
            force_non_empty=bool(args.force_non_empty),
            element_bias_enabled=bool(args.element_bias_enabled),
            precursor_elements=precursor_elements,
            target_hit_bonus=float(args.target_hit_bonus),
            extra_element_penalty=float(args.extra_element_penalty),
            no_overlap_penalty=float(args.no_overlap_penalty),
            stop_bias=float(args.stop_bias),
            ignore_elements=ignore_elements,
        )
        val_pool = collect_candidate_pool(
            model=model,
            loader=val_loader,
            meta_df=val_meta,
            device=device,
            max_traj_len=max_traj_len,
            stop_id=stop_id,
            num_samples=int(args.rerank_num_samples_eval),
            sample_temperatures=sample_temps,
            exact_bonus=float(args.exact_bonus),
            length_penalty=float(args.length_penalty),
            include_greedy=True,
            force_non_empty=bool(args.force_non_empty),
            element_bias_enabled=bool(args.element_bias_enabled),
            precursor_elements=precursor_elements,
            target_hit_bonus=float(args.target_hit_bonus),
            extra_element_penalty=float(args.extra_element_penalty),
            no_overlap_penalty=float(args.no_overlap_penalty),
            stop_bias=float(args.stop_bias),
            ignore_elements=ignore_elements,
        )
        test_pool = collect_candidate_pool(
            model=model,
            loader=test_loader,
            meta_df=test_meta,
            device=device,
            max_traj_len=max_traj_len,
            stop_id=stop_id,
            num_samples=int(args.rerank_num_samples_eval),
            sample_temperatures=sample_temps,
            exact_bonus=float(args.exact_bonus),
            length_penalty=float(args.length_penalty),
            include_greedy=True,
            force_non_empty=bool(args.force_non_empty),
            element_bias_enabled=bool(args.element_bias_enabled),
            precursor_elements=precursor_elements,
            target_hit_bonus=float(args.target_hit_bonus),
            extra_element_penalty=float(args.extra_element_penalty),
            no_overlap_penalty=float(args.no_overlap_penalty),
            stop_bias=float(args.stop_bias),
            ignore_elements=ignore_elements,
        )

        write_json(run_dir / "candidate_pool_train_summary.json", {
            "mean_candidates": float(np.mean([len(x) for x in train_pool["grouped"]])),
            "n_samples": int(len(train_pool["grouped"])),
        })
        write_json(run_dir / "candidate_pool_val_summary.json", {
            "mean_candidates": float(np.mean([len(x) for x in val_pool["grouped"]])),
            "n_samples": int(len(val_pool["grouped"])),
        })
        write_json(run_dir / "candidate_pool_test_summary.json", {
            "mean_candidates": float(np.mean([len(x) for x in test_pool["grouped"]])),
            "n_samples": int(len(test_pool["grouped"])),
        })

        train_feats, train_targets = flatten_candidate_groups_to_dataset(train_pool["grouped"])
        val_feats, val_targets = flatten_candidate_groups_to_dataset(val_pool["grouped"])

        reranker, rerank_logs, rerank_best_state = train_reranker(
            train_feats=train_feats,
            train_targets=train_targets,
            val_feats=val_feats,
            val_targets=val_targets,
            device=device,
            hidden_dims=parse_hidden_dims(args.rerank_hidden_dims),
            dropout=float(args.rerank_dropout),
            lr=float(args.rerank_lr),
            weight_decay=float(args.rerank_weight_decay),
            batch_size=int(args.rerank_batch_size),
            epochs=int(args.rerank_epochs),
            patience=int(args.rerank_patience),
        )

        torch.save({
            **rerank_best_state,
            "config": vars(args),
        }, run_dir / "best_reranker.pt")
        write_json(run_dir / "rerank_train_log.json", rerank_logs)

        val_scored = score_grouped_candidates(
            reranker,
            val_pool["grouped"],
            device=device,
            batch_size=int(args.rerank_batch_size),
        )
        test_scored = score_grouped_candidates(
            reranker,
            test_pool["grouped"],
            device=device,
            batch_size=int(args.rerank_batch_size),
        )

        sample_val_metrics = evaluate_candidate_groups(
            grouped=val_pool["grouped"],
            y_true=val_pool["y_true"],
            fallback_dim=n_precursors,
            prefix="sample_pool_val",
            topk_values=topk_values,
        )
        sample_test_metrics = evaluate_candidate_groups(
            grouped=test_pool["grouped"],
            y_true=test_pool["y_true"],
            fallback_dim=n_precursors,
            prefix="sample_pool_test",
            topk_values=topk_values,
        )
        rerank_val_metrics = evaluate_candidate_groups(
            grouped=val_scored,
            y_true=val_pool["y_true"],
            fallback_dim=n_precursors,
            prefix="rerank_val",
            topk_values=topk_values,
        )
        rerank_test_metrics = evaluate_candidate_groups(
            grouped=test_scored,
            y_true=test_pool["y_true"],
            fallback_dim=n_precursors,
            prefix="rerank_test",
            topk_values=topk_values,
        )

        save_candidate_csv(
            run_dir / "val_candidates_sample_pool.csv",
            val_meta,
            val_pool["grouped"],
            precursor_names,
            topn=int(args.save_topn_candidates),
        )
        save_candidate_csv(
            run_dir / "test_candidates_sample_pool.csv",
            test_meta,
            test_pool["grouped"],
            precursor_names,
            topn=int(args.save_topn_candidates),
        )
        save_candidate_csv(
            run_dir / "val_candidates_reranked.csv",
            val_meta,
            val_scored,
            precursor_names,
            topn=int(args.save_topn_candidates),
        )
        save_candidate_csv(
            run_dir / "test_candidates_reranked.csv",
            test_meta,
            test_scored,
            precursor_names,
            topn=int(args.save_topn_candidates),
        )

        summary["rerank"] = {
            "enabled": True,
            "train_candidate_dataset": {
                "n_rows": int(train_feats.shape[0]),
                "feat_dim": int(train_feats.shape[1]),
            },
            "val_candidate_dataset": {
                "n_rows": int(val_feats.shape[0]),
                "feat_dim": int(val_feats.shape[1]),
            },
            "feature_definition": "[x, candidate_y_multi_hot, candidate_len, trajectory_logprob]",
            "target_definition": "oracle reward = F1 + exact_bonus * exact_match - length_penalty * length_gap",
            "offline_eval_note": (
                "reward and exact_match are used only for supervised reranker training "
                "and offline validation/test diagnostics. They are not available for "
                "unlabeled deployment inference."
            ),
            "element_bias": {
                "enabled": bool(args.element_bias_enabled),
                "target_hit_bonus": float(args.target_hit_bonus),
                "extra_element_penalty": float(args.extra_element_penalty),
                "no_overlap_penalty": float(args.no_overlap_penalty),
                "stop_bias": float(args.stop_bias),
                "ignore_elements": sorted(ignore_elements),
                "note": "This is soft additive logits bias during candidate collection, not hard masking.",
            },
            "sample_pool_val_metrics": sample_val_metrics,
            "sample_pool_test_metrics": sample_test_metrics,
            "rerank_val_metrics": rerank_val_metrics,
            "rerank_test_metrics": rerank_test_metrics,
            "improvement_over_greedy": {
                "val_samples_f1_gain": float(
                    rerank_val_metrics.get("rerank_val_samples_f1", 0.0)
                    - greedy_val_metrics.get("samples_f1", 0.0)
                ),
                "test_samples_f1_gain": float(
                    rerank_test_metrics.get("rerank_test_samples_f1", 0.0)
                    - greedy_test_metrics.get("samples_f1", 0.0)
                ),
                "val_exact_hit@5_gain_vs_greedy": float(
                    rerank_val_metrics.get("rerank_val_exact_hit@5", 0.0)
                    - greedy_val_metrics.get("subset_accuracy", 0.0)
                ),
                "test_exact_hit@5_gain_vs_greedy": float(
                    rerank_test_metrics.get("rerank_test_exact_hit@5", 0.0)
                    - greedy_test_metrics.get("subset_accuracy", 0.0)
                ),
            },
        }
    else:
        summary["rerank"] = {"enabled": False}

    write_json(run_dir / "metrics.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
