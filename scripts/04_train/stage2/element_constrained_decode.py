#!/usr/bin/env python3
"""
Element-constrained decoding for Stage2 GFlowNet precursor prediction.

Core idea: during generation, mask out precursors whose structural metals
are NOT a subset of the target material's metals. This eliminates ~58% of
invalid candidates without retraining.

Usage:
    python element_constrained_decode.py \
        --model_path runs/stage2/gflownet_joint_rerank_hybrid_gold_only_v1/best_model.pt \
        --input_dir data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only \
        --output_dir runs/stage2/gflownet_element_constrained_v1 \
        --num_samples 256 \
        --sample_temperatures 0.8,1.0,1.2
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
for _p in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STOP_TOKEN = "<stop>"

STRUCTURAL_METALS = frozenset({
    "Li", "Na", "K", "Rb", "Cs",
    "Be", "Mg", "Ca", "Sr", "Ba",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
    "Al", "Ga", "In", "Tl", "Ge", "Sn", "Pb", "Sb", "Bi", "Te", "Se",
    "Si", "P", "B", "Th", "U",
})

ELEMENT_PAT = re.compile(r"([A-Z][a-z]?)")


# =========================
# Model (same as train_gflownet_rerank.py)
# =========================
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
        self.x_dim = x_dim
        self.n_precursors = n_precursors
        self.n_actions = n_precursors + 1
        self.max_traj_len = max_traj_len
        self.hidden_dim = hidden_dim

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


# =========================
# Element constraint logic
# =========================
def extract_metals_from_formula(formula: str) -> Set[str]:
    return set(el for el in ELEMENT_PAT.findall(formula) if el in STRUCTURAL_METALS)


def build_precursor_metal_index(precursor_names: List[str]) -> List[Set[str]]:
    """For each precursor, extract its structural metals."""
    return [extract_metals_from_formula(name) for name in precursor_names]


def build_element_mask_for_batch(
    formulas: List[str],
    precursor_metals: List[Set[str]],
    n_precursors: int,
    stop_id: int,
) -> torch.Tensor:
    """
    Build a boolean mask [bsz, n_actions] where True = INVALID (should be masked).
    A precursor is valid if its metals are a subset of the target's metals.
    Stop action is always valid.
    """
    bsz = len(formulas)
    n_actions = n_precursors + 1
    mask = torch.zeros((bsz, n_actions), dtype=torch.bool)

    for i, formula in enumerate(formulas):
        target_metals = extract_metals_from_formula(formula)
        if not target_metals:
            continue
        for j in range(n_precursors):
            prec_metals = precursor_metals[j]
            if not prec_metals:
                continue
            if not prec_metals.issubset(target_metals):
                mask[i, j] = True

    # Stop is always valid
    mask[:, stop_id] = False
    return mask


# =========================
# Constrained decode functions
# =========================
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


@torch.no_grad()
def greedy_decode_constrained(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    element_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    bsz = x.shape[0]
    n_prec = model.n_precursors
    element_mask = element_mask.to(device)

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    stopped = torch.zeros(bsz, dtype=torch.bool, device=device)
    actions = torch.full((bsz, max_traj_len), stop_id, dtype=torch.long, device=device)

    for t in range(max_traj_len):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids)
        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        # Combine with element constraint
        combined_invalid = invalid | element_mask
        logits = logits.masked_fill(combined_invalid, -1e9)
        act = torch.argmax(logits, dim=1)

        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        add_mask = (act != stop_id)
        if add_mask.any():
            next_selected = selected.clone()
            rows = torch.nonzero(add_mask, as_tuple=False).squeeze(1)
            cols = act[rows]
            next_selected[rows, cols] = 1.0
            selected = next_selected

        stopped = stopped | (act == stop_id)

    return actions, selected


@torch.no_grad()
def sample_decode_constrained(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    element_mask: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = x.device
    bsz = x.shape[0]
    n_prec = model.n_precursors
    temperature = max(float(temperature), 1e-6)
    element_mask = element_mask.to(device)

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    stopped = torch.zeros(bsz, dtype=torch.bool, device=device)
    actions = torch.full((bsz, max_traj_len), stop_id, dtype=torch.long, device=device)
    logprob_sum = torch.zeros(bsz, dtype=torch.float32, device=device)

    for t in range(max_traj_len):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids) / temperature
        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        combined_invalid = invalid | element_mask
        logits = logits.masked_fill(combined_invalid, -1e9)
        logp = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logp)

        # Clamp for numerical safety
        probs = probs.clamp_min(1e-10)
        act = torch.multinomial(probs, num_samples=1).squeeze(1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        chosen_logp = logp.gather(1, act.unsqueeze(1)).squeeze(1)
        chosen_logp = torch.where(stopped, torch.zeros_like(chosen_logp), chosen_logp)
        logprob_sum = logprob_sum + chosen_logp

        add_mask = (act != stop_id)
        if add_mask.any():
            next_selected = selected.clone()
            rows = torch.nonzero(add_mask, as_tuple=False).squeeze(1)
            cols = act[rows]
            next_selected[rows, cols] = 1.0
            selected = next_selected

        stopped = stopped | (act == stop_id)

    return actions, selected, logprob_sum


# =========================
# Evaluation
# =========================
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


def multihot_to_label_lists(y: np.ndarray, precursor_names: List[str]) -> List[List[str]]:
    out = []
    for i in range(y.shape[0]):
        idx = np.where(y[i] > 0)[0].tolist()
        out.append([precursor_names[j] for j in idx])
    return out


def reward_from_numpy(
    pred_y: np.ndarray,
    true_y: np.ndarray,
    exact_bonus: float = 4.0,
    length_penalty: float = 0.05,
) -> np.ndarray:
    pred_y = pred_y.astype(np.float32)
    true_y = true_y.astype(np.float32)
    inter = (pred_y * true_y).sum(axis=1)
    pred_cnt = pred_y.sum(axis=1)
    true_cnt = true_y.sum(axis=1)
    f1 = (2.0 * inter) / np.clip(pred_cnt + true_cnt, 1.0, None)
    exact = (pred_y == true_y).all(axis=1).astype(np.float32)
    len_gap = np.abs(pred_cnt - true_cnt)
    reward = f1 + exact_bonus * exact - length_penalty * len_gap
    return np.clip(reward, 1e-4, None)


# =========================
# Dataset
# =========================
class Stage2Dataset(Dataset):
    def __init__(self, x: np.ndarray, y_multi_hot: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_multi_hot = torch.tensor(y_multi_hot, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_multi_hot[idx]


# =========================
# Main
# =========================
def load_model(model_path: Path, device: torch.device) -> Tuple[GFlowNetPolicy, Dict[str, Any]]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    x_dim = ckpt["x_dim"]
    n_precursors = ckpt["n_precursors"]
    max_traj_len = ckpt["max_traj_len"]
    hidden_dim = config.get("hidden_dim", 256)
    x_mlp_hidden_dims = [int(x) for x in str(config.get("x_mlp_hidden_dims", "512,256")).split(",") if x.strip()]
    dropout = config.get("dropout", 0.1)

    model = GFlowNetPolicy(
        x_dim=x_dim,
        n_precursors=n_precursors,
        hidden_dim=hidden_dim,
        max_traj_len=max_traj_len,
        x_mlp_hidden_dims=x_mlp_hidden_dims,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _is_element_consistent(cand_vec: np.ndarray, element_mask_row: np.ndarray) -> bool:
    """Check if a candidate set only uses element-allowed precursors."""
    active_indices = np.where(cand_vec > 0)[0]
    for idx in active_indices:
        if element_mask_row[idx]:
            return False
    return True


def collect_candidates_hybrid(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    y_true: torch.Tensor,
    element_mask: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    num_samples: int,
    sample_temperatures: List[float],
    exact_bonus: float,
    length_penalty: float,
    device: torch.device,
) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
    """
    Hybrid candidate collection:
    1. Constrained greedy decode (best element-consistent prediction)
    2. Unconstrained diverse sampling at high temperatures
    3. Post-hoc element filtering on sampled candidates
    4. Keep some unconstrained candidates for diversity
    """
    bsz = x.shape[0]
    n_prec = model.n_precursors
    x = x.to(device)
    element_mask_dev = element_mask.to(device)
    element_mask_np = element_mask[:, :n_prec].numpy()
    y_np = y_true.numpy().astype(np.int32)

    # Constrained greedy
    _, greedy_selected = greedy_decode_constrained(model, x, max_traj_len, stop_id, element_mask_dev)
    greedy_pred = greedy_selected.cpu().numpy().astype(np.int32)

    # Also unconstrained greedy for comparison
    _, greedy_unconstrained = greedy_decode_unconstrained(model, x, max_traj_len, stop_id)
    greedy_unc_np = greedy_unconstrained.cpu().numpy().astype(np.int32)

    # Per-sample candidate tracking
    grouped: List[List[Dict[str, Any]]] = [[] for _ in range(bsz)]

    def _add_candidate(i: int, cand_vec: np.ndarray, logprob: float, source: str, element_ok: bool):
        key = tuple(np.where(cand_vec > 0)[0].tolist())
        if not key:
            return
        for existing in grouped[i]:
            if existing["key"] == key:
                if logprob > existing["logprob"]:
                    existing["logprob"] = logprob
                    existing["source"] = source
                return
        reward_val = float(reward_from_numpy(cand_vec[None, :], y_np[i:i + 1], exact_bonus, length_penalty)[0])
        exact = int(np.all(cand_vec == y_np[i]))
        grouped[i].append({
            "key": key,
            "cand_vec": cand_vec,
            "logprob": logprob,
            "source": source,
            "reward": reward_val,
            "exact_match": exact,
            "cand_len": int(cand_vec.sum()),
            "element_consistent": element_ok,
        })

    # Add constrained greedy
    for i in range(bsz):
        _add_candidate(i, greedy_pred[i], 0.0, "greedy_constrained", True)

    # Add unconstrained greedy
    for i in range(bsz):
        ec = _is_element_consistent(greedy_unc_np[i], element_mask_np[i])
        _add_candidate(i, greedy_unc_np[i], 0.0, "greedy_unconstrained", ec)

    # Diverse unconstrained sampling with post-hoc filtering
    # Use wider temperature range for diversity
    diverse_temps = sample_temperatures + [1.5, 2.0]
    samples_per_temp = max(1, num_samples // len(diverse_temps))

    for temp in diverse_temps:
        for _ in range(samples_per_temp):
            # Unconstrained sampling
            _, sampled, logprobs = sample_decode_unconstrained(
                model, x, max_traj_len, stop_id, temperature=temp
            )
            sampled_np = sampled.cpu().numpy().astype(np.int32)
            logprobs_np = logprobs.cpu().numpy()
            for i in range(bsz):
                ec = _is_element_consistent(sampled_np[i], element_mask_np[i])
                _add_candidate(i, sampled_np[i], float(logprobs_np[i]), f"sample_t{temp}", ec)

    # Also do some constrained sampling at lower temps for focused search
    constrained_temps = [0.5, 0.8, 1.0]
    constrained_per_temp = max(1, num_samples // (4 * len(constrained_temps)))
    for temp in constrained_temps:
        for _ in range(constrained_per_temp):
            _, sampled, logprobs = sample_decode_constrained(
                model, x, max_traj_len, stop_id, element_mask_dev, temperature=temp
            )
            sampled_np = sampled.cpu().numpy().astype(np.int32)
            logprobs_np = logprobs.cpu().numpy()
            for i in range(bsz):
                _add_candidate(i, sampled_np[i], float(logprobs_np[i]), f"constrained_t{temp}", True)

    # Sort: element-consistent first (by logprob), then others (by logprob)
    # NOTE: sort by logprob, NOT reward, because reward uses ground truth
    for i in range(bsz):
        consistent = [c for c in grouped[i] if c["element_consistent"]]
        inconsistent = [c for c in grouped[i] if not c["element_consistent"]]
        consistent.sort(key=lambda c: -c["logprob"])
        inconsistent.sort(key=lambda c: -c["logprob"])
        grouped[i] = consistent + inconsistent

    return greedy_pred, grouped


@torch.no_grad()
def greedy_decode_unconstrained(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    bsz = x.shape[0]
    n_prec = model.n_precursors

    selected = torch.zeros((bsz, n_prec), dtype=torch.float32, device=device)
    stopped = torch.zeros(bsz, dtype=torch.bool, device=device)
    actions = torch.full((bsz, max_traj_len), stop_id, dtype=torch.long, device=device)

    for t in range(max_traj_len):
        step_ids = torch.full((bsz,), t, dtype=torch.long, device=device)
        logits = model.forward_state(x, selected, step_ids)
        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        logits = logits.masked_fill(invalid, -1e9)
        act = torch.argmax(logits, dim=1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        add_mask = (act != stop_id)
        if add_mask.any():
            next_selected = selected.clone()
            rows = torch.nonzero(add_mask, as_tuple=False).squeeze(1)
            cols = act[rows]
            next_selected[rows, cols] = 1.0
            selected = next_selected
        stopped = stopped | (act == stop_id)

    return actions, selected


@torch.no_grad()
def sample_decode_unconstrained(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        logits = logits.masked_fill(invalid, -1e9)
        logp = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logp).clamp_min(1e-10)

        act = torch.multinomial(probs, num_samples=1).squeeze(1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        chosen_logp = logp.gather(1, act.unsqueeze(1)).squeeze(1)
        chosen_logp = torch.where(stopped, torch.zeros_like(chosen_logp), chosen_logp)
        logprob_sum = logprob_sum + chosen_logp

        add_mask = (act != stop_id)
        if add_mask.any():
            next_selected = selected.clone()
            rows = torch.nonzero(add_mask, as_tuple=False).squeeze(1)
            cols = act[rows]
            next_selected[rows, cols] = 1.0
            selected = next_selected
        stopped = stopped | (act == stop_id)

    return actions, selected, logprob_sum


def main() -> None:
    parser = argparse.ArgumentParser(description="Element-constrained decoding for Stage2 GFlowNet")
    parser.add_argument("--model_path", type=str,
                        default="/Users/wyc/SynPred/runs/stage2/gflownet_joint_rerank_hybrid_gold_only_v1/best_model.pt")
    parser.add_argument("--input_dir", type=str,
                        default="/Users/wyc/SynPred/data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only")
    parser.add_argument("--output_dir", type=str,
                        default="/Users/wyc/SynPred/runs/stage2/gflownet_element_constrained_v1")
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--sample_temperatures", type=str, default="0.8,1.0,1.2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--exact_bonus", type=float, default=4.0)
    parser.add_argument("--length_penalty", type=float, default=0.05)
    parser.add_argument("--save_topn", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"[Info] Loading model from {args.model_path}")
    model, ckpt = load_model(Path(args.model_path), device)
    precursor_names: List[str] = ckpt["precursor_names"]
    action_vocab: List[str] = ckpt["action_vocab"]
    n_precursors = ckpt["n_precursors"]
    max_traj_len = ckpt["max_traj_len"]
    stop_id = action_vocab.index(STOP_TOKEN) if STOP_TOKEN in action_vocab else n_precursors

    # Build precursor metal index
    print("[Info] Building precursor metal index...")
    precursor_metals = build_precursor_metal_index(precursor_names)

    # Load data
    sample_temperatures = [float(t) for t in args.sample_temperatures.split(",") if t.strip()]

    results = {}
    for split in ["train", "val", "test"]:
        print(f"\n[Info] Processing {split} split...")
        npz_path = input_dir / f"{split}.npz"
        meta_path = input_dir / f"{split}_meta.csv"

        data = np.load(npz_path)
        x = data["x"]
        y_multi_hot = data["y_multi_hot"]
        meta_df = pd.read_csv(meta_path)

        formulas = meta_df["formula"].tolist()

        # Build element mask for entire split
        element_mask = build_element_mask_for_batch(
            formulas, precursor_metals, n_precursors, stop_id
        )

        # Stats on element mask
        n_masked = element_mask[:, :n_precursors].sum(dim=1).float()
        n_allowed = n_precursors - n_masked
        print(f"  Element mask: avg {n_allowed.mean():.0f} allowed / {n_precursors} precursors per sample")

        # Run constrained greedy decode
        dataset = Stage2Dataset(x, y_multi_hot)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        all_greedy_pred = []
        all_y_true = []
        all_grouped: List[List[Dict[str, Any]]] = []

        batch_start = 0
        for x_batch, y_batch in loader:
            bsz = x_batch.shape[0]
            batch_mask = element_mask[batch_start:batch_start + bsz]

            greedy_pred, grouped = collect_candidates_hybrid(
                model=model,
                x=x_batch,
                y_true=y_batch,
                element_mask=batch_mask,
                max_traj_len=max_traj_len,
                stop_id=stop_id,
                num_samples=args.num_samples,
                sample_temperatures=sample_temperatures,
                exact_bonus=args.exact_bonus,
                length_penalty=args.length_penalty,
                device=device,
            )
            all_greedy_pred.append(greedy_pred)
            all_y_true.append(y_batch.numpy().astype(np.int32))
            all_grouped.extend(grouped)
            batch_start += bsz

        y_true = np.vstack(all_y_true)
        y_pred_greedy = np.vstack(all_greedy_pred)

        # Greedy metrics
        greedy_metrics = evaluate_from_binary(y_true, y_pred_greedy)
        print(f"  Greedy: exact_match={greedy_metrics['subset_accuracy']:.3f}, "
              f"micro_f1={greedy_metrics['micro_f1']:.3f}, "
              f"samples_f1={greedy_metrics['samples_f1']:.3f}")

        # Oracle and hit@k
        n_samples = len(all_grouped)
        oracle_hits = sum(1 for g in all_grouped if any(c["exact_match"] for c in g))
        oracle_rate = oracle_hits / n_samples

        # Oracle among element-consistent only
        oracle_ec_hits = sum(1 for g in all_grouped
                            if any(c["exact_match"] for c in g if c.get("element_consistent", False)))
        oracle_ec_rate = oracle_ec_hits / n_samples

        hit_at_k = {}
        hit_at_k_ec = {}
        for k in [1, 3, 5, 10, 20, 50]:
            hits = sum(1 for g in all_grouped if any(c["exact_match"] for c in g[:k]))
            hit_at_k[f"hit@{k}"] = hits / n_samples
            # Element-consistent hit@k (candidates are already sorted EC-first)
            hits_ec = sum(1 for g in all_grouped
                         if any(c["exact_match"] for c in g[:k] if c.get("element_consistent", False)))
            hit_at_k_ec[f"hit@{k}_ec"] = hits_ec / n_samples

        print(f"  Oracle (all): {oracle_rate:.3f}, Oracle (element-consistent): {oracle_ec_rate:.3f}")
        print(f"  Hit@1={hit_at_k['hit@1']:.3f}, Hit@5={hit_at_k['hit@5']:.3f}, Hit@10={hit_at_k['hit@10']:.3f}")

        # Mean candidates per sample
        mean_cands = np.mean([len(g) for g in all_grouped])
        mean_ec_cands = np.mean([sum(1 for c in g if c.get("element_consistent", False)) for g in all_grouped])
        print(f"  Mean candidates: {mean_cands:.1f} total, {mean_ec_cands:.1f} element-consistent")

        # Save predictions
        pred_labels = multihot_to_label_lists(y_pred_greedy, precursor_names)
        true_labels = multihot_to_label_lists(y_true, precursor_names)
        pred_df = meta_df.copy()
        pred_df["true_labels"] = [json.dumps(t, ensure_ascii=False) for t in true_labels]
        pred_df["pred_labels"] = [json.dumps(p, ensure_ascii=False) for p in pred_labels]
        pred_df["n_true_labels"] = y_true.sum(axis=1)
        pred_df["n_pred_labels"] = y_pred_greedy.sum(axis=1)
        pred_df.to_csv(output_dir / f"pred_{split}_greedy.csv", index=False)

        # Save candidates
        cand_rows = []
        for sample_idx, cands in enumerate(all_grouped):
            base = meta_df.iloc[sample_idx].to_dict()
            for rank, c in enumerate(cands[:args.save_topn], start=1):
                item = dict(base)
                item["rank"] = rank
                item["source"] = c["source"]
                item["logprob"] = c["logprob"]
                item["reward"] = c["reward"]
                item["exact_match"] = c["exact_match"]
                item["cand_len"] = c["cand_len"]
                item["pred_labels"] = json.dumps(
                    [precursor_names[j] for j in c["key"]], ensure_ascii=False
                )
                item["true_labels"] = json.dumps(true_labels[sample_idx], ensure_ascii=False)
                cand_rows.append(item)
        pd.DataFrame(cand_rows).to_csv(output_dir / f"{split}_candidates.csv", index=False)

        results[split] = {
            "greedy_metrics": greedy_metrics,
            "oracle_exact_match": oracle_rate,
            "oracle_element_consistent": oracle_ec_rate,
            "hit_at_k": hit_at_k,
            "hit_at_k_ec": hit_at_k_ec,
            "mean_candidates_per_sample": float(mean_cands),
            "mean_ec_candidates_per_sample": float(mean_ec_cands),
            "n_samples": n_samples,
        }

    # Save summary
    summary = {
        "config": vars(args),
        "model_path": str(args.model_path),
        "n_precursors": n_precursors,
        "max_traj_len": max_traj_len,
        "results": results,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("COMPARISON: Unconstrained vs Element-Constrained")
    print("=" * 60)
    if "test" in results:
        t = results["test"]
        print(f"  Test Exact Match (greedy): {t['greedy_metrics']['subset_accuracy']:.3f}")
        print(f"  Test Micro F1:             {t['greedy_metrics']['micro_f1']:.3f}")
        print(f"  Test Oracle:               {t['oracle_exact_match']:.3f}")
        print(f"  Test Hit@5:                {t['hit_at_k']['hit@5']:.3f}")
        print(f"  Mean candidates/sample:    {t['mean_candidates_per_sample']:.1f}")
    print(f"\n  Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
