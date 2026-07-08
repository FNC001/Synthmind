#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


STOP_TOKEN = "<stop>"


_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
DEFAULT_IGNORE_ELEMENTS = {"H", "O"}


def parse_ignore_elements(s: str) -> Set[str]:
    if s is None:
        return set(DEFAULT_IGNORE_ELEMENTS)
    s = str(s).strip()
    if not s:
        return set(DEFAULT_IGNORE_ELEMENTS)
    return {x.strip() for x in s.split(",") if x.strip()}


def parse_elements_from_formula_like(text: Any, ignore_elements: Optional[Set[str]] = None) -> Set[str]:
    if ignore_elements is None:
        ignore_elements = set(DEFAULT_IGNORE_ELEMENTS)
    if text is None:
        return set()
    els = set(_ELEMENT_RE.findall(str(text)))
    return {e for e in els if e not in ignore_elements}


def infer_formula_from_meta(meta_df: pd.DataFrame, row_indices: List[int]) -> List[str]:
    formula_cols = ["formula", "formula_x", "formula_y", "material_formula", "composition"]
    out = []
    for i in row_indices:
        row = meta_df.iloc[int(i)]
        val = ""
        for c in formula_cols:
            if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
                val = str(row[c])
                break
        out.append(val)
    return out


def build_composition_bias_matrix(
    precursor_names: List[str],
    target_formulas: List[str],
    stop_id: int,
    target_hit_bonus: float = 4.0,
    extra_element_penalty: float = 1.0,
    no_overlap_penalty: float = 3.0,
    stop_bias: float = 0.0,
    ignore_elements: Optional[Set[str]] = None,
) -> torch.Tensor:
    if ignore_elements is None:
        ignore_elements = set(DEFAULT_IGNORE_ELEMENTS)

    precursor_elements = [
        parse_elements_from_formula_like(p, ignore_elements=ignore_elements)
        for p in precursor_names
    ]

    rows = []
    for formula in target_formulas:
        target_elements = parse_elements_from_formula_like(formula, ignore_elements=ignore_elements)
        bias = []

        for els in precursor_elements:
            hit = els & target_elements
            extra = els - target_elements

            if len(hit) == 0:
                score = -float(no_overlap_penalty)
            else:
                score = float(target_hit_bonus) * float(len(hit))

            if len(extra) > 0:
                score -= float(extra_element_penalty) * float(len(extra))

            bias.append(score)

        full = [0.0] * (len(precursor_names) + 1)
        full[:len(precursor_names)] = bias
        full[stop_id] = float(stop_bias)
        rows.append(full)

    return torch.tensor(rows, dtype=torch.float32)


def summarize_bias_for_debug(
    precursor_names: List[str],
    target_formula: str,
    stop_id: int,
    target_hit_bonus: float,
    extra_element_penalty: float,
    no_overlap_penalty: float,
    stop_bias: float,
    ignore_elements: Set[str],
    top_n: int = 20,
) -> Dict[str, Any]:
    bias = build_composition_bias_matrix(
        precursor_names=precursor_names,
        target_formulas=[target_formula],
        stop_id=stop_id,
        target_hit_bonus=target_hit_bonus,
        extra_element_penalty=extra_element_penalty,
        no_overlap_penalty=no_overlap_penalty,
        stop_bias=stop_bias,
        ignore_elements=ignore_elements,
    )[0].numpy()

    items = []
    for i, p in enumerate(precursor_names):
        items.append({
            "action_id": int(i),
            "precursor": p,
            "bias": float(bias[i]),
            "elements": sorted(parse_elements_from_formula_like(p, ignore_elements=ignore_elements)),
        })

    items = sorted(items, key=lambda x: x["bias"], reverse=True)[:int(top_n)]

    return {
        "target_formula": str(target_formula),
        "target_elements": sorted(parse_elements_from_formula_like(target_formula, ignore_elements=ignore_elements)),
        "ignore_elements": sorted(ignore_elements),
        "top_bias_precursors": items,
    }



def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    arr = np.load(path)
    return {k: arr[k] for k in arr.files}


def parse_hidden_dims(s: str) -> List[int]:
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class Stage2GFlowNetDataset(Dataset):
    def __init__(self, x: np.ndarray, y_multi_hot: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_multi_hot = torch.tensor(y_multi_hot, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_multi_hot[idx], idx


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
        return self.policy_head(state)


def build_invalid_mask(
    selected_mask: torch.Tensor,
    stop_id: int,
    stopped: torch.Tensor | None = None,
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
    next_selected[rows, cols] = 1.0
    return next_selected


@torch.no_grad()
def greedy_decode(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    composition_bias: Optional[torch.Tensor] = None,
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
        if composition_bias is not None:
            logits = logits + composition_bias.to(device=logits.device, dtype=logits.dtype)
        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        logits = logits.masked_fill(invalid, -1e9)
        act = torch.argmax(logits, dim=1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act
        selected = update_selected_non_inplace(selected, act, stop_id)
        stopped = stopped | (act == stop_id)

    return actions, selected


@torch.no_grad()
def sample_decode(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    max_traj_len: int,
    stop_id: int,
    temperature: float = 1.0,
    top_k: int = 0,
    composition_bias: Optional[torch.Tensor] = None,
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
        logits = model.forward_state(x, selected, step_ids)
        if composition_bias is not None:
            logits = logits + composition_bias.to(device=logits.device, dtype=logits.dtype)
        logits = logits / temperature
        invalid = build_invalid_mask(selected, stop_id=stop_id, stopped=stopped)
        logits = logits.masked_fill(invalid, -1e9)

        if top_k > 0 and top_k < logits.shape[1]:
            kth = torch.topk(logits, k=top_k, dim=-1).values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)

        logp = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(logp)

        act = torch.multinomial(probs, num_samples=1).squeeze(1)
        act = torch.where(stopped, torch.full_like(act, stop_id), act)
        actions[:, t] = act

        chosen_logp = logp.gather(1, act.unsqueeze(1)).squeeze(1)
        chosen_logp = torch.where(stopped, torch.zeros_like(chosen_logp), chosen_logp)
        logprob_sum = logprob_sum + chosen_logp

        selected = update_selected_non_inplace(selected, act, stop_id)
        stopped = stopped | (act == stop_id)

    return actions, selected, logprob_sum


def multihot_to_label_lists(y: np.ndarray, precursor_names: List[str]) -> List[List[str]]:
    out = []
    for i in range(y.shape[0]):
        idx = np.where(y[i] > 0)[0].tolist()
        out.append([precursor_names[j] for j in idx])
    return out


def actions_to_label_lists(
    actions: np.ndarray,
    precursor_names: List[str],
    stop_id: int,
) -> List[List[str]]:
    out: List[List[str]] = []
    for row in actions:
        cur = []
        seen = set()
        for a in row.tolist():
            if a == stop_id:
                break
            if 0 <= a < len(precursor_names):
                p = precursor_names[a]
                if p not in seen:
                    cur.append(p)
                    seen.add(p)
        out.append(cur)
    return out


def reconstruct_model_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> GFlowNetPolicy:
    cfg = ckpt["config"]
    model = GFlowNetPolicy(
        x_dim=int(ckpt["x_dim"]),
        n_precursors=int(ckpt["n_precursors"]),
        hidden_dim=int(cfg["hidden_dim"]),
        max_traj_len=int(ckpt["max_traj_len"]),
        x_mlp_hidden_dims=parse_hidden_dims(cfg["x_mlp_hidden_dims"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def sample_candidates(
    model: GFlowNetPolicy,
    loader: DataLoader,
    meta_df: pd.DataFrame,
    precursor_names: List[str],
    stop_id: int,
    max_traj_len: int,
    device: torch.device,
    n_samples: int,
    use_greedy_as_first: bool,
    temperature: float,
    top_k: int,
    composition_constrained: bool = False,
    target_hit_bonus: float = 4.0,
    extra_element_penalty: float = 1.0,
    no_overlap_penalty: float = 3.0,
    stop_bias: float = 0.0,
    ignore_elements: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    unique_counts = []
    pred_len_all = []

    for x, y_multi_hot, batch_idx in loader:
        x = x.to(device)
        batch_indices = batch_idx.numpy().tolist()
        target_formulas = infer_formula_from_meta(meta_df, batch_indices)
        composition_bias = None
        if composition_constrained:
            composition_bias = build_composition_bias_matrix(
                precursor_names=precursor_names,
                target_formulas=target_formulas,
                stop_id=stop_id,
                target_hit_bonus=target_hit_bonus,
                extra_element_penalty=extra_element_penalty,
                no_overlap_penalty=no_overlap_penalty,
                stop_bias=stop_bias,
                ignore_elements=ignore_elements,
            ).to(device)
        y_true_np = y_multi_hot.numpy().astype(int)
        true_lists = multihot_to_label_lists(y_true_np, precursor_names)

        all_action_samples: List[np.ndarray] = []

        if use_greedy_as_first:
            acts, _ = greedy_decode(
                model=model,
                x=x,
                max_traj_len=max_traj_len,
                stop_id=stop_id,
                composition_bias=composition_bias,
            )
            all_action_samples.append(acts.cpu().numpy())

        while len(all_action_samples) < n_samples:
            acts, _, _ = sample_decode(
                model=model,
                x=x,
                max_traj_len=max_traj_len,
                stop_id=stop_id,
                temperature=temperature,
                top_k=top_k,
                composition_bias=composition_bias,
            )
            all_action_samples.append(acts.cpu().numpy())

        stacked = np.stack(all_action_samples, axis=0)  # [S, B, T]

        for local_i, global_i in enumerate(batch_indices):
            meta = meta_df.iloc[global_i].to_dict()
            cand_keys = set()

            for s_idx in range(stacked.shape[0]):
                act_row = stacked[s_idx, local_i]
                pred_labels = actions_to_label_lists(
                    act_row[None, :],
                    precursor_names=precursor_names,
                    stop_id=stop_id,
                )[0]
                pred_len_all.append(len(pred_labels))
                cand_keys.add(tuple(pred_labels))

                row = {
                    **meta,
                    "sample_rank": int(s_idx),
                    "true_labels": json.dumps(true_lists[local_i], ensure_ascii=False),
                    "pred_labels": json.dumps(pred_labels, ensure_ascii=False),
                    "n_pred_labels": int(len(pred_labels)),
                    "decoded_actions": json.dumps(act_row.tolist(), ensure_ascii=False),
                    "decode_method": "greedy" if (use_greedy_as_first and s_idx == 0) else "sample",
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    "composition_constrained": bool(composition_constrained),
                    "target_formula_for_constraint": str(target_formulas[local_i]),
                }
                rows.append(row)

            unique_counts.append(len(cand_keys))

    out_df = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(out_df)),
        "n_base_structures": int(len(meta_df)),
        "n_samples_per_structure": int(n_samples),
        "mean_unique_set_count": float(np.mean(unique_counts)) if unique_counts else 0.0,
        "max_unique_set_count": int(max(unique_counts)) if unique_counts else 0,
        "mean_pred_labels": float(np.mean(pred_len_all)) if pred_len_all else 0.0,
    }
    return out_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample candidate precursor sets from trained GFlowNet-style model.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/MP_exp_doi/data/interim/generative/stage2_gflownet_dataset/hybrid",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/Users/wyc/MP_exp_doi/runs/generative/stage2/gflownet_hybrid_v1/best_model.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "gold_train_holdout"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wyc/MP_exp_doi/runs/generative/stage2/gflownet_hybrid_v1/samples_test",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--use_greedy_as_first", action="store_true")
    parser.add_argument("--composition_constrained", action="store_true")
    parser.add_argument("--target_hit_bonus", type=float, default=4.0)
    parser.add_argument("--extra_element_penalty", type=float, default=1.0)
    parser.add_argument("--no_overlap_penalty", type=float, default=3.0)
    parser.add_argument("--stop_bias", type=float, default=0.0)
    parser.add_argument("--ignore_elements", type=str, default="H,O")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ignore_elements = parse_ignore_elements(args.ignore_elements)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model = reconstruct_model_from_ckpt(ckpt, device)

    action_to_id = ckpt["action_to_id"]
    precursor_names = ckpt["precursor_names"]
    max_traj_len = int(ckpt["max_traj_len"])
    stop_id = int(action_to_id[STOP_TOKEN])

    pack = load_npz(str(Path(args.input_dir) / f"{args.split}.npz"))
    meta_df = pd.read_csv(Path(args.input_dir) / f"{args.split}_meta.csv")

    dataset = Stage2GFlowNetDataset(pack["x"], pack["y_multi_hot"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    out_df, sampling_summary = sample_candidates(
        model=model,
        loader=loader,
        meta_df=meta_df,
        precursor_names=precursor_names,
        stop_id=stop_id,
        max_traj_len=max_traj_len,
        device=device,
        n_samples=args.n_samples,
        use_greedy_as_first=bool(args.use_greedy_as_first),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        composition_constrained=bool(args.composition_constrained),
        target_hit_bonus=float(args.target_hit_bonus),
        extra_element_penalty=float(args.extra_element_penalty),
        no_overlap_penalty=float(args.no_overlap_penalty),
        stop_bias=float(args.stop_bias),
        ignore_elements=ignore_elements,
    )

    csv_path = output_dir / f"{args.split}_samples.csv"
    out_df.to_csv(csv_path, index=False)

    bias_debug = None
    if bool(args.composition_constrained) and len(meta_df) > 0:
        first_formula = infer_formula_from_meta(meta_df, [0])[0]
        bias_debug = summarize_bias_for_debug(
            precursor_names=precursor_names,
            target_formula=first_formula,
            stop_id=stop_id,
            target_hit_bonus=float(args.target_hit_bonus),
            extra_element_penalty=float(args.extra_element_penalty),
            no_overlap_penalty=float(args.no_overlap_penalty),
            stop_bias=float(args.stop_bias),
            ignore_elements=ignore_elements,
            top_n=20,
        )

    summary = {
        "config": {
            "input_dir": args.input_dir,
            "ckpt_path": args.ckpt_path,
            "split": args.split,
            "output_dir": str(output_dir),
            "batch_size": int(args.batch_size),
            "n_samples": int(args.n_samples),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "use_greedy_as_first": bool(args.use_greedy_as_first),
            "composition_constrained": bool(args.composition_constrained),
            "target_hit_bonus": float(args.target_hit_bonus),
            "extra_element_penalty": float(args.extra_element_penalty),
            "no_overlap_penalty": float(args.no_overlap_penalty),
            "stop_bias": float(args.stop_bias),
            "ignore_elements": sorted(ignore_elements),
            "device": str(device),
        },
        "sampling_summary": sampling_summary,
        "composition_bias_debug": bias_debug,
        "output_csv": str(csv_path),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
