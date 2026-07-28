#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.precursor.train_gflownet import GFlowNetPolicy, parse_hidden_dims  # noqa: E402


SetKey = Tuple[int, ...]


@torch.no_grad()
def beam_decode_batch(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    beam_width: int,
    max_traj_len: int,
    stop_id: int,
    force_non_empty: bool,
) -> tuple[List[List[SetKey]], List[List[float]]]:
    device = x.device
    batch_size = x.shape[0]
    n_precursors = model.n_precursors
    n_actions = model.n_actions
    selected = torch.zeros((batch_size, beam_width, n_precursors), device=device)
    stopped = torch.zeros((batch_size, beam_width), dtype=torch.bool, device=device)
    scores = torch.full((batch_size, beam_width), -torch.inf, device=device)
    scores[:, 0] = 0.0
    for step in range(max_traj_len):
        flat_selected = selected.reshape(batch_size * beam_width, n_precursors)
        step_ids = torch.full(
            (batch_size * beam_width,), step, dtype=torch.long, device=device
        )
        logits = model.forward_state(
            x[:, None, :].expand(-1, beam_width, -1).reshape(batch_size * beam_width, -1),
            flat_selected,
            step_ids,
        ).reshape(batch_size, beam_width, n_actions)
        invalid = torch.zeros_like(logits, dtype=torch.bool)
        invalid[:, :, :n_precursors] = selected > 0.5
        if force_non_empty and step == 0:
            invalid[:, :, stop_id] = True
        log_probs = torch.log_softmax(logits.masked_fill(invalid, -1e9), dim=-1)
        if stopped.any():
            log_probs = log_probs.masked_fill(stopped[:, :, None], -torch.inf)
            stop_values = log_probs[:, :, stop_id]
            log_probs[:, :, stop_id] = torch.where(
                stopped, torch.zeros_like(stop_values), stop_values
            )
        joint = scores[:, :, None] + log_probs
        top_scores, top_flat = torch.topk(
            joint.reshape(batch_size, beam_width * n_actions), k=beam_width, dim=1
        )
        parent = torch.div(top_flat, n_actions, rounding_mode="floor")
        action = top_flat.remainder(n_actions)
        selected = torch.gather(
            selected, 1, parent[:, :, None].expand(-1, -1, n_precursors)
        )
        stopped = torch.gather(stopped, 1, parent)
        active = (~stopped) & (action != stop_id)
        if active.any():
            batch_indices, beam_indices = torch.nonzero(active, as_tuple=True)
            selected[batch_indices, beam_indices, action[batch_indices, beam_indices]] = 1.0
        stopped = stopped | (action == stop_id)
        scores = top_scores
    all_candidates: List[List[SetKey]] = []
    all_scores: List[List[float]] = []
    selected_cpu = selected.cpu().numpy()
    scores_cpu = scores.cpu().numpy()
    for batch_index in range(batch_size):
        path_scores: Dict[SetKey, List[float]] = {}
        for beam_index in range(beam_width):
            key = tuple(np.flatnonzero(selected_cpu[batch_index, beam_index] > 0.5).tolist())
            if not key:
                continue
            path_scores.setdefault(key, []).append(float(scores_cpu[batch_index, beam_index]))
        aggregated = [
            (float(np.logaddexp.reduce(values)), key)
            for key, values in path_scores.items()
        ]
        aggregated.sort(key=lambda item: (-item[0], item[1]))
        candidate_scores = [item[0] for item in aggregated]
        candidates = [item[1] for item in aggregated]
        all_candidates.append(candidates)
        all_scores.append(candidate_scores)
    return all_candidates, all_scores


def metrics(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k in (1, 3, 5, 10, 20, 50, 100):
        result[f"exact_hit@{k}"] = float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)])
        )
    result["mean_unique_candidates"] = float(np.mean([len(row) for row in candidates]))
    result["min_unique_candidates"] = int(min(len(row) for row in candidates))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic beam evaluation for a Stage2 GFlowNet policy.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--beam_width", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", default="")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    x = np.asarray(pack["x"], dtype=np.float32)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    n_precursors = int(checkpoint.get("n_precursors", y.shape[1]))
    max_traj_len = int(checkpoint.get("max_traj_len", pack["traj_actions"].shape[1]))
    stop_id = int(checkpoint.get("stop_id", n_precursors))
    model = GFlowNetPolicy(
        x_dim=x.shape[1],
        n_precursors=n_precursors,
        hidden_dim=int(config.get("hidden_dim", 256)),
        max_traj_len=max_traj_len,
        x_mlp_hidden_dims=parse_hidden_dims(str(config.get("x_mlp_hidden_dims", "512"))),
        dropout=float(config.get("dropout", 0.1)),
        family_feature_count=int(config.get("family_feature_count", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=int(args.batch_size), shuffle=False)
    all_candidates: List[List[SetKey]] = []
    all_scores: List[List[float]] = []
    for (batch_x,) in loader:
        candidates, scores = beam_decode_batch(
            model,
            batch_x.to(device),
            int(args.beam_width),
            max_traj_len,
            stop_id,
            bool(config.get("force_non_empty", True)),
        )
        all_candidates.extend(candidates)
        all_scores.extend(scores)
    report = {
        "protocol": f"{args.split}_formula_disjoint_exact_precursor_set",
        "beam_width": int(args.beam_width),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "metrics": metrics(targets, all_candidates),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_candidates_jsonl:
        candidate_output = Path(args.output_candidates_jsonl).resolve()
        with candidate_output.open("w", encoding="utf-8") as handle:
            for index, (row, scores) in enumerate(zip(all_candidates, all_scores)):
                handle.write(json.dumps({"row_index": index, "candidate_label_ids": [list(value) for value in row], "scores": scores}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
