#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.precursor.train_gflownet import GFlowNetPolicy, parse_hidden_dims  # noqa: E402


SetKey = Tuple[int, ...]


def load_candidates(path: Path, n_rows: int, limit: int) -> List[List[SetKey]]:
    rows: List[List[SetKey]] = [[] for _ in range(n_rows)]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            rows[int(item["row_index"])] = [
                tuple(sorted({int(value) for value in candidate}))
                for candidate in item["candidate_label_ids"][:limit]
            ]
    return rows


@torch.no_grad()
def score_order_batch(
    model: GFlowNetPolicy,
    x: torch.Tensor,
    orders: torch.Tensor,
    lengths: torch.Tensor,
    stop_id: int,
) -> torch.Tensor:
    selected = torch.zeros((len(x), model.n_precursors), device=x.device)
    scores = torch.zeros(len(x), device=x.device)
    max_length = int(lengths.max().item())
    for step in range(max_length + 1):
        step_ids = torch.full((len(x),), step, dtype=torch.long, device=x.device)
        logits = model.forward_state(x, selected, step_ids)
        invalid = torch.zeros_like(logits, dtype=torch.bool)
        invalid[:, : model.n_precursors] = selected > 0.5
        log_probs = torch.log_softmax(logits.masked_fill(invalid, -1e9), dim=1)
        active = step <= lengths
        actions = torch.where(active & (step < lengths), orders[:, step], torch.full_like(lengths, stop_id))
        scores += torch.where(active, log_probs.gather(1, actions[:, None]).squeeze(1), 0.0)
        add = active & (step < lengths)
        if add.any():
            row_ids = torch.nonzero(add, as_tuple=False).squeeze(1)
            selected[row_ids, actions[row_ids]] = 1.0
    return scores


def exact_metrics(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore chemistry candidates with GFlowNet teacher-forced set likelihood.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate_source", required=True)
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--reverse_weight", type=float, default=1.0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / "val.npz", allow_pickle=True)
    x_values = np.asarray(pack["x"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    n_precursors = int(checkpoint.get("n_precursors", pack["y_multi_hot"].shape[1]))
    stop_id = int(checkpoint.get("stop_id", n_precursors))
    model = GFlowNetPolicy(
        x_dim=x_values.shape[1], n_precursors=n_precursors,
        hidden_dim=int(config.get("hidden_dim", 256)),
        max_traj_len=int(checkpoint.get("max_traj_len", pack["traj_actions"].shape[1])),
        x_mlp_hidden_dims=parse_hidden_dims(str(config.get("x_mlp_hidden_dims", "512"))),
        dropout=float(config.get("dropout", 0.1)),
        family_feature_count=int(config.get("family_feature_count", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    rows = load_candidates(Path(args.candidate_source).resolve(), len(x_values), args.candidate_limit)
    flat_queries: List[int] = []
    flat_candidates: List[SetKey] = []
    for query_index, candidates in enumerate(rows):
        flat_queries.extend([query_index] * len(candidates))
        flat_candidates.extend(candidates)
    scores = np.empty(len(flat_candidates), dtype=np.float32)
    max_length = max(len(value) for value in flat_candidates)
    for start in range(0, len(flat_candidates), args.batch_size):
        end = min(len(flat_candidates), start + args.batch_size)
        candidates = flat_candidates[start:end]
        lengths = torch.tensor([len(value) for value in candidates], dtype=torch.long, device=device)
        orders = torch.full((len(candidates), max_length + 1), stop_id, dtype=torch.long, device=device)
        reverse_orders = orders.clone()
        for index, candidate in enumerate(candidates):
            orders[index, : len(candidate)] = torch.tensor(candidate, device=device)
            reverse_orders[index, : len(candidate)] = torch.tensor(candidate[::-1], device=device)
        batch_x = torch.from_numpy(x_values[np.asarray(flat_queries[start:end])]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            forward = score_order_batch(model, batch_x, orders, lengths, stop_id)
            reverse = score_order_batch(model, batch_x, reverse_orders, lengths, stop_id)
            same = lengths <= 1
            aggregated = torch.where(
                same,
                forward,
                torch.logaddexp(forward, reverse + math.log(max(args.reverse_weight, 1e-8))),
            )
        scores[start:end] = aggregated.float().cpu().numpy()
        if start % (args.batch_size * 50) == 0:
            print(json.dumps({"scored": end, "total": len(flat_candidates)}), flush=True)
    ranked_rows: List[List[SetKey]] = []
    ranked_scores: List[List[float]] = []
    offset = 0
    for row in rows:
        row_scores = scores[offset : offset + len(row)]
        order = np.argsort(-row_scores, kind="stable")
        ranked_rows.append([row[int(index)] for index in order])
        ranked_scores.append([float(row_scores[int(index)]) for index in order])
        offset += len(row)
    report = {"protocol": "validation_formula_disjoint_policy_set_rescore", "config": vars(args), "validation": exact_metrics(targets, ranked_rows), "mean_candidates": float(np.mean([len(row) for row in rows]))}
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for index, (row, row_scores) in enumerate(zip(ranked_rows, ranked_scores)):
            handle.write(json.dumps({"row_index": index, "candidate_label_ids": [list(value) for value in row], "scores": row_scores}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
