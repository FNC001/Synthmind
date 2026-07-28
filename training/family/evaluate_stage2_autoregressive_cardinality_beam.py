#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from training.family.train_stage2_autoregressive_set import AutoregressiveSetGenerator
from training.family.train_stage2_listwise_ranker import (
    append_query_formula_features,
    precursor_formula_features,
)


SetKey = Tuple[int, ...]


@torch.no_grad()
def fixed_cardinality_paths_batch(
    model: AutoregressiveSetGenerator,
    x: torch.Tensor,
    beam_per_length: int,
    branch_factor: int,
) -> tuple[List[Dict[SetKey, float]], np.ndarray]:
    """Decode each cardinality separately, aggregating equivalent permutations."""
    device = x.device
    batch_size = len(x)
    width = int(beam_per_length)
    target = model.encode_target(x)
    label_representations = model.label_representations()
    length_log_probs = F.log_softmax(model.length_head(target), dim=-1).float().cpu().numpy()
    all_paths: List[Dict[SetKey, float]] = [dict() for _ in range(batch_size)]
    for desired_length in range(1, model.max_set_len + 1):
        selected = torch.zeros((batch_size, width, model.n_labels), device=device)
        scores = torch.full((batch_size, width), -torch.inf, device=device)
        scores[:, 0] = 0.0
        for step in range(desired_length):
            flat_selected = selected.reshape(batch_size * width, model.n_labels)
            flat_target = target[:, None, :].expand(-1, width, -1).reshape(batch_size * width, -1)
            step_ids = torch.full(
                (batch_size * width,), min(step, model.max_set_len), dtype=torch.long, device=device
            )
            logits = model.forward_state(flat_target, flat_selected, step_ids, label_representations)
            label_logits = logits[:, : model.n_labels].masked_fill(flat_selected > 0.5, -1e4)
            log_probs = F.log_softmax(label_logits, dim=-1).reshape(batch_size, width, -1)
            branches = min(int(branch_factor), model.n_labels)
            branch_scores, branch_actions = torch.topk(log_probs, k=branches, dim=-1)
            joint = scores[:, :, None] + branch_scores
            top_scores, top_flat = torch.topk(joint.reshape(batch_size, -1), k=width, dim=-1)
            parent = torch.div(top_flat, branches, rounding_mode="floor")
            actions = torch.gather(branch_actions.reshape(batch_size, -1), 1, top_flat)
            selected = torch.gather(
                selected, 1, parent[:, :, None].expand(-1, -1, model.n_labels)
            )
            rows = torch.arange(batch_size, device=device)[:, None].expand(-1, width)
            beams = torch.arange(width, device=device)[None, :].expand(batch_size, -1)
            selected[rows, beams, actions] = 1.0
            scores = top_scores
        selected_np = selected.cpu().numpy()
        scores_np = scores.float().cpu().numpy()
        for row_index in range(batch_size):
            permutation_scores: Dict[SetKey, List[float]] = {}
            for beam_index in range(width):
                if not np.isfinite(scores_np[row_index, beam_index]):
                    continue
                key = tuple(np.flatnonzero(selected_np[row_index, beam_index] > 0.5).tolist())
                if len(key) == desired_length:
                    permutation_scores.setdefault(key, []).append(float(scores_np[row_index, beam_index]))
            for key, values in permutation_scores.items():
                all_paths[row_index][key] = float(np.logaddexp.reduce(values))
    return all_paths, length_log_probs


def rank_paths(
    paths: Dict[SetKey, float],
    length_log_probs: np.ndarray,
    length_normalization: float,
    length_prior_weight: float,
) -> tuple[List[SetKey], List[float]]:
    values = []
    for key, raw_score in paths.items():
        length = len(key)
        score = raw_score / (float(length) ** float(length_normalization))
        score += float(length_prior_weight) * float(length_log_probs[length - 1])
        values.append((score, key))
    values.sort(key=lambda item: (-item[0], item[1]))
    return [key for _, key in values], [float(score) for score, _ in values]


def exact_metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cardinality-conditioned beam search for precursor sets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--beam_per_length", type=int, default=100)
    parser.add_argument("--branch_factor", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--length_normalizations", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5])
    parser.add_argument("--length_prior_weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--fixed_length_normalization",
        type=float,
        default=None,
        help="Use one validation-selected normalization without tuning on the requested split.",
    )
    parser.add_argument(
        "--fixed_length_prior_weight",
        type=float,
        default=None,
        help="Use one validation-selected cardinality-prior weight without tuning on the requested split.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    x = append_query_formula_features(
        np.asarray(pack["x"], dtype=np.float32), input_dir / f"{args.split}_meta.csv"
    )
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    label_chemistry = torch.from_numpy(precursor_formula_features([str(value) for value in names]))
    model = AutoregressiveSetGenerator(
        int(checkpoint["x_dim"]), int(checkpoint["n_labels"]), int(checkpoint["max_set_len"]),
        int(config["hidden"]), int(config["blocks"]), float(config["dropout"]), label_chemistry,
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    all_paths: List[Dict[SetKey, float]] = []
    all_length_log_probs: List[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            paths, length_log_probs = fixed_cardinality_paths_batch(
                model, batch_x.to(device), args.beam_per_length, args.branch_factor
            )
            all_paths.extend(paths)
            all_length_log_probs.extend(length_log_probs)
    fixed_values = (args.fixed_length_normalization, args.fixed_length_prior_weight)
    if (fixed_values[0] is None) != (fixed_values[1] is None):
        parser.error(
            "--fixed_length_normalization and --fixed_length_prior_weight must be supplied together"
        )
    if args.split == "test" and fixed_values[0] is None:
        parser.error(
            "test evaluation requires validation-selected --fixed_length_normalization and "
            "--fixed_length_prior_weight"
        )
    search_grid = (
        [(float(fixed_values[0]), float(fixed_values[1]))]
        if fixed_values[0] is not None
        else list(itertools.product(args.length_normalizations, args.length_prior_weights))
    )
    trials = []
    best = None
    best_rows: List[List[SetKey]] = []
    best_scores: List[List[float]] = []
    for normalization, prior_weight in search_grid:
        ranked = [
            rank_paths(paths, length_probs, normalization, prior_weight)
            for paths, length_probs in zip(all_paths, all_length_log_probs)
        ]
        rows = [value[0] for value in ranked]
        metrics = exact_metrics(targets, rows)
        trial = {
            "length_normalization": normalization,
            "length_prior_weight": prior_weight,
            **metrics,
        }
        trials.append(trial)
        if best is None or (trial["exact_hit@10"], trial["exact_hit@50"]) > (
            best["exact_hit@10"], best["exact_hit@50"]
        ):
            best = trial
            best_rows = rows
            best_scores = [value[1] for value in ranked]
    assert best is not None
    report = {
        "protocol": f"stage2_{args.split}_formula_disjoint_exact_precursor_set_cardinality_beam",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "beam_per_length": args.beam_per_length,
        "branch_factor": args.branch_factor,
        "selection_mode": "validation_frozen" if fixed_values[0] is not None else "validation_grid_search",
        "best": best,
        "oracle_candidate_recall": float(
            np.mean([target in paths for target, paths in zip(targets, all_paths)])
        ),
        "mean_unique_candidates": float(np.mean([len(paths) for paths in all_paths])),
        "top_trials": sorted(trials, key=lambda row: (-row["exact_hit@10"], -row["exact_hit@50"]))[:20],
    }
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, (row, scores) in enumerate(zip(best_rows, best_scores)):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(key) for key in row],
                "scores": scores,
            }) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
