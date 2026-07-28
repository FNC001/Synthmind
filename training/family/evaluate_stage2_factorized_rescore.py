#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage2_factorized_generator import FactorizedSetGenerator, predict  # noqa: E402


SetKey = Tuple[int, ...]


def load_candidates(path: Path, n_rows: int, limit: int) -> tuple[List[List[SetKey]], List[np.ndarray]]:
    candidates: List[List[SetKey]] = [[] for _ in range(n_rows)]
    scores: List[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in range(n_rows)]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["row_index"])
            values = row["candidate_label_ids"][:limit]
            candidates[index] = [tuple(sorted({int(value) for value in item})) for item in values]
            raw_scores = row.get("scores")
            if raw_scores is None:
                raw_scores = [-float(np.log1p(rank)) for rank in range(len(values))]
            scores[index] = np.asarray(raw_scores[:limit], dtype=np.float32)
    return candidates, scores


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    return (values - values.mean()) / max(float(values.std()), 1e-6)


def metrics(targets: Sequence[SetKey], rows: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500, 1000)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescore chemistry-generated precursor sets with a factorized neural model.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--candidate_source", required=True)
    parser.add_argument("--candidate_limit", type=int, default=1000)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / "val.npz", allow_pickle=True)
    x = np.asarray(pack["x"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in pack["y_multi_hot"]]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_label_logits, all_length_logits = [], []
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = FactorizedSetGenerator(
            int(checkpoint["x_dim"]), int(checkpoint["n_labels"]), int(checkpoint["max_set_len"]),
            int(config["hidden"]), int(config["blocks"]), float(config["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)
        current_labels, current_lengths = predict(model, x, device, int(config.get("batch_size", 256)))
        all_label_logits.append(current_labels)
        all_length_logits.append(current_lengths)
        del model
    label_logits = np.mean(all_label_logits, axis=0)
    length_logits = np.mean(all_length_logits, axis=0)
    length_log_probs = length_logits - np.logaddexp.reduce(length_logits, axis=1, keepdims=True)
    candidates, prior_scores = load_candidates(Path(args.candidate_source).resolve(), len(x), args.candidate_limit)
    neural_label_scores: List[np.ndarray] = []
    neural_length_scores: List[np.ndarray] = []
    for row_index, row in enumerate(candidates):
        label_values = np.asarray([
            float(np.mean(label_logits[row_index, list(candidate)])) if candidate else -100.0
            for candidate in row
        ], dtype=np.float32)
        length_values = np.asarray([
            float(length_log_probs[row_index, min(len(candidate), length_log_probs.shape[1]) - 1])
            for candidate in row
        ], dtype=np.float32)
        neural_label_scores.append(normalize(label_values))
        neural_length_scores.append(normalize(length_values))
    trials: List[Dict[str, Any]] = []
    best_trial = None
    best_rows: List[List[SetKey]] = []
    best_scores: List[List[float]] = []
    for label_weight in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        for length_weight in (0.0, 0.1, 0.25, 0.5, 1.0):
            ranked_rows: List[List[SetKey]] = []
            ranked_scores: List[List[float]] = []
            for row, prior, label_score, length_score in zip(candidates, prior_scores, neural_label_scores, neural_length_scores):
                combined = normalize(prior) + label_weight * label_score + length_weight * length_score
                order = np.argsort(-combined, kind="stable")
                ranked_rows.append([row[int(index)] for index in order])
                ranked_scores.append([float(combined[int(index)]) for index in order])
            trial = {"label_weight": label_weight, "length_weight": length_weight, **metrics(targets, ranked_rows)}
            trials.append(trial)
            if best_trial is None or (trial["exact_hit@10"], trial["exact_hit@50"]) > (best_trial["exact_hit@10"], best_trial["exact_hit@50"]):
                best_trial, best_rows, best_scores = trial, ranked_rows, ranked_scores
    report = {"protocol": "validation_formula_disjoint_factorized_rescore", "config": vars(args), "best": best_trial, "top_trials": sorted(trials, key=lambda row: (-row["exact_hit@10"], -row["exact_hit@50"]))[:10]}
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for index, (row, scores) in enumerate(zip(best_rows, best_scores)):
            handle.write(json.dumps({"row_index": index, "candidate_label_ids": [list(value) for value in row], "scores": scores}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
