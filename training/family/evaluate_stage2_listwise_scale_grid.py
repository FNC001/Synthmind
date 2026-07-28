#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.train_stage2_listwise_ranker import (  # noqa: E402
    CandidateDataset,
    ListwiseSetRanker,
    append_query_formula_features,
    append_source_features,
    load_candidates,
    precursor_formula_features,
)


SetKey = Tuple[int, ...]


def exact_metrics(targets: List[SetKey], rows: List[List[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def masked_exact_metrics(
    targets: List[SetKey], rows: List[List[SetKey]], mask: np.ndarray, prefix: str
) -> Dict[str, float]:
    indices = np.flatnonzero(mask)
    return {
        f"{prefix}_exact_hit@{k}": float(
            np.mean([targets[index] in set(rows[index][:k]) for index in indices])
        ) if len(indices) else 0.0
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation calibration of rank-prior versus neural residual scale.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate_sources", nargs="+", required=True)
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--residual_scales", nargs="+", type=float,
        default=[0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument(
        "--selection_slice", choices=("all", "unseen"), default="all",
        help="Choose residual scale by all validation rows or the train-unseen-label validation slice.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / "val.npz", allow_pickle=True)
    x = np.asarray(pack["x"], dtype=np.float32)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    train_y = np.asarray(
        np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    unseen_mask = np.asarray(
        [any(not bool(train_seen[label]) for label in target) for target in targets], dtype=bool
    )
    candidates, candidate_scores = load_candidates(
        args.candidate_sources, len(x), per_source_limit=int(args.candidate_limit)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if bool(config.get("append_source_features", False)):
        x = append_source_features(x, input_dir / "val_meta.csv", checkpoint.get("source_vocab", []))
    if bool(config.get("append_query_formula_features", False)):
        x = append_query_formula_features(x, input_dir / "val_meta.csv")
    dataset = CandidateDataset(
        x, y, candidates, candidate_scores, args.candidate_limit, args.candidate_limit,
        int(checkpoint["max_set_len"]), int(checkpoint["n_labels"]), False, int(config.get("seed", 0)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    use_relative = bool(config.get("use_relative_chemistry_features", False))
    model = ListwiseSetRanker(
        int(checkpoint["x_dim"]), int(checkpoint["n_labels"]), int(config["hidden"]),
        int(config["blocks"]), float(config["dropout"]), int(checkpoint["max_set_len"]),
        use_membership_energy=int(config.get("membership_pretrain_epochs", 0)) > 0,
        membership_score_scale=float(config.get("membership_score_scale", 1.0)),
        length_score_scale=float(config.get("length_score_scale", 0.0)),
        label_chemistry=(
            torch.from_numpy(precursor_formula_features([
                str(value) for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
            ])) if bool(config.get("use_formula_features", False)) else None
        ),
        query_chemistry_dim=(
            147 if (bool(config.get("aligned_query_formula_encoder", False)) or use_relative) else 0
        ),
        use_relative_chemistry_features=use_relative,
        candidate_transformer_layers=int(config.get("candidate_transformer_layers", 0)),
        candidate_transformer_heads=int(config.get("candidate_transformer_heads", 8)),
        joint_transformer_layers=int(config.get("joint_transformer_layers", 0)),
        joint_transformer_heads=int(config.get("joint_transformer_heads", 8)),
        chemistry_only_label_encoder=bool(config.get("chemistry_only_label_encoder", False)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    residual_rows: List[np.ndarray] = []
    prior_rows: List[np.ndarray] = []
    row_offset = 0
    with torch.no_grad():
        for batch_x, labels, numeric, mask, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                residual = model(batch_x, labels, numeric)
            residual_np = residual.float().cpu().numpy()
            prior_np = numeric[..., 1].float().cpu().numpy()
            mask_np = mask.cpu().numpy()
            for local_index in range(len(batch_x)):
                valid = int(mask_np[local_index].sum())
                residual_rows.append(residual_np[local_index, :valid])
                prior_rows.append(prior_np[local_index, :valid])
            row_offset += len(batch_x)
    trials = []
    best = None
    best_rows: List[List[SetKey]] = []
    for scale in args.residual_scales:
        ranked_rows = []
        for row_index, (residual, prior) in enumerate(zip(residual_rows, prior_rows)):
            score = prior + float(scale) * residual
            order = np.argsort(-score, kind="stable")
            source = candidates[row_index][: args.candidate_limit]
            ranked_rows.append([source[int(index)] for index in order])
        trial = {
            "rank_bias_weight": 1.0,
            "residual_scale": scale,
            **exact_metrics(targets, ranked_rows),
            **masked_exact_metrics(targets, ranked_rows, ~unseen_mask, "seen"),
            **masked_exact_metrics(targets, ranked_rows, unseen_mask, "unseen"),
        }
        trials.append(trial)
        selection_prefix = "" if args.selection_slice == "all" else "unseen_"
        if best is None or (
            trial[f"{selection_prefix}exact_hit@10"],
            trial[f"{selection_prefix}exact_hit@50"],
        ) > (
            best[f"{selection_prefix}exact_hit@10"],
            best[f"{selection_prefix}exact_hit@50"],
        ):
            best = trial
            best_rows = ranked_rows
    assert best is not None
    report = {
        "protocol": "val_formula_disjoint_exact_precursor_set_residual_scale_calibration",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "candidate_sources": args.candidate_sources,
        "selection_slice": args.selection_slice,
        "unseen_rows": int(unseen_mask.sum()),
        "best": best,
        "training_scale": {
            "rank_bias_weight": float(config.get("rank_bias_weight", 1.0)),
            "residual_scale": float(config.get("residual_scale", 0.1)),
        },
        "trials": trials,
    }
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(best_rows):
            handle.write(json.dumps({"row_index": row_index, "candidate_label_ids": [list(key) for key in row]}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
