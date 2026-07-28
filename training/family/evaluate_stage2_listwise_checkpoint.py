#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    append_source_features,
    append_query_formula_features,
    load_candidates,
    precursor_formula_features,
)


SetKey = Tuple[int, ...]


def metrics(targets: List[SetKey], rows: List[List[SetKey]]) -> Dict[str, float]:
    return {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export rankings from a trained Stage2 listwise checkpoint.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate_sources", nargs="+", required=True)
    parser.add_argument("--candidate_limit", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--residual_scale", type=float, default=None,
        help="Frozen validation-selected residual scale. Defaults to the checkpoint training value.",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    x = np.asarray(pack["x"], dtype=np.float32)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    candidates, candidate_scores = load_candidates(args.candidate_sources, len(x))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if bool(config.get("append_source_features", False)):
        x = append_source_features(x, input_dir / f"{args.split}_meta.csv", checkpoint.get("source_vocab", []))
    if bool(config.get("append_query_formula_features", False)):
        x = append_query_formula_features(x, input_dir / f"{args.split}_meta.csv")
    dataset = CandidateDataset(
        x, y, candidates, candidate_scores, args.candidate_limit,
        args.candidate_limit,
        int(checkpoint["max_set_len"]), int(checkpoint["n_labels"]), False, int(config.get("seed", 0)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
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
            147 if (
                bool(config.get("aligned_query_formula_encoder", False))
                or bool(config.get("use_relative_chemistry_features", False))
            ) else 0
        ),
        use_relative_chemistry_features=bool(config.get("use_relative_chemistry_features", False)),
        candidate_transformer_layers=int(config.get("candidate_transformer_layers", 0)),
        candidate_transformer_heads=int(config.get("candidate_transformer_heads", 8)),
        joint_transformer_layers=int(config.get("joint_transformer_layers", 0)),
        joint_transformer_heads=int(config.get("joint_transformer_heads", 8)),
        chemistry_only_label_encoder=bool(config.get("chemistry_only_label_encoder", False)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    rank_bias_weight = float(config.get("rank_bias_weight", 1.0))
    residual_scale = (
        float(args.residual_scale)
        if args.residual_scale is not None
        else float(config.get("residual_scale", 0.1))
    )
    ranked_rows: List[List[SetKey]] = []
    ranked_scores: List[List[float]] = []
    row_offset = 0
    with torch.no_grad():
        for batch_x, labels, numeric, mask, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            numeric = numeric.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                residual = model(batch_x, labels, numeric) * residual_scale
                scores = (rank_bias_weight * numeric[..., 1] + residual).masked_fill(mask < 0.5, -torch.inf)
            scores_np = scores.float().cpu().numpy()
            mask_np = mask.cpu().numpy()
            for local_index in range(len(batch_x)):
                source = candidates[row_offset + local_index][: args.candidate_limit]
                valid_count = int(mask_np[local_index].sum())
                order = np.argsort(-scores_np[local_index, :valid_count], kind="stable")
                ranked_rows.append([source[int(index)] for index in order])
                ranked_scores.append([float(scores_np[local_index, int(index)]) for index in order])
            row_offset += len(batch_x)
    report = {
        "protocol": f"{args.split}_formula_disjoint_exact_precursor_set_listwise_export",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "residual_scale": residual_scale,
        "evaluation": metrics(targets, ranked_rows),
        "mean_candidates": float(np.mean([len(row) for row in ranked_rows])),
    }
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, (row, scores) in enumerate(zip(ranked_rows, ranked_scores)):
            handle.write(json.dumps({"row_index": row_index, "candidate_label_ids": [list(value) for value in row], "scores": scores}) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
