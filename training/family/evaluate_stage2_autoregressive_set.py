#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.family.train_stage2_autoregressive_set import (
    AutoregressiveSetGenerator,
    beam_decode_batch,
)
from training.family.train_stage2_listwise_ranker import (
    append_query_formula_features,
    precursor_formula_features,
)


SetKey = Tuple[int, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export autoregressive precursor-set beam candidates.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--beam_width", type=int, default=500)
    parser.add_argument("--branch_factor", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
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
    precursor_names = json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    label_chemistry = torch.from_numpy(precursor_formula_features([str(value) for value in precursor_names]))
    model = AutoregressiveSetGenerator(
        int(checkpoint["x_dim"]), int(checkpoint["n_labels"]), int(checkpoint["max_set_len"]),
        int(config["hidden"]), int(config["blocks"]), float(config["dropout"]), label_chemistry,
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    rows: List[List[SetKey]] = []
    row_scores: List[List[float]] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            candidates, scores = beam_decode_batch(
                model, batch_x.to(device), args.beam_width, args.branch_factor
            )
            rows.extend(candidates)
            row_scores.extend(scores)
    metrics = {
        f"exact_hit@{k}": float(np.mean([target in set(row[:k]) for target, row in zip(targets, rows)]))
        for k in (1, 3, 5, 10, 20, 50, 100, 500)
    }
    report = {
        "protocol": f"stage2_{args.split}_formula_disjoint_exact_precursor_set_autoregressive_beam",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "beam_width": args.beam_width,
        "branch_factor": args.branch_factor,
        "metrics": metrics,
        "mean_unique_candidates": float(np.mean([len(row) for row in rows])),
    }
    Path(args.output_json).resolve().write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, (row, scores) in enumerate(zip(rows, row_scores)):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(value) for value in row],
                "scores": scores,
            }) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
