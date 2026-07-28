#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def sequence_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def encode_unique_texts(
    values: Sequence[str],
    prompt: Callable[[str], str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    unique_values = list(dict.fromkeys(str(value) for value in values))
    value_to_index = {value: index for index, value in enumerate(unique_values)}
    encoded_batches = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(unique_values), int(batch_size)):
            texts = [prompt(value) for value in unique_values[start : start + int(batch_size)]]
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special = tokens.pop("special_tokens_mask").to(device)
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            content_mask = tokens["attention_mask"].bool() & ~special.bool()
            empty = ~content_mask.any(dim=1)
            if bool(empty.any()):
                content_mask[empty] = tokens["attention_mask"][empty].bool()
            weights = content_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            encoded_batches.append(pooled.float().cpu().numpy())
    unique_encoded = l2_normalize(np.concatenate(encoded_batches, axis=0))
    row_indices = np.asarray([value_to_index[str(value)] for value in values], dtype=np.int64)
    return unique_encoded[row_indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache leakage-safe MatSciBERT representations for Stage2 formulas."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    model_path = Path(args.model).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)

    precursor_names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray("stage2_matscibert_embeddings_v1"),
        "model_path": np.asarray(str(model_path)),
        "hidden_size": np.asarray(int(model.config.hidden_size), dtype=np.int32),
        "precursor_names_sha256": np.asarray(sequence_sha256(precursor_names)),
        "precursor_common_mean": encode_unique_texts(
            precursor_names,
            lambda formula: f"Chemical formula: {formula}.",
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_length,
        ),
        "precursor_role_mean": encode_unique_texts(
            precursor_names,
            lambda formula: f"Precursor chemical formula: {formula}.",
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_length,
        ),
    }
    row_counts: dict[str, int] = {}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        meta_path = input_dir / f"{split}_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        formulas = pd.read_csv(meta_path, usecols=["formula"])["formula"].fillna("").astype(str).tolist()
        row_counts[split] = len(formulas)
        arrays[f"{split}_formula_sha256"] = np.asarray(sequence_sha256(formulas))
        arrays[f"{split}_query_common_mean"] = encode_unique_texts(
            formulas,
            lambda formula: f"Chemical formula: {formula}.",
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_length,
        )
        arrays[f"{split}_query_role_mean"] = encode_unique_texts(
            formulas,
            lambda formula: f"Target material formula: {formula}.",
            tokenizer,
            model,
            device,
            args.batch_size,
            args.max_length,
        )

    np.savez_compressed(output, **arrays)
    report = {
        "schema_version": "stage2_matscibert_embeddings_v1",
        "model": str(model_path),
        "device": str(device),
        "hidden_size": int(model.config.hidden_size),
        "precursor_count": len(precursor_names),
        "rows": row_counts,
        "output": str(output),
        "pooling": "content_token_mean_l2_normalized",
        "views": ["common_formula_prompt", "target_or_precursor_role_prompt"],
        "selection_safety": "No validation or test labels are read; only observable formula strings are encoded.",
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
