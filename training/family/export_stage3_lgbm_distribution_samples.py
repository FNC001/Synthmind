#!/usr/bin/env python3
"""Export probabilistic Stage3 samples from an existing LightGBM artifact.

The residual-bootstrap mode is a leakage-safe generative baseline: residual pools
are built only from the training split, optionally conditioned on a family group.
It is intentionally simple, but it gives VAE/GAN/NF/diffusion models a common
sample-level benchmark instead of comparing them with point-estimate metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


CONTINUOUS_FIELDS = ("temperature_c", "time_h")
DISCRETE_FIELDS = ("atmosphere_coarse", "reaction_method")


def stable_seed(value: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def load_pack(input_dir: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(input_dir / f"{split}.npz", allow_pickle=True) as pack:
        return {key: np.asarray(pack[key]) for key in pack.files}


def build_features(
    pack: dict[str, np.ndarray], schema: dict[str, Any], drop_family_features: bool
) -> np.ndarray:
    structure = np.asarray(pack["x"], dtype=np.float32)
    if drop_family_features:
        structure = structure[:, : int(schema["base_feature_count"])]
    return np.hstack([structure, np.asarray(pack["y_set"], dtype=np.float32)]).astype(
        np.float32
    )


def predict_model(model: Any, features: np.ndarray) -> np.ndarray:
    iteration = getattr(model, "best_iteration", None)
    if iteration is not None and int(iteration) > 0:
        return np.asarray(model.predict(features, num_iteration=int(iteration)))
    return np.asarray(model.predict(features))


def residual_pools(
    residuals: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray,
    clip_quantile: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    valid = np.asarray(residuals[mask], dtype=np.float64)
    if not len(valid):
        raise ValueError("continuous target has no observed training residuals")
    if 0.5 < clip_quantile < 1.0:
        bound = float(np.quantile(np.abs(valid), clip_quantile))
        valid = np.clip(valid, -bound, bound)
    pools: dict[str, np.ndarray] = {}
    for group in np.unique(groups):
        local = np.asarray(residuals[mask & (groups == group)], dtype=np.float64)
        if len(local):
            if 0.5 < clip_quantile < 1.0:
                local = np.clip(local, -bound, bound)
            pools[str(group)] = local
    return valid, pools


def sample_residuals(
    global_pool: np.ndarray,
    group_pool: np.ndarray | None,
    samples: int,
    group_weight: float,
    minimum_group_support: int,
    rng: np.random.Generator,
) -> np.ndarray:
    use_group = group_pool is not None and len(group_pool) >= minimum_group_support
    if not use_group or group_weight <= 0:
        return rng.choice(global_pool, size=samples, replace=True)
    choose_group = rng.random(samples) < group_weight
    result = rng.choice(global_pool, size=samples, replace=True)
    result[choose_group] = rng.choice(group_pool, size=int(choose_group.sum()), replace=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export point or train-residual-bootstrap samples from Stage3 LightGBM."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--mode", choices=("point", "residual_bootstrap"), default="residual_bootstrap")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--group_column", default="family_group_key")
    parser.add_argument("--group_weight", type=float, default=0.7)
    parser.add_argument("--minimum_group_support", type=int, default=20)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--residual_clip_quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if not 0 <= args.group_weight <= 1:
        parser.error("--group_weight must be in [0, 1]")

    input_dir = Path(args.input_dir).expanduser().resolve()
    artifact = joblib.load(Path(args.model).expanduser().resolve())
    schema = artifact["schema"]
    train = load_pack(input_dir, "train")
    target = load_pack(input_dir, args.split)
    train_x = build_features(train, schema, bool(artifact["drop_family_features"]))
    target_x = build_features(target, schema, bool(artifact["drop_family_features"]))
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    target_meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    if args.group_column not in train_meta or args.group_column not in target_meta:
        raise ValueError(f"missing group column {args.group_column!r}")
    train_groups = train_meta[args.group_column].fillna("UNK").astype(str).to_numpy()
    target_groups = target_meta[args.group_column].fillna("UNK").astype(str).to_numpy()

    n_rows = len(target_x)
    continuous = np.empty((n_rows, args.samples, len(CONTINUOUS_FIELDS)), dtype=np.float32)
    discrete = np.empty((n_rows, args.samples, len(DISCRETE_FIELDS)), dtype=np.int16)
    residual_summary: dict[str, Any] = {}

    for field_index, field in enumerate(CONTINUOUS_FIELDS):
        model = artifact["models"][field]
        train_prediction = predict_model(model, train_x).astype(np.float64)
        target_prediction = predict_model(model, target_x).astype(np.float64)
        mask = np.asarray(train["y_cond_continuous_mask"][:, field_index] > 0.5)
        truth = np.asarray(train["y_cond_continuous_raw"][:, field_index], dtype=np.float64)
        if field == "time_h":
            transformed_truth = np.log1p(np.clip(truth, 0.0, None))
            residual = transformed_truth - train_prediction
        else:
            residual = truth - train_prediction
        global_pool, group_pools = residual_pools(
            residual, mask, train_groups, float(args.residual_clip_quantile)
        )
        for row in range(n_rows):
            if args.mode == "point":
                values = np.full(args.samples, target_prediction[row], dtype=np.float64)
            else:
                rng = np.random.default_rng(
                    stable_seed(f"continuous:{field}:{row}", int(args.seed))
                )
                noise = sample_residuals(
                    global_pool,
                    group_pools.get(str(target_groups[row])),
                    int(args.samples),
                    float(args.group_weight),
                    int(args.minimum_group_support),
                    rng,
                )
                values = target_prediction[row] + float(args.residual_scale) * noise
            if field == "time_h":
                values = np.expm1(values)
                values = np.clip(values, 0.0, 10_000.0)
            else:
                values = np.clip(values, 0.0, 2_500.0)
            continuous[row, :, field_index] = values.astype(np.float32)
        residual_summary[field] = {
            "global_observations": int(len(global_pool)),
            "group_pools": int(len(group_pools)),
            "residual_std": float(np.std(global_pool)),
        }

    for field_index, field in enumerate(DISCRETE_FIELDS):
        model = artifact["models"][field]
        probabilities = predict_model(model, target_x).astype(np.float64)
        probabilities = np.clip(probabilities, 0.0, None)
        probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        classes = np.asarray(artifact["class_maps"][field], dtype=np.int64)
        for row in range(n_rows):
            if args.mode == "point":
                values = np.full(args.samples, classes[int(np.argmax(probabilities[row]))])
            else:
                rng = np.random.default_rng(
                    stable_seed(f"discrete:{field}:{row}", int(args.seed))
                )
                values = rng.choice(classes, size=args.samples, replace=True, p=probabilities[row])
            discrete[row, :, field_index] = values.astype(np.int16)

    output_npz = Path(args.output_npz).expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        continuous_samples=continuous,
        discrete_samples=discrete,
        sample_id=np.asarray(target["sample_id"]).astype(str),
    )
    report = {
        "protocol": f"stage3_{args.mode}_distribution_samples",
        "config": vars(args),
        "rows": int(n_rows),
        "samples_per_row": int(args.samples),
        "continuous_shape": list(continuous.shape),
        "discrete_shape": list(discrete.shape),
        "residual_summary": residual_summary,
        "output_npz": str(output_npz),
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
