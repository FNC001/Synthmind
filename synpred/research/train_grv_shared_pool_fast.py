#!/usr/bin/env python3
"""Fast validation-first GRV training on a frozen shared route pool.

The legacy GPU queue is useful for long sweeps, but its validation blend search
is too slow for tight shared-pool iterations.  This runner keeps the same
non-leaky feature schema and checkpoint format, while using a small blend grid
and explicit counterfactual negative sampling.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "10_autorun"
if str(PROJECT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPT_DIR))

from run_autodl_gpu_training_queue import (  # noqa: E402
    RouteRanker,
    atomic_write_json,
    blend_scores,
    build_matrix,
    detect_feature_schema,
    parse_feature_list,
    route_metrics,
    score_in_batches,
    selection_score,
    standardize,
)


BASELINE_SAME_POOL_TOP1 = 0.15500272810459137


@dataclass
class FastVariant:
    seed: int
    hidden: int = 1024
    dropout: float = 0.08
    lr: float = 7e-4
    batch_size: int = 131072
    weight_decay: float = 1e-4
    pairwise_weight: float = 0.7
    pairwise_pairs_per_epoch: int = 131072


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def binary_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(df), dtype=np.float32)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.float32)


def make_targets(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, np.ndarray]:
    relaxed = binary_col(df, "relaxed_route_hit_if_eval")
    strict = binary_col(df, "strict_route_hit_if_eval")
    usable = binary_col(df, "usable_relaxed_route_hit_if_eval")
    if mode == "strict":
        y = np.clip(strict + 0.35 * relaxed + 0.15 * usable, 0.0, 1.0)
        w = 1.0 + 3.0 * strict + 0.8 * relaxed + 0.4 * usable
    elif mode == "usable":
        y = np.clip(usable + 0.35 * relaxed + 0.2 * strict, 0.0, 1.0)
        w = 1.0 + 2.0 * usable + 0.8 * relaxed + 0.8 * strict
    else:
        y = np.clip(relaxed + 0.5 * strict + 0.25 * usable, 0.0, 1.0)
        w = 1.0 + 2.0 * relaxed + 2.0 * strict + 0.7 * usable
    return y.astype(np.float32), w.astype(np.float32)


def build_counterfactual_groups(df: pd.DataFrame) -> list[dict[str, np.ndarray]]:
    relaxed = binary_col(df, "relaxed_route_hit_if_eval") > 0
    strict = binary_col(df, "strict_route_hit_if_eval") > 0
    precursor_exact = binary_col(df, "precursor_exact_if_eval") > 0
    condition_relaxed = binary_col(df, "relaxed_condition_hit_if_eval") > 0
    jaccard = pd.to_numeric(df.get("precursor_jaccard_if_eval", 0), errors="coerce").fillna(0.0).to_numpy()
    sample_ids = df["sample_id"].astype(str).to_numpy()
    groups: list[dict[str, np.ndarray]] = []
    index = np.arange(len(df), dtype=np.int64)
    work = pd.DataFrame({"sample_id": sample_ids, "row": index})
    for _, sub in work.groupby("sample_id", sort=False):
        rows = sub["row"].to_numpy(dtype=np.int64)
        pos = rows[relaxed[rows] | strict[rows]]
        if len(pos) == 0:
            continue
        precursor_conflict = rows[(~precursor_exact[rows]) & condition_relaxed[rows]]
        condition_conflict = rows[precursor_exact[rows] & (~condition_relaxed[rows])]
        near_miss = rows[(jaccard[rows] >= 0.5) & condition_relaxed[rows] & (~precursor_exact[rows])]
        generic = rows[~(relaxed[rows] | strict[rows])]
        neg = np.unique(np.concatenate([precursor_conflict, condition_conflict, near_miss, generic]))
        if len(neg) == 0:
            continue
        groups.append(
            {
                "pos": pos,
                "precursor_conflict": precursor_conflict,
                "condition_conflict": condition_conflict,
                "near_miss": near_miss,
                "generic": generic,
                "neg": neg,
            }
        )
    return groups


def sample_pairs(groups: list[dict[str, np.ndarray]], n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    categories = ["near_miss", "condition_conflict", "precursor_conflict", "generic", "neg"]
    probs = np.array([0.30, 0.25, 0.25, 0.15, 0.05], dtype=np.float64)
    probs = probs / probs.sum()
    pos = np.empty(n_pairs, dtype=np.int64)
    neg = np.empty(n_pairs, dtype=np.int64)
    counts = {c: 0 for c in categories}
    gids = rng.integers(0, len(groups), size=n_pairs)
    chosen_cats = rng.choice(categories, size=n_pairs, p=probs)
    for i, gid in enumerate(gids):
        group = groups[int(gid)]
        pos_pool = group["pos"]
        cat = str(chosen_cats[i])
        neg_pool = group[cat]
        if len(neg_pool) == 0:
            neg_pool = group["neg"]
            cat = "neg"
        pos[i] = pos_pool[int(rng.integers(0, len(pos_pool)))]
        neg[i] = neg_pool[int(rng.integers(0, len(neg_pool)))]
        counts[cat] += 1
    return pos, neg, counts


def base_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"__model_only__": np.zeros(len(df), dtype=np.float64)}
    for col in ["precursor_score_norm", "route_total_score_raw", "condition_score_norm"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            std = vals.std()
            out[col] = (vals - vals.mean()) / std if std > 1e-12 else np.zeros_like(vals)
    return out


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def protocol_labels(df: pd.DataFrame, protocol: str) -> dict[str, np.ndarray]:
    exact = pd.Series(df.get("precursor_exact_if_eval", 0), index=df.index).astype(str).str.lower().isin(
        ["true", "1", "1.0"]
    )
    jac = pd.to_numeric(df.get("precursor_jaccard_if_eval", 0), errors="coerce").fillna(0.0)
    if protocol == "strict_comparable":
        known = pd.to_numeric(df.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        pred_atm = df["atmosphere"].astype(str).str.lower() if "atmosphere" in df.columns else pd.Series("", index=df.index)
        true_atm = (
            df["true_atmosphere"].astype(str).str.lower()
            if "true_atmosphere" in df.columns
            else pd.Series("", index=df.index)
        )
        atm_ok = known & pred_atm.eq(true_atm)
        temp_err = pd.to_numeric(df.get("temp_error", np.inf), errors="coerce").fillna(np.inf)
        time_err = pd.to_numeric(df.get("time_error", np.inf), errors="coerce").fillna(np.inf)
        cond_strict = (temp_err <= 100) & (time_err <= 24) & atm_ok
        cond_relaxed = (temp_err <= 200) & (time_err <= 48) & atm_ok
    else:
        cond_strict = pd.to_numeric(df.get("strict_condition_hit_if_eval", 0), errors="coerce").fillna(0) > 0.5
        cond_relaxed = pd.to_numeric(df.get("relaxed_condition_hit_if_eval", 0), errors="coerce").fillna(0) > 0.5
    return {
        "strict": (exact & cond_strict).to_numpy(dtype=np.int8),
        "relaxed": (exact & cond_relaxed).to_numpy(dtype=np.int8),
        "usable_relaxed": ((jac >= 0.5) & cond_relaxed).to_numpy(dtype=np.int8),
    }


def shared_pool_route_metrics(df: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    """Return old queue metric names using the shared-pool protocol labels.

    The older Stage35 training queue used precomputed hit columns directly.
    Shared-pool comparisons recompute missing-aware, strict-comparable, and
    usable labels from the same frozen candidate rows.  Training selection must
    use the latter protocol, otherwise online validation and checkpoint replay
    are not measuring the same event.
    """

    ranked = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "score": np.asarray(scores, dtype=np.float64),
            "row_index": np.arange(len(df), dtype=np.int64),
        }
    )
    ranked = ranked.sort_values(["sample_id", "score", "row_index"], ascending=[True, False, True], kind="mergesort")
    labels = {
        "missing_aware": protocol_labels(df, "missing_aware"),
        "strict_comparable": protocol_labels(df, "strict_comparable"),
    }
    row_index = ranked["row_index"].to_numpy()
    ranked["missing_relaxed"] = labels["missing_aware"]["relaxed"][row_index]
    ranked["missing_usable_relaxed"] = labels["missing_aware"]["usable_relaxed"][row_index]
    ranked["strict_relaxed"] = labels["strict_comparable"]["relaxed"][row_index]

    metrics: dict[str, float] = {
        "num_samples": float(ranked["sample_id"].nunique()),
        "num_candidates": float(len(ranked)),
    }
    for k in (1, 10, 200):
        top = ranked.groupby("sample_id", sort=False).head(k)
        metrics[f"missing_top{k}_relaxed_route"] = float(top.groupby("sample_id")["missing_relaxed"].max().mean())
        metrics[f"usable_top{k}_relaxed_route"] = float(
            top.groupby("sample_id")["missing_usable_relaxed"].max().mean()
        )
        metrics[f"strict_top{k}_relaxed_route"] = float(top.groupby("sample_id")["strict_relaxed"].max().mean())
    return metrics


def apply_fast_blend(df: pd.DataFrame, model_scores: np.ndarray, blend: dict[str, Any]) -> np.ndarray:
    base_name = str(blend.get("base_score", "__model_only__"))
    alpha = float(blend.get("alpha_model", 1.0))
    model_z = normalize_scores(model_scores)
    if base_name == "__model_only__":
        return model_z
    base_z = base_scores(df).get(base_name)
    if base_z is None:
        return model_z
    return alpha * model_z + (1.0 - alpha) * base_z


def quick_blend_search(df: pd.DataFrame, model_scores: np.ndarray) -> dict[str, Any]:
    model_z = normalize_scores(model_scores)
    alphas = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
    best: dict[str, Any] | None = None
    for base_name, base_z in base_scores(df).items():
        local_alphas = [1.0] if base_name == "__model_only__" else alphas
        for alpha in local_alphas:
            final = model_z if base_name == "__model_only__" else alpha * model_z + (1.0 - alpha) * base_z
            metrics = shared_pool_route_metrics(df, final)
            score = selection_score(metrics)
            candidate = {
                "score": score,
                "metrics": metrics,
                "blend": {
                    "base_score": base_name,
                    "alpha_model": float(alpha),
                    "score_formula": "alpha*z(model_score)+(1-alpha)*z(base_score)",
                },
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    assert best is not None
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_root", default="outputs/autorun/grv_shared_pool_fast_20260623_v1")
    parser.add_argument("--train_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/train_route_candidates.csv")
    parser.add_argument("--val_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv")
    parser.add_argument("--max_minutes", type=float, default=45.0)
    parser.add_argument("--max_epochs", type=int, default=0, help="0 means time-budget controlled.")
    parser.add_argument("--eval_every_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9400)
    parser.add_argument("--label_mode", choices=["route_mix", "usable", "strict"], default="route_mix")
    parser.add_argument("--exclude_features", default="")
    parser.add_argument("--exclude_categorical_features", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_dir = project_root / args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fast_grv_training.log"

    def log(message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_path = project_root / args.train_csv
    val_path = project_root / args.val_csv
    log(f"loading train={train_path}")
    train_df = pd.read_csv(train_path)
    log(f"loading val={val_path}")
    val_df = pd.read_csv(val_path)

    schema = detect_feature_schema(
        train_df,
        parse_feature_list(args.exclude_features),
        parse_feature_list(args.exclude_categorical_features),
    )
    atomic_write_json(out_dir / "feature_schema.json", schema)
    log(f"schema numeric={len(schema['numeric_features'])} categorical={len(schema['categorical_maps'])}")

    train_x_np = build_matrix(train_df, schema)
    val_x_np = build_matrix(val_df, schema)
    train_x_np, [val_x_np], stats = standardize(train_x_np, [val_x_np])
    atomic_write_json(out_dir / "standardization_stats.json", stats)
    train_y_np, train_w_np = make_targets(train_df, args.label_mode)
    groups = build_counterfactual_groups(train_df)
    log(
        "targets="
        + json.dumps(
            {
                "label_mode": args.label_mode,
                "target_mean": float(train_y_np.mean()),
                "weight_mean": float(train_w_np.mean()),
                "counterfactual_groups": len(groups),
            },
            sort_keys=True,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = FastVariant(seed=args.seed)
    model = RouteRanker(train_x_np.shape[1], variant.hidden, variant.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
    pos = float(train_y_np.sum())
    neg = float(len(train_y_np) - pos)
    pos_weight = torch.tensor(min(max(neg / max(pos, 1.0), 1.0), 50.0), device=device)
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    train_x = torch.tensor(train_x_np, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_y_np, dtype=torch.float32, device=device)
    train_w = torch.tensor(train_w_np, dtype=torch.float32, device=device)
    val_x = torch.tensor(val_x_np, dtype=torch.float32, device=device)
    del train_x_np, val_x_np

    rng = np.random.default_rng(args.seed + 19)
    end_time = time.time() + float(args.max_minutes) * 60.0
    best: dict[str, Any] = {"score": -1.0, "epoch": 0, "metrics": {}, "checkpoint": None}
    epoch = 0
    while time.time() < end_time and (int(args.max_epochs) <= 0 or epoch < int(args.max_epochs)):
        epoch += 1
        model.train()
        order = torch.randperm(train_x.shape[0], device=device)
        batch_losses: list[float] = []
        for start in range(0, train_x.shape[0], variant.batch_size):
            idx = order[start : start + variant.batch_size]
            logits = model(train_x[idx])
            loss_vec = criterion(logits, train_y[idx])
            loss = (loss_vec * train_w[idx]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        pair_loss_value = 0.0
        pair_counts: dict[str, int] = {}
        if groups and variant.pairwise_weight > 0 and variant.pairwise_pairs_per_epoch > 0:
            pos_idx_np, neg_idx_np, pair_counts = sample_pairs(groups, variant.pairwise_pairs_per_epoch, rng)
            pos_idx = torch.tensor(pos_idx_np, dtype=torch.long, device=device)
            neg_idx = torch.tensor(neg_idx_np, dtype=torch.long, device=device)
            diff = model(train_x[pos_idx]) - model(train_x[neg_idx])
            pair_loss = torch.nn.functional.softplus(-diff).mean()
            optimizer.zero_grad(set_to_none=True)
            (variant.pairwise_weight * pair_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            pair_loss_value = float(pair_loss.detach().cpu().item())
        log(
            f"epoch={epoch} loss={float(np.mean(batch_losses)):.6f} "
            f"pair_loss={pair_loss_value:.6f} pair_counts={json.dumps(pair_counts, sort_keys=True)}"
        )

        if epoch == 1 or epoch % int(args.eval_every_epochs) == 0 or time.time() >= end_time:
            model_scores = score_in_batches(model, val_x, max(variant.batch_size, 65536))
            pure_metrics = shared_pool_route_metrics(val_df, model_scores)
            blend_result = quick_blend_search(val_df, model_scores)
            metrics = blend_result["metrics"]
            log(
                "VAL "
                + json.dumps(
                    {
                        "epoch": epoch,
                        "score": blend_result["score"],
                        "metrics": metrics,
                        "pure_model_metrics": pure_metrics,
                        "blend": blend_result["blend"],
                    },
                    sort_keys=True,
                )
            )
            latest = {
                "variant": asdict(variant),
                "epoch": epoch,
                "score": blend_result["score"],
                "metrics": metrics,
                "pure_model_metrics": pure_metrics,
                "blend": blend_result["blend"],
                "baseline_same_pool_top1": BASELINE_SAME_POOL_TOP1,
                "top1_delta_vs_same_pool_v3_formula": metrics.get("missing_top1_relaxed_route", 0.0)
                - BASELINE_SAME_POOL_TOP1,
            }
            atomic_write_json(out_dir / "latest_progress.json", latest)
            if blend_result["score"] > best["score"]:
                ckpt_path = out_dir / "checkpoints" / f"fast_grv_seed{variant.seed}_epoch{epoch}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(
                    {
                        "model_state": model_state,
                        "variant": asdict(variant),
                        "metrics": metrics,
                        "full_val_metrics": metrics,
                        "pure_model_metrics": pure_metrics,
                        "blend": blend_result["blend"],
                        "passes_gate": bool(
                            metrics.get("missing_top1_relaxed_route", 0.0) >= BASELINE_SAME_POOL_TOP1 + 0.01
                        ),
                    },
                    ckpt_path,
                )
                replay_model = RouteRanker(train_x.shape[1], variant.hidden, variant.dropout).to(device)
                replay_model.load_state_dict(model_state)
                replay_scores = score_in_batches(replay_model, val_x, max(variant.batch_size, 65536))
                replay_final = apply_fast_blend(val_df, replay_scores, blend_result["blend"])
                replay_metrics = shared_pool_route_metrics(val_df, replay_final)
                self_check = {
                    "checkpoint": str(ckpt_path),
                    "epoch": epoch,
                    "online_metrics": metrics,
                    "reloaded_metrics": replay_metrics,
                    "max_abs_metric_delta": max(
                        abs(float(replay_metrics.get(k, 0.0)) - float(metrics.get(k, 0.0))) for k in metrics
                    ),
                    "consistent_1e_6": all(
                        abs(float(replay_metrics.get(k, 0.0)) - float(metrics.get(k, 0.0))) <= 1e-6 for k in metrics
                    ),
                }
                atomic_write_json(out_dir / "checkpoint_reload_self_check.json", self_check)
                best = {**latest, "checkpoint": str(ckpt_path)}
                atomic_write_json(out_dir / "best_val_model.json", best)

    atomic_write_json(out_dir / "training_summary.json", {"best": best, "selector_update_status": "unchanged"})
    report = [
        "# Fast GRV Shared-Pool Training",
        "",
        "- Split used for selection: validation only",
        "- Test run: no",
        "- Default inference selector updated: no",
        "",
        "```json",
        json.dumps(to_builtin({"best": best}), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (out_dir / "FAST_GRV_TRAINING_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(best), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
