#!/usr/bin/env python3
"""Evaluate GRV/Stage35 rankers on one frozen route-candidate pool.

This runner is intentionally inference-selector safe.  It does not regenerate
candidate pools and does not promote any model.  It freezes a source CSV by
hash, applies multiple rankers to exactly the same rows, and reports paired
sample-level bootstrap intervals for top-1 route hits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "10_autorun"
if str(PROJECT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SCRIPT_DIR))

from run_autodl_gpu_training_queue import (  # noqa: E402
    RouteRanker,
    blend_scores,
    build_matrix,
    score_in_batches,
)


V3_FINAL_WEIGHTS = {
    "route_total_score_raw": 0.8,
    "precursor_rank_score": 0.5,
    "condition_rank_score": 0.5,
    "rank_product_score": 0.7,
    "contains_open_generated_precursor": -0.2,
    "contains_repair_precursor": -0.1,
}


@dataclass(frozen=True)
class RankerSpec:
    name: str
    kind: str
    run_dir: Path | None = None
    checkpoint: Path | None = None
    best_json: Path | None = None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_quantile(values: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo = vals.quantile(0.01)
    hi = vals.quantile(0.99)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return ((vals - lo) / (hi - lo)).clip(0, 1).to_numpy(dtype=np.float64)
    return vals.to_numpy(dtype=np.float64)


def v3_final_scores(df: pd.DataFrame) -> np.ndarray:
    precursor_rank = pd.to_numeric(df.get("precursor_rank", 999), errors="coerce").clip(lower=1).fillna(999)
    condition_rank = pd.to_numeric(df.get("condition_rank", 999), errors="coerce").clip(lower=1).fillna(999)
    features: dict[str, np.ndarray] = {
        "route_total_score_raw": normalize_quantile(df.get("route_total_score_raw", pd.Series(0.0, index=df.index))),
        "precursor_rank_score": (1.0 / precursor_rank).to_numpy(dtype=np.float64),
        "condition_rank_score": (1.0 / condition_rank).to_numpy(dtype=np.float64),
        "rank_product_score": ((1.0 / precursor_rank) * (1.0 / condition_rank)).to_numpy(dtype=np.float64),
        "contains_open_generated_precursor": normalize_quantile(
            df.get("contains_open_generated_precursor", pd.Series(0.0, index=df.index))
        ),
        "contains_repair_precursor": normalize_quantile(
            df.get("contains_repair_precursor", pd.Series(0.0, index=df.index))
        ),
    }
    score = np.zeros(len(df), dtype=np.float64)
    for key, weight in V3_FINAL_WEIGHTS.items():
        score += float(weight) * features[key]
    return score


def raw_route_scores(df: pd.DataFrame) -> np.ndarray:
    if "route_total_score_raw" in df.columns:
        return pd.to_numeric(df["route_total_score_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.zeros(len(df), dtype=np.float64)


def protocol_labels(df: pd.DataFrame, protocol: str) -> dict[str, np.ndarray]:
    exact = pd.Series(df.get("precursor_exact_if_eval", 0), index=df.index).astype(str).str.lower().isin(["true", "1", "1.0"])
    jac = pd.to_numeric(df.get("precursor_jaccard_if_eval", 0), errors="coerce").fillna(0.0)
    if protocol == "strict_comparable":
        known = pd.to_numeric(df.get("atmosphere_known_mask", 0), errors="coerce").fillna(0).astype(int) == 1
        pred_atm = df.get("atmosphere", "").astype(str).str.lower()
        true_atm = df.get("true_atmosphere", "").astype(str).str.lower()
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


def rank_positions(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    ranked = pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str).to_numpy(),
            "score": np.asarray(scores, dtype=np.float64),
            "row_index": np.arange(len(df), dtype=np.int64),
        }
    )
    ranked = ranked.sort_values(["sample_id", "score", "row_index"], ascending=[True, False, True], kind="mergesort")
    ranked["rank"] = ranked.groupby("sample_id", sort=False).cumcount() + 1
    return ranked


def score_metrics(df: pd.DataFrame, scores: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    ranked = rank_positions(df, scores)
    labels = {
        "missing_aware": protocol_labels(df, "missing_aware"),
        "strict_comparable": protocol_labels(df, "strict_comparable"),
    }
    for protocol, cols in labels.items():
        for name, values in cols.items():
            ranked[f"{protocol}_{name}"] = values[ranked["row_index"].to_numpy()]

    out: dict[str, float] = {
        "num_samples": float(ranked["sample_id"].nunique()),
        "num_candidates": float(len(ranked)),
    }
    sample_rows: list[pd.DataFrame] = []
    for protocol in ["missing_aware", "strict_comparable"]:
        for hit_name in ["strict", "relaxed", "usable_relaxed"]:
            col = f"{protocol}_{hit_name}"
            for k in [1, 10, 200]:
                top = ranked[ranked["rank"] <= k]
                out[f"{protocol}_top{k}_{hit_name}_route"] = float(top.groupby("sample_id")[col].max().mean())
            first = ranked[ranked[col] > 0].groupby("sample_id", sort=False)["rank"].min()
            all_ids = pd.Index(sorted(ranked["sample_id"].unique()))
            first = first.reindex(all_ids)
            top1 = (first == 1).fillna(False).astype(int)
            sample_rows.append(
                pd.DataFrame(
                    {
                        "sample_id": all_ids,
                        f"{protocol}_{hit_name}_top1_hit": top1.to_numpy(dtype=np.int8),
                        f"{protocol}_{hit_name}_first_hit_rank": first.fillna(np.inf).to_numpy(dtype=np.float64),
                    }
                )
            )
            finite = first.replace(np.inf, np.nan).dropna()
            out[f"{protocol}_{hit_name}_mean_first_hit_rank"] = float(finite.mean()) if len(finite) else math.inf
            out[f"{protocol}_{hit_name}_ranking_gap"] = float(((first.fillna(999999) - 1).clip(lower=0)).mean())

    merged = sample_rows[0]
    for part in sample_rows[1:]:
        merged = merged.merge(part, on="sample_id", how="outer")
    return out, merged


def paired_bootstrap(
    baseline_hits: np.ndarray,
    candidate_hits: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, float]:
    base = np.asarray(baseline_hits, dtype=np.float64)
    cand = np.asarray(candidate_hits, dtype=np.float64)
    if base.shape != cand.shape:
        raise ValueError("bootstrap hit arrays must have identical shape")
    rng = np.random.default_rng(seed)
    n = len(base)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(cand[idx].mean() - base[idx].mean())
    return {
        "delta": float(cand.mean() - base.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "n_samples": int(n),
        "n_bootstrap": int(n_bootstrap),
    }


def resolve_checkpoint(project_root: Path, spec: RankerSpec) -> tuple[Path, Path, dict[str, Any]]:
    if spec.kind != "checkpoint":
        raise ValueError(spec)
    run_dir = spec.run_dir
    if run_dir is None:
        raise ValueError(f"{spec.name}: missing run_dir")
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    best_json = spec.best_json or (run_dir / "best_val_model.json")
    if not best_json.is_absolute():
        best_json = project_root / best_json
    best_payload: dict[str, Any] = {}
    checkpoint = spec.checkpoint
    if best_json.exists():
        best_payload = json.loads(best_json.read_text(encoding="utf-8"))
        checkpoint = Path(str(best_payload.get("checkpoint", checkpoint or "")))
    if checkpoint is None:
        raise FileNotFoundError(f"{spec.name}: no checkpoint found")
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    if not checkpoint.exists():
        # Older queue JSONs sometimes kept /root/autodl-tmp paths after the
        # project was mirrored to /root.  Recover by matching under run_dir.
        fallback = run_dir / "checkpoints" / checkpoint.name
        if fallback.exists():
            checkpoint = fallback
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    return run_dir, checkpoint, best_payload


def checkpoint_scores(project_root: Path, df: pd.DataFrame, spec: RankerSpec) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    run_dir, checkpoint, best_payload = resolve_checkpoint(project_root, spec)
    schema = json.loads((run_dir / "feature_schema.json").read_text(encoding="utf-8"))
    stats = json.loads((run_dir / "standardization_stats.json").read_text(encoding="utf-8"))
    x_np = build_matrix(df, schema)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    x_np = ((x_np - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    variant = ckpt["variant"]
    model = RouteRanker(x_np.shape[1], int(variant["hidden"]), float(variant["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    x = torch.tensor(x_np, device=device)
    model_scores = score_in_batches(model, x, max(int(variant["batch_size"]) * 2, 65536))
    final_scores = blend_scores(df, model_scores, ckpt.get("blend", {}))
    meta = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "best_json": str(spec.best_json or (run_dir / "best_val_model.json")),
        "variant": variant,
        "blend": ckpt.get("blend", {}),
        "checkpoint_metrics": ckpt.get("metrics", {}),
        "checkpoint_full_val_metrics": ckpt.get("full_val_metrics", {}),
        "best_json_score": best_payload.get("score"),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    return final_scores, meta


def parse_ranker_specs(project_root: Path, text: str) -> list[RankerSpec]:
    specs: list[RankerSpec] = [
        RankerSpec("raw_route_score", "raw"),
        RankerSpec("legacy_v3_final_formula", "v3_formula"),
    ]
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if "=" in item:
            name, run_dir = item.split("=", 1)
        else:
            p = Path(item)
            name = p.name
            run_dir = item
        specs.append(RankerSpec(name=name.strip(), kind="checkpoint", run_dir=Path(run_dir.strip())))
    return specs


def write_report(out_dir: Path, split: str, manifest: dict[str, Any], rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    repro_note = "not applicable"
    repro_path = out_dir / "checkpoint_reproducibility_check.csv"
    if repro_path.exists():
        try:
            repro = pd.read_csv(repro_path)
        except pd.errors.EmptyDataError:
            repro = pd.DataFrame()
        if len(repro):
            inconsistent = int((~repro["consistent_1e_6"].astype(bool)).sum())
            repro_note = f"{inconsistent} metric entries differ from checkpoint-recorded full-val metrics"
    lines = [
        "# GRV Shared-Pool Comparison",
        "",
        f"- Split: `{split}`",
        f"- Frozen pool: `{manifest['pool_path']}`",
        f"- Pool sha256: `{manifest['sha256']}`",
        f"- Samples: {manifest['num_samples']}",
        f"- Candidates: {manifest['num_candidates']}",
        "- Default inference selector updated: no",
        f"- Checkpoint reproducibility check: {repro_note}",
        "",
        "## Same-Pool Metrics",
        "",
        "| ranker | missing top1 | missing top10 | missing top200 | strict-comparable top1 | usable top1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {ranker} | {m1:.4f} | {m10:.4f} | {m200:.4f} | {s1:.4f} | {u1:.4f} |".format(
                ranker=row["ranker"],
                m1=float(row.get("missing_aware_top1_relaxed_route", 0)),
                m10=float(row.get("missing_aware_top10_relaxed_route", 0)),
                m200=float(row.get("missing_aware_top200_relaxed_route", 0)),
                s1=float(row.get("strict_comparable_top1_relaxed_route", 0)),
                u1=float(row.get("missing_aware_top1_usable_relaxed_route", 0)),
            )
        )
    lines += [
        "",
        "## Validation Gate",
        "",
        "```json",
        json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (out_dir / "GRV_SHARED_POOL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--pool_csv", default="outputs/evaluation/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv")
    parser.add_argument("--output_dir", default="outputs/autorun/grv_shared_pool_20260623")
    parser.add_argument(
        "--checkpoint_runs",
        default=(
            "grv_v12=outputs/autorun/gpu_24h_training_20260616_v12_ooftrain_strict_replicate,"
            "grv_v14=outputs/autorun/gpu_24h_training_20260618_v14_ooftrain_strict_replicate,"
            "grv_v15=outputs/autorun/gpu_24h_training_20260618_v15_ooftrain_routemix"
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline_ranker", default="legacy_v3_final_formula")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    pool_csv = Path(args.pool_csv)
    if not pool_csv.is_absolute():
        pool_csv = project_root / pool_csv
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pool_csv)
    manifest = {
        "split": args.split,
        "pool_path": str(pool_csv),
        "sha256": sha256_file(pool_csv),
        "num_samples": int(df["sample_id"].nunique()),
        "num_candidates": int(len(df)),
        "candidate_budget_note": "Frozen existing v3-final route candidate table; every ranker scores identical rows.",
        "selector_update_status": "unchanged",
    }
    write_json(out_dir / "shared_pool_manifest.json", manifest)

    specs = parse_ranker_specs(project_root, args.checkpoint_runs)
    rows: list[dict[str, Any]] = []
    sample_tables: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {}
    for spec in specs:
        if spec.kind == "raw":
            scores = raw_route_scores(df)
            meta = {"kind": spec.kind}
        elif spec.kind == "v3_formula":
            scores = v3_final_scores(df)
            meta = {"kind": spec.kind, "weights": V3_FINAL_WEIGHTS}
        elif spec.kind == "checkpoint":
            scores, meta = checkpoint_scores(project_root, df, spec)
        else:
            raise ValueError(spec)
        metrics, sample_table = score_metrics(df, scores)
        rows.append({"ranker": spec.name, **metrics})
        sample_table = sample_table.add_prefix(f"{spec.name}__").rename(columns={f"{spec.name}__sample_id": "sample_id"})
        sample_tables[spec.name] = sample_table
        metadata[spec.name] = meta

    pd.DataFrame(rows).to_csv(out_dir / "same_pool_comparison.csv", index=False)
    write_json(out_dir / "same_pool_metrics.json", {"manifest": manifest, "rankers": rows, "metadata": metadata})

    repro_rows: list[dict[str, Any]] = []
    row_by_ranker = {row["ranker"]: row for row in rows}
    metric_pairs = {
        "missing_top1_relaxed_route": "missing_aware_top1_relaxed_route",
        "missing_top10_relaxed_route": "missing_aware_top10_relaxed_route",
        "missing_top200_relaxed_route": "missing_aware_top200_relaxed_route",
        "usable_top1_relaxed_route": "missing_aware_top1_usable_relaxed_route",
        "usable_top10_relaxed_route": "missing_aware_top10_usable_relaxed_route",
        "usable_top200_relaxed_route": "missing_aware_top200_usable_relaxed_route",
    }
    if args.split == "validation":
        for ranker, meta in metadata.items():
            recorded = meta.get("checkpoint_full_val_metrics") or {}
            if not recorded:
                continue
            recomputed = row_by_ranker[ranker]
            for recorded_key, recomputed_key in metric_pairs.items():
                if recorded_key not in recorded or recomputed_key not in recomputed:
                    continue
                recorded_value = float(recorded[recorded_key])
                recomputed_value = float(recomputed[recomputed_key])
                repro_rows.append(
                    {
                        "ranker": ranker,
                        "metric": recorded_key,
                        "recorded_full_val": recorded_value,
                        "recomputed_same_pool": recomputed_value,
                        "delta_recomputed_minus_recorded": recomputed_value - recorded_value,
                        "consistent_1e_6": abs(recomputed_value - recorded_value) <= 1e-6,
                    }
                )
    pd.DataFrame(repro_rows).to_csv(out_dir / "checkpoint_reproducibility_check.csv", index=False)

    aligned = None
    for table in sample_tables.values():
        aligned = table if aligned is None else aligned.merge(table, on="sample_id", how="inner")
    if aligned is None:
        raise RuntimeError("No sample-level tables were produced")
    aligned.to_csv(out_dir / "sample_level_top1_hits.csv", index=False)

    baseline_name = args.baseline_ranker
    if baseline_name not in sample_tables:
        raise ValueError(f"Unknown baseline ranker {baseline_name!r}; available={sorted(sample_tables)}")

    ci: dict[str, Any] = {}
    base = aligned
    for name in sample_tables:
        if name == baseline_name:
            continue
        for metric_key in [
            "missing_aware_relaxed_top1_hit",
            "strict_comparable_relaxed_top1_hit",
            "missing_aware_usable_relaxed_top1_hit",
        ]:
            bcol = f"{baseline_name}__{metric_key}"
            ccol = f"{name}__{metric_key}"
            ci[f"{name}_minus_{baseline_name}:{metric_key}"] = paired_bootstrap(
                base[bcol].to_numpy(),
                base[ccol].to_numpy(),
                seed=int(args.seed),
                n_bootstrap=int(args.bootstrap),
            )
    write_json(out_dir / "paired_bootstrap_ci.json", ci)

    baseline_row = next(r for r in rows if r["ranker"] == baseline_name)
    best_row = max(rows, key=lambda r: float(r.get("missing_aware_top1_relaxed_route", 0.0)))
    delta = float(best_row.get("missing_aware_top1_relaxed_route", 0.0)) - float(
        baseline_row.get("missing_aware_top1_relaxed_route", 0.0)
    )
    best_ci = ci.get(f"{best_row['ranker']}_minus_{baseline_name}:missing_aware_relaxed_top1_hit", {})
    gate = {
        "baseline_ranker": baseline_name,
        "best_ranker": best_row["ranker"],
        "missing_aware_top1_delta": delta,
        "required_delta": 0.01,
        "bootstrap_ci": best_ci,
        "passes_same_pool_grv_gate": bool(delta >= 0.01 and float(best_ci.get("ci_low", -1.0)) > 0.0),
        "selector_update_status": "unchanged",
        "test_allowed": bool(args.split == "validation" and delta >= 0.01 and float(best_ci.get("ci_low", -1.0)) > 0.0),
    }
    write_json(out_dir / "validation_gate.json", gate)
    write_report(out_dir, args.split, manifest, rows, gate)
    print(json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
