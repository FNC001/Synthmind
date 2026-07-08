#!/usr/bin/env python3
"""RSP vnext 006: promote skeleton-pruning recoveries under fixed budgets.

v005 proved that leave-one-out pruning recovers many exact skeletons at K=500,
but its conservative `preserve_base_top=200` leaves K=50/K=200 unchanged.  This
experiment keeps the same generated candidates and fixed budgets, then validates
whether earlier insertion of high-ranked pruning candidates improves K=50/200
without materially damaging exact@1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from synthmind.research.run_rsp_expansion import (
    bootstrap_ci,
    metric_dict,
    rank_variant,
    subset_metrics,
    to_builtin,
    write_candidate_table,
    write_json,
)
from synthmind.research.run_rsp_prune_recovery import (
    PruneConfig,
    attach_sample_flags,
    artifact_manifest,
    canonicalize_base,
    evaluate_fixed_params,
    first_existing,
    generate_prune_candidates,
    ints,
    read_config,
    source_ablation_rows,
    train_label_counts,
    write_report as write_prune_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research/configs/rsp_vnext_006_prune_promotion.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--budgets", default="50,200,500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(config_path: str | Path, split: str, budgets_arg: str, seed: int) -> PruneConfig:
    raw = read_config(config_path)
    paths = raw.get("paths", {})
    pruning = raw.get("pruning", {})
    gate = raw.get("validation_gate", {})
    split_key = "validation_candidates" if split == "validation" else "test_candidates"
    budgets = tuple(int(x) for x in budgets_arg.split(",") if x.strip()) or ints(pruning.get("budgets", [50, 200, 500]))
    cfg = PruneConfig(
        output_dir=Path(raw.get("output_dir", "outputs/autorun/rsp_vnext_006_prune_promotion")),
        train_candidates=first_existing(paths.get("train_candidates", [])),
        split_candidates=first_existing(paths.get(split_key, [])),
        budgets=budgets,
        seed_top_grid=ints(pruning.get("grid", {}).get("seed_top", [50, 100, 200])),
        preserve_base_top_grid=ints(pruning.get("grid", {}).get("preserve_base_top", [10, 20, 30, 40, 50, 100, 200])),
        max_candidate_size=int(pruning.get("max_candidate_size", 8)),
        prune_score_offset=float(pruning.get("prune_score_offset", -0.02)),
        max_exact1_drop=float(gate.get("max_exact1_drop_pp", 0.5)) / 100.0,
        bootstrap_iterations=int(gate.get("bootstrap_iterations", 1000)),
        ci=float(gate.get("ci", 0.95)),
        seed=seed,
        raw_config_path=str(config_path),
    )
    return cfg


def evaluate_promotion_gate(result: dict[str, Any], budgets: tuple[int, ...], max_exact1_drop: float) -> dict[str, Any]:
    base = result["variants"]["rsp_v5_baseline"]
    primary = result["variants"]["base_plus_skeleton_prune_promotion"]
    exact1_delta = primary.get("exact@1", 0.0) - base.get("exact@1", 0.0)
    oracle_deltas = {str(k): primary.get(f"skeleton_oracle@{k}", 0.0) - base.get(f"skeleton_oracle@{k}", 0.0) for k in budgets}
    target_delta = max(oracle_deltas.get("50", 0.0), oracle_deltas.get("200", 0.0))
    passed = exact1_delta >= -max_exact1_drop and target_delta > 0
    mode = "standard"
    if exact1_delta < 0 and target_delta > 0:
        mode = "coverage_mode"
    return {
        "passed": bool(passed),
        "status": "passed" if passed else "failed",
        "mode": mode if passed else "not_passed",
        "reason": f"max target oracle delta@50/200={target_delta:.6f}; exact@1 delta={exact1_delta:.6f}",
        "exact1_delta": exact1_delta,
        "oracle_deltas": oracle_deltas,
    }


def select_promotion_params(base: pd.DataFrame, cfg: PruneConfig) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, dict[str, float]], dict[str, int], list[dict[str, Any]]]:
    baseline = rank_variant(base, pd.DataFrame(), preserve_base_top=max(cfg.budgets), budgets=cfg.budgets)
    base_metrics = metric_dict(baseline, cfg.budgets)
    best_score = (-1e9, -1e9, -1e9, -1e9)
    best_params: dict[str, Any] = {}
    best_ranked = {"rsp_v5_baseline": baseline}
    best_metrics = {"rsp_v5_baseline": base_metrics}
    best_counts: dict[str, int] = {}
    grid_rows: list[dict[str, Any]] = []
    for seed_top in cfg.seed_top_grid:
        prune = generate_prune_candidates(base, seed_top, cfg)
        for preserve in cfg.preserve_base_top_grid:
            ranked = rank_variant(base, prune, preserve, cfg.budgets)
            metrics = metric_dict(ranked, cfg.budgets)
            deltas = {k: metrics.get(f"exact@{k}", 0.0) - base_metrics.get(f"exact@{k}", 0.0) for k in cfg.budgets}
            exact1_delta = metrics.get("exact@1", 0.0) - base_metrics.get("exact@1", 0.0)
            top_target = max(deltas.get(50, 0.0), deltas.get(200, 0.0))
            row = {
                "seed_top": seed_top,
                "preserve_base_top": preserve,
                "exact1_delta": exact1_delta,
                "exact50_delta": deltas.get(50, 0.0),
                "exact200_delta": deltas.get(200, 0.0),
                "exact500_delta": deltas.get(500, 0.0),
                "exact@1": metrics.get("exact@1", 0.0),
                "exact@50": metrics.get("exact@50", 0.0),
                "exact@200": metrics.get("exact@200", 0.0),
                "exact@500": metrics.get("exact@500", 0.0),
            }
            grid_rows.append(row)
            print(
                "[rsp_prune_promotion] "
                f"seed_top={seed_top} preserve={preserve} "
                f"d1={exact1_delta:.6f} d50={deltas.get(50, 0.0):.6f} "
                f"d200={deltas.get(200, 0.0):.6f} d500={deltas.get(500, 0.0):.6f}",
                flush=True,
            )
            if exact1_delta < -cfg.max_exact1_drop:
                continue
            # Prefer K=50/200 gains, then K=500 recovery, then exact@1 stability.
            score = (top_target, deltas.get(50, 0.0) + deltas.get(200, 0.0), deltas.get(500, 0.0), exact1_delta)
            if score > best_score:
                best_score = score
                best_params = {"seed_top": seed_top, "preserve_base_top": preserve}
                best_ranked = {"rsp_v5_baseline": baseline, "base_plus_skeleton_prune_promotion": ranked}
                best_metrics = {"rsp_v5_baseline": base_metrics, "base_plus_skeleton_prune_promotion": metrics}
                best_counts = {
                    "prune_candidates": int(len(prune)),
                    "prune_exact_candidate_rows": int(prune["label_exact"].sum()) if not prune.empty else 0,
                }
    if not best_params:
        raise RuntimeError("No promotion parameter setting satisfied exact@1 stability")
    return best_params, best_ranked, best_metrics, best_counts, grid_rows


def evaluate_fixed_promotion_params(base: pd.DataFrame, cfg: PruneConfig, params: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]], dict[str, int]]:
    baseline = rank_variant(base, pd.DataFrame(), preserve_base_top=max(cfg.budgets), budgets=cfg.budgets)
    prune = generate_prune_candidates(base, int(params["seed_top"]), cfg)
    ranked = rank_variant(base, prune, int(params["preserve_base_top"]), cfg.budgets)
    return (
        {"rsp_v5_baseline": baseline, "base_plus_skeleton_prune_promotion": ranked},
        {"rsp_v5_baseline": metric_dict(baseline, cfg.budgets), "base_plus_skeleton_prune_promotion": metric_dict(ranked, cfg.budgets)},
        {"prune_candidates": int(len(prune)), "prune_exact_candidate_rows": int(prune["label_exact"].sum()) if not prune.empty else 0},
    )


def write_report(path: Path, result: dict[str, Any]) -> None:
    result_for_report = dict(result)
    result_for_report["run_id"] = "rsp_vnext_006_prune_promotion"
    write_prune_report(path, result_for_report)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.split, args.budgets, args.seed)
    outdir = cfg.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    validation_result: dict[str, Any] | None = None
    if args.split == "test":
        validation_path = outdir / "metrics.json"
        if not validation_path.exists():
            raise SystemExit("Refusing test: validation metrics.json does not exist")
        validation_result = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation_result.get("validation_gate", {}).get("passed"):
            raise SystemExit("Refusing test: validation gate did not pass")

    print(f"[rsp_prune_promotion] loading split candidates: {cfg.split_candidates}", flush=True)
    base = canonicalize_base(pd.read_csv(cfg.split_candidates))
    counts = train_label_counts(pd.read_csv(cfg.train_candidates))
    base = attach_sample_flags(base, counts)
    print(f"[rsp_prune_promotion] base rows={len(base)} samples={base['sample_id'].nunique()}", flush=True)

    if args.split == "test":
        assert validation_result is not None
        params = validation_result["selected_params"]
        ranked_by_variant, metrics_by_variant, generated_counts = evaluate_fixed_promotion_params(base, cfg, params)
        grid_rows: list[dict[str, Any]] = []
    else:
        params, ranked_by_variant, metrics_by_variant, generated_counts, grid_rows = select_promotion_params(base, cfg)

    baseline = ranked_by_variant["rsp_v5_baseline"]
    primary = ranked_by_variant["base_plus_skeleton_prune_promotion"]
    source_rows = source_ablation_rows("base_plus_skeleton_prune_promotion", primary, baseline, cfg.budgets)
    rare_rows = []
    for name, ranked in ranked_by_variant.items():
        rare_rows.append({"variant": name, "subset": "rare_precursor", **subset_metrics(ranked, "is_rare_reference", cfg.budgets)})
        rare_rows.append({"variant": name, "subset": "oov_precursor", **subset_metrics(ranked, "is_oov_reference", cfg.budgets)})

    print("[rsp_prune_promotion] running paired bootstrap", flush=True)
    bootstrap = bootstrap_ci(baseline, primary, cfg.budgets, cfg.bootstrap_iterations, cfg.ci, cfg.seed)
    result: dict[str, Any] = {
        "run_id": "rsp_vnext_006_prune_promotion",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "split": args.split,
        "config": cfg.raw_config_path,
        "input_artifacts": {"split_candidates": str(cfg.split_candidates), "train_candidates": str(cfg.train_candidates)},
        "budgets": list(cfg.budgets),
        "selected_params": params,
        "generated_candidate_counts": generated_counts,
        "variants": metrics_by_variant,
        "subset_metrics": rare_rows,
        "bootstrap_ci": bootstrap,
        "selector_update_status": "unchanged",
        "test_evaluation": "not_applicable_for_validation" if args.split == "validation" else "run_once_after_validation_gate",
    }
    if args.split == "validation":
        result["validation_gate"] = evaluate_promotion_gate(result, cfg.budgets, cfg.max_exact1_drop)
    else:
        result["validation_gate"] = validation_result.get("validation_gate", {}) if validation_result else {}

    prefix = "val" if args.split == "validation" else "test"
    result["candidate_table"] = write_candidate_table(outdir / f"{prefix}_rsp_prune_promotion_candidates.parquet", primary)
    pd.DataFrame(source_rows).to_csv(outdir / ("candidate_source_ablation.csv" if args.split == "validation" else "test_candidate_source_ablation.csv"), index=False)
    pd.DataFrame(rare_rows).to_csv(outdir / ("rare_precursor_analysis.csv" if args.split == "validation" else "test_rare_precursor_analysis.csv"), index=False)
    if grid_rows:
        pd.DataFrame(grid_rows).to_csv(outdir / "promotion_grid_search.csv", index=False)
    metrics_name = "metrics.json" if args.split == "validation" else "test_metrics_if_gate_passed.json"
    report_name = "RSP_PRUNE_PROMOTION_REPORT.md" if args.split == "validation" else "TEST_RSP_PRUNE_PROMOTION_REPORT.md"
    write_json(outdir / metrics_name, result)
    write_report(outdir / report_name, result)
    manifest_paths = sorted(p for p in outdir.iterdir() if p.is_file() and p.name != "artifact_manifest.json")
    write_json(outdir / "artifact_manifest.json", artifact_manifest(manifest_paths))
    print(json.dumps(to_builtin({"output_dir": str(outdir), "validation_gate": result["validation_gate"]}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
