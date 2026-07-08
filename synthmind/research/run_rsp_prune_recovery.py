#!/usr/bin/env python3
"""RSP vnext 005: fixed-budget skeleton pruning recovery.

This experiment targets a common RSP failure mode: a high-ranked candidate
contains the complete reference skeleton plus one spurious precursor.  It
generates leave-one-out variants from the existing RSP v5 pool, keeps the
candidate budget fixed, and never updates the default inference selector.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from synthmind.research.run_rsp_expansion import (
    bootstrap_ci,
    canonical_set_display,
    canonical_set_key,
    jaccard_from_keys,
    json_list,
    load_config as load_rsp003_config,
    metric_dict,
    normalize_candidates,
    rank_variant,
    subset_metrics,
    to_builtin,
    write_candidate_table,
    write_json,
)


VARIANTS = ["rsp_v5_baseline", "base_plus_skeleton_prune"]


@dataclass(frozen=True)
class PruneConfig:
    output_dir: Path
    train_candidates: Path
    split_candidates: Path
    budgets: tuple[int, ...]
    seed_top_grid: tuple[int, ...]
    preserve_base_top_grid: tuple[int, ...]
    max_candidate_size: int
    prune_score_offset: float
    max_exact1_drop: float
    bootstrap_iterations: int
    ci: float
    seed: int
    raw_config_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research/configs/rsp_vnext_005_prune_recovery.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--budgets", default="50,200,500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} did not contain a mapping")
        return payload


def first_existing(paths: Iterable[str | Path]) -> Path:
    for raw in paths:
        path = Path(raw)
        if path.exists() and not path.name.startswith("._"):
            return path
    raise FileNotFoundError("No configured path exists: " + ", ".join(str(p) for p in paths))


def ints(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in values)


def load_config(config_path: str | Path, split: str, budgets_arg: str, seed: int) -> PruneConfig:
    raw = read_config(config_path)
    paths = raw.get("paths", {})
    pruning = raw.get("pruning", {})
    gate = raw.get("validation_gate", {})
    split_key = "validation_candidates" if split == "validation" else "test_candidates"
    budgets = tuple(int(x) for x in budgets_arg.split(",") if x.strip())
    if not budgets:
        budgets = ints(pruning.get("budgets", [50, 200, 500]))
    return PruneConfig(
        output_dir=Path(raw.get("output_dir", "outputs/autorun/rsp_vnext_005_prune_recovery")),
        train_candidates=first_existing(paths.get("train_candidates", [])),
        split_candidates=first_existing(paths.get(split_key, [])),
        budgets=budgets,
        seed_top_grid=ints(pruning.get("grid", {}).get("seed_top", [50, 100, 200])),
        preserve_base_top_grid=ints(pruning.get("grid", {}).get("preserve_base_top", [100, 200])),
        max_candidate_size=int(pruning.get("max_candidate_size", 8)),
        prune_score_offset=float(pruning.get("prune_score_offset", -0.02)),
        max_exact1_drop=float(gate.get("max_exact1_drop_pp", 0.5)) / 100.0,
        bootstrap_iterations=int(gate.get("bootstrap_iterations", 1000)),
        ci=float(gate.get("ci", 0.95)),
        seed=seed,
        raw_config_path=str(config_path),
    )


def train_label_counts(train_candidates: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for vals in train_candidates.get("true_precursors", []):
        for label in canonical_set_key(json_list(vals)).strip("[]").split(","):
            clean = label.strip().strip('"')
            if clean:
                counts[clean] += 1
    return counts


def attach_sample_flags(base: pd.DataFrame, counts: Counter[str]) -> pd.DataFrame:
    samples = base.sort_values(["sample_id", "rank"]).drop_duplicates("sample_id", keep="first")
    flags = []
    for _, row in samples.iterrows():
        labels = json_list(row.get("true_precursors"))
        canon = json.loads(canonical_set_key(labels))
        flags.append(
            {
                "sample_id": str(row["sample_id"]),
                "is_rare_reference": any(0 < counts.get(x, 0) <= 3 for x in canon),
                "is_oov_reference": any(counts.get(x, 0) == 0 for x in canon),
            }
        )
    flag_df = pd.DataFrame(flags).set_index("sample_id")
    out = base.drop(columns=["is_rare_reference", "is_oov_reference"], errors="ignore").join(flag_df, on="sample_id")
    out["is_rare_reference"] = out["is_rare_reference"].fillna(False).astype(bool)
    out["is_oov_reference"] = out["is_oov_reference"].fillna(False).astype(bool)
    return out


def canonicalize_base(base: pd.DataFrame) -> pd.DataFrame:
    out = normalize_candidates(base)
    out["sample_id"] = out["sample_id"].astype(str)
    out["true_key"] = out["true_precursors"].map(lambda x: canonical_set_key(json_list(x)))
    out["candidate_key"] = out["candidate_set"].map(lambda x: canonical_set_key(json_list(x)))
    out["candidate_set"] = out["candidate_set"].map(lambda x: canonical_set_display(json_list(x)))
    out["label_exact"] = (out["candidate_key"] == out["true_key"]).astype(int)
    out["jaccard_label"] = out.apply(lambda r: jaccard_from_keys(r["true_key"], r["candidate_key"]), axis=1)
    return out


def generate_prune_candidates(base: pd.DataFrame, seed_top: int, cfg: PruneConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seeds = base[base["rank"] <= seed_top].copy()
    for _, row in seeds.iterrows():
        labels = sorted(set(json_list(row.get("candidate_set"))))
        if len(labels) < 2 or len(labels) > cfg.max_candidate_size:
            continue
        for removed in labels:
            pruned = [x for x in labels if x != removed]
            candidate_key = canonical_set_key(pruned)
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "sample_index": row.get("sample_index", np.nan),
                    "id": row.get("id", row["sample_id"]),
                    "formula": row.get("formula", ""),
                    "reaction_method": row.get("reaction_method", ""),
                    "true_precursors": row.get("true_precursors", ""),
                    "pred_precursors": canonical_set_display(pruned),
                    "candidate_set": canonical_set_display(pruned),
                    "candidate_source": "skeleton_prune",
                    "source_group": "prune",
                    "rank": 10**9,
                    "total_score_v5": float(row.get("total_score_v5", 0.0)) + cfg.prune_score_offset - 0.001 * len(labels),
                    "removed_precursor": removed,
                    "prune_seed_rank": int(row.get("rank", 10**9)),
                    "true_key": row["true_key"],
                    "candidate_key": candidate_key,
                    "label_exact": int(candidate_key == row["true_key"]),
                    "jaccard_label": jaccard_from_keys(row["true_key"], candidate_key),
                    "is_rare_reference": bool(row.get("is_rare_reference", False)),
                    "is_oov_reference": bool(row.get("is_oov_reference", False)),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["sample_id", "candidate_key"], keep="first")


def source_ablation_rows(variant_name: str, ranked: pd.DataFrame, baseline_ranked: pd.DataFrame, budgets: Iterable[int]) -> list[dict[str, Any]]:
    rows = []
    for budget in budgets:
        base_hits = (
            baseline_ranked[baseline_ranked["rsp_rank_expanded"] <= budget]
            .groupby("sample_id")["label_exact"]
            .max()
            .rename("base_exact")
        )
        var_top = ranked[ranked["rsp_rank_expanded"] <= budget].copy()
        var_hits = var_top.groupby("sample_id")["label_exact"].max().rename("exact")
        joined = pd.concat([var_hits, base_hits], axis=1).fillna(0)
        new_ids = set(joined[(joined["exact"] > 0) & (joined["base_exact"] <= 0)].index.astype(str))
        exact_new = var_top[(var_top["sample_id"].astype(str).isin(new_ids)) & (var_top["label_exact"] > 0)]
        prune_hits = exact_new[exact_new["source_group"].astype(str).str.contains("prune", na=False)]
        rows.append(
            {
                "variant": variant_name,
                "budget": budget,
                "new_exact_hits": int(len(new_ids)),
                "new_exact_hits_from_prune": int(prune_hits["sample_id"].nunique()),
            }
        )
    return rows


def evaluate_gate(result: dict[str, Any], budgets: Iterable[int], max_exact1_drop: float) -> dict[str, Any]:
    base = result["variants"]["rsp_v5_baseline"]
    primary = result["variants"]["base_plus_skeleton_prune"]
    exact1_delta = primary.get("exact@1", 0.0) - base.get("exact@1", 0.0)
    oracle_deltas = {str(k): primary.get(f"skeleton_oracle@{k}", 0.0) - base.get(f"skeleton_oracle@{k}", 0.0) for k in budgets}
    exact_pass = exact1_delta >= -max_exact1_drop
    oracle_pass = any(v > 0 for v in oracle_deltas.values())
    passed = bool(exact_pass and oracle_pass)
    mode = "standard"
    if exact1_delta < 0 and oracle_pass:
        mode = "coverage_mode"
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "mode": mode if passed else "not_passed",
        "reason": f"max oracle delta={max(oracle_deltas.values()):.6f}; exact@1 delta={exact1_delta:.6f}",
        "exact1_delta": exact1_delta,
        "oracle_deltas": oracle_deltas,
    }


def select_params(base: pd.DataFrame, cfg: PruneConfig) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, dict[str, float]], dict[str, int]]:
    baseline = rank_variant(base, pd.DataFrame(), preserve_base_top=max(cfg.budgets), budgets=cfg.budgets)
    best_score = (-1e9, -1e9, -1e9)
    best_params: dict[str, Any] = {}
    best_ranked = {"rsp_v5_baseline": baseline}
    best_metrics = {"rsp_v5_baseline": metric_dict(baseline, cfg.budgets)}
    best_counts: dict[str, int] = {}
    for seed_top, preserve in itertools.product(cfg.seed_top_grid, cfg.preserve_base_top_grid):
        print(f"[rsp_prune] generating seed_top={seed_top} preserve={preserve}", flush=True)
        prune = generate_prune_candidates(base, seed_top, cfg)
        ranked = {
            "rsp_v5_baseline": baseline,
            "base_plus_skeleton_prune": rank_variant(base, prune, preserve, cfg.budgets),
        }
        metrics = {name: metric_dict(df, cfg.budgets) for name, df in ranked.items()}
        primary = metrics["base_plus_skeleton_prune"]
        base_m = metrics["rsp_v5_baseline"]
        deltas = {k: primary.get(f"exact@{k}", 0.0) - base_m.get(f"exact@{k}", 0.0) for k in cfg.budgets}
        exact1_delta = primary.get("exact@1", 0.0) - base_m.get("exact@1", 0.0)
        # Keep lower budgets stable first; then maximize the residual @500 recovery.
        stable_50 = deltas.get(50, 0.0)
        stable_200 = deltas.get(200, 0.0)
        score = (deltas.get(500, 0.0), exact1_delta, stable_200 + stable_50)
        print(
            "[rsp_prune] metrics "
            f"seed_top={seed_top} preserve={preserve} "
            f"delta50={stable_50:.6f} delta200={stable_200:.6f} delta500={deltas.get(500, 0.0):.6f}",
            flush=True,
        )
        if exact1_delta >= -cfg.max_exact1_drop and stable_50 >= 0 and stable_200 >= 0 and score > best_score:
            best_score = score
            best_params = {"seed_top": seed_top, "preserve_base_top": preserve}
            best_ranked = ranked
            best_metrics = metrics
            best_counts = {
                "prune_candidates": int(len(prune)),
                "prune_exact_candidate_rows": int(prune["label_exact"].sum()) if not prune.empty else 0,
            }
    if not best_params:
        raise RuntimeError("No pruning parameter setting satisfied the stability constraints")
    return best_params, best_ranked, best_metrics, best_counts


def evaluate_fixed_params(base: pd.DataFrame, cfg: PruneConfig, params: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]], dict[str, int]]:
    baseline = rank_variant(base, pd.DataFrame(), preserve_base_top=max(cfg.budgets), budgets=cfg.budgets)
    prune = generate_prune_candidates(base, int(params["seed_top"]), cfg)
    ranked = {
        "rsp_v5_baseline": baseline,
        "base_plus_skeleton_prune": rank_variant(base, prune, int(params["preserve_base_top"]), cfg.budgets),
    }
    metrics = {name: metric_dict(df, cfg.budgets) for name, df in ranked.items()}
    counts = {
        "prune_candidates": int(len(prune)),
        "prune_exact_candidate_rows": int(prune["label_exact"].sum()) if not prune.empty else 0,
    }
    return ranked, metrics, counts


def artifact_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": h.hexdigest(),
                "description": path.name,
                "primary_result": path.name in {"metrics.json", "RSP_PRUNE_RECOVERY_REPORT.md"},
                "diagnostic_result": path.suffix in {".csv", ".json"} or "REPORT" in path.name,
                "reproducible": True,
            }
        )
    return rows


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RSP vnext 005 Skeleton Pruning Recovery Report",
        "",
        f"Created at: {result['created_at']}",
        f"Split: {result['split']}",
        f"Selector update: {result['selector_update_status']}",
        f"Validation gate: {result['validation_gate']['status']}",
        "",
        "## Hypothesis",
        "",
        "Some RSP v5 skeleton misses are near misses where the candidate set contains all reference precursors plus one spurious precursor. Leave-one-out pruning can recover these exact sets under the same K=50/200/500 budget.",
        "",
        "## Selected Parameters",
        "",
        "```json",
        json.dumps(result["selected_params"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
    ]
    for name, metrics in result["variants"].items():
        lines.append(f"### {name}")
        lines.append("")
        for key in sorted(metrics):
            val = metrics[key]
            if isinstance(val, (int, float)):
                lines.append(f"- `{key}`: {val:.6f}")
        lines.append("")
    lines.extend(
        [
            "## Gate Decision",
            "",
            f"- Passed: `{result['validation_gate']['passed']}`",
            f"- Mode: `{result['validation_gate']['mode']}`",
            f"- Reason: {result['validation_gate']['reason']}",
            "",
            "## Test Policy",
            "",
            "Test is run only after validation gate passes. The default inference selector remains unchanged.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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

    print(f"[rsp_prune] loading split candidates: {cfg.split_candidates}", flush=True)
    base = canonicalize_base(pd.read_csv(cfg.split_candidates))
    counts = train_label_counts(pd.read_csv(cfg.train_candidates))
    base = attach_sample_flags(base, counts)
    print(f"[rsp_prune] base rows={len(base)} samples={base['sample_id'].nunique()}", flush=True)

    if args.split == "test":
        assert validation_result is not None
        ranked_by_variant, metrics_by_variant, generated_counts = evaluate_fixed_params(base, cfg, validation_result["selected_params"])
    else:
        params, ranked_by_variant, metrics_by_variant, generated_counts = select_params(base, cfg)

    if args.split == "test":
        params = validation_result["selected_params"] if validation_result else {}
    else:
        params = params

    baseline = ranked_by_variant["rsp_v5_baseline"]
    primary = ranked_by_variant["base_plus_skeleton_prune"]
    source_rows = source_ablation_rows("base_plus_skeleton_prune", primary, baseline, cfg.budgets)
    rare_rows: list[dict[str, Any]] = []
    for name, ranked in ranked_by_variant.items():
        rare_rows.append({"variant": name, "subset": "rare_precursor", **subset_metrics(ranked, "is_rare_reference", cfg.budgets)})
        rare_rows.append({"variant": name, "subset": "oov_precursor", **subset_metrics(ranked, "is_oov_reference", cfg.budgets)})

    print("[rsp_prune] running paired bootstrap", flush=True)
    bootstrap = bootstrap_ci(baseline, primary, cfg.budgets, cfg.bootstrap_iterations, cfg.ci, cfg.seed)
    result: dict[str, Any] = {
        "run_id": "rsp_vnext_005_prune_recovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "config": cfg.raw_config_path,
        "input_artifacts": {
            "split_candidates": str(cfg.split_candidates),
            "train_candidates": str(cfg.train_candidates),
        },
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
        result["validation_gate"] = evaluate_gate(result, cfg.budgets, cfg.max_exact1_drop)
    else:
        result["validation_gate"] = validation_result.get("validation_gate", {}) if validation_result else {}

    prefix = "val" if args.split == "validation" else "test"
    candidate_path = outdir / f"{prefix}_rsp_prune_candidates.parquet"
    result["candidate_table"] = write_candidate_table(candidate_path, primary)
    source_name = "candidate_source_ablation.csv" if args.split == "validation" else "test_candidate_source_ablation.csv"
    rare_name = "rare_precursor_analysis.csv" if args.split == "validation" else "test_rare_precursor_analysis.csv"
    metrics_name = "metrics.json" if args.split == "validation" else "test_metrics_if_gate_passed.json"
    report_name = "RSP_PRUNE_RECOVERY_REPORT.md" if args.split == "validation" else "TEST_RSP_PRUNE_RECOVERY_REPORT.md"
    pd.DataFrame(source_rows).to_csv(outdir / source_name, index=False)
    pd.DataFrame(rare_rows).to_csv(outdir / rare_name, index=False)
    write_json(outdir / metrics_name, result)
    write_report(outdir / report_name, result)
    manifest_paths = sorted(p for p in outdir.iterdir() if p.is_file() and p.name != "artifact_manifest.json")
    write_json(outdir / "artifact_manifest.json", artifact_manifest(manifest_paths))
    print(json.dumps({"output_dir": str(outdir), "validation_gate": result["validation_gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
