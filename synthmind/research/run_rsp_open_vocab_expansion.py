#!/usr/bin/env python3
"""RSP vnext 004: open-vocabulary auxiliary candidate expansion.

This experiment keeps the RSP v5 candidate budget fixed and does not train a
reranker.  It targets the failure mode where the base precursor skeleton is
nearly correct but the reference route contains a low-frequency auxiliary,
solvent, oxidizer, deuterated reagent, or other method-specific reagent that is
outside the train precursor-label vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthmind.research.run_rsp_expansion import (
    VARIANTS,
    bootstrap_ci,
    canonical_label,
    canonical_set_display,
    canonical_set_key,
    evaluate_gate,
    formula_elements,
    infer_target_elements,
    json_list,
    load_config,
    metric_dict,
    normalize_candidates,
    per_sample_hits,
    rank_variant,
    read_json_or_yaml_subset,
    source_ablation_rows,
    subset_metrics,
    write_report,
)


OPEN_VARIANTS = [
    "rsp_v5_baseline",
    "base_plus_open_aux",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="research/configs/rsp_vnext_004_open_vocab.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--budgets", default="50,200,500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def sample_frame_from_base(base: pd.DataFrame, train_counts: Counter[str]) -> pd.DataFrame:
    samples = base.sort_values(["sample_id", "rank"]).drop_duplicates("sample_id", keep="first").copy()
    keep = [
        "sample_id",
        "sample_index",
        "id",
        "formula",
        "reaction_method",
        "true_precursors",
    ]
    samples = samples[[c for c in keep if c in samples.columns]].copy()
    samples["true_key"] = samples["true_precursors"].map(lambda x: canonical_set_key(json_list(x)))
    samples["target_elements"] = samples.apply(lambda row: infer_target_elements(row), axis=1)
    rare_flags: list[bool] = []
    oov_flags: list[bool] = []
    for vals in samples["true_precursors"].map(json_list):
        canon = [canonical_label(x) for x in vals]
        rare_flags.append(any(0 < train_counts.get(x, 0) <= 3 for x in canon))
        oov_flags.append(any(train_counts.get(x, 0) == 0 for x in canon))
    samples["is_rare_reference"] = rare_flags
    samples["is_oov_reference"] = oov_flags
    return samples


def label_is_auxiliary(label: str, target_elements: set[str]) -> bool:
    elems = formula_elements(label)
    if not elems:
        return False
    if elems.isdisjoint(target_elements):
        return True
    light = {"H", "C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "B", "D"}
    return elems.issubset(light)


def build_auxiliary_bank(train_candidates: pd.DataFrame, min_count: int) -> tuple[dict[str, Counter[str]], Counter[str], Counter[str]]:
    if "sample_id" not in train_candidates.columns:
        train_candidates["sample_id"] = train_candidates.get("id", train_candidates.get("sample_index")).astype(str)
    rank_col = "rank" if "rank" in train_candidates.columns else "precursor_rank" if "precursor_rank" in train_candidates.columns else None
    sort_cols = ["sample_id"] + ([rank_col] if rank_col else [])
    samples = train_candidates.sort_values(sort_cols).drop_duplicates("sample_id", keep="first").copy()
    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    global_aux: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    for _, row in samples.iterrows():
        method = str(row.get("reaction_method", "")).strip()
        target_elements = infer_target_elements(row)
        labels = [canonical_label(x) for x in json_list(row.get("true_precursors"))]
        for label in labels:
            if not label:
                continue
            train_counts[label] += 1
            if label_is_auxiliary(label, target_elements):
                by_method[method][label] += 1
                global_aux[label] += 1
    by_method = {k: Counter({p: c for p, c in v.items() if c >= min_count}) for k, v in by_method.items()}
    global_aux = Counter({p: c for p, c in global_aux.items() if c >= min_count})
    return by_method, global_aux, train_counts


def open_choices_for_sample(
    method: str,
    by_method: dict[str, Counter[str]],
    global_aux: Counter[str],
    universal_auxiliaries: list[str],
    *,
    method_top: int,
    global_top: int,
) -> list[tuple[str, float, str]]:
    choices: list[tuple[str, float, str]] = []
    for label, count in by_method.get(method, Counter()).most_common(method_top):
        choices.append((label, 1.0 + float(np.log1p(count)), "method_aux"))
    seen = {x[0] for x in choices}
    for label, count in global_aux.most_common(global_top):
        if label not in seen:
            choices.append((label, 0.7 + float(np.log1p(count)), "global_aux"))
            seen.add(label)
    for idx, label in enumerate(universal_auxiliaries):
        canon = canonical_label(label)
        if canon and canon not in seen:
            choices.append((canon, 0.45 - 0.005 * idx, "universal_aux"))
            seen.add(canon)
    return choices


def generate_open_aux_candidates(
    base: pd.DataFrame,
    samples: pd.DataFrame,
    by_method: dict[str, Counter[str]],
    global_aux: Counter[str],
    universal_auxiliaries: list[str],
    *,
    base_seed_top: int,
    min_sample_oov_risk: float,
    method_top: int,
    global_top: int,
    max_per_sample: int,
    max_candidate_size: int,
) -> pd.DataFrame:
    sample_map = samples.set_index("sample_id").to_dict(orient="index")
    seed_rows = base[base["rank"] <= base_seed_top].copy()
    rows: list[dict[str, Any]] = []
    for sample_id, group in seed_rows.groupby("sample_id", sort=False):
        sample = sample_map.get(str(sample_id))
        if sample is None:
            continue
        if min_sample_oov_risk > 0:
            risk = pd.to_numeric(group.get("oov_risk_score", pd.Series(0.0, index=group.index)), errors="coerce").fillna(0.0)
            if float(risk.max()) < min_sample_oov_risk:
                continue
        method = str(sample.get("reaction_method", "")).strip()
        choices = open_choices_for_sample(
            method,
            by_method,
            global_aux,
            universal_auxiliaries,
            method_top=method_top,
            global_top=global_top,
        )
        if not choices:
            continue
        generated = 0
        true_key = str(sample["true_key"])
        for _, seed in group.sort_values("rank").iterrows():
            base_labels = [canonical_label(x) for x in json_list(seed.get("candidate_set"))]
            if not base_labels or len(base_labels) >= max_candidate_size:
                continue
            base_key = canonical_set_key(base_labels)
            base_score = float(seed.get("total_score_v5", 0.0))
            for aux, aux_score, aux_source in choices:
                if aux in base_labels:
                    continue
                labels = base_labels + [aux]
                if len(labels) > max_candidate_size:
                    continue
                candidate_key = canonical_set_key(labels)
                if candidate_key == base_key:
                    continue
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "sample_index": sample.get("sample_index", np.nan),
                        "id": sample.get("id", sample_id),
                        "formula": sample.get("formula", ""),
                        "reaction_method": method,
                        "true_precursors": sample.get("true_precursors", ""),
                        "pred_precursors": canonical_set_display(labels),
                        "candidate_set": canonical_set_display(labels),
                        "candidate_source": f"open_aux:{aux_source}",
                        "source_group": "open",
                        "rank": 10**9,
                        "total_score_v5": base_score + 0.03 * aux_score - 0.01 * len(labels),
                        "family_score": np.nan,
                        "open_vocab_score": aux_score,
                        "element_coverage": seed.get("element_coverage", np.nan),
                        "missing_element_count": seed.get("missing_element_count", np.nan),
                        "extra_element_count": seed.get("extra_element_count", np.nan),
                        "candidate_size": float(len(labels)),
                        "target_elements": sample.get("target_elements", set()),
                        "candidate_elements": formula_elements("".join(labels)),
                        "true_key": true_key,
                        "candidate_key": candidate_key,
                        "label_exact": int(candidate_key == true_key),
                        "jaccard_label": 0.0,
                        "is_rare_reference": bool(sample.get("is_rare_reference", False)),
                        "is_oov_reference": bool(sample.get("is_oov_reference", False)),
                    }
                )
                generated += 1
                if generated >= max_per_sample:
                    break
            if generated >= max_per_sample:
                break
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    true_sets = out["true_key"].map(json.loads)
    cand_sets = out["candidate_key"].map(json.loads)
    out["jaccard_label"] = [
        len(set(a) & set(b)) / len(set(a) | set(b)) if set(a) or set(b) else 1.0
        for a, b in zip(true_sets, cand_sets)
    ]
    return out


def open_source_rows(ranked: pd.DataFrame, baseline_ranked: pd.DataFrame, budgets: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        base_ps = per_sample_hits(baseline_ranked, budget).set_index("sample_id")
        var_ps = per_sample_hits(ranked, budget).set_index("sample_id")
        joined = var_ps.join(base_ps[["exact"]].rename(columns={"exact": "base_exact"}), how="left").fillna(0)
        new_ids = set(joined[(joined["exact"] > 0) & (joined["base_exact"] <= 0)].index.astype(str))
        sub = ranked[(ranked["rsp_rank_expanded"] <= budget) & (ranked["sample_id"].astype(str).isin(new_ids))]
        exact = sub[sub["label_exact"] > 0].copy()
        source_counts = exact["candidate_source"].astype(str).value_counts().to_dict() if len(exact) else {}
        rows.append(
            {
                "variant": "base_plus_open_aux",
                "budget": budget,
                "new_exact_hits": len(new_ids),
                "new_exact_hits_from_open": int(sum(source_counts.values())),
                "open_source_breakdown": json.dumps(source_counts, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def write_artifact_manifest(outdir: Path, files: list[Path]) -> None:
    rows = []
    for path in files:
        if not path.exists():
            continue
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "description": path.name,
                "primary_result": path.name in {"metrics.json", "RSP_OPEN_VOCAB_REPORT.md"},
                "diagnostic_result": "analysis" in path.name or "ablation" in path.name,
                "reproducible": True,
            }
        )
    (outdir / "artifact_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    raw = read_json_or_yaml_subset(Path(args.config))
    base_config = raw.get("base_config", "research/configs/rsp_vnext_003.yaml")
    cfg = load_config(base_config, args.split, args.budgets, args.seed)
    open_cfg = raw.get("open_vocab", {})
    outdir = Path(raw.get("output_dir", "outputs/autorun/rsp_vnext_004_open_vocab_expansion"))
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[rsp_open_vocab] loading train candidates: {cfg.train_candidates}", flush=True)
    train_candidates = pd.read_csv(cfg.train_candidates)
    by_method, global_aux, train_counts = build_auxiliary_bank(
        train_candidates, int(open_cfg.get("min_aux_train_count", 1))
    )
    print(f"[rsp_open_vocab] method_aux_methods={len(by_method)} global_aux={len(global_aux)}", flush=True)

    print(f"[rsp_open_vocab] loading split candidates: {cfg.split_candidates}", flush=True)
    base = normalize_candidates(pd.read_csv(cfg.split_candidates))
    samples = sample_frame_from_base(base, train_counts)
    sample_flags = samples.set_index("sample_id")[["is_rare_reference", "is_oov_reference"]]
    base = base.drop(columns=["is_rare_reference", "is_oov_reference"], errors="ignore").join(sample_flags, on="sample_id")
    base["is_rare_reference"] = base["is_rare_reference"].fillna(False).astype(bool)
    base["is_oov_reference"] = base["is_oov_reference"].fillna(False).astype(bool)

    budgets = list(cfg.budgets)
    baseline = rank_variant(base, pd.DataFrame(), preserve_base_top=max(budgets), budgets=budgets)
    open_exp = generate_open_aux_candidates(
        base,
        samples,
        by_method,
        global_aux,
        [str(x) for x in open_cfg.get("universal_auxiliaries", [])],
        base_seed_top=int(open_cfg.get("base_seed_top", 30)),
        min_sample_oov_risk=float(open_cfg.get("min_sample_oov_risk", 0.0)),
        method_top=int(open_cfg.get("method_aux_top", 8)),
        global_top=int(open_cfg.get("global_aux_top", 8)),
        max_per_sample=int(open_cfg.get("max_open_generated_per_sample", 40)),
        max_candidate_size=int(open_cfg.get("max_candidate_size", 9)),
    )
    print(f"[rsp_open_vocab] open_aux candidates={len(open_exp)}", flush=True)
    open_ranked = rank_variant(
        base,
        open_exp,
        preserve_base_top=int(open_cfg.get("preserve_base_top", 20)),
        budgets=budgets,
    )
    metrics = {
        "rsp_v5_baseline": metric_dict(baseline, budgets),
        "base_plus_open_aux": metric_dict(open_ranked, budgets),
    }
    result = {
        "run_id": raw.get("run_id", "rsp_vnext_004_open_vocab_expansion"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "config": str(args.config),
        "budgets": budgets,
        "selected_params": open_cfg,
        "generated_candidate_counts": {"open_aux_candidates": int(len(open_exp))},
        "variants": metrics,
        "selector_update_status": "unchanged",
        "input_artifacts": {
            "train_candidates": str(cfg.train_candidates),
            "split_candidates": str(cfg.split_candidates),
        },
    }
    # Reuse the existing gate by mapping the open candidate to the primary name.
    gate_payload = {
        "variants": {
            "rsp_v5_baseline": metrics["rsp_v5_baseline"],
            "base_plus_family_rare_chemistry": metrics["base_plus_open_aux"],
        }
    }
    gate = evaluate_gate(gate_payload, budgets, float(open_cfg.get("max_exact1_drop_pp", 0.5)) / 100.0)
    result["validation_gate"] = gate
    result["bootstrap_ci"] = bootstrap_ci(
        baseline,
        open_ranked,
        budgets,
        int(open_cfg.get("bootstrap_iterations", 1000)),
        0.95,
        args.seed,
    )
    subset_rows = []
    for name, ranked in [("rsp_v5_baseline", baseline), ("base_plus_open_aux", open_ranked)]:
        subset_rows.append({"variant": name, "subset": "rare_precursor", **subset_metrics(ranked, "is_rare_reference", budgets)})
        subset_rows.append({"variant": name, "subset": "oov_precursor", **subset_metrics(ranked, "is_oov_reference", budgets)})
    result["subset_metrics"] = subset_rows
    source_rows = source_ablation_rows("base_plus_open_aux", open_ranked, baseline, budgets)
    source_rows.extend(open_source_rows(open_ranked, baseline, budgets))

    metrics_name = "metrics.json" if args.split == "validation" else "test_metrics_if_gate_passed.json"
    candidate_name = (
        "val_rsp_open_vocab_candidates.parquet" if args.split == "validation" else "test_rsp_open_vocab_candidates.parquet"
    )
    source_name = "candidate_source_ablation.csv" if args.split == "validation" else "test_candidate_source_ablation.csv"
    rare_name = "rare_precursor_analysis.csv" if args.split == "validation" else "test_rare_precursor_analysis.csv"
    report_name = "RSP_OPEN_VOCAB_REPORT.md" if args.split == "validation" else "TEST_RSP_OPEN_VOCAB_REPORT.md"

    (outdir / metrics_name).write_text(json.dumps(to_builtin(result), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    pd.DataFrame(source_rows).to_csv(outdir / source_name, index=False)
    pd.DataFrame(subset_rows).to_csv(outdir / rare_name, index=False)
    try:
        open_ranked.to_parquet(outdir / candidate_name, index=False)
    except Exception as exc:
        csv_path = outdir / f"{candidate_name}.csv"
        open_ranked.to_csv(csv_path, index=False)
        (outdir / candidate_name).write_text(
            f"Parquet engine unavailable; see CSV fallback: {csv_path.name}\n{exc!r}\n",
            encoding="utf-8",
        )
    report_lines = [
        "# RSP vnext 004 Open-Vocabulary Candidate Expansion",
        "",
        f"- Split: `{args.split}`",
        "- Selector update: unchanged",
        f"- Open auxiliary candidates: {len(open_exp)}",
        f"- Validation gate: {gate.get('status')}",
        "",
        "## Metrics",
        "",
    ]
    for variant, vals in metrics.items():
        report_lines.append(f"### {variant}")
        report_lines.append("")
        for key in sorted(vals):
            report_lines.append(f"- `{key}`: {float(vals[key]):.6f}")
        report_lines.append("")
    report_lines += [
        "## Gate",
        "",
        "```json",
        json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    (outdir / report_name).write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_artifact_manifest(
        outdir,
        [
            outdir / metrics_name,
            outdir / candidate_name,
            outdir / source_name,
            outdir / rare_name,
            outdir / report_name,
        ],
    )
    print(json.dumps(to_builtin(gate), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
