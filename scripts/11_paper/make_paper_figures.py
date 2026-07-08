#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_DIR = SCRIPT_DIR.parent / "08_auto_improve"
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from metrics_registry import build_registry  # noqa: E402


def pct(x: Any) -> float:
    try:
        return 100 * float(x)
    except Exception:
        return 0.0


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create SynPred paper figures.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/autorun/24h_optimization_20260613/07_figures")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    reg = build_registry(root, out.parent.parent, include_experiments=True).records["baselines"]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})

    # Figure 1
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    steps = ["Structure /\ncomposition", "Stage2\nprecursors", "Chemistry\ncheck", "Stage3\nconditions", "Stage35\nroute ranking", "Recommended\nroutes"]
    for i, label in enumerate(steps):
        ax.text(i, 0.5, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#eef3f8", ec="#334"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.42, 0.5), xytext=(i + 0.58, 0.5), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.set_ylim(0, 1)
    pd.DataFrame({"step": steps}).to_csv(out / "fig1_pipeline_schematic_source.csv", index=False)
    save(fig, out, "fig1_pipeline_schematic")

    # Figure 2
    target_dir = root / "data/interim/generative/stage3_condition_targets_v3_20260610"
    rows = []
    for split in ["train", "val", "test"]:
        p = target_dir / f"{split}.csv"
        if p.exists():
            df = pd.read_csv(p)
            for method, n in df.get("reaction_method", pd.Series(dtype=str)).value_counts().items():
                rows.append({"split": split, "reaction_method": method, "n": n})
    dist = pd.DataFrame(rows)
    dist.to_csv(out / "fig2_dataset_distribution_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not dist.empty:
        pivot = dist.pivot_table(index="reaction_method", columns="split", values="n", fill_value=0)
        pivot.plot(kind="bar", stacked=False, ax=ax)
    ax.set_ylabel("Samples")
    ax.set_title("Dataset Distribution by Reaction Method")
    save(fig, out, "fig2_dataset_distribution")

    # Figure 3
    stage2 = reg["stage2_v5_all_test"]["metrics"]
    core = reg["stage2_core_calibrated_test"]["metrics"]
    ks = [1, 10, 50, 100, 200, 500]
    rec_rows = []
    for model, m in [("v5 all-method", stage2), ("core final", core)]:
        for k in ks:
            rec_rows.append({"model": model, "K": k, "topK_exact_pct": pct(m.get(f"top{k}_exact"))})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(out / "fig3_stage2_topk_recall_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, g in rec.groupby("model"):
        ax.plot(g["K"], g["topK_exact_pct"], marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("Exact recall (%)")
    ax.legend()
    save(fig, out, "fig3_stage2_topk_recall")

    # Figure 4
    rows = []
    for model, miss, strict in [
        ("v3 final", reg["stage3_v3_missing_aware_test"]["metrics"], reg["stage3_v3_strict_comparable_test"]["metrics"]),
        ("v4 alignment", reg["stage3_v4_missing_aware_test"]["metrics"], reg["stage3_v4_strict_comparable_test"]["metrics"]),
    ]:
        rows += [
            {"model": model, "metric": "missing top1 relaxed", "value": pct(miss.get("top1_relaxed_condition"))},
            {"model": model, "metric": "missing top10 relaxed", "value": pct(miss.get("top10_relaxed_condition"))},
            {"model": model, "metric": "strict top1 relaxed", "value": pct(strict.get("top1_relaxed_condition"))},
            {"model": model, "metric": "strict top10 relaxed", "value": pct(strict.get("top10_relaxed_condition"))},
        ]
    s3 = pd.DataFrame(rows)
    s3.to_csv(out / "fig4_stage3_condition_metrics_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s3.pivot(index="metric", columns="model", values="value").plot(kind="bar", ax=ax)
    ax.set_ylabel("Success (%)")
    save(fig, out, "fig4_stage3_condition_metrics")

    # Figure 5
    rows = []
    for model, miss, strict in [
        ("v3 final", reg["stage35_v3_final_missing_aware_test"]["metrics"], reg["stage35_v3_final_strict_comparable_test"]["metrics"]),
        ("v4 alignment", reg["stage35_v4_missing_aware_test"]["metrics"], reg["stage35_v4_strict_comparable_test"]["metrics"]),
    ]:
        rows += [
            {"model": model, "metric": "missing top1", "value": pct(miss.get("top1_relaxed_route"))},
            {"model": model, "metric": "missing top10", "value": pct(miss.get("top10_relaxed_route"))},
            {"model": model, "metric": "strict top1", "value": pct(strict.get("top1_relaxed_route"))},
            {"model": model, "metric": "strict top10", "value": pct(strict.get("top10_relaxed_route"))},
        ]
    route = pd.DataFrame(rows)
    route.to_csv(out / "fig5_stage35_route_success_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    route.pivot(index="metric", columns="model", values="value").plot(kind="bar", ax=ax)
    ax.set_ylabel("Route success (%)")
    save(fig, out, "fig5_stage35_route_success")

    # Figure 6
    coupling = root / "outputs/autorun/24h_optimization_20260613/02_stage2_diagnosis/stage2_stage3_coupling_diagnosis.json"
    rows = []
    if coupling.exists():
        obj = json.loads(coupling.read_text())
        rows = obj.get("bucket_counts", [])
    err = pd.DataFrame(rows)
    err.to_csv(out / "fig6_error_decomposition_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not err.empty:
        err.plot(kind="bar", x="bucket", y="fraction", ax=ax, legend=False)
        ax.set_ylabel("Fraction")
    save(fig, out, "fig6_error_decomposition")

    # Figure 7
    by = pd.DataFrame(reg["stage2_v5_by_reaction_method"].get("records", []))
    by.to_csv(out / "fig7_core_vs_all_methods_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not by.empty:
        show = by[by["reaction_method"].isin(["solid_state", "solution", "melt_arc", "other", "hydro_solvothermal"])]
        show.plot(kind="bar", x="reaction_method", y=["top1_exact", "top10_exact", "top500_exact"], ax=ax)
        ax.set_ylabel("Rate")
    save(fig, out, "fig7_core_vs_all_methods")

    print(json.dumps({"figures_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

