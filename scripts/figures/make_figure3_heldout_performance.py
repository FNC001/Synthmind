#!/usr/bin/env python3
"""Create main-text Figure 3 for held-out Synthmind performance.

The script intentionally reads existing test artifacts instead of embedding
performance values in the plotting code. If a required Top10 metric is absent,
it stops and writes a blocker report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


RSP_COLOR = "#2166AC"
CDG_COLOR = "#1F9E89"
GRV_COLOR = "#0B7285"
STRICT_COLOR = "#6BAED6"
SUCCESS_COLOR = "#74C476"
RSP_MISS_COLOR = "#2166AC"
CDG_MISS_COLOR = "#1F9E89"
GRV_MISS_COLOR = "#3B6FB6"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#172033"

TOPK_DEFAULT = [1, 10, 50]
TOLERANCE_PP = 0.05


SANITY_CHECKS = {
    ("RSP", "precursor_set_exact", 1): 35.89,
    ("RSP", "precursor_set_exact", 10): 59.74,
    ("RSP", "precursor_set_exact", 50): 71.51,
    ("CDG", "missing_aware_relaxed_condition", 1): 54.56,
    ("CDG", "missing_aware_relaxed_condition", 50): 76.01,
    ("CDG", "strict_comparable_relaxed_condition", 1): 24.84,
    ("CDG", "strict_comparable_relaxed_condition", 50): 39.57,
    ("GRV", "missing_aware_relaxed_route", 1): 21.15,
    ("GRV", "missing_aware_relaxed_route", 50): 41.46,
    ("GRV", "strict_comparable_relaxed_route", 1): 10.32,
    ("GRV", "strict_comparable_relaxed_route", 50): 22.95,
}


ERROR_ATTRIBUTION = [
    {
        "category": "success",
        "sample_count": 64,
        "percent": 10.13,
        "color": SUCCESS_COLOR,
        "notes": "Complete route succeeds under the Figure 3 error-attribution protocol.",
    },
    {
        "category": "RSP skeleton miss",
        "sample_count": 352,
        "percent": 55.70,
        "color": RSP_MISS_COLOR,
        "notes": "Correct precursor skeleton is not recovered in the upstream RSP candidate pool.",
    },
    {
        "category": "CDG condition miss",
        "sample_count": 45,
        "percent": 7.12,
        "color": CDG_MISS_COLOR,
        "notes": "Precursor skeleton is available, but no compatible condition tuple is found.",
    },
    {
        "category": "GRV ranking miss",
        "sample_count": 171,
        "percent": 27.06,
        "color": GRV_MISS_COLOR,
        "notes": "A usable route exists in the candidate space but is not ranked to Top1.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=Path("outputs"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/figures/figure3"))
    parser.add_argument("--topk", default="1,10,50")
    parser.add_argument("--style", default="ncs_blue_teal")
    parser.add_argument("--strict-data-check", action="store_true")
    return parser.parse_args()


def setup_matplotlib():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
        style_name = "scienceplots: science,no-latex"
    except Exception:
        style_name = "scienceplots-compatible fallback"

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.8,
            "legend.title_fontsize": 6.8,
            "figure.titlesize": 10.0,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#4B5563",
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.75,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#D1D5DB",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )
    return plt, style_name


def pct(value: float) -> float:
    return 100.0 * float(value)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def metric_value(df: pd.DataFrame, module: str, metric: str, top_k: int, data_source: Path) -> dict[str, Any]:
    part = df[(df["metric"] == metric) & (df["topk"] == top_k)]
    if part.empty:
        raise KeyError(f"Missing {module} metric={metric} topk={top_k} in {data_source}")
    row = part.iloc[0]
    return {
        "module": module,
        "metric": metric,
        "top_k": top_k,
        "value_percent": pct(row["value"]),
        "split": row.get("split", "test"),
        "model_name": row.get("artifact", ""),
        "sample_count": int(row.get("n_samples", 0)) if not pd.isna(row.get("n_samples", 0)) else "",
        "candidate_pool": row.get("candidate_pool", ""),
        "data_source": str(data_source),
    }


def blocker(outdir: Path, missing: list[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    text = [
        "# BLOCKER_FIGURE3_TOP10_DATA",
        "",
        "Figure 3 was not generated because required Top10 data were missing.",
        "",
        "Missing data items:",
    ]
    text.extend(f"- {item}" for item in missing)
    (outdir / "BLOCKER_FIGURE3_TOP10_DATA.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def require_topk(df: pd.DataFrame, metrics: list[str], topk: list[int], source: Path) -> list[str]:
    missing = []
    for metric in metrics:
        for k in topk:
            if df[(df["metric"] == metric) & (df["topk"] == k)].empty:
                missing.append(f"{source}: metric={metric}, top_k={k}")
    return missing


def source_row(
    *,
    panel: str,
    module: str,
    task: str,
    protocol: str,
    metric: str,
    top_k: int | str,
    value_percent: float,
    split: str,
    model_name: str,
    candidate_pool: str,
    sample_count: int | str,
    data_source: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "figure_panel": panel,
        "module": module,
        "task": task,
        "protocol": protocol,
        "metric": metric,
        "top_k": top_k,
        "value_percent": value_percent,
        "lower_ci": "",
        "upper_ci": "",
        "split": split,
        "model_name": model_name,
        "candidate_pool": candidate_pool,
        "sample_count": sample_count,
        "data_source": data_source,
        "notes": notes,
    }


def build_source_data(metrics_root: Path, outdir: Path, topk: list[int], strict: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    base = metrics_root / "autorun" / "final_precision_topk_20260624"
    rsp_path = base / "stage2_rsp_topk_test.csv"
    cdg_path = base / "stage3_cdg_topk_test.csv"
    grv_path = base / "stage35_final_v005_grv14_topk_test.csv"
    ranker_path = base / "model_evidence_scienceplots" / "grv_fig2_same_pool_ranker_source.csv"

    rsp = read_csv(rsp_path)
    cdg = read_csv(cdg_path)
    grv = read_csv(grv_path)
    ranker = read_csv(ranker_path) if ranker_path.exists() else pd.DataFrame()

    required = []
    required += require_topk(rsp, ["precursor_set_exact", "best_jaccard"], topk, rsp_path)
    required += require_topk(cdg, ["missing_aware_relaxed_condition", "strict_comparable_relaxed_condition"], topk, cdg_path)
    required += require_topk(grv, ["missing_aware_relaxed_route", "strict_comparable_relaxed_route"], topk, grv_path)
    if required:
        blocker(outdir, required)
        if strict:
            raise RuntimeError("Missing required Figure 3 TopK data. See BLOCKER_FIGURE3_TOP10_DATA.md")

    rows: list[dict[str, Any]] = []

    # Panel a: task-level comparison.
    panel_a_specs = [
        (rsp, rsp_path, "RSP", "precursor-set prediction", "exact set match", "precursor_set_exact", "RSP-Promote", "RSP v006 prune promotion"),
        (cdg, cdg_path, "CDG", "condition-tuple prediction", "missing-aware relaxed", "missing_aware_relaxed_condition", "CDG-Joint", "Joint tuple: temperature + time + atmosphere + solvent"),
        (grv, grv_path, "GRV", "complete-route prediction", "missing-aware relaxed", "missing_aware_relaxed_route", "GRV-Strict-II", "Complete precursor + condition + method route"),
    ]
    for df, path, module, task, protocol, metric, model_label, notes in panel_a_specs:
        for k in topk:
            val = metric_value(df, module, metric, k, path)
            rows.append(
                source_row(
                    panel="a",
                    module=module,
                    task=task,
                    protocol=protocol,
                    metric=metric,
                    top_k=k,
                    value_percent=val["value_percent"],
                    split=val["split"],
                    model_name=model_label,
                    candidate_pool=val.get("candidate_pool", ""),
                    sample_count=val["sample_count"],
                    data_source=str(path),
                    notes=notes,
                )
            )

    # Panel b.
    for metric, protocol, notes in [
        ("precursor_set_exact", "exact precursor set", "K fixed during evaluation"),
        ("best_jaccard", "best Jaccard overlap", "Best overlap within fixed RSP budget"),
    ]:
        for k in topk:
            val = metric_value(rsp, "RSP", metric, k, rsp_path)
            rows.append(
                source_row(
                    panel="b",
                    module="RSP",
                    task="fixed-budget precursor proposal",
                    protocol=protocol,
                    metric=metric,
                    top_k=k,
                    value_percent=val["value_percent"],
                    split=val["split"],
                    model_name="RSP-Promote",
                    candidate_pool="fixed TopK budget",
                    sample_count=val["sample_count"],
                    data_source=str(rsp_path),
                    notes=notes,
                )
            )

    # Panel c.
    for metric, protocol in [
        ("missing_aware_relaxed_condition", "missing-aware relaxed"),
        ("strict_comparable_relaxed_condition", "strict-comparable relaxed"),
    ]:
        for k in topk:
            val = metric_value(cdg, "CDG", metric, k, cdg_path)
            rows.append(
                source_row(
                    panel="c",
                    module="CDG",
                    task="joint condition-tuple prediction",
                    protocol=protocol,
                    metric=metric,
                    top_k=k,
                    value_percent=val["value_percent"],
                    split=val["split"],
                    model_name="CDG-Joint",
                    candidate_pool="condition tuple candidates",
                    sample_count=val["sample_count"],
                    data_source=str(cdg_path),
                    notes="Joint tuple: temperature + time + atmosphere + solvent",
                )
            )

    # Panel d.
    for metric, protocol in [
        ("missing_aware_relaxed_route", "missing-aware relaxed"),
        ("strict_comparable_relaxed_route", "strict-comparable relaxed"),
    ]:
        for k in topk:
            val = metric_value(grv, "GRV", metric, k, grv_path)
            rows.append(
                source_row(
                    panel="d",
                    module="GRV",
                    task="complete-route ranking",
                    protocol=protocol,
                    metric=metric,
                    top_k=k,
                    value_percent=val["value_percent"],
                    split=val["split"],
                    model_name="GRV-Strict-II",
                    candidate_pool="RSP v005 K500 + CDG route pool",
                    sample_count=val["sample_count"],
                    data_source=str(grv_path),
                    notes="Complete-route TopK performance",
                )
            )

    if not ranker.empty:
        ranker_map = {
            "GRV-Raw": "raw route score",
            "GRV-Legacy": "legacy blend",
            "GRV-Strict-II": "GRV-Strict-II",
        }
        for ranker_name, label in ranker_map.items():
            part = ranker[ranker["ranker"] == ranker_name]
            if not part.empty:
                rows.append(
                    source_row(
                        panel="d_inset",
                        module="GRV",
                        task="same-pool ranker comparison",
                        protocol="missing-aware relaxed",
                        metric="route@1",
                        top_k=1,
                        value_percent=float(part.iloc[0]["missing_top1"]),
                        split="test",
                        model_name=label,
                        candidate_pool="shared route candidate pool",
                        sample_count="",
                        data_source=str(ranker_path),
                        notes="Same-pool ranker comparison; separates ranking from candidate-pool effects.",
                    )
                )

    error_df = pd.DataFrame(ERROR_ATTRIBUTION)
    error_df.to_csv(outdir / "figure3_error_attribution.csv", index=False)
    for row in ERROR_ATTRIBUTION:
        rows.append(
            source_row(
                panel="e",
                module="pipeline",
                task="error attribution",
                protocol="Top1 error decomposition",
                metric=row["category"],
                top_k=1,
                value_percent=row["percent"],
                split="test",
                model_name="RSP-CDG-GRV",
                candidate_pool="Figure 3 error-attribution subset",
                sample_count=row["sample_count"],
                data_source="error-attribution summary specified for Figure 3 task",
                notes=row["notes"],
            )
        )

    source_df = pd.DataFrame(rows)
    source_df.to_csv(outdir / "figure3_source_data.csv", index=False)

    warnings = sanity_check(source_df)
    if warnings and strict:
        # The instruction asks to warn/explain, not block, when sanity values differ.
        pass
    return source_df, error_df, ranker, warnings


def sanity_check(source_df: pd.DataFrame) -> list[str]:
    warnings = []
    lookup = {}
    for _, row in source_df.iterrows():
        key = (row["module"], row["metric"], int(row["top_k"]) if str(row["top_k"]).isdigit() else row["top_k"])
        lookup[key] = float(row["value_percent"])
    for key, expected in SANITY_CHECKS.items():
        actual = lookup.get(key)
        if actual is None:
            warnings.append(f"Sanity value missing for {key}: expected {expected:.2f}%.")
        elif abs(actual - expected) > TOLERANCE_PP:
            warnings.append(
                f"Sanity mismatch for {key}: artifact={actual:.3f}%, expected={expected:.2f}% "
                f"(delta={actual - expected:+.3f} pp)."
            )
    return warnings


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.07, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")


def tidy_axes(ax, ylabel: str = "Hit rate (%)", ylim: tuple[float, float] | None = None) -> None:
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.75)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#6B7280")


def legend(ax, **kwargs) -> None:
    defaults = dict(
        frameon=True,
        framealpha=0.92,
        edgecolor="#D1D5DB",
        fontsize=6.4,
        handlelength=1.35,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.25,
    )
    defaults.update(kwargs)
    ax.legend(**defaults)


def line_plot(ax, data: pd.DataFrame, metric: str, label: str, color: str, marker: str, linestyle: str = "-") -> None:
    part = data[data["metric"] == metric].sort_values("top_k")
    ax.plot(
        part["top_k"],
        part["value_percent"],
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.65,
        markersize=4.4,
        label=label,
    )


def draw_figure(source_df: pd.DataFrame, error_df: pd.DataFrame, ranker: pd.DataFrame, outdir: Path, style_name: str) -> None:
    plt, _ = setup_matplotlib()
    import matplotlib.pyplot as plt_imported
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig = plt_imported.figure(figsize=(7.25, 6.05))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], wspace=1.15, hspace=0.58)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    # Panel a: grouped bars.
    a = source_df[source_df["figure_panel"] == "a"].copy()
    order_k = [1, 10, 50]
    module_order = ["RSP", "CDG", "GRV"]
    colors = {"RSP": RSP_COLOR, "CDG": CDG_COLOR, "GRV": GRV_COLOR}
    x = list(range(len(order_k)))
    width = 0.23
    for idx, module in enumerate(module_order):
        values = [
            float(a[(a["module"] == module) & (a["top_k"] == k)]["value_percent"].iloc[0])
            for k in order_k
        ]
        ax_a.bar(
            [xx + (idx - 1) * width for xx in x],
            values,
            width=width,
            color=colors[module],
            edgecolor="#1F2937",
            linewidth=0.35,
            label=module,
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"Top{k}" for k in order_k])
    ax_a.set_title("Task-level performance")
    tidy_axes(ax_a, ylim=(0, 85))
    legend(ax_a, loc="upper left", ncols=1)
    panel_label(ax_a, "a")

    # Panel b.
    b = source_df[source_df["figure_panel"] == "b"].copy()
    line_plot(ax_b, b, "precursor_set_exact", "Exact set match", RSP_COLOR, "o")
    line_plot(ax_b, b, "best_jaccard", "Best Jaccard", "#2CA25F", "s")
    ax_b.set_xticks(order_k)
    ax_b.set_xticklabels([str(k) for k in order_k])
    ax_b.set_xlabel("Top-K")
    ax_b.set_title("RSP precursor proposal")
    tidy_axes(ax_b, ylabel="Hit / overlap (%)", ylim=(30, 90))
    legend(ax_b, loc="lower right")
    ax_b.text(
        0.03,
        0.94,
        "K fixed during evaluation",
        transform=ax_b.transAxes,
        fontsize=6.6,
        color=TEXT_COLOR,
        va="top",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", linewidth=0.55),
    )
    panel_label(ax_b, "b")

    # Panel c.
    c = source_df[source_df["figure_panel"] == "c"].copy()
    line_plot(ax_c, c, "missing_aware_relaxed_condition", "Missing-aware", CDG_COLOR, "o")
    line_plot(ax_c, c, "strict_comparable_relaxed_condition", "Strict-comparable", CDG_COLOR, "^", "--")
    ax_c.set_xticks(order_k)
    ax_c.set_xticklabels([str(k) for k in order_k])
    ax_c.set_xlabel("Top-K")
    ax_c.set_title("CDG condition tuples")
    tidy_axes(ax_c, ylim=(15, 82))
    legend(ax_c, loc="lower right")
    ax_c.text(
        0.03,
        0.94,
        "T + time + atmosphere + solvent",
        transform=ax_c.transAxes,
        fontsize=6.3,
        color=TEXT_COLOR,
        va="top",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", linewidth=0.55),
    )
    panel_label(ax_c, "c")

    # Panel d.
    d = source_df[source_df["figure_panel"] == "d"].copy()
    line_plot(ax_d, d, "missing_aware_relaxed_route", "Missing-aware", GRV_COLOR, "o")
    line_plot(ax_d, d, "strict_comparable_relaxed_route", "Strict-comparable", GRV_COLOR, "^", "--")
    ax_d.set_xticks(order_k)
    ax_d.set_xticklabels([str(k) for k in order_k])
    ax_d.set_xlabel("Top-K")
    ax_d.set_title("GRV complete-route ranking")
    tidy_axes(ax_d, ylim=(5, 46))
    legend(ax_d, loc="lower right")
    panel_label(ax_d, "d")

    if not ranker.empty:
        inset_rows = source_df[source_df["figure_panel"] == "d_inset"].copy()
        if not inset_rows.empty:
            iax = inset_axes(ax_d, width="42%", height="47%", loc="upper left", borderpad=1.1)
            labels = ["raw route score", "legacy blend", "GRV-Strict-II"]
            short_labels = ["Raw", "Legacy", "GRV"]
            vals = [
                float(inset_rows[inset_rows["model_name"] == label]["value_percent"].iloc[0])
                for label in labels
                if not inset_rows[inset_rows["model_name"] == label].empty
            ]
            short_labels = short_labels[: len(vals)]
            iax.bar(short_labels, vals, color=["#9CA3AF", "#5B677A", GRV_COLOR], edgecolor="#1F2937", linewidth=0.3)
            iax.set_title("same pool @1", fontsize=6.2, pad=2)
            iax.set_ylim(0, max(vals) * 1.28 if vals else 1)
            iax.tick_params(axis="x", labelsize=5.5, rotation=20)
            iax.tick_params(axis="y", labelsize=5.5)
            iax.grid(axis="y", color=GRID_COLOR, linewidth=0.4)

    # Panel e.
    labels = error_df["category"].tolist()
    percents = error_df["percent"].tolist()
    colors_e = error_df["color"].tolist()
    left = 0.0
    for label, percent, color in zip(labels, percents, colors_e):
        ax_e.barh([0], [percent], left=left, height=0.42, color=color, edgecolor="white", linewidth=0.8)
        if percent >= 6:
            ax_e.text(left + percent / 2, 0, f"{percent:.1f}%", ha="center", va="center", color="white", fontsize=7.0, fontweight="bold")
        left += percent
    ax_e.set_xlim(0, 100)
    ax_e.set_yticks([])
    ax_e.set_xlabel("Samples (%)")
    ax_e.set_title("Error attribution identifies precursor coverage as the dominant bottleneck")
    tidy_axes(ax_e, ylabel="", ylim=(-0.65, 0.65))
    ax_e.grid(axis="x", color=GRID_COLOR, linewidth=0.55, alpha=0.75)
    ax_e.grid(axis="y", visible=False)
    handles = [plt_imported.Rectangle((0, 0), 1, 1, color=color) for color in colors_e]
    ax_e.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncols=2, fontsize=6.2)
    panel_label(ax_e, "e")

    fig.suptitle("Held-out predictive performance and error propagation across the Synthmind pipeline", y=0.985, fontsize=10.5, fontweight="bold")
    fig.text(
        0.5,
        0.006,
        "All Top-K metrics are read from held-out test artifacts; CI columns are left blank because bootstrap intervals were not available for these final TopK tables.",
        ha="center",
        fontsize=6.8,
        color="#4B5563",
    )
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(outdir / f"Figure3_heldout_performance.{suffix}", bbox_inches="tight", dpi=450)
    plt_imported.close(fig)


def write_caption(outdir: Path) -> None:
    caption = """# Figure 3 Caption

**Figure 3 | Held-out predictive performance and error propagation across the Synthmind pipeline.**
a, Task-level performance at Top1, Top10 and Top50 for precursor-set prediction, condition-tuple prediction and complete-route prediction. b, RSP precursor-set performance under a fixed candidate budget, reported as exact set match and best Jaccard overlap. c, CDG condition-tuple prediction under missing-aware and strict-comparable protocols, highlighting joint ranking of temperature, time, atmosphere and solvent. d, GRV complete-route ranking under the same evaluation protocols; the inset compares route rankers under a shared candidate pool to separate ranking improvements from candidate-pool effects. e, Cross-module error attribution showing that precursor-skeleton miss remains the dominant failure mode, followed by route-ranking miss and condition-generation miss.
"""
    (outdir / "Figure3_CAPTION.md").write_text(caption, encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(outdir: Path) -> None:
    rows = []
    for path in sorted(outdir.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "description": {
                        ".png": "High-resolution raster figure",
                        ".pdf": "Vector PDF figure",
                        ".svg": "Editable vector SVG figure",
                        ".csv": "Figure source data",
                        ".md": "Caption or report",
                        ".json": "Artifact manifest",
                    }.get(path.suffix, "Figure artifact"),
                }
            )
    (outdir / "artifact_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_report(outdir: Path, source_df: pd.DataFrame, error_df: pd.DataFrame, warnings: list[str], style_name: str) -> None:
    def get(panel: str, module: str, metric: str, k: int) -> float:
        part = source_df[
            (source_df["figure_panel"] == panel)
            & (source_df["module"] == module)
            & (source_df["metric"] == metric)
            & (source_df["top_k"].astype(str) == str(k))
        ]
        return float(part.iloc[0]["value_percent"])

    lines = [
        "# Figure 3 Report",
        "",
        "## Inputs",
        "",
        "- RSP: `outputs/autorun/final_precision_topk_20260624/stage2_rsp_topk_test.csv`",
        "- CDG: `outputs/autorun/final_precision_topk_20260624/stage3_cdg_topk_test.csv`",
        "- GRV: `outputs/autorun/final_precision_topk_20260624/stage35_final_v005_grv14_topk_test.csv`",
        "- Same-pool inset: `outputs/autorun/final_precision_topk_20260624/model_evidence_scienceplots/grv_fig2_same_pool_ranker_source.csv`",
        "- Error attribution: Figure 3 task-specified error-attribution summary; no separately named local/remote source file was found during search.",
        "",
        f"Matplotlib style: {style_name}.",
        "",
        "## Main Values",
        "",
        f"- RSP exact: Top1 {get('b', 'RSP', 'precursor_set_exact', 1):.2f}%, Top10 {get('b', 'RSP', 'precursor_set_exact', 10):.2f}%, Top50 {get('b', 'RSP', 'precursor_set_exact', 50):.2f}%.",
        f"- RSP Best Jaccard: Top1 {get('b', 'RSP', 'best_jaccard', 1):.2f}%, Top10 {get('b', 'RSP', 'best_jaccard', 10):.2f}%, Top50 {get('b', 'RSP', 'best_jaccard', 50):.2f}%.",
        f"- CDG missing-aware relaxed: Top1 {get('c', 'CDG', 'missing_aware_relaxed_condition', 1):.2f}%, Top10 {get('c', 'CDG', 'missing_aware_relaxed_condition', 10):.2f}%, Top50 {get('c', 'CDG', 'missing_aware_relaxed_condition', 50):.2f}%.",
        f"- CDG strict-comparable relaxed: Top1 {get('c', 'CDG', 'strict_comparable_relaxed_condition', 1):.2f}%, Top10 {get('c', 'CDG', 'strict_comparable_relaxed_condition', 10):.2f}%, Top50 {get('c', 'CDG', 'strict_comparable_relaxed_condition', 50):.2f}%.",
        f"- GRV missing-aware relaxed: Top1 {get('d', 'GRV', 'missing_aware_relaxed_route', 1):.2f}%, Top10 {get('d', 'GRV', 'missing_aware_relaxed_route', 10):.2f}%, Top50 {get('d', 'GRV', 'missing_aware_relaxed_route', 50):.2f}%.",
        f"- GRV strict-comparable relaxed: Top1 {get('d', 'GRV', 'strict_comparable_relaxed_route', 1):.2f}%, Top10 {get('d', 'GRV', 'strict_comparable_relaxed_route', 10):.2f}%, Top50 {get('d', 'GRV', 'strict_comparable_relaxed_route', 50):.2f}%.",
        "",
        "## Error Attribution",
        "",
    ]
    for row in error_df.itertuples(index=False):
        lines.append(f"- {row.category}: {row.sample_count} samples, {row.percent:.2f}%.")
    lines.extend(["", "## Sanity Check", ""])
    if warnings:
        lines.extend(f"- WARNING: {w}" for w in warnings)
    else:
        lines.append("- All checked values match the requested sanity values within +/-0.05 percentage point.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `Figure3_heldout_performance.png`",
            "- `Figure3_heldout_performance.pdf`",
            "- `Figure3_heldout_performance.svg`",
            "- `figure3_source_data.csv`",
            "- `figure3_error_attribution.csv`",
            "- `Figure3_CAPTION.md`",
            "- `artifact_manifest.json`",
        ]
    )
    (outdir / "Figure3_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    topk = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    if topk != TOPK_DEFAULT:
        raise ValueError("Figure 3 main-text plot is constrained to Top1, Top10 and Top50.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    plt, style_name = setup_matplotlib()
    source_df, error_df, ranker_df, warnings = build_source_data(args.metrics_root, args.outdir, topk, args.strict_data_check)
    draw_figure(source_df, error_df, ranker_df, args.outdir, style_name)
    write_caption(args.outdir)
    write_report(args.outdir, source_df, error_df, warnings, style_name)
    write_manifest(args.outdir)


if __name__ == "__main__":
    main()
