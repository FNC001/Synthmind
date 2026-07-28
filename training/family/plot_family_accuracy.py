#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_labels(value: Any) -> Set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    text = "" if value is None else str(value).strip()
    if not text:
        return set()
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, (list, tuple, set)):
                return {str(item) for item in parsed}
        except Exception:
            pass
    return {item.strip() for item in text.split(";") if item.strip()}


def row_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, float]:
    intersection = len(true_set & pred_set)
    precision = intersection / len(pred_set) if pred_set else float(not true_set)
    recall = intersection / len(true_set) if true_set else float(not pred_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(true_set | pred_set)
    return {
        "exact": float(true_set == pred_set),
        "precision": float(precision),
        "recall": float(recall),
        "samples_f1": float(f1),
        "jaccard": float(intersection / union) if union else 1.0,
    }


def aggregate(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.groupby(column, dropna=False)
        .agg(
            n=("exact", "size"),
            exact_accuracy=("exact", "mean"),
            samples_f1=("samples_f1", "mean"),
            jaccard=("jaccard", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
        )
        .reset_index()
        .sort_values(["n", "samples_f1"], ascending=[False, False])
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.set_axisbelow(True)


def save_overview(df: pd.DataFrame, output: Path) -> None:
    names = ["Exact accuracy", "Samples F1", "Jaccard", "Precision", "Recall"]
    values = [
        df["exact"].mean(),
        df["samples_f1"].mean(),
        df["jaccard"].mean(),
        df["precision"].mean(),
        df["recall"].mean(),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(names, values, color=["#2166ac", "#4393c3", "#92c5de", "#4d9221", "#7fbc41"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(f"Stage 2 full-database test performance (n={len(df):,})")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_family_scatter(family: pd.DataFrame, output: Path) -> None:
    plot = family.loc[family["n_train"] > 0].copy()
    sizes = 22 + 10 * np.sqrt(plot["n"].to_numpy())
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    scatter = ax.scatter(
        plot["n_train"],
        plot["samples_f1"],
        s=sizes,
        c=plot["exact_accuracy"],
        cmap="viridis",
        vmin=0,
        vmax=1,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Training samples in primary cation family (log scale)")
    ax.set_ylabel("Test samples F1")
    ax.set_title("Per-family accuracy and training support")
    for _, row in plot.nlargest(12, "n").iterrows():
        ax.annotate(str(row["family_signature_primary"]), (row["n_train"], row["samples_f1"]), fontsize=7)
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Exact accuracy")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_group_comparison(table: pd.DataFrame, group_col: str, title: str, output: Path) -> None:
    show = table.head(16).copy()
    x = np.arange(len(show))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8.2, len(show) * 0.65), 5.2))
    ax.bar(x - width / 2, show["exact_accuracy"], width, label="Exact accuracy", color="#2166ac")
    ax.bar(x + width / 2, show["samples_f1"], width, label="Samples F1", color="#b2182b")
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in show[group_col]], rotation=35, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(frameon=False)
    for index, count in enumerate(show["n"]):
        ax.text(index, 0.02, f"n={int(count)}", rotation=90, ha="center", va="bottom", fontsize=7, color="white")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_training_curves(train_log_path: Path, output: Path) -> None:
    logs = json.loads(train_log_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(logs)
    frame["val_samples_f1"] = [
        float(item.get("samples_f1", np.nan)) for item in frame["val_metrics"]
    ]
    fig, left = plt.subplots(figsize=(8.4, 5.0))
    right = left.twinx()
    left.plot(frame["epoch"], frame["train_sup_loss"], label="Train supervised loss", color="#2166ac")
    left.plot(frame["epoch"], frame["val_loss"], label="Validation loss", color="#92c5de")
    right.plot(frame["epoch"], frame["val_samples_f1"], label="Validation samples F1", color="#b2182b", linewidth=2)
    best_index = frame["val_samples_f1"].idxmax()
    right.scatter([frame.loc[best_index, "epoch"]], [frame.loc[best_index, "val_samples_f1"]], color="#b2182b", zorder=5)
    left.set_xlabel("Epoch")
    left.set_ylabel("Loss")
    right.set_ylabel("Validation samples F1")
    right.set_ylim(0, 1)
    left.set_title("Training convergence and selected validation epoch")
    lines = [*left.get_lines(), *right.get_lines()]
    left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="center right")
    style_axis(left)
    right.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def prediction_metrics(path: Path) -> Dict[str, float]:
    frame = pd.read_csv(path)
    rows = [
        row_metrics(parse_labels(true_value), parse_labels(pred_value))
        for true_value, pred_value in zip(frame["true_labels"], frame["pred_labels"])
    ]
    metric_frame = pd.DataFrame(rows)
    return {
        "exact_accuracy": float(metric_frame["exact"].mean()),
        "samples_f1": float(metric_frame["samples_f1"].mean()),
        "jaccard": float(metric_frame["jaccard"].mean()),
    }


def save_ablation_comparison(current: Dict[str, float], baseline: Dict[str, float], output: Path) -> None:
    labels = ["Exact accuracy", "Samples F1", "Jaccard"]
    keys = ["exact_accuracy", "samples_f1", "jaccard"]
    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(x - width / 2, [baseline[key] for key in keys], width, label="No-family ablation", color="#bdbdbd")
    ax.bar(x + width / 2, [current[key] for key in keys], width, label="Cation-family conditioned", color="#2166ac")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Target-cation family feature ablation")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage2 accuracy by target cation family.")
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--dataset_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_log", default="")
    parser.add_argument("--baseline_prediction_csv", default="")
    args = parser.parse_args()

    prediction_path = Path(args.prediction_csv).expanduser().resolve()
    summary_path = Path(args.dataset_summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prediction_path)
    required = {"true_labels", "pred_labels", "family_signature_primary"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"prediction table lacks required columns: {missing}")

    true_sets = [parse_labels(value) for value in df["true_labels"]]
    pred_sets = [parse_labels(value) for value in df["pred_labels"]]
    metric_rows = [row_metrics(true_set, pred_set) for true_set, pred_set in zip(true_sets, pred_sets)]
    metrics = pd.DataFrame(metric_rows)
    for column in metrics:
        df[column] = metrics[column].to_numpy()
    df.to_csv(output_dir / "test_predictions_with_row_metrics.csv", index=False)

    family = aggregate(df, "family_signature_primary")
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train_counts = dataset_summary["splits"]["train"]["family_counts"]
    family["n_train"] = family["family_signature_primary"].map(train_counts).fillna(0).astype(int)
    family["support_band"] = pd.cut(
        family["n_train"],
        bins=[-1, 49, 99, 199, 499, np.inf],
        labels=["<50", "50-99", "100-199", "200-499", ">=500"],
    ).astype(str)
    family.to_csv(output_dir / "accuracy_by_primary_family.csv", index=False)

    source = aggregate(df, "source_dataset") if "source_dataset" in df else pd.DataFrame()
    quality = aggregate(df, "quality_tier") if "quality_tier" in df else pd.DataFrame()
    support = aggregate(
        df.merge(family[["family_signature_primary", "support_band"]], on="family_signature_primary"),
        "support_band",
    )
    source.to_csv(output_dir / "accuracy_by_source.csv", index=False)
    quality.to_csv(output_dir / "accuracy_by_quality.csv", index=False)
    support.to_csv(output_dir / "accuracy_by_support_band.csv", index=False)

    overall = {
        "n_test": int(len(df)),
        "exact_accuracy": float(df["exact"].mean()),
        "samples_f1": float(df["samples_f1"].mean()),
        "jaccard": float(df["jaccard"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "n_test_families": int(df["family_signature_primary"].nunique()),
        "prediction_csv": str(prediction_path),
    }
    if str(args.baseline_prediction_csv).strip():
        baseline_path = Path(args.baseline_prediction_csv).expanduser().resolve()
        baseline = prediction_metrics(baseline_path)
        overall["no_family_ablation"] = baseline
        overall["delta_vs_no_family"] = {
            key: float(overall[key] - baseline[key])
            for key in ("exact_accuracy", "samples_f1", "jaccard")
        }
        save_ablation_comparison(overall, baseline, output_dir / "fig_family_ablation_comparison.png")
    (output_dir / "accuracy_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    save_overview(df, output_dir / "fig_accuracy_overview.png")
    save_family_scatter(family, output_dir / "fig_accuracy_by_family_support.png")
    save_group_comparison(family, "family_signature_primary", "Largest primary cation families", output_dir / "fig_top_families.png")
    if not source.empty:
        save_group_comparison(source, "source_dataset", "Accuracy by source dataset", output_dir / "fig_accuracy_by_source.png")
    save_group_comparison(support, "support_band", "Accuracy by family support", output_dir / "fig_accuracy_by_support_band.png")
    if str(args.train_log).strip():
        save_training_curves(
            Path(args.train_log).expanduser().resolve(),
            output_dir / "fig_training_curves.png",
        )

    lines = [
        "# Full-database cation-family accuracy report",
        "",
        f"- Test rows: {overall['n_test']:,}",
        f"- Exact set accuracy: {overall['exact_accuracy']:.2%}",
        f"- Samples F1: {overall['samples_f1']:.2%}",
        f"- Samples Jaccard: {overall['jaccard']:.2%}",
        f"- Test primary families: {overall['n_test_families']}",
        "",
        "Generated figures:",
        "",
        "- `fig_accuracy_overview.png`",
        "- `fig_accuracy_by_family_support.png`",
        "- `fig_top_families.png`",
        "- `fig_accuracy_by_source.png`",
        "- `fig_accuracy_by_support_band.png`",
    ]
    (output_dir / "accuracy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
