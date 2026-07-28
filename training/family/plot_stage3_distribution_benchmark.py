#!/usr/bin/env python3
"""Create DiffSyn-style model-comparison figures from Stage3 coverage reports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), Path(path).expanduser().resolve()


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def field_names(report: dict) -> List[str]:
    return [*report["continuous"].keys(), *report["discrete"].keys()]


def coverage_values(report: dict, fields: List[str]) -> np.ndarray:
    values = []
    for field in fields:
        section = report["continuous"].get(field, report["discrete"].get(field))
        values.append(float(section["coverage_f1_macro"]))
    return np.asarray(values, dtype=np.float64)


def normalized_wasserstein(report: dict) -> float:
    values = []
    for section in report["continuous"].values():
        threshold = max(float(section["threshold"]), 1e-12)
        values.append(float(section["wasserstein_macro"]) / threshold)
    return float(np.mean(values)) if values else float("nan")


def plot_distance(reports: Dict[str, dict], output_dir: Path, dpi: int) -> None:
    names = list(reports)
    values = [normalized_wasserstein(reports[name]) for name in names]
    figure, axis = plt.subplots(figsize=(max(6.4, 0.9 * len(names)), 4.5))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(names)))
    bars = axis.bar(np.arange(len(names)), values, color=colors, width=0.72)
    axis.set_ylabel("Normalized Wasserstein-1 (lower is better)")
    axis.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_figure(figure, output_dir, "figure_a_normalized_wasserstein", dpi)


def plot_radar(reports: Dict[str, dict], output_dir: Path, dpi: int) -> None:
    first = next(iter(reports.values()))
    fields = field_names(first)
    angles = np.linspace(0, 2 * math.pi, len(fields), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    figure, axis = plt.subplots(figsize=(7.2, 6.4), subplot_kw={"projection": "polar"})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(reports), 2)))
    for color, (name, report) in zip(colors, reports.items()):
        values = coverage_values(report, fields)
        closed_values = np.concatenate([values, values[:1]])
        axis.plot(closed_angles, closed_values, marker="o", linewidth=1.8, label=name, color=color)
        axis.fill(closed_angles, closed_values, alpha=0.06, color=color)
    axis.set_xticks(angles, fields)
    axis.set_ylim(0, 1)
    axis.set_yticks([0.25, 0.5, 0.75, 1.0])
    axis.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    axis.set_title("Coverage-F1 by synthesis field", pad=22)
    axis.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False)
    save_figure(figure, output_dir, "figure_b_coverage_radar", dpi)


def plot_error_heatmap(reports: Dict[str, dict], output_dir: Path, dpi: int) -> None:
    names = list(reports)
    fields = field_names(next(iter(reports.values())))
    raw = np.zeros((len(names), len(fields)), dtype=np.float64)
    units: List[str] = []
    for column, field in enumerate(fields):
        if field in next(iter(reports.values()))["continuous"]:
            units.append("mean MAE")
            for row, name in enumerate(names):
                raw[row, column] = float(
                    reports[name]["continuous"][field]["mean_absolute_error_macro"]
                )
        else:
            units.append("JS")
            for row, name in enumerate(names):
                raw[row, column] = float(
                    reports[name]["discrete"][field]["jensen_shannon_macro"]
                )
    column_max = np.nanmax(raw, axis=0, keepdims=True)
    normalized = raw / np.maximum(column_max, 1e-12)
    figure, axis = plt.subplots(
        figsize=(max(7.0, 1.4 * len(fields)), max(3.5, 0.55 * len(names) + 1.5))
    )
    image = axis.imshow(normalized, cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(fields)), fields, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(names)), names)
    for row in range(len(names)):
        for column in range(len(fields)):
            axis.text(
                column,
                row,
                f"{raw[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if normalized[row, column] > 0.55 else "black",
                fontsize=9,
            )
    axis.set_xlabel("Continuous columns show distribution-mean MAE; categorical columns show JS")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Within-field normalized error")
    save_figure(figure, output_dir, "figure_c_field_error_heatmap", dpi)


def plot_distributions(
    input_dir: Path,
    split: str,
    predictions: Dict[str, Path],
    output_dir: Path,
    dpi: int,
    sample_limit: int,
    seed: int,
) -> None:
    if not predictions:
        return
    true_pack = np.load(input_dir / f"{split}.npz", allow_pickle=True)
    truth = np.asarray(true_pack["y_cond_continuous_raw"], dtype=np.float64)
    mask = np.asarray(true_pack["y_cond_continuous_mask"], dtype=np.float32) > 0.5
    fields = ["temperature_c", "time_h"]
    rng = np.random.default_rng(seed)
    figure, axes = plt.subplots(
        len(predictions) + 1,
        len(fields),
        figsize=(10.5, 2.15 * (len(predictions) + 1)),
        sharex="col",
        squeeze=False,
    )
    for column, field in enumerate(fields):
        true_values = truth[:, column][mask[:, column]]
        axes[0, column].hist(true_values, bins=35, density=True, color="0.55", alpha=0.9)
        axes[0, column].set_title(field)
        axes[0, column].set_ylabel("True")
    for row, (name, path) in enumerate(predictions.items(), start=1):
        with np.load(path, allow_pickle=False) as pack:
            generated = np.asarray(pack["continuous_samples"], dtype=np.float64)
        for column, field in enumerate(fields):
            values = generated[:, :, column].reshape(-1)
            values = values[np.isfinite(values)]
            if len(values) > sample_limit:
                values = values[rng.choice(len(values), sample_limit, replace=False)]
            axes[row, column].hist(
                values,
                bins=35,
                density=True,
                color=plt.cm.tab10((row - 1) % 10),
                alpha=0.82,
            )
            axes[row, column].set_ylabel(name)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    axes[-1, 0].set_xlabel("Temperature (°C)")
    axes[-1, 1].set_xlabel("Time (h)")
    figure.tight_layout()
    save_figure(figure, output_dir, "figure_d_condition_distributions", dpi)


def plot_composite(output_dir: Path, dpi: int) -> None:
    """Assemble the four benchmark views into one paper-style A-D panel."""
    stems = [
        "figure_a_normalized_wasserstein",
        "figure_b_coverage_radar",
        "figure_c_field_error_heatmap",
        "figure_d_condition_distributions",
    ]
    paths = [output_dir / f"{stem}.png" for stem in stems]
    if not all(path.exists() for path in paths):
        return
    figure, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for label, axis, path in zip("abcd", axes.flat, paths):
        axis.imshow(plt.imread(path))
        axis.axis("off")
        axis.text(
            0.01,
            0.99,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=20,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "none", "pad": 2.0},
        )
    save_figure(figure, output_dir, "figure_abcd_generative_benchmark", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a unified Stage3 generative benchmark.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--report", action="append", default=[], help="Repeat NAME=coverage.json")
    parser.add_argument(
        "--predictions", action="append", default=[], help="Repeat NAME=prediction_samples.npz"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--distribution_sample_limit", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    if not args.report:
        parser.error("at least one --report NAME=PATH is required")
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in map(parse_named_path, args.report)
    }
    predictions = dict(map(parse_named_path, args.predictions))
    unknown = set(predictions) - set(reports)
    if unknown:
        parser.error(f"prediction models have no matching reports: {sorted(unknown)}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    plot_distance(reports, output_dir, int(args.dpi))
    plot_radar(reports, output_dir, int(args.dpi))
    plot_error_heatmap(reports, output_dir, int(args.dpi))
    plot_distributions(
        Path(args.input_dir).resolve(),
        str(args.split),
        predictions,
        output_dir,
        int(args.dpi),
        int(args.distribution_sample_limit),
        int(args.seed),
    )
    plot_composite(output_dir, int(args.dpi))
    manifest = {
        "protocol": f"{args.split}_stage3_distribution_benchmark_figures",
        "reports": {name: str(path) for name, path in map(parse_named_path, args.report)},
        "predictions": {name: str(path) for name, path in predictions.items()},
        "figures": sorted(path.name for path in output_dir.glob("figure_*")),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
