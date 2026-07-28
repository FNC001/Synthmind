#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_overview(metrics: Dict[str, Any], baseline: Dict[str, Any], output: Path) -> pd.DataFrame:
    current = metrics.get("test_metrics") or metrics["metrics"]["test"]
    base = baseline.get("test_metrics") or baseline["metrics"]["test"]
    rows = [
        {"task": "Temperature", "metric": "MAE (°C)", "family": current["continuous"]["temperature_c"]["mae"], "no_family": base["continuous"]["temperature_c"]["mae"], "higher_is_better": False},
        {"task": "Time", "metric": "MAE (h)", "family": current["continuous"]["time_h"]["mae"], "no_family": base["continuous"]["time_h"]["mae"], "higher_is_better": False},
        {"task": "Atmosphere", "metric": "Accuracy", "family": current["discrete"]["atmosphere_coarse"]["accuracy"], "no_family": base["discrete"]["atmosphere_coarse"]["accuracy"], "higher_is_better": True},
        {"task": "Method", "metric": "Accuracy", "family": current["discrete"]["reaction_method"]["accuracy"], "no_family": base["discrete"]["reaction_method"]["accuracy"], "higher_is_better": True},
    ]
    frame = pd.DataFrame(rows)
    frame["relative_improvement_pct"] = np.where(
        frame["higher_is_better"],
        100 * (frame["family"] - frame["no_family"]) / frame["no_family"],
        100 * (frame["no_family"] - frame["family"]) / frame["no_family"],
    )
    frame.to_csv(output / "stage3_ablation_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    for axis, row in zip(axes, rows):
        values = [row["no_family"], row["family"]]
        axis.bar(["No family", "Family"], values, color=["#9aa0a6", "#2878b5"])
        axis.set_title(f"{row['task']}\n{row['metric']}")
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Stage 3 test performance: cation-family features vs ablation", y=1.04, fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "fig_stage3_ablation_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return frame


def save_regression_scatter(pred: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, name, label in zip(axes, ("temperature_c", "time_h"), ("Temperature (°C)", "Time (h)")):
        valid = pred[f"has_{name}"] > 0.5
        truth = pred.loc[valid, f"true_{name}"].to_numpy(float)
        estimate = pred.loc[valid, f"pred_{name}"].to_numpy(float)
        if name == "time_h":
            truth = np.log1p(np.clip(truth, 0, None))
            estimate = np.log1p(np.clip(estimate, 0, None))
            label = "log1p(Time [h])"
        axis.hexbin(truth, estimate, gridsize=45, mincnt=1, cmap="Blues")
        low = float(min(truth.min(), estimate.min()))
        high = float(max(truth.max(), estimate.max()))
        axis.plot([low, high], [low, high], "--", color="#d62728", linewidth=1)
        axis.set_xlabel(f"True {label}")
        axis.set_ylabel(f"Predicted {label}")
        axis.grid(alpha=0.15)
    fig.suptitle("Stage 3 held-out test predictions", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "fig_stage3_regression_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def family_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in pred.groupby("family_signature_primary", dropna=False):
        row: Dict[str, Any] = {"family": str(family), "support": int(len(group))}
        for name in ("temperature_c", "time_h"):
            valid = group[f"has_{name}"] > 0.5
            row[f"{name}_n"] = int(valid.sum())
            row[f"{name}_mae"] = (
                float(mean_absolute_error(group.loc[valid, f"true_{name}"], group.loc[valid, f"pred_{name}"]))
                if valid.sum() else np.nan
            )
        for name in ("atmosphere_coarse", "reaction_method"):
            valid = group[f"has_{name}"] > 0.5
            row[f"{name}_n"] = int(valid.sum())
            row[f"{name}_accuracy"] = (
                float(accuracy_score(group.loc[valid, f"true_{name}_id"], group.loc[valid, f"pred_{name}_id"]))
                if valid.sum() else np.nan
            )
            row[f"{name}_macro_f1"] = (
                float(f1_score(group.loc[valid, f"true_{name}_id"], group.loc[valid, f"pred_{name}_id"], average="macro", zero_division=0))
                if valid.sum() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["support", "family"], ascending=[False, True])


def save_family_plots(table: pd.DataFrame, output: Path) -> None:
    eligible = table[table["support"] >= 10].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specifications = [
        ("temperature_c_mae", "Temperature MAE (°C)"),
        ("time_h_mae", "Time MAE (h)"),
        ("atmosphere_coarse_accuracy", "Atmosphere accuracy"),
        ("reaction_method_accuracy", "Method accuracy"),
    ]
    for axis, (column, label) in zip(axes.ravel(), specifications):
        valid = eligible.dropna(subset=[column])
        axis.scatter(valid["support"], valid[column], s=26, alpha=0.75, color="#2878b5")
        axis.set_xscale("log")
        axis.set_xlabel("Test family support (log scale)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    fig.suptitle("Stage 3 performance by cation family (support ≥ 10)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "fig_stage3_by_family_support.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Stage3 family-conditioned accuracy and ablation.")
    parser.add_argument("--metrics_json", required=True)
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--baseline_metrics_json", required=True)
    parser.add_argument("--baseline_prediction_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = load_json(Path(args.metrics_json))
    baseline = load_json(Path(args.baseline_metrics_json))
    prediction = pd.read_csv(args.prediction_csv, low_memory=False)
    baseline_prediction = pd.read_csv(args.baseline_prediction_csv, low_memory=False)
    if prediction["sample_id"].astype(str).tolist() != baseline_prediction["sample_id"].astype(str).tolist():
        raise ValueError("family and ablation predictions are not aligned by sample_id")
    overview = save_overview(metrics, baseline, output)
    save_regression_scatter(prediction, output)
    per_family = family_table(prediction)
    per_family.to_csv(output / "stage3_metrics_by_family.csv", index=False)
    save_family_plots(per_family, output)
    report = {
        "n_test": int(len(prediction)),
        "n_test_families": int(per_family.shape[0]),
        "ablation": overview.to_dict(orient="records"),
    }
    (output / "stage3_accuracy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
