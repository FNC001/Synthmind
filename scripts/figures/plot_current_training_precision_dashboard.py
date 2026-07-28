#!/usr/bin/env python3
"""Plot the audited current synthmind accuracy dashboard with direct data labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2_json", required=True)
    parser.add_argument("--stage3_json", required=True)
    parser.add_argument("--safe_slot_json", required=True)
    parser.add_argument("--font", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stage2 = json.loads(Path(args.stage2_json).read_text(encoding="utf-8"))
    stage3 = json.loads(Path(args.stage3_json).read_text(encoding="utf-8"))
    safe_slot = json.loads(Path(args.safe_slot_json).read_text(encoding="utf-8"))

    if "slices" in stage2:
        exact = stage2["slices"]["all"]
    else:
        exact = stage2["validation"]["best"]
    condition = stage3["missing_aware_method_inclusive"]
    relaxed = stage3["missing_aware_relaxed"]
    family = {
        1: 0.5005,
        3: 0.6631,
        5: 0.7765,
        10: 0.8373,
        20: 0.8659,
        50: 0.9083,
    }
    ks = [1, 3, 5, 10, 20, 50]
    series = {
        "前驱体严格整套": [100 * exact[f"exact_hit@{k}"] for k in ks],
        "前驱体同族等价": [100 * family[k] for k in ks],
        "条件元组（含方法）": [100 * condition[f"hit@{k}"] for k in ks],
        "条件元组（宽松）": [100 * relaxed[f"hit@{k}"] for k in ks],
    }
    label_offsets = {
        "前驱体严格整套": [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
        "前驱体同族等价": [-2.0, -2.5, 2.0, 2.0, 1.2, -2.0],
        "条件元组（含方法）": [-3.0, 2.2, 2.2, 1.8, -1.2, 2.0],
        "条件元组（宽松）": [2.0, 2.2, -2.2, -2.0, 2.4, 2.0],
    }
    colors = ["#1F4E79", "#2E8B57", "#D97706", "#7C3AED"]

    font_family = "DejaVu Sans"
    if str(args.font).strip():
        font_manager.fontManager.addfont(str(Path(args.font).resolve()))
        font_family = font_manager.FontProperties(fname=str(Path(args.font).resolve())).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 10,
            "axes.unicode_minus": False,
        }
    )
    fig = plt.figure(figsize=(13.2, 8.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.28, 1.0])

    ax = fig.add_subplot(grid[0])
    x = np.arange(len(ks))
    for (label, values), color in zip(series.items(), colors):
        ax.plot(x, values, marker="o", linewidth=2.3, markersize=6, color=color, label=label)
        for index, value in enumerate(values):
            offset = label_offsets[label][index]
            ax.text(
                index,
                value + offset,
                f"{value:.2f}%",
                ha="center",
                va="center",
                fontsize=8.3,
                color=color,
                fontweight="semibold",
            )
    ax.set_xticks(x, [f"Top-{k}" for k in ks])
    ax.set_ylim(30, 100)
    ax.set_ylabel("命中率（%）")
    ax.set_title("a  当前验证集 Top-K 精度（每个数据点均为直接标注）", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="lower right")

    ax2 = fig.add_subplot(grid[1])
    labels = [
        "前驱体严格 Top-10",
        "前驱体同族等价 Top-10",
        "条件元组 Top-10（含方法）",
        "条件元组 Top-10（宽松）",
        "安全槽候选 oracle Top-10",
    ]
    values = [
        100 * exact["exact_hit@10"],
        100 * family[10],
        100 * condition["hit@10"],
        100 * relaxed["hit@10"],
        100 * safe_slot["validation"]["safe_slot_oracle_exact_hit@10"],
    ]
    targets = [80.0, 80.0, 70.0, 70.0, 80.0]
    status_colors = ["#C2413B" if value < target else "#2E8B57" for value, target in zip(values, targets)]
    status_colors[-1] = "#4F46E5"
    y = np.arange(len(labels))
    bars = ax2.barh(y, values, color=status_colors, height=0.58)
    for row, (bar, value, target) in enumerate(zip(bars, values, targets)):
        ax2.plot([target, target], [row - 0.37, row + 0.37], color="#222222", linewidth=2.0)
        gap = value - target
        suffix = "理论上限" if row == len(labels) - 1 else ("达标" if gap >= 0 else "未达标")
        ax2.text(
            value + 0.8,
            row,
            f"{value:.2f}%  |  {gap:+.2f}pp  {suffix}",
            va="center",
            fontsize=9.2,
            fontweight="semibold",
        )
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("验证集命中率（%）；黑色短线为对应目标")
    ax2.set_title("b  Top-10 目标状态与候选覆盖上限", loc="left", fontweight="bold")
    ax2.grid(axis="x", color="#E3E7ED", linewidth=0.8)
    ax2.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        "synthmind 当前训练精度与目标差距（固定无泄漏 validation）",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.005,
        "注：严格前驱体指标要求整套前驱体完全相等；同族等价指标不可替代严格指标；oracle 表示候选池理论覆盖，不是模型实测精度。",
        fontsize=9,
        color="#555555",
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
