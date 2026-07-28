#!/usr/bin/env python3
"""Plot formal validation progress and clearly separated exploratory ceilings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def _top10(path: str) -> float:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "validation" in payload:
        return float(payload["validation"]["best"]["exact_hit@10"])
    return float(payload["slices"]["all"]["exact_hit@10"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide32", required=True)
    parser.add_argument("--g02", required=True)
    parser.add_argument("--font", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    font_family = "DejaVu Sans"
    if args.font:
        font_manager.fontManager.addfont(str(Path(args.font).resolve()))
        font_family = font_manager.FontProperties(fname=str(Path(args.font).resolve())).get_name()
    plt.rcParams.update({"font.family": font_family, "axes.unicode_minus": False})

    labels = [
        "同族候选基础排序",
        "MatSciBERT + 稠密材料特征",
        "多正例 + 宽候选池（32）",
        "宽候选 + G02 族专家（当前最佳）",
        "族×元素数×阴离子路由",
        "88 源按族直接选优",
        "88 源逐样本并集 oracle",
    ]
    values = [
        0.7497695853,
        0.7635944700,
        _top10(args.wide32),
        _top10(args.g02),
        0.7834101382,
        0.7986175115,
        0.8400921659,
    ]
    classes = ["正式固定验证"] * 4 + ["探索性（非正式精度）"] * 2 + ["Oracle（不可部署）"]
    colors = ["#2F6FED"] * 4 + ["#F28A20"] * 2 + ["#858A93"]

    figure, axis = plt.subplots(figsize=(13.2, 7.8))
    y = list(range(len(labels)))
    bars = axis.barh(y, [100 * value for value in values], color=colors, height=0.58)
    axis.axvline(80, color="#D93B34", linewidth=2.2)
    axis.text(80.15, -0.60, "目标 80%", color="#D93B34", fontsize=11, fontweight="bold")
    for bar, value, result_class in zip(bars, values, classes):
        axis.text(
            100 * value + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{100 * value:.2f}%  |  {result_class}",
            va="center",
            fontsize=10.2,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(70, 85.8)
    axis.set_xlabel("严格整套前驱体 Top-10 命中率（%）")
    figure.suptitle(
        "前驱体严格 Top-10：当前正式验证进展与探索性覆盖上限",
        x=0.18,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    axis.set_title(
        "固定验证集 n=2,170；整套规范化前驱体集合完全一致才计为命中",
        loc="left",
        color="#586174",
        fontsize=10.5,
        pad=12,
    )
    axis.grid(axis="x", color="#E2E7EF", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.text(
        0.02,
        0.015,
        "标注：蓝色模型只用训练标签拟合，在固定 validation 上选模；橙色结果使用当前 validation 标签进行路由选择，仅作诊断；灰色 oracle 需要预先知道正确答案。冻结 test 未开启。",
        fontsize=9.2,
        color="#4E586B",
    )
    figure.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.12)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
