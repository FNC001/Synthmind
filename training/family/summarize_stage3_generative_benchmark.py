#!/usr/bin/env python3
"""Summarize Stage 3 generative models under one validation-only protocol."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_model(value: str) -> tuple[str, Path, Path, Path | None]:
    if "=" not in value:
        raise ValueError(f"expected NAME=TOPK,COVERAGE[,TRAIN], got {value!r}")
    name, paths = value.split("=", 1)
    parts = [Path(item).expanduser().resolve() for item in paths.split(",")]
    if len(parts) not in {2, 3}:
        raise ValueError(f"expected two or three paths for {name!r}, got {len(parts)}")
    return name.strip(), parts[0], parts[1], parts[2] if len(parts) == 3 else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: dict[str, Any], section: str, key: str) -> float | None:
    value = report.get(section, {}).get(key)
    return None if value is None else float(value)


def coverage_summary(report: dict[str, Any]) -> dict[str, float]:
    continuous = list(report.get("continuous", {}).values())
    discrete = list(report.get("discrete", {}).values())
    fields = [*continuous, *discrete]
    coverage = [float(field["coverage_f1_macro"]) for field in fields]
    normalized_wasserstein = [
        float(field["wasserstein_macro"]) / max(float(field["threshold"]), 1e-12)
        for field in continuous
    ]
    return {
        "field_macro_coverage_f1": fmean(coverage) if coverage else float("nan"),
        "continuous_normalized_wasserstein": (
            fmean(normalized_wasserstein) if normalized_wasserstein else float("nan")
        ),
    }


def training_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {
            "parameter_count": None,
            "best_epoch": None,
            "elapsed_seconds": None,
            "peak_cuda_memory_mb": None,
        }
    peak = report.get("peak_cuda_memory_mb")
    if peak is None:
        peak = report.get("peak_cuda_memory_bytes")
        peak = None if peak is None else float(peak) / (1024.0 * 1024.0)
    return {
        "parameter_count": report.get("parameter_count"),
        "best_epoch": report.get("best_epoch"),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "peak_cuda_memory_mb": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeat NAME=topk.json,coverage.json[,training_metrics.json]",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    sources: dict[str, dict[str, str | None]] = {}
    for value in args.model:
        name, topk_path, coverage_path, training_path = parse_model(value)
        topk = read_json(topk_path)
        coverage = read_json(coverage_path)
        training = read_json(training_path) if training_path is not None else None
        row: dict[str, Any] = {
            "model": name,
            "rows": int(topk["rows"]),
            "samples_per_row": int(topk["samples_per_row"]),
            "mean_unique_condition_buckets": float(topk["mean_unique_condition_buckets"]),
            "method_top1": metric(topk, "missing_aware_method_inclusive", "hit@1"),
            "method_top3": metric(topk, "missing_aware_method_inclusive", "hit@3"),
            "method_top5": metric(topk, "missing_aware_method_inclusive", "hit@5"),
            "method_top10": metric(topk, "missing_aware_method_inclusive", "hit@10"),
            "method_top20": metric(topk, "missing_aware_method_inclusive", "hit@20"),
            "method_top50": metric(topk, "missing_aware_method_inclusive", "hit@50"),
            "relaxed_top10": metric(topk, "missing_aware_relaxed", "hit@10"),
            **coverage_summary(coverage),
            **training_summary(training),
        }
        rows.append(row)
        sources[name] = {
            "topk": str(topk_path),
            "coverage": str(coverage_path),
            "training": None if training_path is None else str(training_path),
        }

    rows.sort(key=lambda item: float(item["method_top10"] or 0.0), reverse=True)
    output = {
        "protocol": "validation_only_formula_disjoint_stage3_generative_model_benchmark",
        "selection_note": "No frozen test data are used for model selection.",
        "models": rows,
        "sources": sources,
    }
    output_json = Path(args.output_json).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
