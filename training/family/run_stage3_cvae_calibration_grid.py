#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, List


def float_list(value: str) -> List[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation grid for hybrid-CVAE sampling diversity.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--latent_scales", default="0.4,0.7,1.0")
    parser.add_argument("--continuous_noise_scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--categorical_temperatures", default="0.7,1.0")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    sampler = script_dir / "sample_stage3_hybrid_cvae.py"
    topk_evaluator = script_dir / "evaluate_stage3_sample_topk.py"
    coverage_evaluator = script_dir / "evaluate_stage3_distribution_coverage.py"
    rows: List[dict[str, Any]] = []
    combinations = itertools.product(
        float_list(args.latent_scales),
        float_list(args.continuous_noise_scales),
        float_list(args.categorical_temperatures),
    )
    for index, (latent_scale, noise_scale, categorical_temperature) in enumerate(combinations):
        tag = (
            f"l{latent_scale:g}_n{noise_scale:g}_c{categorical_temperature:g}"
            .replace(".", "p")
        )
        sample_path = output_dir / f"{tag}_samples.npz"
        topk_path = output_dir / f"{tag}_topk.json"
        coverage_path = output_dir / f"{tag}_coverage.json"
        subprocess.run([
            sys.executable, str(sampler),
            "--input_dir", args.input_dir,
            "--checkpoint", args.checkpoint,
            "--output_npz", str(sample_path),
            "--samples", str(args.samples),
            "--batch_size", "64",
            "--latent_scale", str(latent_scale),
            "--continuous_noise_scale", str(noise_scale),
            "--categorical_temperature", str(categorical_temperature),
            "--seed", str(int(args.seed) + index * 1009),
            "--device", args.device,
        ], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([
            sys.executable, str(topk_evaluator),
            "--input_dir", args.input_dir,
            "--split", "val",
            "--predictions_npz", str(sample_path),
            "--output_json", str(topk_path),
        ], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([
            sys.executable, str(coverage_evaluator),
            "--input_dir", args.input_dir,
            "--split", "val",
            "--predictions_npz", str(sample_path),
            "--output_json", str(coverage_path),
        ], check=True, stdout=subprocess.DEVNULL)
        topk = json.loads(topk_path.read_text(encoding="utf-8"))
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        row = {
            "tag": tag,
            "latent_scale": latent_scale,
            "continuous_noise_scale": noise_scale,
            "categorical_temperature": categorical_temperature,
            "method_inclusive_hit@10": topk["missing_aware_method_inclusive"]["hit@10"],
            "relaxed_hit@10": topk["missing_aware_relaxed"]["hit@10"],
            "coverage_f1_field_macro": coverage["coverage_f1_field_macro"],
            "temperature_wasserstein": coverage["continuous"]["temperature_c"]["wasserstein_macro"],
            "time_wasserstein": coverage["continuous"]["time_h"]["wasserstein_macro"],
            "atmosphere_js": coverage["discrete"]["atmosphere_coarse"]["jensen_shannon_macro"],
            "method_js": coverage["discrete"]["reaction_method"]["jensen_shannon_macro"],
            "samples_npz": str(sample_path),
            "topk_json": str(topk_path),
            "coverage_json": str(coverage_path),
        }
        row["selection_score"] = (
            float(row["method_inclusive_hit@10"])
            + 0.25 * float(row["coverage_f1_field_macro"])
        )
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "protocol": "validation_only_hybrid_cvae_sampling_calibration_grid",
        "config": vars(args),
        "trials": rows,
        "best_by_selection_score": max(rows, key=lambda row: row["selection_score"]),
        "best_by_method_inclusive_hit@10": max(rows, key=lambda row: row["method_inclusive_hit@10"]),
        "best_by_coverage_f1": max(rows, key=lambda row: row["coverage_f1_field_macro"]),
        "best_by_temperature_wasserstein": min(rows, key=lambda row: row["temperature_wasserstein"]),
        "best_by_time_wasserstein": min(rows, key=lambda row: row["time_wasserstein"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
