#!/usr/bin/env python3
"""Validate a completed Synthmind GNoME prediction directory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path, decompress_gzip: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompress_gzip else Path.open
    if decompress_gzip:
        handle = opener(path, "rb")
    else:
        handle = opener(path, "rb")
    with handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_json_cell(value: Any, expected_type: type) -> bool:
    try:
        return isinstance(json.loads(str(value)), expected_type)
    except Exception:
        return False


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-zero-failures", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    predictions_path = output_dir / "all_predictions.csv.gz"
    top100_path = output_dir / "top100_most_synthesizable.csv"
    manifest_path = output_dir / "model_manifest.json"
    summary_path = output_dir / "summary.json"
    required_files = [
        predictions_path,
        top100_path,
        output_dir / "top100_most_synthesizable.md",
        manifest_path,
        summary_path,
    ]
    checks: list[dict[str, Any]] = []
    missing = [str(path) for path in required_files if not path.is_file()]
    checks.append(check("required_output_files", not missing, {"missing": missing}))
    if missing:
        raise SystemExit(json.dumps(checks, ensure_ascii=False, indent=2))

    input_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".cif", ".vasp", ".poscar"}
    )
    input_ids = [path.stem for path in input_files]
    input_set = set(input_ids)
    frame = pd.read_csv(predictions_path, low_memory=False)
    output_ids = frame["sample_id"].astype(str).tolist()
    output_set = set(output_ids)

    checks.extend(
        [
            check(
                "row_count_matches_input",
                len(frame) == len(input_files),
                {"input": len(input_files), "output": len(frame)},
            ),
            check(
                "output_ids_unique",
                len(output_ids) == len(output_set),
                {"rows": len(output_ids), "unique": len(output_set)},
            ),
            check(
                "input_output_id_set_equal",
                input_set == output_set,
                {
                    "missing_from_output": sorted(input_set - output_set)[:20],
                    "unexpected_output": sorted(output_set - input_set)[:20],
                },
            ),
        ]
    )

    status_counts = {
        str(key): int(value)
        for key, value in frame["prediction_status"].value_counts(dropna=False).items()
    }
    if args.require_zero_failures:
        checks.append(
            check(
                "zero_prediction_failures",
                status_counts == {"ok": len(frame)},
                status_counts,
            )
        )
    ok = frame.loc[frame["prediction_status"].eq("ok")].copy()
    required_columns = [
        "sample_id",
        "input_sha256",
        "formula",
        "predicted_precursors",
        "top10_precursor_sets",
        "pred_reaction_method",
        "pred_temperature_c_median",
        "pred_temperature_c_p25",
        "pred_temperature_c_p75",
        "pred_time_h_median",
        "pred_time_h_p25",
        "pred_time_h_p75",
        "pred_atmosphere",
        "stage3_ensemble_consensus",
        "synthesizability_rank_score",
        "quality_flags",
    ]
    missing_columns = [name for name in required_columns if name not in frame.columns]
    checks.append(
        check("required_columns_present", not missing_columns, missing_columns)
    )
    if not missing_columns:
        missing_values = ok[required_columns].isna().sum()
        missing_values = {
            str(key): int(value)
            for key, value in missing_values.items()
            if int(value) != 0
        }
        checks.append(
            check("required_values_nonmissing", not missing_values, missing_values)
        )

    numeric_columns = [
        "pred_temperature_c_median",
        "pred_temperature_c_p25",
        "pred_temperature_c_p75",
        "pred_time_h_median",
        "pred_time_h_p25",
        "pred_time_h_p75",
        "stage3_ensemble_consensus",
        "synthesizability_rank_score",
        "precursor_target_required_element_coverage",
    ]
    numeric = ok[numeric_columns].apply(pd.to_numeric, errors="coerce")
    checks.extend(
        [
            check(
                "numeric_values_finite",
                bool(np.isfinite(numeric.to_numpy(dtype=float)).all()),
                {
                    name: int((~np.isfinite(numeric[name].to_numpy(dtype=float))).sum())
                    for name in numeric_columns
                },
            ),
            check(
                "temperature_median_positive",
                bool((numeric["pred_temperature_c_median"] > 0).all()),
                {
                    "minimum": float(
                        numeric["pred_temperature_c_median"].min()
                    )
                },
            ),
            check(
                "time_median_nonnegative",
                bool((numeric["pred_time_h_median"] >= 0).all()),
                {"minimum": float(numeric["pred_time_h_median"].min())},
            ),
            check(
                "quartiles_ordered",
                bool(
                    (
                        numeric["pred_temperature_c_p25"]
                        <= numeric["pred_temperature_c_median"]
                    ).all()
                    and (
                        numeric["pred_temperature_c_median"]
                        <= numeric["pred_temperature_c_p75"]
                    ).all()
                    and (
                        numeric["pred_time_h_p25"]
                        <= numeric["pred_time_h_median"]
                    ).all()
                    and (
                        numeric["pred_time_h_median"]
                        <= numeric["pred_time_h_p75"]
                    ).all()
                ),
                {},
            ),
            check(
                "ranking_score_in_unit_interval",
                bool(
                    numeric["synthesizability_rank_score"].between(0, 1).all()
                ),
                {
                    "minimum": float(
                        numeric["synthesizability_rank_score"].min()
                    ),
                    "maximum": float(
                        numeric["synthesizability_rank_score"].max()
                    ),
                },
            ),
            check(
                "required_elements_fully_covered",
                bool(
                    np.isclose(
                        numeric[
                            "precursor_target_required_element_coverage"
                        ].to_numpy(dtype=float),
                        1.0,
                    ).all()
                ),
                {
                    "minimum": float(
                        numeric[
                            "precursor_target_required_element_coverage"
                        ].min()
                    )
                },
            ),
        ]
    )

    json_checks = {
        "predicted_precursors": list,
        "top10_precursor_sets": list,
        "quality_flags": list,
    }
    for column, expected_type in json_checks.items():
        valid = ok[column].map(
            lambda value: parse_json_cell(value, expected_type)
        )
        checks.append(
            check(
                f"{column}_valid_json",
                bool(valid.all()),
                {"invalid_rows": int((~valid).sum())},
            )
        )
    sha_valid = ok["input_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}")
    checks.append(
        check(
            "input_sha256_format",
            bool(sha_valid.all()),
            {"invalid_rows": int((~sha_valid).sum())},
        )
    )

    top100 = pd.read_csv(top100_path, low_memory=False)
    top_scores = pd.to_numeric(
        top100["synthesizability_rank_score"], errors="coerce"
    )
    expected_top_count = min(100, len(ok))
    checks.extend(
        [
            check(
                "top100_row_count",
                len(top100) == expected_top_count,
                {"expected": expected_top_count, "actual": len(top100)},
            ),
            check(
                "top100_ids_subset",
                set(top100["sample_id"].astype(str)).issubset(output_set),
                {},
            ),
            check(
                "top100_ranks_contiguous",
                top100["synthesizability_rank"].astype(int).tolist()
                == list(range(1, len(top100) + 1)),
                {},
            ),
            check(
                "top100_scores_descending",
                bool(top_scores.is_monotonic_decreasing),
                {},
            ),
        ]
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weight_checks = []
    for item in manifest["stage2"]["experts"]:
        path = Path(item["checkpoint"])
        actual = sha256_file(path) if path.is_file() else ""
        weight_checks.append(
            {
                "model_id": item["model_id"],
                "expected": item["checkpoint_sha256"],
                "actual": actual,
                "passed": actual == item["checkpoint_sha256"],
            }
        )
    for name, item in manifest["stage3"]["models"].items():
        path = Path(item["checkpoint"])
        actual = sha256_file(path) if path.is_file() else ""
        weight_checks.append(
            {
                "model_id": name,
                "expected": item["checkpoint_sha256"],
                "actual": actual,
                "passed": actual == item["checkpoint_sha256"],
            }
        )
    checks.append(
        check(
            "model_weight_sha256",
            all(item["passed"] for item in weight_checks),
            weight_checks,
        )
    )
    checks.append(
        check(
            "stage3_sample_count_64_each",
            int(manifest["stage3"]["samples_per_model"]) == 64,
            manifest["stage3"]["samples_per_model"],
        )
    )

    passed = all(item["passed"] for item in checks)
    report = {
        "passed": passed,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_count": len(input_files),
        "output_count": len(frame),
        "status_counts": status_counts,
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "artifact_sha256": {
            "all_predictions_gzip": sha256_file(predictions_path),
            "all_predictions_decompressed": sha256_file(
                predictions_path, decompress_gzip=True
            ),
            "top100_csv": sha256_file(top100_path),
            "top100_markdown": sha256_file(
                output_dir / "top100_most_synthesizable.md"
            ),
            "model_manifest": sha256_file(manifest_path),
            "summary": sha256_file(summary_path),
        },
        "statistics": {
            "external_elemental_completion_rows": int(
                ok["quality_flags"]
                .astype(str)
                .str.contains("external_elemental_completion", regex=False)
                .sum()
            ),
            "stage3_oov_rows": int(
                ok["quality_flags"]
                .astype(str)
                .str.contains("stage3_precursor_oov", regex=False)
                .sum()
            ),
            "composition_prior_fallback_rows": int(
                ok["quality_flags"]
                .astype(str)
                .str.contains(
                    "composition_training_prior_fallback", regex=False
                )
                .sum()
            ),
            "temperature_median_quantiles_c": {
                str(key): float(value)
                for key, value in numeric[
                    "pred_temperature_c_median"
                ].quantile([0, 0.1, 0.5, 0.9, 1]).items()
            },
            "time_median_quantiles_h": {
                str(key): float(value)
                for key, value in numeric["pred_time_h_median"]
                .quantile([0, 0.1, 0.5, 0.9, 1])
                .items()
            },
            "ranking_score_quantiles": {
                str(key): float(value)
                for key, value in numeric[
                    "synthesizability_rank_score"
                ].quantile([0, 0.1, 0.5, 0.9, 1]).items()
            },
        },
    }
    write_json(output_dir / "validation_report.json", report)

    lines = [
        "# Synthmind GNoME 全量输出验证报告",
        "",
        f"- 总体结论：{'通过' if passed else '未通过'}",
        f"- 检查项：{report['checks_passed']}/{report['checks_total']}",
        f"- 输入结构：{len(input_files):,}",
        f"- 输出记录：{len(frame):,}",
        f"- 状态统计：`{json.dumps(status_counts, ensure_ascii=False)}`",
        "",
        "## 检查明细",
        "",
        "| 检查 | 结果 | 说明 |",
        "|---|---|---|",
    ]
    for item in checks:
        detail = json.dumps(item["detail"], ensure_ascii=False)
        if len(detail) > 300:
            detail = detail[:297] + "..."
        lines.append(
            f"| {item['name']} | {'通过' if item['passed'] else '失败'} | "
            f"`{detail}` |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `synthesizability_rank_score` 是未标定排序代理值，不是实验成功概率。",
            "- `composition_training_prior_fallback` 表示模型候选元素不完整时使用了冻结训练库中的真实前驱体频率先验。",
            "- `external_elemental_completion` 表示冻结词表没有覆盖某个目标元素，使用该元素单质显式补全。",
            "- Stage3 OOV 路线的条件置信度低于全部前驱体均在词表内的路线，实验前必须人工复核。",
        ]
    )
    (output_dir / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    checksum_lines = [
        f"{digest}  {name}"
        for name, digest in report["artifact_sha256"].items()
    ]
    (output_dir / "ARTIFACT_SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
