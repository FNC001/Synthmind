#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/auto_improve/synpred_auto_v1_20260612")


STAGE2_TOPK_KEYS = [
    "top1_exact",
    "top3_exact",
    "top5_exact",
    "top10_exact",
    "top20_exact",
    "top50_exact",
    "top100_exact",
    "top200_exact",
    "top500_exact",
    "top500_best_f1",
    "top500_best_jaccard",
    "oov_top500_exact",
    "non_oov_top500_exact",
]

STAGE3_KEYS = [
    "top1_strict_condition",
    "top1_relaxed_condition",
    "top10_strict_condition",
    "top10_relaxed_condition",
    "oracle_relaxed_condition",
    "temperature_MAE",
    "time_MAE",
    "atmosphere_accuracy",
]

STAGE35_KEYS = [
    "top1_strict_route",
    "top1_relaxed_route",
    "top10_strict_route",
    "top10_relaxed_route",
    "top200_relaxed_route",
    "top200_usable_relaxed_route",
    "top1_usable_relaxed_route",
]


BASELINE_THRESHOLDS = {
    "stage2": {
        "all_top1_exact": 0.3947,
        "all_top10_exact": 0.6335,
        "all_top500_exact": 0.8024,
        "core_top1_exact": 0.4615,
        "core_top10_exact": 0.6917,
        "core_top500_exact": 0.8533,
    },
    "stage35": {
        "strict_comparable_top1_relaxed_route": 0.1045,
        "missing_aware_top1_relaxed_route": 0.2072,
        "strict_comparable_top10_relaxed_route": 0.1804,
        "missing_aware_top10_relaxed_route": 0.3455,
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def rel(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{100.0 * float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pick_metrics_blob(obj: Dict[str, Any], split: str = "test", protocol: Optional[str] = None) -> Dict[str, Any]:
    if protocol and protocol in obj and isinstance(obj[protocol], dict):
        return pick_metrics_blob(obj[protocol], split=split)
    if protocol:
        for prefix in ("blend", "raw", "calibrated"):
            key = f"{prefix}_{protocol}"
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if split in obj and isinstance(obj[split], dict):
        split_obj = obj[split]
        if "metrics" in split_obj and isinstance(split_obj["metrics"], dict):
            return split_obj["metrics"]
        return split_obj
    for key in (f"{split}_metrics", "metrics"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def subset_metrics(metrics: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: metrics[key] for key in keys if key in metrics}


def read_csv_records(path: Path, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if max_records is not None:
        df = df.head(max_records)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def csv_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    df = pd.read_csv(path)
    return {
        "exists": True,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


@dataclass
class MetricSource:
    name: str
    stage: str
    path: Path
    split: str = "test"
    protocol: Optional[str] = None
    metric_keys: Tuple[str, ...] = ()


class MetricsRegistry:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.records: Dict[str, Any] = {
            "project_root": str(project_root),
            "thresholds": BASELINE_THRESHOLDS,
            "baselines": {},
            "experiments": {},
            "sources": {},
        }

    def _abs(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def add_json_metrics(
        self,
        *,
        bucket: str,
        name: str,
        stage: str,
        path: str | Path,
        split: str = "test",
        protocol: Optional[str] = None,
        keys: Iterable[str] = (),
    ) -> Dict[str, Any]:
        p = self._abs(path)
        item: Dict[str, Any] = {
            "stage": stage,
            "split": split,
            "protocol": protocol,
            "path": rel(self.project_root, p),
            "exists": p.exists(),
            "metrics": {},
        }
        if p.exists():
            obj = read_json(p)
            metrics = pick_metrics_blob(obj, split=split, protocol=protocol)
            selected = subset_metrics(metrics, keys) if keys else dict(metrics)
            item["metrics"] = selected
            if isinstance(obj, dict) and "best_val" in obj:
                item["best_val_metrics"] = pick_metrics_blob(obj["best_val"], split="metrics")
            if isinstance(obj, dict) and "config" in obj:
                item["config"] = obj["config"]
        self.records.setdefault(bucket, {})[name] = item
        self.records["sources"][name] = item["path"]
        return item

    def add_csv_table(self, *, bucket: str, name: str, stage: str, path: str | Path) -> Dict[str, Any]:
        p = self._abs(path)
        item = {"stage": stage, **csv_summary(p)}
        if item.get("exists"):
            item["path"] = rel(self.project_root, p)
            item["records"] = read_csv_records(p)
        self.records.setdefault(bucket, {})[name] = item
        self.records["sources"][name] = item.get("path", str(p))
        return item

    def collect_current_baselines(self) -> Dict[str, Any]:
        self.add_json_metrics(
            bucket="baselines",
            name="stage2_v5_all_test",
            stage="stage2",
            path="outputs/evaluation/stage2_score_calibration_v5_20260610/test_calibrated_metrics.json",
            keys=STAGE2_TOPK_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage2_v5_all_val",
            stage="stage2",
            path="outputs/evaluation/stage2_score_calibration_v5_20260610/test_calibrated_metrics.json",
            split="best_val",
            keys=STAGE2_TOPK_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage2_core_calibrated_test",
            stage="stage2",
            path="outputs/evaluation/stage2_score_calibration_core_methods_20260610/test_core_calibrated_metrics.json",
            keys=STAGE2_TOPK_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage2_train_oof_v4_approx",
            stage="stage2",
            path="outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612/train_oof_generation_summary.json",
            keys=(),
        )
        self.add_csv_table(
            bucket="baselines",
            name="stage2_v5_by_reaction_method",
            stage="stage2",
            path="outputs/evaluation/stage2_score_calibration_v5_20260610/test_by_reaction_method.csv",
        )
        self.add_csv_table(
            bucket="baselines",
            name="stage2_v5_by_failure_type",
            stage="stage2",
            path="outputs/evaluation/stage2_score_calibration_v5_20260610/test_by_failure_type.csv",
        )
        self.add_csv_table(
            bucket="baselines",
            name="stage2_v5_by_candidate_source",
            stage="stage2",
            path="outputs/evaluation/stage2_score_calibration_v5_20260610/test_by_candidate_source.csv",
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage3_v3_missing_aware_test",
            stage="stage3",
            path="outputs/evaluation/stage3_stage35_v3_dual_protocol_20260612/stage3_condition_metrics_missing_aware.json",
            protocol="missing_aware",
            keys=STAGE3_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage3_v3_strict_comparable_test",
            stage="stage3",
            path="outputs/evaluation/stage3_stage35_v3_dual_protocol_20260612/stage3_condition_metrics_strict_comparable.json",
            protocol="strict_comparable",
            keys=STAGE3_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage3_v4_missing_aware_test",
            stage="stage3",
            path="outputs/evaluation/stage3_condition_calibration_v4_20260612/test_metrics_missing_aware.json",
            protocol="missing_aware",
            keys=STAGE3_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage3_v4_strict_comparable_test",
            stage="stage3",
            path="outputs/evaluation/stage3_condition_calibration_v4_20260612/test_metrics_strict_comparable.json",
            protocol="strict_comparable",
            keys=STAGE3_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage35_v3_final_missing_aware_test",
            stage="stage35",
            path="runs/stage35/route_reranker_v3_final_20260612/test_metrics.json",
            protocol="missing_aware",
            keys=STAGE35_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage35_v3_final_strict_comparable_test",
            stage="stage35",
            path="runs/stage35/route_reranker_v3_final_20260612/test_metrics.json",
            protocol="strict_comparable",
            keys=STAGE35_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage35_v4_missing_aware_test",
            stage="stage35",
            path="outputs/evaluation/stage35_route_candidates_v4_20260612/test_route_candidate_metrics.json",
            protocol="missing_aware",
            keys=STAGE35_KEYS,
        )
        self.add_json_metrics(
            bucket="baselines",
            name="stage35_v4_strict_comparable_test",
            stage="stage35",
            path="outputs/evaluation/stage35_route_candidates_v4_20260612/test_route_candidate_metrics.json",
            protocol="strict_comparable",
            keys=STAGE35_KEYS,
        )
        return self.records

    def collect_experiment_metrics(self, output_dir: Path) -> None:
        root = self._abs(output_dir)
        if not root.exists():
            return
        for path in sorted(root.rglob("*.json")):
            name = path.relative_to(root).with_suffix("").as_posix().replace("/", "__")
            try:
                obj = read_json(path)
            except json.JSONDecodeError:
                continue
            metrics = pick_metrics_blob(obj)
            if not isinstance(metrics, dict):
                continue
            metric_like = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            if not metric_like:
                continue
            stage = infer_stage_from_name(name)
            self.records["experiments"][name] = {
                "stage": stage,
                "path": rel(self.project_root, path),
                "metrics": metric_like,
            }

    def write(self, output_dir: Path) -> Tuple[Path, Path]:
        out = self._abs(output_dir) / "metrics_registry"
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "metrics_registry.json"
        md_path = out / "metrics_registry.md"
        write_json(json_path, self.records)
        md_path.write_text(render_registry_markdown(self.records), encoding="utf-8")
        return json_path, md_path


def infer_stage_from_name(name: str) -> str:
    low = name.lower()
    if "stage35" in low or "route" in low:
        return "stage35"
    if "stage3" in low or "condition" in low:
        return "stage3"
    if "stage2" in low or "precursor" in low:
        return "stage2"
    return "unknown"


def render_metric_row(name: str, metrics: Dict[str, Any], keys: Iterable[str]) -> str:
    cells = [name]
    for key in keys:
        cells.append(pct(metrics.get(key)))
    return "| " + " | ".join(cells) + " |"


def render_registry_markdown(records: Dict[str, Any]) -> str:
    baselines = records.get("baselines", {})
    lines = [
        "# SynPred Auto Improvement Metrics Registry",
        "",
        "This registry normalizes the current Stage2, Stage3, and Stage35 baseline metrics plus any discovered experiment metrics.",
        "",
        "## Stage2",
        "",
        "| source | top1 | top10 | top200 | top500 | best F1@500 | best Jaccard@500 | OOV top500 | non-OOV top500 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("stage2_v5_all_test", "stage2_core_calibrated_test"):
        item = baselines.get(name, {})
        lines.append(
            render_metric_row(
                name,
                item.get("metrics", {}),
                [
                    "top1_exact",
                    "top10_exact",
                    "top200_exact",
                    "top500_exact",
                    "top500_best_f1",
                    "top500_best_jaccard",
                    "oov_top500_exact",
                    "non_oov_top500_exact",
                ],
            )
        )
    oof = baselines.get("stage2_train_oof_v4_approx", {}).get("metrics", {})
    if oof:
        lines.extend(
            [
                "",
                "## Stage2 Train OOF Approximation",
                "",
                "| source | top1 | top10 | top20 | mean F1@top1 | mean best F1@20 | open generated | repair | chemistry ok |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                render_metric_row(
                    "stage2_train_oof_v4_approx",
                    oof,
                    [
                        "top1_exact",
                        "top10_exact",
                        "top20_exact",
                        "mean_f1_top1",
                        "mean_best_f1_top20",
                        "open_generated_rate",
                        "repair_rate",
                        "chemistry_ok_rate",
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Stage3 Condition",
            "",
            "| source | protocol | top1 strict | top1 relaxed | top10 strict | top10 relaxed | oracle relaxed |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "stage3_v3_missing_aware_test",
        "stage3_v3_strict_comparable_test",
        "stage3_v4_missing_aware_test",
        "stage3_v4_strict_comparable_test",
    ):
        item = baselines.get(name, {})
        m = item.get("metrics", {})
        lines.append(
            f"| {name} | {item.get('protocol') or ''} | {pct(m.get('top1_strict_condition'))} | {pct(m.get('top1_relaxed_condition'))} | {pct(m.get('top10_strict_condition'))} | {pct(m.get('top10_relaxed_condition'))} | {pct(m.get('oracle_relaxed_condition'))} |"
        )
    lines.extend(
        [
            "",
            "## Stage35 Route",
            "",
            "| source | protocol | top1 strict | top1 relaxed | top10 strict | top10 relaxed | top200 relaxed | usable relaxed top200 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("stage35_v3_final_missing_aware_test", "stage35_v3_final_strict_comparable_test"):
        item = baselines.get(name, {})
        m = item.get("metrics", {})
        lines.append(
            f"| {name} | {item.get('protocol') or ''} | {pct(m.get('top1_strict_route'))} | {pct(m.get('top1_relaxed_route'))} | {pct(m.get('top10_strict_route'))} | {pct(m.get('top10_relaxed_route'))} | {pct(m.get('top200_relaxed_route'))} | {pct(m.get('top200_usable_relaxed_route'))} |"
        )
    if records.get("experiments"):
        lines.extend(["", "## Discovered Experiment Metric Files", ""])
        for name, item in sorted(records["experiments"].items()):
            lines.append(f"- `{name}` ({item.get('stage')}): `{item.get('path')}`")
    lines.append("")
    return "\n".join(lines)


def build_registry(project_root: Path, output_dir: Path, include_experiments: bool = True) -> MetricsRegistry:
    registry = MetricsRegistry(project_root)
    registry.collect_current_baselines()
    if include_experiments:
        registry.collect_experiment_metrics(output_dir)
    return registry


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a normalized SynPred Stage2/Stage3/Stage35 metrics registry.")
    ap.add_argument("--project_root", default=".", help="Repository root.")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Auto-improvement output directory.")
    ap.add_argument("--include_experiments", type=int, default=1, help="Scan output_dir for extra metrics JSON files.")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    registry = build_registry(project_root, Path(args.output_dir), include_experiments=bool(args.include_experiments))
    json_path, md_path = registry.write(Path(args.output_dir))
    print(json.dumps({"registry_json": str(json_path), "registry_report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
