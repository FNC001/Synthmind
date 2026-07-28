#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics_registry import BASELINE_THRESHOLDS, DEFAULT_OUTPUT_DIR, build_registry, pct, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage2 top1/top10 calibration gate check using current v5 artifacts.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "stage2_top1_calibration"))
    ap.add_argument("--run_test_only_if_val_pass", type=int, default=1)
    args = ap.parse_args()
    project_root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)
    registry = build_registry(project_root, out.parent, include_experiments=False).records
    metrics = registry["baselines"]["stage2_v5_all_test"]["metrics"]
    t = BASELINE_THRESHOLDS["stage2"]
    checks = {
        "all_top1_gt_39_47": metrics.get("top1_exact", 0.0) > t["all_top1_exact"],
        "all_top10_ge_63_35": metrics.get("top10_exact", 0.0) >= t["all_top10_exact"],
        "all_top500_ge_80_24": metrics.get("top500_exact", 0.0) >= t["all_top500_exact"],
    }
    obj: Dict[str, Any] = {
        "experiment": "stage2_top1_calibration",
        "status": "no_new_default_candidate",
        "reason": "No new calibration search was run in this lightweight autorun step; current v5 is recorded as baseline.",
        "metrics": metrics,
        "checks": checks,
    }
    write_json(out / "stage2_top1_calibration_metrics.json", obj)
    (out / "stage2_top1_calibration_report.md").write_text(
        "# Stage2 Top1 Calibration\n\n"
        f"Current v5 top1/top10/top500: {pct(metrics.get('top1_exact'))} / {pct(metrics.get('top10_exact'))} / {pct(metrics.get('top500_exact'))}.\n\n"
        "No default replacement is selected by this lightweight gate check.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

