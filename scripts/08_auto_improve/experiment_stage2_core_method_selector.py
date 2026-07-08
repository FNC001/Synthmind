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


def build(project_root: Path) -> Dict[str, Any]:
    registry = build_registry(project_root, DEFAULT_OUTPUT_DIR, include_experiments=False).records
    all_m = registry["baselines"]["stage2_v5_all_test"]["metrics"]
    core_m = registry["baselines"]["stage2_core_calibrated_test"]["metrics"]
    t = BASELINE_THRESHOLDS["stage2"]
    checks = {
        "core_top1_gt_46_15": core_m.get("top1_exact", 0.0) > t["core_top1_exact"],
        "core_top10_ge_69_17": core_m.get("top10_exact", 0.0) >= t["core_top10_exact"],
        "core_top500_ge_85_33": core_m.get("top500_exact", 0.0) >= t["core_top500_exact"],
    }
    mode = "core_mode_candidate" if all(checks.values()) else "diagnostic_only"
    return {
        "experiment": "stage2_core_method_selector",
        "status": mode,
        "all_method_metrics": all_m,
        "core_method_metrics": core_m,
        "checks": checks,
        "decision": "Do not replace default Stage2 unless core top1/top10/top500 gates all pass.",
    }


def render(obj: Dict[str, Any]) -> str:
    c = obj["core_method_metrics"]
    return "\n".join(
        [
            "# Stage2 Core Method Selector",
            "",
            f"Status: `{obj['status']}`",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| core top1 exact | {pct(c.get('top1_exact'))} |",
            f"| core top10 exact | {pct(c.get('top10_exact'))} |",
            f"| core top500 exact | {pct(c.get('top500_exact'))} |",
            "",
            obj["decision"],
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate whether core methods should use core-specific Stage2 artifacts.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "stage2_core_selector"))
    args = ap.parse_args()
    project_root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)
    obj = build(project_root)
    write_json(out / "stage2_model_selection.json", obj)
    (out / "stage2_core_method_selector_report.md").write_text(render(obj), encoding="utf-8")
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

