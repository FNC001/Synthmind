#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics_registry import DEFAULT_OUTPUT_DIR, build_registry, pct, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Stage3 condition calibration candidates against v3 final gates.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "condition_calibration_search"))
    args = ap.parse_args()
    project_root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)
    reg = build_registry(project_root, out.parent, include_experiments=False).records["baselines"]
    v3m = reg["stage3_v3_missing_aware_test"]["metrics"]
    v3s = reg["stage3_v3_strict_comparable_test"]["metrics"]
    v4m = reg["stage3_v4_missing_aware_test"]["metrics"]
    v4s = reg["stage3_v4_strict_comparable_test"]["metrics"]
    checks = {
        "strict_top1_relaxed_improves": v4s.get("top1_relaxed_condition", 0.0) > v3s.get("top1_relaxed_condition", 0.0),
        "missing_top1_relaxed_not_lower": v4m.get("top1_relaxed_condition", 0.0) >= v3m.get("top1_relaxed_condition", 0.0),
        "strict_top10_relaxed_not_lower": v4s.get("top10_relaxed_condition", 0.0) >= v3s.get("top10_relaxed_condition", 0.0),
        "missing_top10_relaxed_not_lower": v4m.get("top10_relaxed_condition", 0.0) >= v3m.get("top10_relaxed_condition", 0.0),
    }
    obj = {
        "experiment": "condition_calibration_search",
        "status": "no_replacement" if not all(checks.values()) else "candidate",
        "baseline_v3": {"missing_aware": v3m, "strict_comparable": v3s},
        "candidate_v4": {"missing_aware": v4m, "strict_comparable": v4s},
        "checks": checks,
    }
    write_json(out / "condition_calibration_search_metrics.json", obj)
    (out / "condition_calibration_search_report.md").write_text(
        "# Stage3 Condition Calibration Search\n\n"
        f"v3 missing-aware top1 relaxed: {pct(v3m.get('top1_relaxed_condition'))}; v4: {pct(v4m.get('top1_relaxed_condition'))}.\n\n"
        f"v3 strict-comparable top1 relaxed: {pct(v3s.get('top1_relaxed_condition'))}; v4: {pct(v4s.get('top1_relaxed_condition'))}.\n\n"
        f"Decision: `{obj['status']}`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

