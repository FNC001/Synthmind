#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_DIR = SCRIPT_DIR.parent / "08_auto_improve"
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from metrics_registry import build_registry, pct  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a compact SynPred results summary for reports.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/autorun/24h_optimization_20260613")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    reg = build_registry(root, out, include_experiments=True).records["baselines"]
    s2 = reg["stage2_v5_all_test"]["metrics"]
    s2c = reg["stage2_core_calibrated_test"]["metrics"]
    s3 = reg["stage3_v3_missing_aware_test"]["metrics"]
    r3 = reg["stage35_v3_final_missing_aware_test"]["metrics"]
    text = "\n".join(
        [
            "# SynPred Results Summary",
            "",
            f"- Stage2 v5 all-method top1/top10/top500 exact: {pct(s2.get('top1_exact'))} / {pct(s2.get('top10_exact'))} / {pct(s2.get('top500_exact'))}.",
            f"- Stage2 core top1/top10/top500 exact: {pct(s2c.get('top1_exact'))} / {pct(s2c.get('top10_exact'))} / {pct(s2c.get('top500_exact'))}.",
            f"- Stage3 v3 missing-aware top1/top10 relaxed condition: {pct(s3.get('top1_relaxed_condition'))} / {pct(s3.get('top10_relaxed_condition'))}.",
            f"- Stage35 v3 final missing-aware top1/top10 relaxed route: {pct(r3.get('top1_relaxed_route'))} / {pct(r3.get('top10_relaxed_route'))}.",
            "",
        ]
    )
    (out / "RESULTS_SUMMARY.md").write_text(text, encoding="utf-8")
    print(json.dumps({"summary": str(out / "RESULTS_SUMMARY.md")}, indent=2))


if __name__ == "__main__":
    main()

