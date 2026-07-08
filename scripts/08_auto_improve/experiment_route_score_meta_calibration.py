#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare route score meta-calibration candidates and classify optional modes.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/auto_improve/synpred_auto_v1_20260612/route_score_meta_calibration")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    v3 = read_json(root / "runs/stage35/route_reranker_v3_final_20260612/test_metrics.json")
    v4 = read_json(root / "runs/stage35/route_reranker_v4_20260612/test_metrics.json")
    v3m = v3.get("blend_missing_aware", {})
    v3s = v3.get("blend_strict_comparable", {})
    v4m = v4.get("blend_missing_aware", {})
    v4s = v4.get("blend_strict_comparable", {})
    coverage_mode = bool(
        v4m.get("top200_relaxed_route", 0.0) > v3m.get("top200_relaxed_route", 0.0)
        or v4s.get("top10_relaxed_route", 0.0) > v3s.get("top10_relaxed_route", 0.0)
    )
    top1_loss = bool(
        v4m.get("top1_relaxed_route", 0.0) < v3m.get("top1_relaxed_route", 0.0)
        or v4s.get("top1_relaxed_route", 0.0) < v3s.get("top1_relaxed_route", 0.0)
    )
    status = "coverage_mode" if coverage_mode and top1_loss else "no_replacement"
    obj: Dict[str, Any] = {
        "experiment": "route_score_meta_calibration",
        "status": status,
        "baseline_v3": {"missing_aware": v3m, "strict_comparable": v3s},
        "candidate_v4": {"missing_aware": v4m, "strict_comparable": v4s},
    }
    (out / "route_score_meta_calibration_metrics.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    (out / "route_score_meta_calibration_report.md").write_text(
        "# Route Score Meta Calibration\n\n"
        f"Status: `{status}`. v4 is not selected as default when top1 relaxed route is lower than v3 final.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

