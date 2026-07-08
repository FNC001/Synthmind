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
    ap = argparse.ArgumentParser(description="Stage35 pairwise hard-negative ranker autorun hook.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/auto_improve/synpred_auto_v1_20260612/pairwise_route_ranker")
    ap.add_argument("--max_train_minutes", type=float, default=120.0)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    v3 = read_json(root / "runs/stage35/route_reranker_v3_final_20260612/test_metrics.json")
    v4 = read_json(root / "runs/stage35/route_reranker_v4_20260612/test_metrics.json")
    obj: Dict[str, Any] = {
        "experiment": "pairwise_route_ranker",
        "status": "skipped_training_in_lightweight_cycle",
        "reason": "A new pairwise ranker is not trained unless the 24h controller allocates remaining train budget and validation data gates are satisfied.",
        "max_train_minutes": args.max_train_minutes,
        "baseline_v3_blend_missing_aware": v3.get("blend_missing_aware", {}),
        "baseline_v3_blend_strict_comparable": v3.get("blend_strict_comparable", {}),
        "v4_blend_missing_aware": v4.get("blend_missing_aware", {}),
        "v4_blend_strict_comparable": v4.get("blend_strict_comparable", {}),
    }
    (out / "pairwise_route_ranker_metrics.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")
    (out / "pairwise_route_ranker_report.md").write_text(
        "# Pairwise Route Ranker\n\n"
        "No new default candidate was trained in this lightweight autorun hook. Existing v3/v4 route-ranker metrics were recorded for model selection.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

