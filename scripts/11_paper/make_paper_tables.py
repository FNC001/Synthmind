#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
AUTO_DIR = SCRIPT_DIR.parent / "08_auto_improve"
if str(AUTO_DIR) not in sys.path:
    sys.path.insert(0, str(AUTO_DIR))

from metrics_registry import build_registry  # noqa: E402


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(row[c]) else str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def pct(x: Any) -> Any:
    try:
        return round(100 * float(x), 2)
    except Exception:
        return x


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_table(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(md_table(df), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create SynPred paper-ready result tables.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_dir", default="outputs/autorun/24h_optimization_20260613/08_tables")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    reg = build_registry(root, out.parent.parent, include_experiments=True).records["baselines"]

    # Table 1
    rows: List[Dict[str, Any]] = []
    target_dir = root / "data/interim/generative/stage3_condition_targets_v3_20260610"
    for split in ["train", "val", "test"]:
        csv_path = target_dir / f"{split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            rows.append(
                {
                    "split": split,
                    "n_samples": len(df),
                    "n_reaction_methods": df.get("reaction_method", pd.Series(dtype=str)).nunique(),
                    "core_samples": int(df.get("reaction_method", pd.Series(dtype=str)).isin(["solid_state", "solution", "melt_arc"]).sum()),
                    "condition_label_columns": ",".join([c for c in ["temperature_c", "time_h", "atmosphere", "solvent"] if c in df.columns]),
                }
            )
    save_table(pd.DataFrame(rows), out, "table1_dataset_statistics")

    # Table 2
    stage2_rows = []
    for label, key in [
        ("v5 all-method", "stage2_v5_all_test"),
        ("core final", "stage2_core_calibrated_test"),
    ]:
        m = reg.get(key, {}).get("metrics", {})
        stage2_rows.append(
            {
                "model": label,
                "top1_exact_pct": pct(m.get("top1_exact")),
                "top10_exact_pct": pct(m.get("top10_exact")),
                "top200_exact_pct": pct(m.get("top200_exact")),
                "top500_exact_pct": pct(m.get("top500_exact")),
                "best_jaccard_at_500_pct": pct(m.get("top500_best_jaccard")),
            }
        )
    save_table(pd.DataFrame(stage2_rows), out, "table2_stage2_ablation")

    # Table 3
    stage3_rows = []
    for label, miss_key, strict_key in [
        ("v3 final", "stage3_v3_missing_aware_test", "stage3_v3_strict_comparable_test"),
        ("v4 alignment", "stage3_v4_missing_aware_test", "stage3_v4_strict_comparable_test"),
    ]:
        mm = reg.get(miss_key, {}).get("metrics", {})
        sm = reg.get(strict_key, {}).get("metrics", {})
        stage3_rows.append(
            {
                "model": label,
                "missing_top1_relaxed_pct": pct(mm.get("top1_relaxed_condition")),
                "missing_top10_relaxed_pct": pct(mm.get("top10_relaxed_condition")),
                "strict_top1_relaxed_pct": pct(sm.get("top1_relaxed_condition")),
                "strict_top10_relaxed_pct": pct(sm.get("top10_relaxed_condition")),
                "temp_MAE": mm.get("temperature_MAE", ""),
                "time_MAE": mm.get("time_MAE", ""),
                "atmosphere_acc": mm.get("atmosphere_accuracy", ""),
            }
        )
    save_table(pd.DataFrame(stage3_rows), out, "table3_stage3_condition_prediction")

    # Table 4
    stage35_rows = []
    for label, miss_key, strict_key in [
        ("v3 final blend", "stage35_v3_final_missing_aware_test", "stage35_v3_final_strict_comparable_test"),
        ("v4 alignment", "stage35_v4_missing_aware_test", "stage35_v4_strict_comparable_test"),
    ]:
        mm = reg.get(miss_key, {}).get("metrics", {})
        sm = reg.get(strict_key, {}).get("metrics", {})
        stage35_rows.append(
            {
                "model": label,
                "missing_top1_relaxed_pct": pct(mm.get("top1_relaxed_route")),
                "missing_top10_relaxed_pct": pct(mm.get("top10_relaxed_route")),
                "missing_top200_relaxed_pct": pct(mm.get("top200_relaxed_route")),
                "strict_top1_relaxed_pct": pct(sm.get("top1_relaxed_route")),
                "strict_top10_relaxed_pct": pct(sm.get("top10_relaxed_route")),
                "strict_top200_relaxed_pct": pct(sm.get("top200_relaxed_route")),
            }
        )
    save_table(pd.DataFrame(stage35_rows), out, "table4_stage35_route_prediction")

    # Table 5
    by_method = reg.get("stage2_v5_by_reaction_method", {}).get("records", [])
    save_table(pd.DataFrame(by_method), out, "table5_core_method_performance")

    # Table 6
    failure = reg.get("stage2_v5_by_failure_type", {}).get("records", [])
    save_table(pd.DataFrame(failure), out, "table6_failure_analysis")

    print(json.dumps({"tables_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

