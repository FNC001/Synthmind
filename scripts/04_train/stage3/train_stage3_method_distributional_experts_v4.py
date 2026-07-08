#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


MISSING = "<UNK_OR_MISSING>"


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    pq = data_dir / f"{split}.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    return pd.read_csv(data_dir / f"{split}.csv")


def mode_or_missing(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    s = s[s.ne(MISSING)]
    return str(s.mode().iloc[0]) if len(s) else MISSING


def build_templates(train: pd.DataFrame) -> Dict[str, Any]:
    templates: Dict[str, Any] = {}
    groups = {"global": train}
    groups.update({f"method::{m}": g for m, g in train.groupby("reaction_method", sort=False)})
    for name, g in groups.items():
        temp = pd.to_numeric(g["temperature_clipped"], errors="coerce").dropna()
        time = pd.to_numeric(g["time_h_clipped"], errors="coerce").dropna()
        known_atm = g[pd.to_numeric(g["atmosphere_known_mask"], errors="coerce").fillna(0).astype(int) == 1]
        known_solv = g[pd.to_numeric(g["solvent_known_mask"], errors="coerce").fillna(0).astype(int) == 1]
        templates[name] = {
            "rows": int(len(g)),
            "temperature_point": float(temp.median()) if len(temp) else 700.0,
            "temperature_p10": float(temp.quantile(0.10)) if len(temp) else 500.0,
            "temperature_p50": float(temp.quantile(0.50)) if len(temp) else 700.0,
            "temperature_p90": float(temp.quantile(0.90)) if len(temp) else 1000.0,
            "time_point": float(time.median()) if len(time) else 24.0,
            "time_p10": float(time.quantile(0.10)) if len(time) else 6.0,
            "time_p50": float(time.quantile(0.50)) if len(time) else 24.0,
            "time_p90": float(time.quantile(0.90)) if len(time) else 72.0,
            "atmosphere": mode_or_missing(known_atm["atmosphere_target_class"]) if len(known_atm) else MISSING,
            "solvent": mode_or_missing(known_solv["solvent_target_class"]) if len(known_solv) else MISSING,
            "low_confidence": bool(name == "method::melt_arc" and len(g) < 500),
        }
    return templates


def template_metrics(df: pd.DataFrame, templates: Dict[str, Any]) -> Dict[str, Any]:
    rows = []
    for _, r in df.iterrows():
        tpl = templates.get(f"method::{r['reaction_method']}", templates["global"])
        temp_err = abs(float(r["temperature_clipped"]) - float(tpl["temperature_point"]))
        time_err = abs(float(r["time_h_clipped"]) - float(tpl["time_point"]))
        known = int(r["atmosphere_known_mask"]) == 1
        atm_ok = (not known) or str(tpl["atmosphere"]) == str(r["atmosphere_target_class"])
        rows.append({
            "reaction_method": str(r["reaction_method"]),
            "strict": int(temp_err <= 100 and time_err <= 24 and atm_ok),
            "relaxed": int(temp_err <= 200 and time_err <= 48 and atm_ok),
            "temp_abs_error": temp_err,
            "time_abs_error": time_err,
            "atm_ok": int(atm_ok),
        })
    mdf = pd.DataFrame(rows)
    out = {
        "all": {
            "temp_mae": float(mdf["temp_abs_error"].mean()),
            "time_mae": float(mdf["time_abs_error"].mean()),
            "strict_condition": float(mdf["strict"].mean()),
            "relaxed_condition": float(mdf["relaxed"].mean()),
            "atmosphere_known_acc_proxy": float(mdf["atm_ok"].mean()),
        },
        "by_method": {},
    }
    for method, g in mdf.groupby("reaction_method", sort=False):
        out["by_method"][str(method)] = {
            "rows": int(len(g)),
            "temp_mae": float(g["temp_abs_error"].mean()),
            "time_mae": float(g["time_abs_error"].mean()),
            "strict_condition": float(g["strict"].mean()),
            "relaxed_condition": float(g["relaxed"].mean()),
            "atmosphere_known_acc_proxy": float(g["atm_ok"].mean()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Stage3 method-specific distributional experts v4.")
    ap.add_argument("--dataset_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v4_20260612")
    ap.add_argument("--run_dir", default="runs/stage3/method_distributional_experts_v4_20260612")
    ap.add_argument("--python", default="/Users/lihonglin/miniconda3/envs/py311/bin/python")
    ap.add_argument("--n_estimators_reg", type=int, default=360)
    ap.add_argument("--n_estimators_clf", type=int, default=260)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    global_dir = run_dir / "global_expert"
    cmd = [
        args.python,
        "scripts/04_train/stage3/train_stage3_distributional_condition_v3.py",
        "--input_dir", args.dataset_dir,
        "--run_dir", str(global_dir),
        "--n_estimators_reg", str(args.n_estimators_reg),
        "--n_estimators_clf", str(args.n_estimators_clf),
    ]
    subprocess.run(cmd, check=True)
    src = global_dir / "stage3_distributional_condition_v3.joblib"
    if src.exists():
        shutil.copy2(src, run_dir / "stage3_method_distributional_experts_v4.joblib")
    train = read_split(Path(args.dataset_dir), "train")
    val = read_split(Path(args.dataset_dir), "val")
    test = read_split(Path(args.dataset_dir), "test")
    templates = build_templates(train)
    joblib.dump(templates, run_dir / "method_expert_templates.joblib")
    metrics = {
        "global_expert_metrics": json.loads((global_dir / "metrics.json").read_text(encoding="utf-8")) if (global_dir / "metrics.json").exists() else {},
        "method_template_metrics": {"val": template_metrics(val, templates), "test": template_metrics(test, templates)},
        "experts": sorted(templates),
        "config": vars(args),
    }
    write_json(run_dir / "metrics.json", metrics)
    report = ["# Stage3 Method Distributional Experts v4", "", "```json", json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2), "```"]
    (run_dir / "method_distributional_experts_v4_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
