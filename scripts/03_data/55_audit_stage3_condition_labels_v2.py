#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def describe_numeric(df: pd.DataFrame, col: str, group_cols: List[str] | None = None) -> pd.DataFrame:
    d = df[df[col].notna()].copy()
    if group_cols:
        rows = []
        for key, sub in d.groupby(group_cols, dropna=False):
            key_vals = key if isinstance(key, tuple) else (key,)
            vals = sub[col].astype(float)
            rec = {g: v for g, v in zip(group_cols, key_vals)}
            rec.update(summary_stats(vals))
            rows.append(rec)
        return pd.DataFrame(rows)
    rec = summary_stats(d[col].astype(float))
    return pd.DataFrame([rec])


def summary_stats(vals: pd.Series) -> Dict[str, float]:
    qs = vals.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "p01": float(qs.loc[0.01]),
        "p05": float(qs.loc[0.05]),
        "p25": float(qs.loc[0.25]),
        "median": float(qs.loc[0.50]),
        "p75": float(qs.loc[0.75]),
        "p95": float(qs.loc[0.95]),
        "p99": float(qs.loc[0.99]),
        "max": float(vals.max()),
    }


def value_distribution(df: pd.DataFrame, col: str, group_col: str | None = None) -> pd.DataFrame:
    if group_col is None:
        out = df[col].fillna("<NA>").astype(str).value_counts().rename_axis(col).reset_index(name="count")
        out["fraction"] = out["count"] / max(len(df), 1)
        return out
    rows = []
    for method, sub in df.groupby(group_col, dropna=False):
        counts = sub[col].fillna("<NA>").astype(str).value_counts()
        for label, count in counts.items():
            rows.append({group_col: method, col: label, "count": int(count), "fraction": float(count / max(len(sub), 1))})
    return pd.DataFrame(rows)


def condition_multimodal_groups(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["formula", "predicted_precursor_set_chem_checked", "reaction_method"]
    rows = []
    for key, sub in df.groupby(key_cols, dropna=False):
        temps = sub["temperature_c"].dropna().astype(float)
        times = sub["time_h"].dropna().astype(float)
        if len(sub) < 2:
            continue
        temp_range = float(temps.max() - temps.min()) if len(temps) else 0.0
        time_range = float(times.max() - times.min()) if len(times) else 0.0
        if temp_range >= 200.0 or time_range >= 48.0:
            rec = {c: v for c, v in zip(key_cols, key)}
            rec.update({
                "n_rows": int(len(sub)),
                "temperature_min": float(temps.min()) if len(temps) else np.nan,
                "temperature_max": float(temps.max()) if len(temps) else np.nan,
                "temperature_range": temp_range,
                "time_min_h": float(times.min()) if len(times) else np.nan,
                "time_max_h": float(times.max()) if len(times) else np.nan,
                "time_range_h": time_range,
                "sample_ids": json.dumps(sub["sample_id"].head(20).tolist(), ensure_ascii=False),
            })
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["temperature_range", "time_range_h"], ascending=False) if rows else pd.DataFrame()


def implausible_condition_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        method = str(r.get("reaction_method", ""))
        t = float(r.get("temperature_c", np.nan))
        h = float(r.get("time_h", np.nan))
        reason = []
        if method == "solid_state" and np.isfinite(t) and t < 250:
            reason.append("solid_state_low_temperature")
        if method == "solution" and np.isfinite(t) and t > 900:
            reason.append("solution_high_temperature")
        if method == "melt_arc" and np.isfinite(h) and h > 240:
            reason.append("melt_arc_long_time")
        if method == "melt_arc" and np.isfinite(t) and t < 300:
            reason.append("melt_arc_low_reported_temperature")
        if reason:
            rec = r[["sample_id", "formula", "reaction_method", "temperature_c", "time_h", "atmosphere", "solvent"]].to_dict()
            rec["noise_reason"] = ";".join(reason)
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Stage3 condition labels, units, outliers, and multimodality.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_dataset_chem_checked/method_stratified_v5_20260610")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_condition_label_audit_v2_20260610")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    split_dfs = []
    for split in ["train", "val", "test"]:
        p = input_dir / f"{split}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        split_dfs.append(df)
    df_all = pd.concat(split_dfs, ignore_index=True)
    df_all["core_group"] = np.where(df_all["reaction_method"].isin(CORE_METHODS), "core", "non_core")

    temperature_all = describe_numeric(df_all, "temperature_c")
    temperature_by_method = describe_numeric(df_all, "temperature_c", ["reaction_method"])
    temperature_by_source = describe_numeric(df_all, "temperature_c", ["source_dataset"])
    temperature_by_core = describe_numeric(df_all, "temperature_c", ["core_group"])
    time_all = describe_numeric(df_all, "time_h")
    time_by_method = describe_numeric(df_all, "time_h", ["reaction_method"])
    time_by_source = describe_numeric(df_all, "time_h", ["source_dataset"])
    time_by_core = describe_numeric(df_all, "time_h", ["core_group"])

    temperature_outliers = df_all[(df_all["temperature_c"] < 20) | (df_all["temperature_c"] > 1800)].copy()
    time_outliers = df_all[(df_all["time_h"] < 0.1) | (df_all["time_h"] > 1000)].copy()
    multimodal = condition_multimodal_groups(df_all)
    noise = implausible_condition_rows(df_all)
    method_dist = df_all.groupby(["split", "reaction_method"]).agg(
        n=("sample_id", "count"),
        temp_median=("temperature_c", "median"),
        time_median=("time_h", "median"),
        atmosphere_mode=("atmosphere", lambda x: x.fillna("<NA>").astype(str).mode().iloc[0] if len(x.mode()) else "<NA>"),
        solvent_mode=("solvent", lambda x: x.fillna("<NA>").astype(str).mode().iloc[0] if len(x.mode()) else "<NA>"),
    ).reset_index()

    artifacts = {
        "temperature_distribution_all.csv": temperature_all,
        "temperature_distribution_by_method.csv": temperature_by_method,
        "temperature_distribution_by_source.csv": temperature_by_source,
        "temperature_distribution_by_core.csv": temperature_by_core,
        "time_distribution_all.csv": time_all,
        "time_distribution_by_method.csv": time_by_method,
        "time_distribution_by_source.csv": time_by_source,
        "time_distribution_by_core.csv": time_by_core,
        "atmosphere_distribution.csv": value_distribution(df_all, "atmosphere"),
        "atmosphere_distribution_by_method.csv": value_distribution(df_all, "atmosphere", "reaction_method"),
        "solvent_distribution.csv": value_distribution(df_all, "solvent"),
        "solvent_distribution_by_method.csv": value_distribution(df_all, "solvent", "reaction_method"),
        "temperature_outliers.csv": temperature_outliers,
        "time_outliers.csv": time_outliers,
        "condition_multimodal_groups.csv": multimodal,
        "condition_noise_candidates.csv": noise,
        "method_condition_distribution.csv": method_dist,
    }
    for name, table in artifacts.items():
        table.to_csv(output_dir / name, index=False)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_rows": int(len(df_all)),
        "split_counts": df_all["split"].value_counts().to_dict(),
        "method_counts": df_all["reaction_method"].value_counts().to_dict(),
        "core_counts": df_all["core_group"].value_counts().to_dict(),
        "temperature_outliers": int(len(temperature_outliers)),
        "time_outliers": int(len(time_outliers)),
        "multimodal_groups": int(len(multimodal)),
        "noise_candidate_rows": int(len(noise)),
        "temperature_all": temperature_all.iloc[0].to_dict(),
        "time_all": time_all.iloc[0].to_dict(),
        "top_atmosphere": value_distribution(df_all, "atmosphere").head(10).to_dict("records"),
        "top_solvent": value_distribution(df_all, "solvent").head(10).to_dict("records"),
    }
    write_json(output_dir / "condition_label_audit_summary.json", summary)

    lines = [
        "# Stage3 Condition Label Audit v2",
        "",
        f"- Input: `{input_dir}`",
        f"- Rows: {summary['n_rows']}",
        f"- Temperature outliers (<20 C or >1800 C): {summary['temperature_outliers']}",
        f"- Time outliers (<0.1 h or >1000 h): {summary['time_outliers']}",
        f"- Multimodal formula+precursor+method groups: {summary['multimodal_groups']}",
        f"- Reaction-method noise candidate rows: {summary['noise_candidate_rows']}",
        "",
        "## Overall Temperature",
        "",
        temperature_all.to_markdown(index=False),
        "",
        "## Overall Time",
        "",
        time_all.to_markdown(index=False),
        "",
        "## By Reaction Method Temperature",
        "",
        temperature_by_method.sort_values("n", ascending=False).to_markdown(index=False),
        "",
        "## By Reaction Method Time",
        "",
        time_by_method.sort_values("n", ascending=False).to_markdown(index=False),
        "",
        "## Top Atmosphere Labels",
        "",
        value_distribution(df_all, "atmosphere").head(15).to_markdown(index=False),
        "",
        "## Top Solvent Labels",
        "",
        value_distribution(df_all, "solvent").head(15).to_markdown(index=False),
        "",
        "No rows were deleted by this audit. Outliers and noise candidates are reported only.",
    ]
    (output_dir / "condition_label_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
