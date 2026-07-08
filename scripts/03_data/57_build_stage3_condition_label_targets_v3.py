#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


MISSING_VALUES = {"", "nan", "none", "null", "<unk_or_missing>", "<UNK_OR_MISSING>", "unknown", "missing"}


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
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_table(path_base: Path) -> pd.DataFrame:
    csv_path = path_base.with_suffix(".csv")
    parquet_path = path_base.with_suffix(".parquet")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    return pd.read_csv(csv_path)


def write_table(df: pd.DataFrame, path_base: Path) -> None:
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    try:
        df.to_parquet(path_base.with_suffix(".parquet"), index=False)
    except Exception as exc:
        path_base.with_suffix(".parquet.SKIPPED.txt").write_text(str(exc), encoding="utf-8")


def norm_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "<UNK_OR_MISSING>"
    s = str(value).strip().lower()
    if s in MISSING_VALUES:
        return "<UNK_OR_MISSING>"
    aliases = {
        "argon": "ar",
        "oxygen": "o2",
        "nitrogen": "n2",
        "air atmosphere": "air",
        "water": "water",
        "h2o": "water",
        "ethyl alcohol": "ethanol",
        "etoh": "ethanol",
    }
    return aliases.get(s, s)


def temp_bin(method: str, temp: float) -> str:
    if not np.isfinite(temp):
        return "missing"
    m = str(method)
    if m == "solid_state":
        bins = [400, 600, 800, 1000, 1200, 1400]
        labels = ["<400", "400-600", "600-800", "800-1000", "1000-1200", "1200-1400", ">1400"]
    elif m in {"solution", "hydro_solvothermal", "precipitation", "sol_gel"}:
        bins = [80, 150, 250, 400, 700]
        labels = ["<80", "80-150", "150-250", "250-400", "400-700", ">700"]
    elif m == "melt_arc":
        bins = [500, 900, 1200]
        labels = ["not_applicable", "post_anneal_low", "post_anneal_medium", "post_anneal_high"]
    else:
        bins = [100, 300, 600, 900, 1200]
        labels = ["<100", "100-300", "300-600", "600-900", "900-1200", ">1200"]
    idx = int(np.searchsorted(np.asarray(bins, dtype=float), float(temp), side="right"))
    return labels[min(idx, len(labels) - 1)]


def temp_bin_center(method: str, label: str) -> float:
    centers = {
        "<400": 300, "400-600": 500, "600-800": 700, "800-1000": 900, "1000-1200": 1100, "1200-1400": 1300, ">1400": 1500,
        "<80": 60, "80-150": 115, "150-250": 200, "250-400": 325, "400-700": 550, ">700": 800,
        "not_applicable": 25, "post_anneal_low": 700, "post_anneal_medium": 1050, "post_anneal_high": 1350,
        "<100": 60, "100-300": 200, "300-600": 450, "600-900": 750, "900-1200": 1050, ">1200": 1350,
    }
    return float(centers.get(str(label), np.nan))


def time_bin(time_h: float) -> str:
    if not np.isfinite(time_h):
        return "missing"
    bins = [1, 6, 12, 24, 48, 96]
    labels = ["<1h", "1-6h", "6-12h", "12-24h", "24-48h", "48-96h", ">96h"]
    idx = int(np.searchsorted(np.asarray(bins, dtype=float), float(time_h), side="right"))
    return labels[min(idx, len(labels) - 1)]


def time_bin_center(label: str) -> float:
    return float({"<1h": 0.5, "1-6h": 3, "6-12h": 9, "12-24h": 18, "24-48h": 36, "48-96h": 72, ">96h": 144}.get(str(label), np.nan))


def group_key(row: pd.Series) -> str:
    fam_cols = [c for c in row.index if c.startswith("precursor_family_count__")]
    fam = [c.replace("precursor_family_count__", "") for c in fam_cols if float(row.get(c, 0) or 0) > 0]
    return "|".join([str(row.get("formula", "")), str(row.get("reaction_method", "")), "+".join(sorted(fam))])


def add_targets(df: pd.DataFrame, group_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    temp = pd.to_numeric(out["temperature_c"], errors="coerce")
    time_h = pd.to_numeric(out["time_h"], errors="coerce")
    out["temperature_c_raw"] = temp
    out["time_h_raw"] = time_h
    out["log_time_h"] = np.log1p(time_h.clip(lower=0))
    out["temperature_clipped"] = temp.clip(lower=20, upper=1800)
    out["time_h_clipped"] = time_h.clip(lower=0.05, upper=500)
    out["log_time_clipped"] = np.log1p(out["time_h_clipped"])
    out["temperature_bin"] = [temp_bin(m, t) for m, t in zip(out["reaction_method"], out["temperature_clipped"])]
    out["temperature_bin_center"] = [temp_bin_center(m, b) for m, b in zip(out["reaction_method"], out["temperature_bin"])]
    out["time_bin"] = [time_bin(t) for t in out["time_h_clipped"]]
    out["time_bin_center"] = [time_bin_center(b) for b in out["time_bin"]]
    out["atmosphere_raw"] = out.get("atmosphere", "<UNK_OR_MISSING>")
    out["solvent_raw"] = out.get("solvent", "<UNK_OR_MISSING>")
    out["atmosphere_normalized"] = out["atmosphere_raw"].apply(norm_label)
    out["solvent_normalized"] = out["solvent_raw"].apply(norm_label)
    out["atmosphere_known_mask"] = (out["atmosphere_normalized"] != "<UNK_OR_MISSING>").astype(int)
    out["solvent_known_mask"] = (out["solvent_normalized"] != "<UNK_OR_MISSING>").astype(int)
    out["atmosphere_target_class"] = out["atmosphere_normalized"].where(out["atmosphere_known_mask"].astype(bool), "MASKED_MISSING")
    out["solvent_target_class"] = out["solvent_normalized"].where(out["solvent_known_mask"].astype(bool), "MASKED_MISSING")
    out["atmosphere_missing_reason"] = np.where(out["atmosphere_known_mask"] == 1, "known", "not_reported_or_unparsed")
    out["solvent_missing_reason"] = np.where(out["solvent_known_mask"] == 1, "known", "not_reported_or_unparsed")
    out["condition_group_key"] = out.apply(group_key, axis=1)
    if group_stats is not None:
        out = out.merge(group_stats, on="condition_group_key", how="left")
    return out


def make_group_stats(train: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, g in train.groupby("condition_group_key", sort=False):
        temp = pd.to_numeric(g["temperature_clipped"], errors="coerce").dropna()
        time_h = pd.to_numeric(g["time_h_clipped"], errors="coerce").dropna()
        if len(temp) == 0 or len(time_h) == 0:
            continue
        rec: Dict[str, Any] = {
            "condition_group_key": key,
            "multimodal_group_id": abs(hash(key)) % 10_000_000_000,
            "multimodal_group_size": int(len(g)),
        }
        for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
            suffix = str(int(q * 100))
            rec[f"temperature_p{suffix}"] = float(temp.quantile(q))
            rec[f"time_p{suffix}"] = float(time_h.quantile(q))
        rec["temperature_iqr"] = rec["temperature_p75"] - rec["temperature_p25"]
        rec["time_iqr"] = rec["time_p75"] - rec["time_p25"]
        rec["is_multimodal_group"] = int(rec["multimodal_group_size"] >= 3 and (rec["temperature_iqr"] > 200 or rec["time_iqr"] > 48))
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "temperature_bins": {str(k): int(v) for k, v in df["temperature_bin"].value_counts().items()},
        "time_bins": {str(k): int(v) for k, v in df["time_bin"].value_counts().items()},
        "atmosphere_known": int(df["atmosphere_known_mask"].sum()),
        "atmosphere_missing": int((1 - df["atmosphere_known_mask"]).sum()),
        "solvent_known": int(df["solvent_known_mask"].sum()),
        "solvent_missing": int((1 - df["solvent_known_mask"]).sum()),
        "multimodal_rows": int(pd.to_numeric(df.get("is_multimodal_group", 0), errors="coerce").fillna(0).sum()),
        "temperature_outliers": int(((pd.to_numeric(df["temperature_c_raw"], errors="coerce") < 20) | (pd.to_numeric(df["temperature_c_raw"], errors="coerce") > 1800)).sum()),
        "time_outliers": int(((pd.to_numeric(df["time_h_raw"], errors="coerce") < 0.05) | (pd.to_numeric(df["time_h_raw"], errors="coerce") > 500)).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage3 v3 distributional condition targets and missing-aware labels.")
    ap.add_argument("--input_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v3_20260610")
    ap.add_argument("--output_dir", default="data/interim/generative/stage3_condition_targets_v3_20260610")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_schema = json.loads((input_dir / "schema.json").read_text(encoding="utf-8"))
    train0 = add_targets(read_table(input_dir / "train"))
    group_stats = make_group_stats(train0)
    train = add_targets(read_table(input_dir / "train"), group_stats)
    val = add_targets(read_table(input_dir / "val"), group_stats)
    test = add_targets(read_table(input_dir / "test"), group_stats)
    fill_cols = [c for c in group_stats.columns if c != "condition_group_key"]
    for df in [train, val, test]:
        for col in fill_cols:
            if col not in df.columns:
                df[col] = np.nan
        df["multimodal_group_size"] = pd.to_numeric(df["multimodal_group_size"], errors="coerce").fillna(1).astype(int)
        df["is_multimodal_group"] = pd.to_numeric(df["is_multimodal_group"], errors="coerce").fillna(0).astype(int)
        for col in ["temperature_p10", "temperature_p25", "temperature_p50", "temperature_p75", "temperature_p90"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["temperature_clipped"])
        for col in ["time_p10", "time_p25", "time_p50", "time_p75", "time_p90"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df["time_h_clipped"])
        df["temperature_iqr"] = pd.to_numeric(df["temperature_iqr"], errors="coerce").fillna(0.0)
        df["time_iqr"] = pd.to_numeric(df["time_iqr"], errors="coerce").fillna(0.0)

    for split, df in [("train", train), ("val", val), ("test", test)]:
        write_table(df, output_dir / split)
    group_stats.to_csv(output_dir / "train_condition_group_distribution_stats.csv", index=False)
    schema = dict(base_schema)
    schema["stage3_condition_targets_v3"] = {
        "continuous_targets": ["temperature_clipped", "log_time_clipped"],
        "bin_targets": ["temperature_bin", "time_bin"],
        "missing_aware_targets": ["atmosphere_target_class", "solvent_target_class"],
    }
    write_json(output_dir / "schema.json", schema)
    summary = {"config": vars(args), "group_stats_rows": int(len(group_stats)), "splits": {s: summarize(d) for s, d in [("train", train), ("val", val), ("test", test)]}}
    write_json(output_dir / "summary.json", summary)
    report = ["# Stage3 Condition Targets v3", "", json.dumps(to_builtin(summary), ensure_ascii=False, indent=2)]
    (output_dir / "stage3_condition_targets_v3_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
