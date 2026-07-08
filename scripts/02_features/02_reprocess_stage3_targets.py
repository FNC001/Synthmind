#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def normalize_text_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s or s == "nan":
        return None
    return s


def map_atmosphere_coarse(x: Any) -> Optional[str]:
    s = normalize_text_label(x)
    if s is None:
        return None

    # oxidizing / air-like
    if s in {"air", "o2", "co2"}:
        return "air_or_oxidizing"

    # inert family
    if s in {"ar", "n2", "he", "inert"}:
        return "inert"

    # reducing family
    if s in {"h2", "nh3"}:
        return "reducing"

    if s == "vacuum":
        return "vacuum"

    return "other"


def map_time_bucket(x: Any) -> Optional[str]:
    v = safe_float(x)
    if v is None:
        return None
    if v <= 5:
        return "short"
    if v <= 24:
        return "medium"
    return "long"


def add_stage3_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # normalized time
    time_vals = []
    time_log_vals = []
    time_bucket_vals = []

    for x in out["target_time_h"].tolist():
        v = safe_float(x)
        if v is None or v < 0:
            time_vals.append(np.nan)
            time_log_vals.append(np.nan)
            time_bucket_vals.append(None)
        else:
            time_vals.append(v)
            time_log_vals.append(math.log1p(v))
            time_bucket_vals.append(map_time_bucket(v))

    out["target_time_h_clean"] = time_vals
    out["target_time_h_log1p"] = time_log_vals
    out["target_time_bucket"] = time_bucket_vals

    # temperature cleaned copy
    temp_vals = []
    for x in out["target_temperature_c"].tolist():
        v = safe_float(x)
        if v is None or v < 0 or v > 2000:
            temp_vals.append(np.nan)
        else:
            temp_vals.append(v)
    out["target_temperature_c_clean"] = temp_vals

    # atmosphere coarse
    out["target_atmosphere_coarse"] = out["target_atmosphere"].apply(map_atmosphere_coarse)

    # solvent normalized copy
    out["target_solvent_clean"] = out["target_solvent"].apply(normalize_text_label)

    return out


def summarize_df(df: pd.DataFrame) -> Dict[str, Any]:
    def vc(series: pd.Series, topn: int = 20) -> Dict[str, int]:
        s = series.dropna().astype(str)
        return {k: int(v) for k, v in s.value_counts().head(topn).to_dict().items()}

    return {
        "n_rows": int(len(df)),
        "n_temp_nonnull": int(df["target_temperature_c_clean"].notna().sum()),
        "n_time_nonnull": int(df["target_time_h_clean"].notna().sum()),
        "n_time_log_nonnull": int(df["target_time_h_log1p"].notna().sum()),
        "n_time_bucket_nonnull": int(df["target_time_bucket"].notna().sum()),
        "n_atmosphere_nonnull": int(df["target_atmosphere"].notna().sum()),
        "n_atmosphere_coarse_nonnull": int(df["target_atmosphere_coarse"].notna().sum()),
        "n_solvent_nonnull": int(df["target_solvent_clean"].notna().sum()),
        "time_bucket_dist": vc(df["target_time_bucket"]),
        "atmosphere_coarse_dist": vc(df["target_atmosphere_coarse"]),
        "solvent_dist_top10": vc(df["target_solvent_clean"], topn=10),
    }


def process_one_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(input_path)
    df2 = add_stage3_targets(df)
    df2.to_csv(output_path, index=False)
    return summarize_df(df2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reprocess stage3 targets: log1p(time), bucketed time, coarse atmosphere.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/features/structdesc_features",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/features/structdesc_features_stage3_v2",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # copy stage2 files through unchanged for convenience
    stage2_files = [
        "stage2_train_raw.csv",
        "stage2_train_ml.csv",
        "stage2_val_raw.csv",
        "stage2_val_ml.csv",
        "stage2_test_raw.csv",
        "stage2_test_ml.csv",
        "stage2_gold_train_holdout_raw.csv",
        "stage2_gold_train_holdout_ml.csv",
    ]
    for fn in stage2_files:
        src = input_dir / fn
        dst = output_dir / fn
        if src.exists():
            df = pd.read_csv(src)
            df.to_csv(dst, index=False)

    # stage3 files to transform
    stage3_files = [
        "stage3_train_raw.csv",
        "stage3_val_raw.csv",
        "stage3_test_raw.csv",
        "stage3_gold_train_holdout_raw.csv",
    ]

    summary: Dict[str, Any] = {
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
        "files": {},
    }

    for fn in stage3_files:
        src = input_dir / fn
        dst = output_dir / fn
        if src.exists():
            summary["files"][fn] = process_one_file(src, dst)

    # copy meta if present
    meta_in = input_dir / "meta"
    meta_out = output_dir / "meta"
    ensure_dir(meta_out)
    if meta_in.exists():
        for p in meta_in.glob("*"):
            if p.is_file():
                try:
                    if p.suffix.lower() == ".json":
                        with open(p, "r", encoding="utf-8") as f:
                            obj = json.load(f)
                        write_json(meta_out / p.name, obj)
                    else:
                        (meta_out / p.name).write_bytes(p.read_bytes())
                except Exception:
                    pass

    write_json(output_dir / "meta" / "stage3_reprocess_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
