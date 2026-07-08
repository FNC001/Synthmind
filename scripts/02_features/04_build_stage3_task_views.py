#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def detect_feature_cols(df: pd.DataFrame) -> List[str]:
    return [
        c for c in df.columns
        if c.startswith("feat_")
        or c.startswith("graph_emb_")
        or c.startswith("cgcnn_graph_emb_")
        or c.startswith("alignn_graph_emb_")
    ]


def detect_meta_cols(df: pd.DataFrame) -> List[str]:
    preferred = [
        "id",
        "material_id",
        "formula",
        "doi",
        "split_group",
        "source_dataset",
        "synthesis_type",
    ]
    return [c for c in preferred if c in df.columns]


def normalize_text_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s or s == "nan":
        return None
    return s


def build_single_target_view(
    df: pd.DataFrame,
    preferred_target_cols: List[str],
    canonical_target_col: str,
    extra_keep_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    feature_cols = detect_feature_cols(df)
    meta_cols = detect_meta_cols(df)

    target_col = None
    for c in preferred_target_cols:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise ValueError(f"No target column found for {canonical_target_col}. Tried: {preferred_target_cols}")

    keep_cols = meta_cols + feature_cols + [target_col]
    if extra_keep_cols:
        keep_cols.extend([c for c in extra_keep_cols if c in df.columns])

    # 去重保序
    seen = set()
    keep_cols_unique = []
    for c in keep_cols:
        if c not in seen:
            keep_cols_unique.append(c)
            seen.add(c)

    out = df[keep_cols_unique].copy()

    if target_col != canonical_target_col:
        out = out.rename(columns={target_col: canonical_target_col})

    # 统一做轻量规范化
    if out[canonical_target_col].dtype == object:
        out[canonical_target_col] = out[canonical_target_col].apply(normalize_text_label)

    out["has_target"] = out[canonical_target_col].notna().astype(int)
    return out


def build_temperature_view(df: pd.DataFrame) -> pd.DataFrame:
    return build_single_target_view(
        df=df,
        preferred_target_cols=[
            "target_temperature_c_clean",
            "temperature_c_clean",
            "target_temperature_c",
            "temperature_c",
        ],
        canonical_target_col="target_temperature_c_clean",
        extra_keep_cols=[],
    )


def build_time_bucket_view(df: pd.DataFrame) -> pd.DataFrame:
    return build_single_target_view(
        df=df,
        preferred_target_cols=[
            "target_time_bucket",
            "time_bucket",
        ],
        canonical_target_col="target_time_bucket",
        extra_keep_cols=[
            "target_time_h_log1p",
            "time_h_log1p",
            "target_time_h_clean",
            "time_h_clean",
            "target_time_h",
            "time_h",
        ],
    )


def build_atmosphere_view(df: pd.DataFrame) -> pd.DataFrame:
    return build_single_target_view(
        df=df,
        preferred_target_cols=[
            "target_atmosphere_coarse",
            "atmosphere_coarse",
            "target_atmosphere",
            "atmosphere",
        ],
        canonical_target_col="target_atmosphere_coarse",
        extra_keep_cols=[
            "target_atmosphere",
            "atmosphere",
        ],
    )


def build_solvent_view(df: pd.DataFrame) -> pd.DataFrame:
    return build_single_target_view(
        df=df,
        preferred_target_cols=[
            "target_solvent_clean",
            "solvent_clean",
            "target_solvent",
            "solvent",
        ],
        canonical_target_col="target_solvent_clean",
        extra_keep_cols=[
            "target_solvent",
            "solvent",
        ],
    )


def build_synthesis_type_view(df: pd.DataFrame) -> pd.DataFrame:
    return build_single_target_view(
        df=df,
        preferred_target_cols=[
            "synthesis_type",
            "target_synthesis_type",
        ],
        canonical_target_col="synthesis_type",
        extra_keep_cols=[],
    )


def summarize_view(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_features": int(len(detect_feature_cols(df))),
        "n_target_nonnull": int(df[target_col].notna().sum()),
        "n_has_target": int(df["has_target"].sum()) if "has_target" in df.columns else None,
    }

    s = df[target_col].dropna()
    if len(s) == 0:
        return summary

    if pd.api.types.is_numeric_dtype(s):
        summary["target_stats"] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
        }
    else:
        vc = s.astype(str).value_counts().to_dict()
        summary["target_dist"] = {str(k): int(v) for k, v in vc.items()}

    return summary


def process_split(
    split_name: str,
    input_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    in_path = input_dir / f"stage3_{split_name}_raw.csv"
    df = load_csv(str(in_path))

    temp_df = build_temperature_view(df)
    time_df = build_time_bucket_view(df)
    atm_df = build_atmosphere_view(df)
    solv_df = build_solvent_view(df)
    synth_df = build_synthesis_type_view(df)

    temp_out = output_dir / f"temperature_{split_name}.csv"
    time_out = output_dir / f"time_bucket_{split_name}.csv"
    atm_out = output_dir / f"atmosphere_coarse_{split_name}.csv"
    solv_out = output_dir / f"solvent_{split_name}.csv"
    synth_out = output_dir / f"synthesis_type_{split_name}.csv"

    temp_df.to_csv(temp_out, index=False)
    time_df.to_csv(time_out, index=False)
    atm_df.to_csv(atm_out, index=False)
    solv_df.to_csv(solv_out, index=False)
    synth_df.to_csv(synth_out, index=False)

    return {
        "input_csv": str(in_path),
        "temperature_csv": str(temp_out),
        "time_bucket_csv": str(time_out),
        "atmosphere_coarse_csv": str(atm_out),
        "solvent_csv": str(solv_out),
        "synthesis_type_csv": str(synth_out),
        "temperature": summarize_view(temp_df, "target_temperature_c_clean"),
        "time_bucket": summarize_view(time_df, "target_time_bucket"),
        "atmosphere_coarse": summarize_view(atm_df, "target_atmosphere_coarse"),
        "solvent": summarize_view(solv_df, "target_solvent_clean"),
        "synthesis_type": summarize_view(synth_df, "synthesis_type"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stage3 task views for temperature, time bucket, atmosphere, solvent, and synthesis type."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/features/structdesc_features_stage3_v2",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/features/stage3_task_views",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    summary: Dict[str, Any] = {
        "config": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
        "splits": {},
    }

    for split_name in ["train", "val", "test", "gold_train_holdout"]:
        summary["splits"][split_name] = process_split(split_name, input_dir, output_dir)

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
