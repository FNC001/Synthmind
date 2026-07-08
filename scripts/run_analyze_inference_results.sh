#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# SynPred Analyze Inference Results
#
# Purpose:
#   Analyze parallel or serial inference outputs.
#   Recover precursor information when possible.
#   Convert standardized Stage3 condition outputs back to readable values
#   when schema/stat files are available.
#
# Usage:
#   bash scripts/run_analyze_inference_results.sh [PROJECT_ROOT]
#
# Example:
#   FORCE=1 bash /Users/wyc/SynPred/scripts/run_analyze_inference_results.sh /Users/wyc/SynPred
# ============================================================

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"

FORCE="${FORCE:-0}"

# Default layout for your current parallel inference.
INFERENCE_ROOT="${INFERENCE_ROOT:-${PROJECT_ROOT}/outputs/inference_batches_parallel}"
INPUT_BATCH_ROOT="${INPUT_BATCH_ROOT:-${PROJECT_ROOT}/data/infer/GNoME_selected_Wyckoff_CLscore_vasp_batches}"
INTERIM_INFER_ROOT="${INTERIM_INFER_ROOT:-${PROJECT_ROOT}/data/interim/infer}"
STAGE3_SCHEMA_JSON="${STAGE3_SCHEMA_JSON:-${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/schema.json}"
STAGE3_DATASET_DIR="${STAGE3_DATASET_DIR:-${PROJECT_ROOT}/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1}"

ANALYSIS_DIR="${ANALYSIS_DIR:-${PROJECT_ROOT}/outputs/analysis/GNoME_batch_summary}"
LOG_DIR="${PROJECT_ROOT}/outputs/logs/analyze_inference_results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${ANALYSIS_DIR}" "${LOG_DIR}"

SCRIPT_PY="${LOG_DIR}/analyze_inference_results.py"
LOG_FILE="${LOG_DIR}/analyze.log"

cat > "${SCRIPT_PY}" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
INFERENCE_ROOT = Path(os.environ["INFERENCE_ROOT"])
INPUT_BATCH_ROOT = Path(os.environ["INPUT_BATCH_ROOT"])
INTERIM_INFER_ROOT = Path(os.environ["INTERIM_INFER_ROOT"])
STAGE3_SCHEMA_JSON = Path(os.environ["STAGE3_SCHEMA_JSON"])
STAGE3_DATASET_DIR = Path(os.environ["STAGE3_DATASET_DIR"])
ANALYSIS_DIR = Path(os.environ["ANALYSIS_DIR"])

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = ANALYSIS_DIR / "all_batch_candidates_summary.csv"
STATUS_CSV = ANALYSIS_DIR / "batch_status_summary.csv"
TOP_LIST_CSV = ANALYSIS_DIR / "most_synthesizable_top_list.csv"
SHORT_LIST_CSV = ANALYSIS_DIR / "most_synthesizable_short_list.csv"
SHORT_LIST_WITH_PRECURSORS_CSV = ANALYSIS_DIR / "most_synthesizable_short_list_with_precursors.csv"
TOP_LIST_WITH_PRECURSORS_CSV = ANALYSIS_DIR / "most_synthesizable_top_list_with_precursors.csv"
MANIFEST_JSON = ANALYSIS_DIR / "analysis_manifest.json"

PREFERRED_RESULT_FILES = [
    "export/all_test_candidates_flat.csv",
    "combined/*/test_candidates_flat.csv",
    "stage3/test_candidates_flat.csv",
]

PREC_COL_CANDIDATES = [
    "precursor_set",
    "precursors",
    "precursor_names",
    "precursor_formula",
    "precursor_formulas",
    "pred_precursors",
    "candidate_precursors",
    "top_precursors",
    "retrieved_precursors",
    "stage2_precursors",
]

KEY_COL_CANDIDATES = [
    "material_id",
    "target_structure_id",
    "structure_id",
    "sample_id",
    "sample_index",
    "parent_index",
    "row_id",
    "case_name",
]

SCORE_PRIORITY = [
    "synthesizability_score_final",
    "joint_score",
    "safe_strict_score",
    "confidence_score",
    "condition_confidence",
    "route_confidence_score",
    "route_score",
    "score",
    "model_score_raw",
    "model_score",
    "rerank_score",
    "candidate_score",
    "log_prob",
    "trajectory_logprob",
]


def read_csv_safe(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"[WARN] failed to read CSV: {path} | {e}")
    return None


def read_json_safe(path: Path) -> Optional[dict]:
    try:
        if path.exists() and path.stat().st_size > 0:
            return json.loads(path.read_text())
    except Exception as e:
        print(f"[WARN] failed to read JSON: {path} | {e}")
    return None


def to_num(s, default=np.nan):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def normalize_series(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    if v.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    lo = float(v.min())
    hi = float(v.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return ((v - lo) / (hi - lo)).fillna(0.0)


def find_score_col(df: pd.DataFrame) -> Optional[str]:
    for c in SCORE_PRIORITY:
        if c in df.columns:
            return c
    score_cols = [c for c in df.columns if "score" in c.lower()]
    return score_cols[0] if score_cols else None


def find_rank_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["rank", "final_rank", "joint_rank", "safe_strict_rank", "rank_in_file", "condition_rank"]:
        if c in df.columns:
            return c
    return None


def find_prec_col(df: pd.DataFrame) -> Optional[str]:
    for c in PREC_COL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = c.lower()
        if "precursor" in lc and not any(x in lc for x in ["score", "qc", "frequency", "count", "rank"]):
            return c
    return None


def find_key_cols(left: pd.DataFrame, right: pd.DataFrame) -> List[str]:
    if "material_id" in left.columns and "material_id" in right.columns:
        return ["material_id"]
    for c in KEY_COL_CANDIDATES:
        if c in left.columns and c in right.columns:
            return [c]
    return []


def infer_batch_name(path: Path) -> str:
    for part in path.parts:
        if re.match(r"batch_\d+", part):
            return part
    parent = path.parent
    while parent != parent.parent:
        if re.match(r"batch_\d+", parent.name):
            return parent.name
        parent = parent.parent
    return ""


def infer_case_name(path: Path) -> str:
    if path.parent.name not in ["export", "stage3", "combined"]:
        return path.parent.name
    return infer_batch_name(path)


def discover_result_files() -> List[Path]:
    files: List[Path] = []
    if not INFERENCE_ROOT.exists():
        print(f"[WARN] INFERENCE_ROOT does not exist: {INFERENCE_ROOT}")
        return files

    for batch_dir in sorted(INFERENCE_ROOT.glob("batch_*")):
        if not batch_dir.is_dir():
            continue
        for pattern in PREFERRED_RESULT_FILES:
            files.extend(sorted(batch_dir.glob(pattern)))

    # Fallback: any test_candidates_flat.csv under inference root.
    if not files:
        files = sorted(INFERENCE_ROOT.glob("**/test_candidates_flat.csv"))

    # Remove duplicates.
    seen = set()
    out = []
    for f in files:
        if f.exists() and f.stat().st_size > 0:
            key = str(f.resolve())
            if key not in seen:
                out.append(f)
                seen.add(key)
    return out


def load_all_candidates() -> pd.DataFrame:
    files = discover_result_files()
    rows = []
    status_rows = []

    for f in files:
        batch = infer_batch_name(f)
        case_name = infer_case_name(f)
        df = read_csv_safe(f)
        if df is None or df.empty:
            status_rows.append({
                "batch": batch,
                "case_name": case_name,
                "status": "empty_or_read_error",
                "route_csv_path": str(f),
                "n_rows": 0,
            })
            continue

        df = df.copy()

        # Add metadata columns safely.
        # Some inference outputs may already contain case_name / batch / source_layout,
        # so do not use DataFrame.insert blindly.
        meta = {
            "case_name": case_name,
            "batch": batch,
            "source_layout": "parallel" if "inference_batches_parallel" in str(f) else "unknown",
            "route_csv_path": str(f),
        }

        for col, value in meta.items():
            if col not in df.columns:
                df[col] = value
            else:
                df[col] = df[col].fillna("")
                df[col] = df[col].astype(str)
                df.loc[df[col].str.len() == 0, col] = value

        # Move metadata columns to the front without duplicating columns.
        front_cols = ["case_name", "batch", "source_layout", "route_csv_path"]
        other_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + other_cols]

        if "rank_in_file" not in df.columns:
            if "rank" in df.columns:
                df["rank_in_file"] = pd.to_numeric(df["rank"], errors="coerce")
            else:
                df["rank_in_file"] = np.arange(1, len(df) + 1)

        rows.append(df)
        status_rows.append({
            "batch": batch,
            "case_name": case_name,
            "status": "ok",
            "route_csv_path": str(f),
            "n_rows": int(len(df)),
        })

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(STATUS_CSV, index=False)

    if not rows:
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True, sort=False)
    all_df.to_csv(SUMMARY_CSV, index=False)
    return all_df


def collect_precursor_lookup_for_batch(batch: str) -> List[pd.DataFrame]:
    files: List[Path] = []

    batch_out = INFERENCE_ROOT / batch
    files += list(batch_out.glob("combined/*/debug_parent_candidates.csv"))
    files += list(batch_out.glob("combined/*/test_candidates_flat.csv"))
    files += list(batch_out.glob("**/*conditioned*.csv"))
    files += list(batch_out.glob("stage2/**/*.csv"))

    files += list((INTERIM_INFER_ROOT / "GNoME_selected_Wyckoff_CLscore_vasp_batches" / batch).glob("**/*.csv"))
    files += list((INTERIM_INFER_ROOT / batch).glob("**/*.csv"))
    files += list((INPUT_BATCH_ROOT / batch).glob("**/*.csv"))

    # Remove duplicates.
    unique_files = []
    seen = set()
    for f in files:
        if f.exists() and f.stat().st_size > 0:
            key = str(f.resolve())
            if key not in seen:
                unique_files.append(f)
                seen.add(key)

    lookups: List[pd.DataFrame] = []

    for f in unique_files:
        df = read_csv_safe(f)
        if df is None or df.empty:
            continue

        prec_col = find_prec_col(df)
        if prec_col is None:
            continue

        key_cols = [c for c in KEY_COL_CANDIDATES if c in df.columns]
        if not key_cols:
            continue

        use_cols = key_cols + [prec_col]
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in ["route", "template", "atmosphere", "temperature", "time", "confidence", "qc"]):
                if c not in use_cols:
                    use_cols.append(c)

        sub = df[use_cols].copy()
        sub = sub.dropna(subset=[prec_col], how="all")
        if sub.empty:
            continue

        sub = sub.rename(columns={prec_col: "precursor_set_recovered"})
        sub["precursor_source_file"] = str(f)
        lookups.append(sub)

        print(f"[FOUND precursor] batch={batch}")
        print(f"  file={f}")
        print(f"  precursor_col={prec_col}")
        print(f"  keys={key_cols}")

    return lookups


def recover_precursors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "precursor_set" not in df.columns:
        df["precursor_set"] = ""

    df["precursor_set"] = df["precursor_set"].fillna("").astype(str)

    if "batch" not in df.columns:
        print("[WARN] No batch column, cannot recover precursors by batch.")
        return df

    for batch in sorted(df["batch"].fillna("").astype(str).unique()):
        if not batch:
            continue

        mask_batch = df["batch"].astype(str) == batch
        need = mask_batch & (df["precursor_set"].fillna("").astype(str).str.len() == 0)

        if not need.any():
            continue

        lookups = collect_precursor_lookup_for_batch(batch)

        for lk in lookups:
            need = mask_batch & (df["precursor_set"].fillna("").astype(str).str.len() == 0)
            if not need.any():
                break

            keys = find_key_cols(df, lk)
            if not keys:
                continue

            lk2 = lk.drop_duplicates(subset=keys).copy()
            left = df.loc[need, keys].copy()
            merged = left.merge(lk2, on=keys, how="left")

            recovered = merged["precursor_set_recovered"].fillna("").astype(str)
            hit = recovered.str.len() > 0

            if hit.any():
                idx = df.loc[need].index[hit.values]
                df.loc[idx, "precursor_set"] = recovered[hit].values
                if "precursor_source_file" in merged.columns:
                    df.loc[idx, "precursor_source_file"] = merged.loc[hit, "precursor_source_file"].values
                print(f"[MERGE precursor] batch={batch} keys={keys} filled={int(hit.sum())}")

    return df


def infer_poscar_file(row: pd.Series) -> str:
    batch = str(row.get("batch", "")).strip()
    material_id = str(row.get("material_id", "")).strip()

    if not batch:
        return ""

    poscar_dir = INPUT_BATCH_ROOT / batch / "poscars"
    if not poscar_dir.exists():
        return ""

    poscars = sorted(list(poscar_dir.glob("*.vasp")) + list(poscar_dir.glob("*.POSCAR")) + list(poscar_dir.glob("POSCAR*")))
    if not poscars:
        return ""

    if material_id:
        key = material_id.split("__", 1)[-1]
        for pattern in [f"*{key}*.vasp", f"*{material_id}*.vasp", f"*{key}*", f"*{material_id}*"]:
            matches = sorted(poscar_dir.glob(pattern))
            if matches:
                return str(matches[0])

        try:
            prefix = material_id.split("__", 1)[0]
            if prefix.startswith("infer_"):
                idx = int(prefix.replace("infer_", "")) - 1
                if 0 <= idx < len(poscars):
                    return str(poscars[idx])
        except Exception:
            pass

    return str(poscars[0]) if len(poscars) == 1 else ""


def load_stage3_scaler_from_npz() -> Dict[str, Tuple[float, float]]:
    """
    Try to infer continuous target standardization parameters from npz/schema.
    This supports several common key names.
    """
    out: Dict[str, Tuple[float, float]] = {}

    schema = read_json_safe(STAGE3_SCHEMA_JSON) or {}
    cont_cols = []
    for key in ["cont_col_names", "continuous_col_names", "continuous_targets", "cont_cols"]:
        if isinstance(schema.get(key), list):
            cont_cols = list(schema[key])
            break

    if not cont_cols:
        # Your logs show these two names.
        cont_cols = ["target_temperature_c_clean", "target_time_h_clean"]

    # schema may already contain mean/std.
    for name in cont_cols:
        candidates = [
            ("target_mean", "target_std"),
            ("cont_mean", "cont_std"),
            ("continuous_mean", "continuous_std"),
            ("y_cont_mean", "y_cont_std"),
        ]
        for mean_key, std_key in candidates:
            mean_obj = schema.get(mean_key)
            std_obj = schema.get(std_key)
            if isinstance(mean_obj, dict) and isinstance(std_obj, dict) and name in mean_obj and name in std_obj:
                out[name] = (float(mean_obj[name]), float(std_obj[name]))
            elif isinstance(mean_obj, list) and isinstance(std_obj, list):
                try:
                    i = cont_cols.index(name)
                    out[name] = (float(mean_obj[i]), float(std_obj[i]))
                except Exception:
                    pass

    # npz may contain mean/std arrays.
    train_npz = STAGE3_DATASET_DIR / "train.npz"
    if train_npz.exists():
        try:
            z = np.load(train_npz, allow_pickle=True)
            keys = list(z.keys())

            mean_arr = None
            std_arr = None
            for mk in ["y_cont_mean", "cont_mean", "target_mean", "continuous_mean", "y_mean"]:
                if mk in z:
                    mean_arr = np.asarray(z[mk]).astype(float)
                    break
            for sk in ["y_cont_std", "cont_std", "target_std", "continuous_std", "y_std"]:
                if sk in z:
                    std_arr = np.asarray(z[sk]).astype(float)
                    break

            if mean_arr is not None and std_arr is not None:
                for i, name in enumerate(cont_cols[: len(mean_arr)]):
                    out[name] = (float(mean_arr[i]), float(std_arr[i]))

            print(f"[INFO] train.npz keys = {keys}")
            if out:
                print(f"[INFO] recovered scalers from npz/schema: {out}")
        except Exception as e:
            print(f"[WARN] failed to inspect train.npz scaler: {e}")

    return out


def add_readable_conditions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scalers = load_stage3_scaler_from_npz()

    temp_candidates = [
        "temperature_c",
        "temperature_C",
        "target_temperature_c",
        "target_temperature_c_clean",
        "pred_temperature_c",
    ]
    time_candidates = [
        "time_h",
        "target_time_h",
        "target_time_h_clean",
        "pred_time_h",
    ]

    temp_col = next((c for c in temp_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if temp_col:
        raw = pd.to_numeric(df[temp_col], errors="coerce")
        if temp_col in scalers:
            mean, std = scalers[temp_col]
            df["temperature_c_recovered"] = raw * std + mean
            df["temperature_c"] = df["temperature_c_recovered"]
        else:
            # If values already look like Celsius, keep them. Otherwise mark as standardized.
            if raw.dropna().between(100, 2000).mean() > 0.5:
                df["temperature_c"] = raw
            else:
                df["temperature_c_standardized"] = raw
                if "temperature_c" not in df.columns:
                    df["temperature_c"] = np.nan

    if time_col:
        raw = pd.to_numeric(df[time_col], errors="coerce")
        if time_col in scalers:
            mean, std = scalers[time_col]
            df["time_h_recovered"] = raw * std + mean
            df["time_h"] = df["time_h_recovered"]
        else:
            if raw.dropna().between(0, 500).mean() > 0.5:
                df["time_h"] = raw
            else:
                df["time_h_standardized"] = raw
                if "time_h" not in df.columns:
                    df["time_h"] = np.nan

    return df


def compute_synthesizability_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    score_col = find_score_col(df)
    if score_col:
        df["model_score_raw"] = pd.to_numeric(df[score_col], errors="coerce")
        df["model_score_norm"] = normalize_series(df["model_score_raw"])
    else:
        df["model_score_raw"] = 0.0
        df["model_score_norm"] = 0.0

    rank_col = find_rank_col(df)
    if rank_col:
        rank = pd.to_numeric(df[rank_col], errors="coerce").fillna(100).clip(lower=1)
        df["rank_bonus"] = 1.0 / rank
    else:
        df["rank_bonus"] = 0.0

    if "precursor_set" in df.columns:
        prec = df["precursor_set"].fillna("").astype(str)
        freq = prec.value_counts()
        df["precursor_frequency"] = prec.map(freq).fillna(1)
    else:
        df["precursor_frequency"] = 1

    max_freq = max(float(pd.to_numeric(df["precursor_frequency"], errors="coerce").max()), 1.0)
    df["frequency_bonus"] = pd.to_numeric(df["precursor_frequency"], errors="coerce").fillna(1.0) / max_freq

    if "temperature_c" in df.columns:
        temp = pd.to_numeric(df["temperature_c"], errors="coerce")
        df["temperature_reasonable"] = ((temp >= 300) & (temp <= 1200)).astype(float)
        df.loc[temp.isna(), "temperature_reasonable"] = 0.5
    else:
        df["temperature_reasonable"] = 0.5

    if "time_h" in df.columns:
        t = pd.to_numeric(df["time_h"], errors="coerce")
        df["time_reasonable"] = ((t >= 0.1) & (t <= 72)).astype(float)
        df.loc[t.isna(), "time_reasonable"] = 0.5
    else:
        df["time_reasonable"] = 0.5

    df["synthesizability_score_final"] = (
        0.45 * pd.to_numeric(df["model_score_norm"], errors="coerce").fillna(0)
        + 0.25 * pd.to_numeric(df["rank_bonus"], errors="coerce").fillna(0)
        + 0.15 * pd.to_numeric(df["frequency_bonus"], errors="coerce").fillna(0)
        + 0.10 * pd.to_numeric(df["temperature_reasonable"], errors="coerce").fillna(0)
        + 0.05 * pd.to_numeric(df["time_reasonable"], errors="coerce").fillna(0)
    )

    return df


def choose_columns(df: pd.DataFrame, mode: str = "top") -> List[str]:
    front = [
        "material_id",
        "target_structure_id",
        "structure_id",
        "poscar_file",
        "poscar_path",
        "batch",
        "case_name",
        "source_layout",
        "synthesizability_score_final",
        "model_score_raw",
        "model_score_norm",
        "rank_in_file",
        "rank",
        "precursor_set",
        "precursor_source_file",
        "precursor_frequency",
        "temperature_c",
        "time_h",
        "temperature_c_standardized",
        "time_h_standardized",
        "target_temperature_c_clean",
        "target_time_h_clean",
        "pred_atmosphere",
        "target_atmosphere_coarse",
        "synthesis_type",
        "route_confidence_score",
        "route_confidence_level",
        "precursor_qc_status",
        "condition_distribution_confidence_level",
        "route_template_primary",
        "route_recommendation_status",
        "route_csv_path",
    ]

    keep = []
    for c in front:
        if c in df.columns and c not in keep:
            keep.append(c)

    if mode == "top":
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in [
                "formula",
                "material",
                "target",
                "structure",
                "poscar",
                "filename",
                "file_name",
                "input_file",
                "source_file",
                "composition",
                "element",
                "precursor",
                "temp",
                "time",
                "atmosphere",
                "score",
                "rank",
                "route",
                "confidence",
                "qc",
                "condition",
                "synthesis",
            ]):
                if c not in keep:
                    keep.append(c)

    return [c for c in keep if c in df.columns]


def main():
    print("============================================================")
    print("SynPred inference result analysis")
    print("============================================================")
    print(f"PROJECT_ROOT       = {PROJECT_ROOT}")
    print(f"INFERENCE_ROOT     = {INFERENCE_ROOT}")
    print(f"INPUT_BATCH_ROOT   = {INPUT_BATCH_ROOT}")
    print(f"INTERIM_INFER_ROOT = {INTERIM_INFER_ROOT}")
    print(f"STAGE3_SCHEMA_JSON = {STAGE3_SCHEMA_JSON}")
    print(f"STAGE3_DATASET_DIR = {STAGE3_DATASET_DIR}")
    print(f"ANALYSIS_DIR       = {ANALYSIS_DIR}")

    df = load_all_candidates()
    if df.empty:
        print("[WARN] No inference candidate rows found.")
        MANIFEST_JSON.write_text(json.dumps({
            "status": "no_candidates",
            "inference_root": str(INFERENCE_ROOT),
        }, ensure_ascii=False, indent=2))
        return

    print(f"[INFO] loaded candidate rows: {len(df)}")

    df = recover_precursors(df)
    df = add_readable_conditions(df)

    if "poscar_path" not in df.columns:
        df["poscar_path"] = ""
    df["poscar_path"] = df["poscar_path"].fillna("").astype(str)

    need_poscar = df["poscar_path"].str.len() == 0
    if need_poscar.any():
        df.loc[need_poscar, "poscar_path"] = df.loc[need_poscar].apply(infer_poscar_file, axis=1)

    df["poscar_file"] = df["poscar_path"].fillna("").astype(str).map(lambda x: Path(x).name if x else "")

    df = compute_synthesizability_score(df)
    df = df.sort_values("synthesizability_score_final", ascending=False)

    top_cols = choose_columns(df, mode="top")
    short_cols = choose_columns(df, mode="short")

    df[top_cols].head(1000).to_csv(TOP_LIST_CSV, index=False)
    df[short_cols].head(500).to_csv(SHORT_LIST_CSV, index=False)

    # For compatibility with earlier naming.
    df[top_cols].head(1000).to_csv(TOP_LIST_WITH_PRECURSORS_CSV, index=False)
    df[short_cols].head(500).to_csv(SHORT_LIST_WITH_PRECURSORS_CSV, index=False)

    n_prec = 0
    if "precursor_set" in df.columns:
        n_prec = int((df["precursor_set"].fillna("").astype(str).str.len() > 0).sum())

    manifest = {
        "status": "ok",
        "n_rows": int(len(df)),
        "n_rows_with_precursors": n_prec,
        "n_batches": int(df["batch"].nunique()) if "batch" in df.columns else None,
        "summary_csv": str(SUMMARY_CSV),
        "status_csv": str(STATUS_CSV),
        "top_list_csv": str(TOP_LIST_CSV),
        "short_list_csv": str(SHORT_LIST_CSV),
        "top_list_with_precursors_csv": str(TOP_LIST_WITH_PRECURSORS_CSV),
        "short_list_with_precursors_csv": str(SHORT_LIST_WITH_PRECURSORS_CSV),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print()
    print("[DONE] summary:", SUMMARY_CSV)
    print("[DONE] status: ", STATUS_CSV)
    print("[DONE] top:    ", TOP_LIST_CSV)
    print("[DONE] short:  ", SHORT_LIST_CSV)
    print("[DONE] manifest:", MANIFEST_JSON)
    print()
    print(f"[INFO] rows with recovered precursors: {n_prec}/{len(df)}")

    print()
    print("Top 30 synthesizable candidates:")
    show_cols = [c for c in short_cols if c in df.columns]
    print(df[show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
PY

export PROJECT_ROOT
export INFERENCE_ROOT
export INPUT_BATCH_ROOT
export INTERIM_INFER_ROOT
export STAGE3_SCHEMA_JSON
export STAGE3_DATASET_DIR
export ANALYSIS_DIR

echo "============================================================"
echo "SynPred Analyze Inference Results"
echo "============================================================"
echo "PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "INFERENCE_ROOT     = ${INFERENCE_ROOT}"
echo "INPUT_BATCH_ROOT   = ${INPUT_BATCH_ROOT}"
echo "INTERIM_INFER_ROOT = ${INTERIM_INFER_ROOT}"
echo "STAGE3_SCHEMA_JSON = ${STAGE3_SCHEMA_JSON}"
echo "STAGE3_DATASET_DIR = ${STAGE3_DATASET_DIR}"
echo "ANALYSIS_DIR       = ${ANALYSIS_DIR}"
echo "LOG_DIR            = ${LOG_DIR}"
echo

python "${SCRIPT_PY}" 2>&1 | tee "${LOG_FILE}"

echo
echo "============================================================"
echo "[DONE] Analysis complete"
echo "============================================================"
echo "Log:"
echo "  ${LOG_FILE}"
echo
echo "Outputs:"
echo "  ${ANALYSIS_DIR}/all_batch_candidates_summary.csv"
echo "  ${ANALYSIS_DIR}/batch_status_summary.csv"
echo "  ${ANALYSIS_DIR}/most_synthesizable_top_list.csv"
echo "  ${ANALYSIS_DIR}/most_synthesizable_short_list.csv"
echo "  ${ANALYSIS_DIR}/most_synthesizable_top_list_with_precursors.csv"
echo "  ${ANALYSIS_DIR}/most_synthesizable_short_list_with_precursors.csv"
echo "  ${ANALYSIS_DIR}/analysis_manifest.json"
