#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_REGISTRY = (
    "scripts/07_infer/structure_to_synthesis_route/pipeline_v3/"
    "configs/stage3_reference_registry.json"
)

SOURCE_PRIORITY = {
    # Highest priority: manually/explicitly registered partial real Flow patch.
    # This should override older current/V32/V34 rows during deduplication when the
    # same condition candidate exists with weaker provenance.
    "v45b_registered_v44_regenerated_flow_patch": -2,
    "v40e_registered_partial_flow_patch": -1,
    "v34_discovered_existing_stage3_rows": 0,
    "v32_stage3_candidates_with_metadata": 1,
    "existing_current_reference": 2,
}

V33_V34_REQUESTED_TARGETS_DEFAULT = [
    "mp-2652",      # Y2O3
    "mp-1986",      # ZnO
    "mp-1190568",   # BaSO4
    "mp-12372",     # CaSO4
    "mp-22851",     # NaCl
    "mp-2251",      # Li3N
    "mp-1330",      # AlN
    "mp-1079918",   # CaCO3
    "mp-1198150",   # Bi2Se3
]


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_json_read_error": str(e), "_json_path": str(path)}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def rpath(project_root: Path, x: str | Path | None) -> Path:
    if x is None:
        return Path("")
    p = Path(x)
    return p if p.is_absolute() else project_root / p


def load_registry(project_root: Path, registry_json: str | Path) -> dict:
    p = rpath(project_root, registry_json)
    if not p.exists():
        raise FileNotFoundError(f"Missing stage3 reference registry: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def safe_read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"[SKIP] {label}: missing {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"[LOAD] {label}: {path} | rows={len(df)} cols={len(df.columns)}")
        return df
    except Exception as e:
        print(f"[WARN] failed to read {label}: {path} | {e}")
        return pd.DataFrame()


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_elements(x) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip()
    if not s:
        return ""

    s = s.replace(",", ";").replace("-", ";").replace("|", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return ";".join(sorted(set(parts)))


def infer_family_from_elements(elements: str) -> str:
    elems = set([e for e in str(elements).split(";") if e])

    if not elems:
        return "unknown"
    if "O" in elems and "P" in elems:
        return "phosphate_or_oxide"
    if "O" in elems and "S" in elems:
        return "sulfate_or_oxide"
    if "O" in elems and "C" in elems:
        return "carbonate_or_oxide"
    if "O" in elems:
        return "oxide"
    if elems & {"F", "Cl", "Br", "I"}:
        return "halide"
    if "N" in elems:
        return "nitride"
    if "S" in elems:
        return "sulfide"
    if "Se" in elems:
        return "selenide"
    if "P" in elems:
        return "phosphide_or_phosphate_like"

    return "non_oxide"


def clean_id_series(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .replace({"nan": "", "None": "", "NaN": "", "<NA>": ""})
        .str.strip()
    )


def normalize_stage3_candidate_table(
    df: pd.DataFrame,
    source_label: str,
    source_path: Path,
) -> pd.DataFrame:
    """
    Convert heterogeneous Stage3 candidate tables into the current reference schema.

    Required downstream columns:
      case_id
      candidate_temperature_c
      candidate_time_h
      candidate_condition_score

    Extra useful metadata:
      mp_id
      formula
      elements
      mp_family
      candidate_source
      candidate_condition_rank
      reference_source_label
      reference_source_path
    """
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)

    mp_col = first_existing_col(df, ["mp_id", "case_id", "material_id", "target_mp_id"])
    case_col = first_existing_col(df, ["case_id", "mp_id", "material_id", "target_mp_id"])

    formula_col = first_existing_col(
        df,
        ["formula", "mp_formula", "target_formula", "pretty_formula", "reduced_formula"],
    )
    elements_col = first_existing_col(
        df,
        ["elements", "mp_elements", "target_elements", "chemsys"],
    )
    family_col = first_existing_col(
        df,
        ["mp_family", "target_family", "family", "route_family"],
    )

    temp_col = first_existing_col(
        df,
        [
            "candidate_temperature_c",
            "temperature_c",
            "v28_temperature_c",
            "pred_temperature_c",
            "temperature",
        ],
    )
    time_col = first_existing_col(
        df,
        [
            "candidate_time_h",
            "time_h",
            "v28_time_h",
            "pred_time_h",
            "time",
        ],
    )
    score_col = first_existing_col(
        df,
        [
            "candidate_condition_score",
            "stage3_score",
            "condition_score",
            "flow_score",
            "mdn_score",
            "score",
        ],
    )
    source_col = first_existing_col(
        df,
        ["candidate_source", "condition_source", "source", "model_source"],
    )
    rank_col = first_existing_col(
        df,
        ["candidate_condition_rank", "condition_rank", "rank"],
    )

    if temp_col is None or time_col is None:
        print(
            f"[SKIP] {source_label}: missing temperature/time columns. "
            f"temp_col={temp_col}, time_col={time_col}"
        )
        return pd.DataFrame()

    if mp_col is not None:
        out["mp_id"] = clean_id_series(df[mp_col])
    else:
        out["mp_id"] = ""

    if case_col is not None:
        out["case_id"] = clean_id_series(df[case_col])
    else:
        out["case_id"] = out["mp_id"]

    if formula_col is not None:
        out["formula"] = df[formula_col].fillna("").astype(str).str.strip()
    else:
        out["formula"] = ""

    if elements_col is not None:
        out["elements"] = df[elements_col].map(normalize_elements)
    else:
        out["elements"] = ""

    if family_col is not None:
        out["mp_family"] = df[family_col].fillna("").astype(str).str.strip()
    else:
        out["mp_family"] = out["elements"].map(infer_family_from_elements)

    out["candidate_temperature_c"] = pd.to_numeric(df[temp_col], errors="coerce")
    out["candidate_time_h"] = pd.to_numeric(df[time_col], errors="coerce")

    if score_col is not None:
        out["candidate_condition_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    else:
        out["candidate_condition_score"] = 1.0

    if source_col is not None:
        out["candidate_source"] = df[source_col].fillna("").astype(str).str.strip()
    else:
        out["candidate_source"] = source_label

    if rank_col is not None:
        out["candidate_condition_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    else:
        out["candidate_condition_rank"] = np.nan

    out["reference_source_label"] = source_label
    out["reference_source_path"] = str(source_path)

    out["has_mp_metadata"] = out["formula"].fillna("").astype(str).str.len().gt(0)
    out["has_mp_elements"] = out["elements"].fillna("").astype(str).str.len().gt(0)

    out = out[
        out["candidate_temperature_c"].notna()
        & out["candidate_time_h"].notna()
    ].copy()

    missing_case = out["case_id"].astype(str).str.len() == 0
    out.loc[missing_case, "case_id"] = out.loc[missing_case, "mp_id"]

    missing_mpid = out["mp_id"].astype(str).str.len() == 0
    case_looks_mp = out["case_id"].astype(str).str.startswith("mp-")
    out.loc[missing_mpid & case_looks_mp, "mp_id"] = out.loc[missing_mpid & case_looks_mp, "case_id"]

    return out.reset_index(drop=True)


def backup_existing_file(path: Path, backup_dir: Path) -> str:
    if not path.exists():
        return ""

    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / f"{path.stem}.backup_{now_tag()}{path.suffix}"
    shutil.copy2(path, dst)
    print(f"[BACKUP] {path} -> {dst}")
    return str(dst)


def run_v34_discovery_if_requested(
    project_root: Path,
    registry: dict,
    run_v34_discovery: bool,
) -> None:
    if not run_v34_discovery:
        return

    v34 = registry.get("v34", {})
    check_script = rpath(project_root, v34.get("check_script", ""))
    discover_script = rpath(project_root, v34.get("discover_script", ""))
    check_out = rpath(project_root, v34.get("existing_check_output_dir", ""))
    discover_out = rpath(project_root, v34.get("discovery_output_dir", ""))

    if check_script.exists():
        cmd = [
            "python",
            str(check_script),
            "--project_root",
            str(project_root),
            "--output_dir",
            str(check_out),
        ]
        print("[RUN]", " ".join(f'"{x}"' if " " in x else x for x in cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"[SKIP] V34 check script missing: {check_script}")

    if discover_script.exists():
        cmd = [
            "python",
            str(discover_script),
            "--project_root",
            str(project_root),
            "--output_dir",
            str(discover_out),
        ]
        print("[RUN]", " ".join(f'"{x}"' if " " in x else x for x in cmd))
        subprocess.run(cmd, check=True)
    else:
        print(f"[SKIP] V34 discovery script missing: {discover_script}")


def add_source_priority(ref: pd.DataFrame) -> pd.DataFrame:
    out = ref.copy()
    if "reference_source_label" not in out.columns:
        out["reference_source_label"] = ""

    out["_source_priority"] = (
        out["reference_source_label"]
        .map(SOURCE_PRIORITY)
        .fillna(9)
        .astype(int)
    )

    return out


def build_requested_target_coverage(
    ref: pd.DataFrame,
    registry: dict,
) -> dict:
    requested_targets = V33_V34_REQUESTED_TARGETS_DEFAULT

    v33_cfg = registry.get("v33", {})
    manifest_path = (
        v33_cfg.get("request_manifest_csv")
        or v33_cfg.get("stage3_expansion_request_manifest_csv")
        or ""
    )

    if not manifest_path:
        manifest_path = (
            "outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/"
            "stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv"
        )

    if "mp_id" not in ref.columns:
        return {
            "requested_targets": requested_targets,
            "requested_target_counts": {},
            "n_requested_targets": len(requested_targets),
            "n_requested_targets_covered": 0,
            "n_requested_targets_missing": len(requested_targets),
            "requested_targets_missing": requested_targets,
            "manifest_path": manifest_path,
        }

    counts = (
        ref.loc[ref["mp_id"].astype(str).isin(requested_targets), "mp_id"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    covered = sorted(counts.keys())
    missing = sorted(set(requested_targets) - set(covered))

    return {
        "requested_targets": requested_targets,
        "requested_target_counts": counts,
        "n_requested_targets": len(requested_targets),
        "n_requested_targets_covered": len(covered),
        "n_requested_targets_missing": len(missing),
        "requested_targets_missing": missing,
        "manifest_path": manifest_path,
    }


def get_enabled_extra_sources(registry: dict) -> list[dict]:
    """
    Collect optional reference sources.

    Supported registry styles:

    1. Dedicated V40e block:
       "v40e": {
         "enabled": true,
         "registered_patch_csv": "...",
         "source_label": "v40e_registered_partial_flow_patch"
       }

    2. Backward compatible path key:
       "v40e": {
         "enabled": true,
         "path": "...",
         "source_label": "v40e_registered_partial_flow_patch"
       }

    3. Generic future extension:
       "extra_reference_sources": [
         {
           "name": "...",
           "enabled": true,
           "path": "...",
           "source_label": "...",
           "priority": 0
         }
       ]
    """
    sources: list[dict] = []

    v40e_cfg = registry.get("v40e", {})
    if isinstance(v40e_cfg, dict) and v40e_cfg.get("enabled", False):
        v40e_path = (
            v40e_cfg.get("registered_patch_csv")
            or v40e_cfg.get("path")
            or v40e_cfg.get("csv")
            or ""
        )
        if v40e_path:
            sources.append(
                {
                    "name": v40e_cfg.get("name", "v40e_registered_partial_flow_patch"),
                    "source_label": v40e_cfg.get(
                        "source_label",
                        "v40e_registered_partial_flow_patch",
                    ),
                    "path": v40e_path,
                    "claim_boundary": v40e_cfg.get(
                        "claim_boundary",
                        "partial_real_stage3_flow_patch_not_experimental_validation",
                    ),
                }
            )

    extra = registry.get("extra_reference_sources", [])
    if isinstance(extra, list):
        for item in extra:
            if not isinstance(item, dict):
                continue
            if not item.get("enabled", False):
                continue
            p = item.get("path") or item.get("csv") or item.get("registered_patch_csv") or ""
            if not p:
                continue
            sources.append(
                {
                    "name": item.get("name", item.get("source_label", "extra_reference_source")),
                    "source_label": item.get("source_label", item.get("name", "extra_reference_source")),
                    "path": p,
                    "claim_boundary": item.get("claim_boundary", ""),
                }
            )

    return sources


def load_and_normalize_source(
    project_root: Path,
    path_value: str,
    source_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    p = rpath(project_root, path_value)
    raw = safe_read_csv(p, source_label)
    norm = normalize_stage3_candidate_table(raw, source_label, p)
    return raw, norm, p


def build_reference_library(
    project_root: Path,
    registry: dict,
    include_existing_current: bool = True,
    include_v32: bool = True,
    include_v34_discovered: bool = True,
    include_extra_sources: bool = True,
    prefer_source_priority: bool = True,
) -> tuple[pd.DataFrame, dict]:
    current_csv = rpath(
        project_root,
        registry.get("current_reference", {}).get(
            "output_reference_csv",
            "data/interim/references/stage3_condition_reference/current/stage3_condition_reference.csv",
        ),
    )

    frames = []
    source_summaries = []

    # Optional high-priority patch sources, including V40e.
    if include_extra_sources:
        for src in get_enabled_extra_sources(registry):
            source_label = src["source_label"]
            raw, norm, p = load_and_normalize_source(
                project_root=project_root,
                path_value=src["path"],
                source_label=source_label,
            )
            if len(norm):
                frames.append(norm)
            source_summaries.append(
                {
                    "source": source_label,
                    "path": str(p),
                    "raw_rows": int(len(raw)),
                    "normalized_rows": int(len(norm)),
                    "claim_boundary": src.get("claim_boundary", ""),
                }
            )

    if include_existing_current and current_csv.exists():
        df0 = safe_read_csv(current_csv, "existing_current_reference")
        norm0 = normalize_stage3_candidate_table(
            df0,
            "existing_current_reference",
            current_csv,
        )
        if len(norm0):
            frames.append(norm0)
        source_summaries.append(
            {
                "source": "existing_current_reference",
                "path": str(current_csv),
                "raw_rows": int(len(df0)),
                "normalized_rows": int(len(norm0)),
            }
        )

    if include_v32:
        v32_csv = rpath(
            project_root,
            registry.get("v32", {}).get("stage3_candidates_with_metadata_csv", ""),
        )
        df32 = safe_read_csv(v32_csv, "v32_stage3_candidates_with_metadata")
        norm32 = normalize_stage3_candidate_table(
            df32,
            "v32_stage3_candidates_with_metadata",
            v32_csv,
        )
        if len(norm32):
            frames.append(norm32)
        source_summaries.append(
            {
                "source": "v32_stage3_candidates_with_metadata",
                "path": str(v32_csv),
                "raw_rows": int(len(df32)),
                "normalized_rows": int(len(norm32)),
            }
        )

    if include_v34_discovered:
        v34_csv = rpath(
            project_root,
            registry.get("v34", {}).get("discovered_rows_csv", ""),
        )
        df34 = safe_read_csv(v34_csv, "v34_discovered_existing_stage3_rows")
        norm34 = normalize_stage3_candidate_table(
            df34,
            "v34_discovered_existing_stage3_rows",
            v34_csv,
        )
        if len(norm34):
            frames.append(norm34)
        source_summaries.append(
            {
                "source": "v34_discovered_existing_stage3_rows",
                "path": str(v34_csv),
                "raw_rows": int(len(df34)),
                "normalized_rows": int(len(norm34)),
            }
        )
    extra_sources = registry.get("extra_sources", [])
    if isinstance(extra_sources, list):
        for src in extra_sources:
            if not isinstance(src, dict):
                continue

            if not bool(src.get("enabled", True)):
                continue

            source_label = str(src.get("source_label") or src.get("name") or "extra_stage3_source")
            src_path = rpath(project_root, src.get("path", ""))

            df_extra = safe_read_csv(src_path, source_label)
            norm_extra = normalize_stage3_candidate_table(
                df_extra,
                source_label,
                src_path,
            )

            if len(norm_extra):
                frames.append(norm_extra)

            source_summaries.append(
                {
                    "source": source_label,
                    "path": str(src_path),
                    "raw_rows": int(len(df_extra)),
                    "normalized_rows": int(len(norm_extra)),
                }
            )
    if not frames:
        return pd.DataFrame(), {
            "source_summaries": source_summaries,
            "status": "blocked_no_reference_rows",
        }

    ref = pd.concat(frames, ignore_index=True, sort=False)

    before = len(ref)

    for c in [
        "case_id",
        "mp_id",
        "formula",
        "elements",
        "mp_family",
        "candidate_source",
        "reference_source_label",
        "reference_source_path",
    ]:
        if c not in ref.columns:
            ref[c] = ""

    ref["case_id"] = clean_id_series(ref["case_id"])
    ref["mp_id"] = clean_id_series(ref["mp_id"])
    ref["formula"] = ref["formula"].fillna("").astype(str).str.strip()
    ref["elements"] = ref["elements"].fillna("").astype(str).map(normalize_elements)
    ref["mp_family"] = ref["mp_family"].fillna("").astype(str).str.strip()
    ref["candidate_source"] = ref["candidate_source"].fillna("").astype(str).str.strip()
    ref["reference_source_label"] = ref["reference_source_label"].fillna("").astype(str).str.strip()
    ref["reference_source_path"] = ref["reference_source_path"].fillna("").astype(str).str.strip()

    ref["candidate_temperature_c"] = pd.to_numeric(ref["candidate_temperature_c"], errors="coerce")
    ref["candidate_time_h"] = pd.to_numeric(ref["candidate_time_h"], errors="coerce")
    ref["candidate_condition_score"] = pd.to_numeric(
        ref["candidate_condition_score"], errors="coerce"
    ).fillna(0.0)

    if "candidate_condition_rank" in ref.columns:
        ref["candidate_condition_rank"] = pd.to_numeric(ref["candidate_condition_rank"], errors="coerce")
    else:
        ref["candidate_condition_rank"] = np.nan

    ref = ref[
        ref["candidate_temperature_c"].notna()
        & ref["candidate_time_h"].notna()
    ].copy()

    missing_case = ref["case_id"].astype(str).str.len() == 0
    ref.loc[missing_case, "case_id"] = ref.loc[missing_case, "mp_id"]

    missing_mpid = ref["mp_id"].astype(str).str.len() == 0
    case_looks_mp = ref["case_id"].astype(str).str.startswith("mp-")
    ref.loc[missing_mpid & case_looks_mp, "mp_id"] = ref.loc[missing_mpid & case_looks_mp, "case_id"]

    missing_family = ref["mp_family"].astype(str).str.len() == 0
    ref.loc[missing_family, "mp_family"] = ref.loc[missing_family, "elements"].map(infer_family_from_elements)

    ref["has_mp_metadata"] = ref["formula"].fillna("").astype(str).str.len().gt(0)
    ref["has_mp_elements"] = ref["elements"].fillna("").astype(str).str.len().gt(0)

    dedup_cols = [
        "case_id",
        "mp_id",
        "formula",
        "elements",
        "candidate_temperature_c",
        "candidate_time_h",
        "candidate_condition_score",
        "candidate_source",
    ]
    dedup_cols = [c for c in dedup_cols if c in ref.columns]

    if prefer_source_priority:
        ref = add_source_priority(ref)

        sort_for_dedup = [
            "_source_priority",
            "mp_id",
            "case_id",
            "candidate_temperature_c",
            "candidate_time_h",
            "candidate_condition_score",
        ]
        sort_for_dedup = [c for c in sort_for_dedup if c in ref.columns]

        ascending = []
        for c in sort_for_dedup:
            if c == "candidate_condition_score":
                ascending.append(False)
            else:
                ascending.append(True)

        ref = ref.sort_values(sort_for_dedup, ascending=ascending).reset_index(drop=True)

    ref = ref.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    after = len(ref)

    if "_source_priority" in ref.columns:
        ref = ref.drop(columns=["_source_priority"])

    if "stage3_reference_row_id" in ref.columns:
        ref = ref.drop(columns=["stage3_reference_row_id"])

    ref.insert(0, "stage3_reference_row_id", np.arange(1, len(ref) + 1))

    sort_cols = [
        c for c in ["case_id", "mp_id", "candidate_condition_score"] if c in ref.columns
    ]
    if sort_cols:
        ascending = [True] * len(sort_cols)
        if "candidate_condition_score" in sort_cols:
            ascending[sort_cols.index("candidate_condition_score")] = False
        ref = ref.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        ref["stage3_reference_row_id"] = np.arange(1, len(ref) + 1)

    requested_coverage = build_requested_target_coverage(ref, registry)

    summary = {
        "status": "pass",
        "source_summaries": source_summaries,
        "source_priority": SOURCE_PRIORITY,
        "prefer_source_priority": bool(prefer_source_priority),
        "n_rows_before_dedup": int(before),
        "n_rows_after_dedup": int(after),
        "n_reference_rows": int(len(ref)),
        "n_unique_case_id": int(ref["case_id"].nunique()) if "case_id" in ref.columns else 0,
        "n_unique_mp_id": int(ref["mp_id"].nunique()) if "mp_id" in ref.columns else 0,
        "n_with_formula": int(ref["formula"].fillna("").astype(str).str.len().gt(0).sum()),
        "n_with_elements": int(ref["elements"].fillna("").astype(str).str.len().gt(0).sum()),
        "candidate_source_counts": ref["candidate_source"].value_counts(dropna=False).to_dict(),
        "reference_source_label_counts": ref["reference_source_label"].value_counts(dropna=False).to_dict(),
        "family_counts": ref["mp_family"].value_counts(dropna=False).head(30).to_dict(),
        "v33_v34_requested_target_coverage": requested_coverage,
        "extra_sources_enabled": get_enabled_extra_sources(registry),
    }

    return ref, summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Refresh current real Stage3 condition-reference library by merging "
            "current/V32/V34 and optional registered Stage3 reference patches."
        )
    )
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--registry_json", default=DEFAULT_REGISTRY)
    ap.add_argument("--output_csv", default="")
    ap.add_argument("--output_summary_json", default="")
    ap.add_argument("--output_md", default="")
    ap.add_argument("--backup_existing", action="store_true")
    ap.add_argument("--no_existing_current", action="store_true")
    ap.add_argument("--no_v32", action="store_true")
    ap.add_argument("--no_v34_discovered", action="store_true")
    ap.add_argument("--no_extra_sources", action="store_true")
    ap.add_argument("--run_v34_discovery", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--no_prefer_source_priority", action="store_true")

    args = ap.parse_args()

    project_root = Path(args.project_root)
    registry = load_registry(project_root, args.registry_json)

    current_ref_csv = rpath(
        project_root,
        registry.get("current_reference", {}).get(
            "output_reference_csv",
            "data/interim/references/stage3_condition_reference/current/stage3_condition_reference.csv",
        ),
    )

    output_csv = rpath(project_root, args.output_csv) if args.output_csv else current_ref_csv

    refresh_out_dir = rpath(
        project_root,
        registry.get("current_reference", {}).get(
            "refresh_output_dir",
            "outputs/stage3_reference_refresh",
        ),
    )
    refresh_out_dir.mkdir(parents=True, exist_ok=True)

    output_summary_json = (
        rpath(project_root, args.output_summary_json)
        if args.output_summary_json
        else refresh_out_dir / "stage3_reference_refresh_summary.json"
    )
    output_md = (
        rpath(project_root, args.output_md)
        if args.output_md
        else refresh_out_dir / "stage3_reference_refresh_report.md"
    )

    print("============================================================")
    print("Refresh Stage3 reference library")
    print(f"project_root      = {project_root}")
    print(f"registry_json     = {rpath(project_root, args.registry_json)}")
    print(f"output_csv        = {output_csv}")
    print(f"summary_json      = {output_summary_json}")
    print(f"output_md         = {output_md}")
    print("============================================================")

    run_v34_discovery_if_requested(
        project_root=project_root,
        registry=registry,
        run_v34_discovery=bool(args.run_v34_discovery),
    )

    ref, summary = build_reference_library(
        project_root=project_root,
        registry=registry,
        include_existing_current=not args.no_existing_current,
        include_v32=not args.no_v32,
        include_v34_discovered=not args.no_v34_discovered,
        include_extra_sources=not args.no_extra_sources,
        prefer_source_priority=not args.no_prefer_source_priority,
    )

    summary.update(
        {
            "project_root": str(project_root),
            "registry_json": str(rpath(project_root, args.registry_json)),
            "output_csv": str(output_csv),
            "output_summary_json": str(output_summary_json),
            "output_md": str(output_md),
            "backup_existing": bool(args.backup_existing),
            "dry_run": bool(args.dry_run),
            "claim_boundary": (
                "stage3_reference_refresh_merges_internal_mdn_flow_condition_candidates_"
                "and_registered_reference_patches_not_experimental_validation"
            ),
            "interpretation": (
                "This refresh step merges current/V32/V34 real Stage3-style condition candidate rows "
                "and optional registered reference patches such as V40e into the active Stage3 reference "
                "library used by pipeline_v3 reliability scoring. It does not generate new Stage3 candidates "
                "by itself. Source-priority deduplication preserves higher-priority patch/V34/V32 provenance "
                "over already-refreshed current-reference rows."
            ),
        }
    )

    if ref.empty:
        summary["status"] = "blocked_no_reference_rows"
        write_json(output_summary_json, summary)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(
            "# Stage3 Reference Refresh Report\n\n"
            "Status: `blocked_no_reference_rows`\n\n"
            "No usable Stage3 reference rows were found.\n",
            encoding="utf-8",
        )
        print("[BLOCKED] no usable reference rows")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    backup_path = ""
    if args.backup_existing and output_csv.exists() and not args.dry_run:
        backup_path = backup_existing_file(output_csv, refresh_out_dir / "backups")

    summary["backup_path"] = backup_path

    if not args.dry_run:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        ref.to_csv(output_csv, index=False)
        print(f"[SAVE] {output_csv}")
    else:
        print("[DRY RUN] not writing refreshed reference CSV")

    write_json(output_summary_json, summary)
    print(f"[SAVE] {output_summary_json}")

    preview_cols = [
        "stage3_reference_row_id",
        "case_id",
        "mp_id",
        "formula",
        "elements",
        "mp_family",
        "candidate_temperature_c",
        "candidate_time_h",
        "candidate_condition_score",
        "candidate_source",
        "reference_source_label",
    ]
    preview_cols = [c for c in preview_cols if c in ref.columns]

    requested_targets = summary.get("v33_v34_requested_target_coverage", {})
    requested_counts = requested_targets.get("requested_target_counts", {})

    lines = []
    lines.append("# Stage3 Reference Refresh Report\n")
    lines.append(f"- status: `{summary.get('status')}`")
    lines.append(f"- output_csv: `{output_csv}`")
    lines.append(f"- n_reference_rows: {summary.get('n_reference_rows')}")
    lines.append(f"- n_unique_mp_id: {summary.get('n_unique_mp_id')}")
    lines.append(f"- n_unique_case_id: {summary.get('n_unique_case_id')}")
    lines.append(f"- n_with_formula: {summary.get('n_with_formula')}")
    lines.append(f"- n_with_elements: {summary.get('n_with_elements')}")
    lines.append(f"- prefer_source_priority: {summary.get('prefer_source_priority')}")
    lines.append("")
    lines.append("## Source priority\n")
    lines.append(pd.DataFrame(
        [{"source": k, "priority": v} for k, v in SOURCE_PRIORITY.items()]
    ).to_markdown(index=False))
    lines.append("")
    lines.append("## Source summaries\n")
    lines.append(pd.DataFrame(summary.get("source_summaries", [])).to_markdown(index=False))
    lines.append("")
    lines.append("## Reference source label counts\n")
    lines.append(pd.DataFrame(
        [{"reference_source_label": k, "n_rows": v}
         for k, v in summary.get("reference_source_label_counts", {}).items()]
    ).to_markdown(index=False))
    lines.append("")
    lines.append("## Candidate source counts\n")
    lines.append(pd.DataFrame(
        [{"candidate_source": k, "n_rows": v}
         for k, v in summary.get("candidate_source_counts", {}).items()]
    ).head(50).to_markdown(index=False))
    lines.append("")
    lines.append("## V33/V34 requested target coverage\n")
    lines.append(f"- n_requested_targets: {requested_targets.get('n_requested_targets')}")
    lines.append(f"- n_requested_targets_covered: {requested_targets.get('n_requested_targets_covered')}")
    lines.append(f"- n_requested_targets_missing: {requested_targets.get('n_requested_targets_missing')}")
    lines.append(f"- requested_targets_missing: `{requested_targets.get('requested_targets_missing')}`")
    lines.append("")
    if requested_counts:
        lines.append(pd.DataFrame(
            [{"mp_id": k, "n_rows": v} for k, v in requested_counts.items()]
        ).to_markdown(index=False))
    else:
        lines.append("No requested target rows found.")
    lines.append("")
    lines.append("## Preview\n")
    lines.append(ref[preview_cols].head(50).to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(summary["interpretation"])
    lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {output_md}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
