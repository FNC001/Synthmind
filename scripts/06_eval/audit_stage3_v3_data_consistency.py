#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


SPLITS = ["train", "val", "test"]


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
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_table(path_base: Path) -> pd.DataFrame:
    csv = path_base.with_suffix(".csv")
    parquet = path_base.with_suffix(".parquet")
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    return pd.read_csv(csv)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open(encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def source_flag(series: pd.Series, patterns: Iterable[str]) -> pd.Series:
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series(False, index=series.index)
    for pat in patterns:
        mask = mask | text.str.contains(pat, regex=False)
    return mask.astype(int)


def summarize_dataset(pred_dir: Path, target_dir: Path) -> tuple[List[Dict[str, Any]], Dict[str, set[str]]]:
    rows: List[Dict[str, Any]] = []
    ids: Dict[str, set[str]] = {}
    for split in SPLITS:
        pred = read_table(pred_dir / split)
        tgt = read_table(target_dir / split)
        ids[split] = set(pred["sample_id"].astype(str))
        rows.append(
            {
                "section": "dataset",
                "split": split,
                "pred_rows": int(len(pred)),
                "target_rows": int(len(tgt)),
                "sample_id_aligned": bool(set(pred["sample_id"].astype(str)) == set(tgt["sample_id"].astype(str))),
                "chemistry_ok_rows": int((pred["precursor_check_status"].astype(str) == "ok").sum()),
                "mean_precursor_f1": float(pd.to_numeric(pred["precursor_f1_to_true"], errors="coerce").mean()),
                "mean_precursor_jaccard": float(pd.to_numeric(pred["precursor_jaccard_to_true"], errors="coerce").mean()),
                "open_generated_rows": int(pd.to_numeric(pred.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0).sum()),
                "repair_rows": int(pd.to_numeric(pred.get("contains_repair_precursor", 0), errors="coerce").fillna(0).sum()),
                "input_modes": json.dumps(pred["precursor_input_mode"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    return rows, ids


def summarize_stage2(stage2_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, name in [("val", "val_candidate_sets_repaired.csv"), ("test", "test_candidate_sets_repaired.csv")]:
        path = stage2_dir / name
        if not path.exists():
            rows.append({"section": "stage2", "split": split, "exists": False, "path": str(path)})
            continue
        flags = {"open_from_mix": 0, "repair_from_mix": 0, "rows": 0}
        for chunk in pd.read_csv(path, usecols=lambda c: c in {"candidate_source_mix", "candidate_source", "pred_precursors"}, chunksize=250_000):
            flags["rows"] += len(chunk)
            mix = chunk.get("candidate_source_mix", pd.Series("", index=chunk.index))
            src = chunk.get("candidate_source", pd.Series("", index=chunk.index))
            joined = mix.fillna("").astype(str) + " " + src.fillna("").astype(str)
            flags["open_from_mix"] += int(source_flag(joined, ["open_generated", "generated", "open_vocab"]).sum())
            flags["repair_from_mix"] += int(source_flag(joined, ["repair", "known_vocab_repair"]).sum())
        rows.append({"section": "stage2", "split": split, "exists": True, "path": str(path), **flags})
    return rows


def summarize_candidates(cond_dir: Path, route_dir: Path, dataset_ids: Dict[str, set[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in SPLITS:
        cond_path = cond_dir / f"{split}_condition_candidates_calibrated.csv"
        if cond_path.exists():
            cond = pd.read_csv(cond_path, usecols=lambda c: c in {"sample_id", "temperature_c", "time_h", "atmosphere", "condition_source", "condition_calibrated_score_v3", "condition_rank_calibrated_v3", "open_generated_penalty", "repair_penalty"})
            rows.append(
                {
                    "section": "condition_candidates",
                    "split": split,
                    "exists": True,
                    "rows": int(len(cond)),
                    "n_samples": int(cond["sample_id"].nunique()),
                    "sample_ids_match_dataset": bool(set(cond["sample_id"].astype(str)) == dataset_ids.get(split, set())),
                    "nan_scores": int(cond[["condition_calibrated_score_v3"]].isna().sum().sum()),
                    "empty_condition_rows": int((cond["temperature_c"].isna() | cond["time_h"].isna() | cond["atmosphere"].isna()).sum()),
                    "open_penalty_rows": int((pd.to_numeric(cond.get("open_generated_penalty", 0), errors="coerce").fillna(0) > 0).sum()),
                    "repair_penalty_rows": int((pd.to_numeric(cond.get("repair_penalty", 0), errors="coerce").fillna(0) > 0).sum()),
                }
            )
        else:
            rows.append({"section": "condition_candidates", "split": split, "exists": False, "path": str(cond_path)})

        route_path = route_dir / f"{split}_route_candidates.csv"
        if route_path.exists():
            first = pd.read_csv(route_path, nrows=1)
            cols = set(first.columns)
            required_flags = {"candidate_source_mix", "contains_open_generated_precursor", "contains_repair_precursor"}
            accum = {
                "rows": 0,
                "n_samples": set(),
                "nan_scores": 0,
                "empty_precursor_rows": 0,
                "empty_condition_rows": 0,
                "open_from_mix": 0,
                "repair_from_mix": 0,
                "explicit_open_rows": 0,
                "explicit_repair_rows": 0,
            }
            usecols = lambda c: c in {
                "sample_id",
                "pred_precursors",
                "candidate_source_mix",
                "candidate_source",
                "contains_open_generated_precursor",
                "contains_repair_precursor",
                "temperature_c",
                "time_h",
                "atmosphere",
                "precursor_score",
                "condition_score",
                "route_total_score_raw",
            }
            for chunk in pd.read_csv(route_path, usecols=usecols, chunksize=250_000):
                accum["rows"] += len(chunk)
                accum["n_samples"].update(chunk["sample_id"].astype(str).unique().tolist())
                score_cols = [c for c in ["precursor_score", "condition_score", "route_total_score_raw"] if c in chunk.columns]
                accum["nan_scores"] += int(chunk[score_cols].isna().sum().sum()) if score_cols else 0
                accum["empty_precursor_rows"] += int(chunk.get("pred_precursors", pd.Series("", index=chunk.index)).fillna("").astype(str).str.len().eq(0).sum())
                accum["empty_condition_rows"] += int((chunk["temperature_c"].isna() | chunk["time_h"].isna() | chunk["atmosphere"].isna()).sum())
                mix = chunk.get("candidate_source_mix", pd.Series("", index=chunk.index)).fillna("").astype(str) + " " + chunk.get("candidate_source", pd.Series("", index=chunk.index)).fillna("").astype(str)
                accum["open_from_mix"] += int(source_flag(mix, ["open_generated", "generated", "open_vocab"]).sum())
                accum["repair_from_mix"] += int(source_flag(mix, ["repair", "known_vocab_repair"]).sum())
                if "contains_open_generated_precursor" in chunk.columns:
                    accum["explicit_open_rows"] += int(pd.to_numeric(chunk["contains_open_generated_precursor"], errors="coerce").fillna(0).sum())
                if "contains_repair_precursor" in chunk.columns:
                    accum["explicit_repair_rows"] += int(pd.to_numeric(chunk["contains_repair_precursor"], errors="coerce").fillna(0).sum())
            rows.append(
                {
                    "section": "route_candidates",
                    "split": split,
                    "exists": True,
                    "rows": int(accum["rows"]),
                    "n_samples": int(len(accum["n_samples"])),
                    "sample_ids_match_dataset": bool(accum["n_samples"] == dataset_ids.get(split, set())),
                    "has_required_source_fields": bool(required_flags <= cols),
                    "missing_required_source_fields": ",".join(sorted(required_flags - cols)),
                    "nan_scores": int(accum["nan_scores"]),
                    "empty_precursor_rows": int(accum["empty_precursor_rows"]),
                    "empty_condition_rows": int(accum["empty_condition_rows"]),
                    "open_from_mix_rows": int(accum["open_from_mix"]),
                    "repair_from_mix_rows": int(accum["repair_from_mix"]),
                    "explicit_open_rows": int(accum["explicit_open_rows"]),
                    "explicit_repair_rows": int(accum["explicit_repair_rows"]),
                }
            )
        else:
            rows.append({"section": "route_candidates", "split": split, "exists": False, "path": str(route_path)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit Stage3/Stage35 v3 data consistency before final reranker training.")
    ap.add_argument("--predprec_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v3_20260610")
    ap.add_argument("--targets_dir", default="data/interim/generative/stage3_condition_targets_v3_20260610")
    ap.add_argument("--condition_dir", default="outputs/evaluation/stage3_condition_calibration_v3_20260610")
    ap.add_argument("--route_dir", default="outputs/evaluation/stage35_route_candidates_v3_20260610")
    ap.add_argument("--stage2_dir", default="outputs/evaluation/stage2_candidate_pool_v5_20260610")
    ap.add_argument("--output_dir", default="outputs/evaluation/stage3_stage35_v3_final_audit_20260612")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, ids = summarize_dataset(Path(args.predprec_dir), Path(args.targets_dir))
    rows.extend(summarize_stage2(Path(args.stage2_dir)))
    rows.extend(summarize_candidates(Path(args.condition_dir), Path(args.route_dir), ids))
    df = pd.DataFrame(rows)
    df.to_csv(out / "data_consistency_audit.csv", index=False)
    issues = []
    def as_int(value: Any) -> int:
        try:
            if pd.isna(value):
                return 0
            return int(value)
        except Exception:
            return 0
    for _, row in df.iterrows():
        if row.get("exists") is False:
            issues.append(f"Missing {row.get('section')} {row.get('split')}: {row.get('path')}")
        if row.get("has_required_source_fields") is False:
            issues.append(f"Route {row.get('split')} missing source fields: {row.get('missing_required_source_fields')}")
        if as_int(row.get("nan_scores", 0)) > 0:
            issues.append(f"{row.get('section')} {row.get('split')} has NaN scores: {row.get('nan_scores')}")
        if as_int(row.get("empty_precursor_rows", 0)) > 0:
            issues.append(f"{row.get('section')} {row.get('split')} has empty precursor rows")
        if as_int(row.get("empty_condition_rows", 0)) > 0:
            issues.append(f"{row.get('section')} {row.get('split')} has empty condition rows")
    report = [
        "# Stage3/Stage35 v3 Data Consistency Audit",
        "",
        f"Inputs: `{args.predprec_dir}`, `{args.targets_dir}`, `{args.condition_dir}`, `{args.route_dir}`",
        "",
        "## Findings",
        "",
    ]
    if issues:
        report.extend([f"- {x}" for x in issues])
    else:
        report.append("- No blocking consistency issues detected.")
    report.extend(["", "## Audit Table", "", df.to_markdown(index=False)])
    (out / "data_consistency_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "data_consistency_audit.json", {"issues": issues, "rows": rows})
    print(json.dumps(to_builtin({"issues": issues, "audit_csv": str(out / "data_consistency_audit.csv")}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
