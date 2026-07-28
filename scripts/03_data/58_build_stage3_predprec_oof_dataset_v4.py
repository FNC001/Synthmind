#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
FAMILY_PATTERNS = {
    "acetate": re.compile(r"CH3COO|C2H3O2|acetate", re.I),
    "carbonate": re.compile(r"CO3|carbonate", re.I),
    "elemental": re.compile(r"^[A-Z][a-z]?$"),
    "halide": re.compile(r"Cl|Br|I|F|chloride|bromide|iodide|fluoride", re.I),
    "hydroxide": re.compile(r"OH|hydroxide", re.I),
    "nitrate": re.compile(r"NO3|nitrate", re.I),
    "oxide": re.compile(r"O[0-9]*|oxide", re.I),
    "phosphate": re.compile(r"PO4|phosphate", re.I),
    "sulfate": re.compile(r"SO4|sulfate", re.I),
}


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


def parse_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x) for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"\s*\+\s*|;", text) if x.strip()]


def dump_list(items: Sequence[str]) -> str:
    return json.dumps([str(x) for x in items if str(x).strip()], ensure_ascii=False)


def elements(text: str) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def target_source_elements(formula: str) -> Set[str]:
    elems = elements(formula) - {"O"}
    return elems or elements(formula)


def set_metrics(true_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, float]:
    t = set(str(x) for x in true_labels if str(x).strip())
    p = set(str(x) for x in pred_labels if str(x).strip())
    inter = len(t & p)
    precision = inter / len(p) if p else 0.0
    recall = inter / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = len(t | p)
    return {
        "precursor_exact": float(t == p),
        "precursor_precision": precision,
        "precursor_recall": recall,
        "precursor_f1_to_true": f1,
        "precursor_jaccard_to_true": inter / union if union else 1.0,
    }


def family_counts(labels: Sequence[str]) -> Dict[str, float]:
    counts = {name: 0.0 for name in FAMILY_PATTERNS}
    counts["other_salt"] = 0.0
    counts["unknown"] = 0.0
    for lab in labels:
        matched = False
        for name, pat in FAMILY_PATTERNS.items():
            if pat.search(str(lab)):
                counts[name] += 1.0
                matched = True
                break
        if not matched:
            counts["unknown"] += 1.0
    n = max(float(len(labels)), 1.0)
    out = {}
    for name, val in counts.items():
        out[f"precursor_family_count__{name}"] = float(val)
        out[f"precursor_family_frac__{name}"] = float(val / n)
    return out


def read_ranked_candidates(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rank_col = "precursor_rank"
    if rank_col not in df.columns:
        if "calibrated_rank_v5" in df.columns:
            rank_col = "calibrated_rank_v5"
        elif "rank" in df.columns:
            rank_col = "rank"
        else:
            df["_rank"] = df.groupby("id" if "id" in df.columns else "sample_id", sort=False).cumcount() + 1
            rank_col = "_rank"
    sid_col = "sample_id" if "sample_id" in df.columns else "id"
    set_col = "precursor_set" if "precursor_set" in df.columns else "pred_precursors"
    score_col = "calibrated_score_v5" if "calibrated_score_v5" in df.columns else "total_score_v5" if "total_score_v5" in df.columns else "precursor_score"
    source_mix = df["candidate_source_mix"] if "candidate_source_mix" in df.columns else df["precursor_source_mix"] if "precursor_source_mix" in df.columns else pd.Series("", index=df.index)
    repair_flag = df["contains_repair_precursor"] if "contains_repair_precursor" in df.columns else source_mix.astype(str).str.contains("repair", regex=False).astype(int)
    out = pd.DataFrame({
        "sample_id": df[sid_col].astype(str),
        "precursor_rank": pd.to_numeric(df[rank_col], errors="coerce").fillna(999).astype(int),
        "precursor_set": df[set_col].astype(str),
        "precursor_score": pd.to_numeric(df[score_col], errors="coerce").fillna(0.0),
        "precursor_source_mix": source_mix.astype(str),
        "contains_open_generated_precursor": df.get("contains_open_generated_precursor", 0),
        "contains_repair_precursor": repair_flag,
        "chemistry_check_status": df.get("chemistry_check_status", "ok"),
        "missing_source_elements": df.get("missing_source_elements", "[]"),
        "extra_forbidden_elements": df.get("extra_forbidden_elements", "[]"),
        "precursor_exact_if_eval": pd.to_numeric(df.get("exact", df.get("precursor_exact_if_eval", 0)), errors="coerce").fillna(0.0),
        "precursor_f1_if_eval": pd.to_numeric(df.get("f1", df.get("precursor_f1_if_eval", 0)), errors="coerce").fillna(0.0),
        "precursor_jaccard_if_eval": pd.to_numeric(df.get("jaccard", df.get("precursor_jaccard_if_eval", 0)), errors="coerce").fillna(0.0),
    })
    out["split"] = split
    return out[out["precursor_rank"] <= 20].copy()


def candidate_summary(cands: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    by_sid: Dict[str, Dict[str, Any]] = {}
    for sid, g in cands.sort_values(["sample_id", "precursor_rank"]).groupby("sample_id", sort=False):
        sets = [parse_list(x) for x in g["precursor_set"].tolist()]
        scores = pd.to_numeric(g["precursor_score"], errors="coerce").fillna(0.0).to_numpy(float)
        top1 = sets[0] if sets else []
        by_sid[str(sid)] = {
            "top1": top1,
            "top5": sets[:5],
            "top20": sets[:20],
            "top1_score": float(scores[0]) if len(scores) else 0.0,
            "score_mean5": float(np.mean(scores[:5])) if len(scores) else 0.0,
            "score_max20": float(np.max(scores[:20])) if len(scores) else 0.0,
            "score_std20": float(np.std(scores[:20])) if len(scores) else 0.0,
            "source_mix": "|".join(sorted(set(g["precursor_source_mix"].astype(str).tolist()))),
            "contains_open": int(pd.to_numeric(g["contains_open_generated_precursor"], errors="coerce").fillna(0).max()),
            "contains_repair": int(pd.to_numeric(g["contains_repair_precursor"], errors="coerce").fillna(0).max()),
            "chemistry_check_status": "ok" if (g["chemistry_check_status"].astype(str) == "ok").any() else str(g["chemistry_check_status"].iloc[0]),
            "missing_source_elements": str(g["missing_source_elements"].iloc[0]) if "missing_source_elements" in g else "[]",
            "extra_forbidden_elements": str(g["extra_forbidden_elements"].iloc[0]) if "extra_forbidden_elements" in g else "[]",
        }
    return by_sid


def apply_candidates(base: pd.DataFrame, cands: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = base.copy()
    summaries = candidate_summary(cands)
    for i, row in out.iterrows():
        sid = str(row["sample_id"])
        rec = summaries.get(sid)
        if rec is None:
            top1 = parse_list(row.get("predicted_precursor_set_chem_checked", "[]"))
            rec = {
                "top1": top1, "top5": [top1], "top20": [top1], "top1_score": float(row.get("precursor_confidence_score", 0)),
                "score_mean5": float(row.get("precursor_confidence_score", 0)), "score_max20": float(row.get("precursor_confidence_score", 0)),
                "score_std20": 0.0, "source_mix": str(row.get("precursor_source_mix", "")), "contains_open": int(row.get("contains_open_generated_precursor", 0)),
                "contains_repair": int(row.get("contains_repair_precursor", 0)), "chemistry_check_status": str(row.get("precursor_check_status", "ok")),
                "missing_source_elements": str(row.get("missing_source_elements", "[]")), "extra_forbidden_elements": str(row.get("extra_forbidden_elements", "[]")),
            }
        top1 = rec["top1"]
        true_set = parse_list(row.get("true_precursor_set", "[]"))
        metrics = set_metrics(true_set, top1)
        out.at[i, "raw_predicted_precursor_set"] = dump_list(top1)
        out.at[i, "predicted_precursor_set_chem_checked"] = dump_list(top1)
        out.at[i, "precursors_text"] = " + ".join(top1)
        out.at[i, "predicted_precursor_top1"] = dump_list(top1)
        out.at[i, "predicted_precursor_top5"] = json.dumps(rec["top5"], ensure_ascii=False)
        out.at[i, "predicted_precursor_top20"] = json.dumps(rec["top20"], ensure_ascii=False)
        out.at[i, "precursor_top1_score"] = rec["top1_score"]
        out.at[i, "precursor_top5_score_stats"] = json.dumps({"mean": rec["score_mean5"]}, ensure_ascii=False)
        out.at[i, "precursor_top20_uncertainty"] = rec["score_std20"]
        out.at[i, "precursor_input_mode"] = mode
        out.at[i, "precursor_input_source"] = mode
        out.at[i, "precursor_source_mix"] = rec["source_mix"]
        out.at[i, "contains_open_generated_precursor"] = rec["contains_open"]
        out.at[i, "contains_repair_precursor"] = rec["contains_repair"]
        out.at[i, "contains_raw_model_precursor"] = int("model" in rec["source_mix"])
        out.at[i, "precursor_set_size"] = len(top1)
        target = target_source_elements(str(row["formula"]))
        covered = set().union(*(elements(p) & target for p in top1)) if top1 else set()
        out.at[i, "target_source_elements"] = dump_list(sorted(target))
        out.at[i, "covered_source_elements"] = dump_list(sorted(covered))
        out.at[i, "missing_source_elements"] = rec["missing_source_elements"]
        out.at[i, "extra_forbidden_elements"] = rec["extra_forbidden_elements"]
        out.at[i, "precursor_check_status"] = rec["chemistry_check_status"]
        out.at[i, "chemistry_check_status"] = rec["chemistry_check_status"]
        out.at[i, "precursor_confidence_score"] = rec["top1_score"]
        for k, v in metrics.items():
            out.at[i, k] = v
        fam = family_counts(top1)
        for k, v in fam.items():
            if k not in out.columns:
                out[k] = 0.0
            out.at[i, k] = v
    return out


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "precursor_input_mode": {str(k): int(v) for k, v in df["precursor_input_mode"].value_counts().items()},
        "mean_precursor_f1": float(pd.to_numeric(df["precursor_f1_to_true"], errors="coerce").mean()),
        "mean_precursor_jaccard": float(pd.to_numeric(df["precursor_jaccard_to_true"], errors="coerce").mean()),
        "open_generated_rate": float(pd.to_numeric(df.get("contains_open_generated_precursor", 0), errors="coerce").fillna(0).mean()),
        "repair_rate": float(pd.to_numeric(df.get("contains_repair_precursor", 0), errors="coerce").fillna(0).mean()),
        "chemistry_ok_rate": float((df.get("precursor_check_status", "ok").astype(str) == "ok").mean()),
    }


def write_table(df: pd.DataFrame, out_base: Path) -> None:
    try:
        df.to_parquet(out_base.with_suffix(".parquet"), index=False)
    except Exception as exc:
        out_base.with_suffix(".parquet.SKIPPED.txt").write_text(str(exc), encoding="utf-8")
    df.to_csv(out_base.with_suffix(".csv"), index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage3 predicted-precursor OOF dataset v4 from train OOF top20 and Stage2 v5 val/test candidates.")
    ap.add_argument("--v3_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v3_20260610")
    ap.add_argument("--targets_dir", default="data/interim/generative/stage3_condition_targets_v3_20260610")
    ap.add_argument("--train_candidates", default="outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612/train_oof_top20_precursor_candidates.csv")
    ap.add_argument("--val_candidates", default="outputs/evaluation/stage2_candidate_pool_v5_20260610/val_candidate_sets_repaired.csv")
    ap.add_argument("--test_candidates", default="outputs/evaluation/stage2_score_calibration_v5_20260610/test_candidate_sets_calibrated_v5.csv")
    ap.add_argument("--output_dir", default="data/interim/generative/stage3_condition_dataset_predprec_oof_v4_20260612")
    args = ap.parse_args()

    v3 = Path(args.v3_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = {s: pd.read_csv(v3 / f"{s}.csv") for s in ["train", "val", "test"]}
    targets_dir = Path(args.targets_dir)
    for s in ["train", "val", "test"]:
        target_path = targets_dir / f"{s}.csv"
        if not target_path.exists():
            continue
        tgt = pd.read_csv(target_path)
        add_cols = [c for c in tgt.columns if c not in base[s].columns or c in {
            "temperature_c_raw", "time_h_raw", "log_time_h", "temperature_clipped", "time_h_clipped",
            "log_time_clipped", "temperature_bin", "temperature_bin_center", "time_bin",
            "time_bin_center", "atmosphere_raw", "solvent_raw", "atmosphere_normalized",
            "solvent_normalized", "atmosphere_known_mask", "solvent_known_mask",
            "atmosphere_target_class", "solvent_target_class", "atmosphere_missing_reason",
            "solvent_missing_reason", "condition_group_key", "multimodal_group_id",
            "multimodal_group_size", "temperature_p10", "time_p10", "temperature_p25",
            "time_p25", "temperature_p50", "time_p50", "temperature_p75", "time_p75",
            "temperature_p90", "time_p90", "temperature_iqr", "time_iqr", "is_multimodal_group",
        }]
        merged = base[s].merge(tgt[["sample_id"] + add_cols].drop_duplicates("sample_id"), on="sample_id", how="left", suffixes=("", "__target"))
        for c in add_cols:
            alt = f"{c}__target"
            if alt in merged.columns:
                merged[c] = merged[alt].combine_first(merged[c]) if c in merged.columns else merged[alt]
                merged = merged.drop(columns=[alt])
        base[s] = merged
    old_summary = {s: summarize(df) for s, df in base.items()}
    cands = {
        "train": read_ranked_candidates(Path(args.train_candidates), "train"),
        "val": read_ranked_candidates(Path(args.val_candidates), "val"),
        "test": read_ranked_candidates(Path(args.test_candidates), "test"),
    }
    modes = {"train": "stage2_v4_oof_top1", "val": "stage2_v5_val_top1", "test": "stage2_v5_test_top1"}
    outputs = {s: apply_candidates(base[s], cands[s], modes[s]) for s in ["train", "val", "test"]}
    for s, df in outputs.items():
        df["split"] = s
        write_table(df, outdir / s)
        meta_cols = [c for c in [
            "sample_index", "sample_id", "formula", "reaction_method", "split", "precursor_input_mode",
            "predicted_precursor_top1", "precursor_f1_to_true", "precursor_jaccard_to_true",
            "contains_open_generated_precursor", "contains_repair_precursor", "precursor_check_status",
        ] if c in df.columns]
        df[meta_cols].to_csv(outdir / f"{s}_meta.csv", index=False)
    new_summary = {s: summarize(df) for s, df in outputs.items()}
    schema = json.loads((v3 / "schema.json").read_text(encoding="utf-8")) if (v3 / "schema.json").exists() else {}
    schema["stage3_predprec_oof_v4"] = {"config": vars(args), "candidate_modes": modes}
    write_json(outdir / "schema.json", schema)
    comparison = {}
    for s in ["train", "val", "test"]:
        comparison[s] = {
            "old_v3_precursor_f1": old_summary[s]["mean_precursor_f1"],
            "new_v4_precursor_f1": new_summary[s]["mean_precursor_f1"],
            "old_v3_jaccard": old_summary[s]["mean_precursor_jaccard"],
            "new_v4_jaccard": new_summary[s]["mean_precursor_jaccard"],
            "old_v3_repair_rate": old_summary[s]["repair_rate"],
            "new_v4_repair_rate": new_summary[s]["repair_rate"],
        }
    summary = {"config": vars(args), "old_v3": old_summary, "new_v4": new_summary, "comparison": comparison}
    write_json(outdir / "predprec_oof_v4_dataset_summary.json", summary)
    report = ["# Stage3 Predicted-Precursor OOF Dataset v4", "", "```json", json.dumps(to_builtin(summary), ensure_ascii=False, indent=2), "```"]
    (outdir / "predprec_oof_v4_dataset_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
