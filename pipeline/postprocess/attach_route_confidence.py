#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return int(float(x))
    except Exception:
        return default


def norm_clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def infer_condition_reasonable(row):
    temp = safe_float(row.get("temperature_c"), None)
    time_h = safe_float(row.get("time_h"), None)

    warnings = []
    score = 1.0

    if temp is None:
        warnings.append("missing_temperature")
        score -= 0.30
    else:
        if temp < 300:
            warnings.append("low_temperature_warning")
            score -= 0.25
        elif temp > 1600:
            warnings.append("extreme_high_temperature_warning")
            score -= 0.35
        elif temp > 1300:
            warnings.append("high_temperature_warning")
            score -= 0.15

    if time_h is None:
        warnings.append("missing_time")
        score -= 0.30
    else:
        if time_h < 0.1:
            warnings.append("too_short_time_warning")
            score -= 0.25
        elif time_h > 240:
            warnings.append("too_long_time_warning")
            score -= 0.35
        elif time_h > 72:
            warnings.append("long_time_warning")
            score -= 0.10

    condition_source = str(row.get("condition_source", "")).strip()
    if condition_source == "top_component_mean":
        score += 0.05
    elif condition_source == "flow_sample":
        score -= 0.03

    return norm_clip(score), warnings


def _resolve_element_coverage(row):
    """Resolve element_coverage from multiple possible column names."""
    for col in ("element_coverage", "v3_element_coverage"):
        v = safe_float(row.get(col), None)
        if v is not None and v > 0.0:
            return v

    has_missing = safe_int(row.get("has_missing_target_elements_qc"), -1)
    if has_missing == 0:
        return 1.0
    elif has_missing == 1:
        n_missing = safe_int(row.get("v43_safe_n_missing_core"), 1)
        return max(0.0, 1.0 - n_missing * 0.25)

    return 0.5


def _resolve_missing_count(row):
    """Resolve missing_count from multiple possible column names."""
    for col in ("missing_count", "v3_missing_count"):
        v = safe_int(row.get(col), None)
        if v is not None and v > 0:
            return v

    return safe_int(row.get("v43_safe_n_missing_core"), 0)


def compute_route_confidence(row, max_rank):
    warnings = []

    element_coverage = _resolve_element_coverage(row)
    missing_count = _resolve_missing_count(row)
    extra_penalty = safe_float(row.get("extra_element_penalty"), safe_float(row.get("v3_extra_element_penalty"), 0.0))
    precursor_rank = safe_int(row.get("precursor_rank"), max_rank)
    stage3_score = safe_float(row.get("stage3_score"), 0.0)
    stage35_score = safe_float(row.get("stage35_v21_score"), safe_float(row.get("stage35_score"), 0.0))
    stage35_prob = safe_float(row.get("stage35_v2_prob"), safe_float(row.get("stage35_prob"), 0.0))

    # 1. element confidence
    element_score = element_coverage
    if missing_count > 0:
        element_score -= min(0.6, 0.2 * missing_count)
        warnings.append("missing_target_elements")
    if extra_penalty > 0:
        element_score -= min(0.4, 0.1 * extra_penalty)
        warnings.append("extra_element_penalty")
    element_score = norm_clip(element_score)

    # 2. precursor rank confidence
    # precursor_rank sometimes starts from 0.
    rank_score = 1.0 - (max(precursor_rank, 0) / max(max_rank, 1))
    rank_score = norm_clip(rank_score)

    # 3. stage3 condition score
    condition_model_score = norm_clip(stage3_score)

    # 4. condition physical reasonableness
    condition_rule_score, condition_warnings = infer_condition_reasonable(row)
    warnings.extend(condition_warnings)

    # 5. learned ranker probability is often poorly calibrated, so use weak weight
    prob_score = norm_clip(stage35_prob)

    # 6. route-score relative signal, compressed
    # If stage35_score is already large, this mostly avoids destroying confidence.
    route_score_component = norm_clip(stage35_score / 12.0)

    confidence_score = (
        0.30 * element_score
        + 0.18 * rank_score
        + 0.17 * condition_model_score
        + 0.17 * condition_rule_score
        + 0.08 * prob_score
        + 0.10 * route_score_component
    )
    confidence_score = norm_clip(confidence_score)

    if confidence_score >= 0.78:
        confidence_level = "high_confidence"
        recommendation_status = "recommended"
    elif confidence_score >= 0.58:
        confidence_level = "medium_confidence"
        recommendation_status = "recommended_with_validation"
    else:
        confidence_level = "low_confidence"
        recommendation_status = "review_required"

    # Upgrade/downgrade by hard warnings — only for severe element mismatch
    if element_coverage < 0.5 or missing_count >= 2:
        confidence_level = "low_confidence"
        recommendation_status = "review_required"

    if "extreme_high_temperature_warning" in warnings or "too_long_time_warning" in warnings:
        if confidence_level == "high_confidence":
            confidence_level = "medium_confidence"
            recommendation_status = "recommended_with_validation"

    if len(warnings) == 0:
        warning_level = "no_warning"
    elif any(w in warnings for w in ["missing_target_elements", "extreme_high_temperature_warning", "too_long_time_warning"]):
        warning_level = "major_warning"
    else:
        warning_level = "minor_warning"

    return {
        "route_confidence_score": round(confidence_score, 4),
        "route_confidence_level": confidence_level,
        "route_warning_level": warning_level,
        "route_recommendation_status": recommendation_status,
        "route_warnings": ";".join(sorted(set(warnings))) if warnings else "",
        "model_claim_boundary": "model_suggested_not_source_backed",
    }


def write_markdown(df, output_md, top_n):
    cols = [
        "final_route_rank",
        "precursor_set",
        "temperature_c",
        "time_h",
        "condition_source",
        "stage3_score",
        "element_coverage",
        "route_confidence_level",
        "route_confidence_score",
        "route_warning_level",
        "route_recommendation_status",
        "route_warnings",
    ]
    cols = [c for c in cols if c in df.columns]

    out = []
    out.append("# Final Top Routes with Confidence\n")
    out.append("")
    out.append("This file reports model-suggested synthesis routes with confidence and warning annotations.")
    out.append("")
    out.append("Important: `model_suggested_not_source_backed` means the route is generated and ranked by the model, but has not yet been verified against external literature or experimental evidence.")
    out.append("")

    show = df.head(top_n).copy()

    for _, row in show.iterrows():
        rank = row.get("final_route_rank", row.get("stage35_v21_rank", ""))
        precursor_set = row.get("precursor_set", "")
        temp = safe_float(row.get("temperature_c"), float("nan"))
        time_h = safe_float(row.get("time_h"), float("nan"))
        confidence = row.get("route_confidence_level", "")
        conf_score = row.get("route_confidence_score", "")
        status = row.get("route_recommendation_status", "")
        warning_level = row.get("route_warning_level", "")
        warnings = row.get("route_warnings", "")

        out.append(f"## Route {rank}")
        out.append("")
        out.append(f"- **Precursor set:** {precursor_set}")
        out.append(f"- **Temperature:** {temp:.2f} °C")
        out.append(f"- **Time:** {time_h:.2f} h")
        out.append(f"- **Confidence:** {confidence} ({conf_score})")
        out.append(f"- **Status:** {status}")
        out.append(f"- **Warning level:** {warning_level}")
        if str(warnings).strip():
            out.append(f"- **Warnings:** {warnings}")
        else:
            out.append("- **Warnings:** none")
        out.append(f"- **Claim boundary:** {row.get('model_claim_boundary', 'model_suggested_not_source_backed')}")
        out.append("")

    output_md.write_text("\n".join(out), encoding="utf-8")



def normalize_level_value(x: object) -> str:
    """Normalize empty/NaN-like values to an empty string."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    return s


def max_warning_level(a: object, b: object) -> str:
    """Return the stronger warning level."""
    order = {
        "no_warning": 0,
        "minor_warning": 1,
        "major_warning": 2,
    }
    aa = normalize_level_value(a) or "no_warning"
    bb = normalize_level_value(b) or "no_warning"
    return aa if order.get(aa, 0) >= order.get(bb, 0) else bb


def max_recommendation_status(a: object, b: object) -> str:
    """Return the more conservative recommendation status."""
    order = {
        "recommended": 0,
        "recommended_with_validation": 1,
        "review_required": 2,
    }
    aa = normalize_level_value(a) or "recommended"
    bb = normalize_level_value(b) or "recommended"
    return aa if order.get(aa, 0) >= order.get(bb, 0) else bb


def merge_warning_text(a: object, b: object) -> str:
    """Merge warning strings while preserving readable semicolon-separated form."""
    vals = []
    for x in [a, b]:
        s = normalize_level_value(x)
        if not s:
            continue
        for part in str(s).split(";"):
            part = part.strip()
            if part and part not in vals:
                vals.append(part)
    return "; ".join(vals)


def apply_precursor_qc_to_confidence(df):
    """
    Use precursor-level QC to adjust final route confidence.

    Required behavior:
    - precursor_qc_level == pass:
        no downgrade.
    - precursor_qc_level == minor_warning:
        small confidence penalty;
        route_warning_level at least minor_warning;
        route_recommendation_status at least recommended_with_validation.
    - precursor_qc_level == major_warning:
        stronger confidence penalty;
        route_warning_level at least major_warning;
        route_recommendation_status at least review_required.
    """
    if "precursor_qc_level" not in df.columns:
        return df

    if "route_confidence_score" not in df.columns:
        df["route_confidence_score"] = 0.5
    if "route_warning_level" not in df.columns:
        df["route_warning_level"] = "no_warning"
    if "route_recommendation_status" not in df.columns:
        df["route_recommendation_status"] = "recommended_with_validation"
    if "route_warnings" not in df.columns:
        df["route_warnings"] = ""

    for i, row in df.iterrows():
        qc_level = normalize_level_value(row.get("precursor_qc_level", ""))
        qc_status = normalize_level_value(row.get("precursor_qc_status", ""))
        qc_warnings = normalize_level_value(row.get("precursor_qc_warnings", ""))

        if qc_level == "major_warning" or qc_status == "review_required":
            df.at[i, "route_confidence_score"] = max(
                0.0,
                float(row.get("route_confidence_score", 0.5)) - 0.15,
            )
            df.at[i, "route_warning_level"] = max_warning_level(
                row.get("route_warning_level", "no_warning"),
                "major_warning",
            )
            df.at[i, "route_recommendation_status"] = max_recommendation_status(
                row.get("route_recommendation_status", "recommended_with_validation"),
                "review_required",
            )
            df.at[i, "route_warnings"] = merge_warning_text(
                row.get("route_warnings", ""),
                "precursor_qc_major_warning" + (f":{qc_warnings}" if qc_warnings else ""),
            )

        elif qc_level == "minor_warning" or qc_status == "recommended_with_validation":
            df.at[i, "route_confidence_score"] = max(
                0.0,
                float(row.get("route_confidence_score", 0.5)) - 0.05,
            )
            df.at[i, "route_warning_level"] = max_warning_level(
                row.get("route_warning_level", "no_warning"),
                "minor_warning",
            )
            df.at[i, "route_recommendation_status"] = max_recommendation_status(
                row.get("route_recommendation_status", "recommended"),
                "recommended_with_validation",
            )
            if qc_warnings:
                df.at[i, "route_warnings"] = merge_warning_text(
                    row.get("route_warnings", ""),
                    "precursor_qc_minor_warning:" + qc_warnings,
                )

    # Recompute route_confidence_level after QC penalty.
    def score_to_level(x):
        try:
            x = float(x)
        except Exception:
            x = 0.0
        if x >= 0.78:
            return "high_confidence"
        if x >= 0.55:
            return "medium_confidence"
        return "low_confidence"

    df["route_confidence_level"] = df["route_confidence_score"].apply(score_to_level)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", required=True)
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    summary_json = Path(args.summary_json)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input_csv: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise RuntimeError(f"Empty input_csv: {input_csv}")

    if "final_route_rank" not in df.columns:
        if "stage35_v21_rank" in df.columns:
            df["final_route_rank"] = df["stage35_v21_rank"]
        else:
            df["final_route_rank"] = range(1, len(df) + 1)

    max_rank = max(1, safe_int(df.get("precursor_rank", pd.Series([len(df)])).max(), len(df)))

    conf_rows = []
    for _, row in df.iterrows():
        conf_rows.append(compute_route_confidence(row, max_rank=max_rank))

    conf_df = pd.DataFrame(conf_rows)
    out_df = pd.concat([df.reset_index(drop=True), conf_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_csv, index=False)
    write_markdown(out_df, output_md, top_n=args.top_n)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_md": str(output_md),
        "n_routes": int(len(out_df)),
        "top_n": int(args.top_n),
        "confidence_level_counts": out_df["route_confidence_level"].value_counts().to_dict(),
        "warning_level_counts": out_df["route_warning_level"].value_counts().to_dict(),
        "recommendation_status_counts": out_df["route_recommendation_status"].value_counts().to_dict(),
        "claim_boundary": "model_suggested_not_source_backed",
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")
    print(f"[SAVE] {summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# ============================================================
# Precursor-QC-aware confidence post-processing
# ============================================================

def _as_clean_str(x):
    """Convert NaN/None-like values to an empty string."""
    try:
        import pandas as _pd
        if _pd.isna(x):
            return ""
    except Exception:
        pass
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _merge_warning_text(*items):
    """Merge warning strings while keeping order and removing empties/duplicates."""
    out = []
    seen = set()
    for item in items:
        s = _as_clean_str(item).strip()
        if not s:
            continue
        parts = [x.strip() for x in s.split(";") if x.strip()]
        for part in parts:
            if part not in seen:
                seen.add(part)
                out.append(part)
    return "; ".join(out) if out else ""


def apply_precursor_qc_to_route_confidence(df):
    """
    Propagate precursor-level QC risk into route-level confidence.

    Expected optional columns from qc_route_precursors.py:
      - precursor_qc_score
      - precursor_qc_level
      - precursor_qc_status
      - precursor_qc_warnings

    Policy:
      pass:
        keep original confidence result.
      minor_warning:
        reduce confidence score moderately;
        route_warning_level must be at least minor_warning;
        route_recommendation_status must be at most recommended_with_validation.
      major_warning:
        reduce confidence score strongly;
        route_warning_level becomes major_warning;
        route_recommendation_status becomes review_required.
    """
    if df is None or len(df) == 0:
        return df

    if "precursor_qc_level" not in df.columns:
        return df

    import pandas as pd

    if "route_confidence_score" in df.columns:
        df["route_confidence_score"] = pd.to_numeric(
            df["route_confidence_score"], errors="coerce"
        ).fillna(0.0)

    # Ensure downstream columns exist.
    if "route_warning_level" not in df.columns:
        df["route_warning_level"] = "no_warning"
    if "route_recommendation_status" not in df.columns:
        df["route_recommendation_status"] = "recommended_with_validation"
    if "route_warnings" not in df.columns:
        df["route_warnings"] = ""

    for idx, row in df.iterrows():
        qc_level = _as_clean_str(row.get("precursor_qc_level", "")).strip()
        qc_warnings = _as_clean_str(row.get("precursor_qc_warnings", "")).strip()

        if not qc_level or qc_level == "pass":
            continue

        if qc_level == "minor_warning":
            # Penalize confidence but do not force review_required.
            if "route_confidence_score" in df.columns:
                df.at[idx, "route_confidence_score"] = max(
                    0.0, float(df.at[idx, "route_confidence_score"]) - 0.05
                )

            old_warning_level = _as_clean_str(row.get("route_warning_level", "no_warning"))
            if old_warning_level == "no_warning":
                df.at[idx, "route_warning_level"] = "minor_warning"

            old_status = _as_clean_str(row.get("route_recommendation_status", "recommended"))
            if old_status == "recommended":
                df.at[idx, "route_recommendation_status"] = "recommended_with_validation"

            qc_msg = "precursor_qc_minor_warning"
            if qc_warnings:
                qc_msg += f":{qc_warnings}"
            df.at[idx, "route_warnings"] = _merge_warning_text(
                row.get("route_warnings", ""),
                qc_msg,
            )

        elif qc_level == "major_warning":
            # Major precursor QC risk must dominate route-level status.
            if "route_confidence_score" in df.columns:
                df.at[idx, "route_confidence_score"] = max(
                    0.0, float(df.at[idx, "route_confidence_score"]) - 0.15
                )

            df.at[idx, "route_warning_level"] = "major_warning"
            df.at[idx, "route_recommendation_status"] = "review_required"

            qc_msg = "precursor_qc_major_warning"
            if qc_warnings:
                qc_msg += f":{qc_warnings}"
            df.at[idx, "route_warnings"] = _merge_warning_text(
                row.get("route_warnings", ""),
                qc_msg,
            )

        else:
            # Unknown QC levels are kept as minor conservative warnings.
            if "route_confidence_score" in df.columns:
                df.at[idx, "route_confidence_score"] = max(
                    0.0, float(df.at[idx, "route_confidence_score"]) - 0.03
                )

            old_warning_level = _as_clean_str(row.get("route_warning_level", "no_warning"))
            if old_warning_level == "no_warning":
                df.at[idx, "route_warning_level"] = "minor_warning"

            old_status = _as_clean_str(row.get("route_recommendation_status", "recommended"))
            if old_status == "recommended":
                df.at[idx, "route_recommendation_status"] = "recommended_with_validation"

            qc_msg = f"precursor_qc_unknown_level:{qc_level}"
            if qc_warnings:
                qc_msg += f":{qc_warnings}"
            df.at[idx, "route_warnings"] = _merge_warning_text(
                row.get("route_warnings", ""),
                qc_msg,
            )

    # Recompute confidence level from adjusted score, if the column exists.
    if "route_confidence_score" in df.columns:
        def _score_to_level(x):
            try:
                x = float(x)
            except Exception:
                x = 0.0
            if x >= 0.78:
                return "high_confidence"
            if x >= 0.50:
                return "medium_confidence"
            return "low_confidence"

        df["route_confidence_level"] = df["route_confidence_score"].apply(_score_to_level)

    # Convert empty strings in route_warnings back to NA-like blanks for cleaner CSV/MD.
    if "route_warnings" in df.columns:
        df["route_warnings"] = df["route_warnings"].apply(lambda x: _as_clean_str(x).strip())
        df.loc[df["route_warnings"] == "", "route_warnings"] = pd.NA

    return df
