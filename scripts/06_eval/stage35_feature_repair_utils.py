#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
IGNORED_SOURCE_ELEMENTS = {"H", "O", "C", "N"}
CHEM_FEATURE_COLUMNS = [
    "element_coverage",
    "missing_element_count",
    "extra_element_count",
    "candidate_size",
]


def parse_precursor_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value if str(x)]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, (list, tuple, set)):
                return [str(x) for x in obj if str(x)]
        except Exception:
            pass
    for sep in (";", "|"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    return [text]


def elements(text: Any) -> set[str]:
    return set(ELEMENT_RE.findall(str(text)))


def target_elements(formula: Any) -> set[str]:
    elems = elements(formula)
    without_oxygen = elems - {"O"}
    return without_oxygen or elems


def precursor_source_elements(labels: Sequence[str]) -> set[str]:
    out: set[str] = set()
    for label in labels:
        out |= elements(label) - IGNORED_SOURCE_ELEMENTS
    return out


def coverage_stats(labels: Sequence[str], formula: Any) -> dict[str, float]:
    target = target_elements(formula)
    present = precursor_source_elements(labels)
    return {
        "element_coverage": float(len(target & present) / len(target)) if target else 1.0,
        "missing_element_count": float(len(target - present)),
        "extra_element_count": float(len(present - target)),
        "candidate_size": float(len(labels)),
    }


def _needs_repair(series: pd.Series, force: bool) -> pd.Series:
    if force:
        return pd.Series(True, index=series.index)
    values = pd.to_numeric(series, errors="coerce")
    return values.isna()


def repair_precursor_chem_features(
    df: pd.DataFrame,
    *,
    force_columns: Iterable[str] | None = None,
    repair_zeroish_columns: bool = False,
) -> pd.DataFrame:
    """Repair safe chemistry features from formula and predicted precursors.

    The Stage35 v3/v4 route builders can receive precursor CSVs with different
    schemas across splits.  When test precursors lack element-coverage columns,
    old builders silently filled them with zero, causing a validation/test
    feature shift.  These features are inference-safe because they only use the
    target formula and the predicted precursor set.
    """

    if df.empty:
        return df
    if "formula" not in df.columns:
        return df

    pred_col = next((c for c in ("pred_precursors", "candidate_set", "precursor_set") if c in df.columns), None)
    if pred_col is None:
        return df

    out = df.copy()
    force_set = set(force_columns or [])
    parsed = out[pred_col].map(parse_precursor_list)
    stats = [
        coverage_stats(labels, formula)
        for labels, formula in zip(parsed.tolist(), out["formula"].tolist())
    ]
    stats_df = pd.DataFrame(stats, index=out.index)

    for col in CHEM_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = stats_df[col]
            continue
        current = pd.to_numeric(out[col], errors="coerce")
        force = col in force_set
        mask = _needs_repair(current, force)
        if repair_zeroish_columns and not force:
            nonempty = parsed.map(bool)
            mask = mask | (nonempty & current.fillna(0.0).eq(0.0))
        out.loc[mask, col] = stats_df.loc[mask, col]

    if "chemistry_check_status" not in out.columns or "chemistry_check_status" in force_set:
        missing = pd.to_numeric(out["missing_element_count"], errors="coerce").fillna(0.0)
        extra = pd.to_numeric(out["extra_element_count"], errors="coerce").fillna(0.0)
        out["chemistry_check_status"] = np.where((missing <= 0) & (extra <= 0), "ok", "failed")

    if "precursor_condition_compatibility_score" in out.columns or "precursor_condition_compatibility_score" in force_set:
        out["precursor_condition_compatibility_score"] = pd.to_numeric(
            out["element_coverage"], errors="coerce"
        ).fillna(0.0)

    return out


def repair_summary(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(after))}
    for col in CHEM_FEATURE_COLUMNS + ["precursor_condition_compatibility_score"]:
        if col not in after.columns:
            continue
        b = pd.to_numeric(before[col], errors="coerce") if col in before.columns else pd.Series(np.nan, index=after.index)
        a = pd.to_numeric(after[col], errors="coerce")
        changed = (~b.fillna(np.inf).eq(a.fillna(np.inf))).mean() if len(after) else 0.0
        summary[col] = {
            "before_mean": None if b.notna().sum() == 0 else float(b.mean()),
            "after_mean": None if a.notna().sum() == 0 else float(a.mean()),
            "after_min": None if a.notna().sum() == 0 else float(a.min()),
            "after_max": None if a.notna().sum() == 0 else float(a.max()),
            "changed_fraction": float(changed),
        }
    if "chemistry_check_status" in after.columns:
        summary["chemistry_check_status_counts"] = after["chemistry_check_status"].fillna("<MISSING>").astype(str).value_counts().to_dict()
    return summary
