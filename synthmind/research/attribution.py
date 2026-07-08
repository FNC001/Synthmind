from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ATTRIBUTION_CATEGORIES = (
    "reference_invalid",
    "success",
    "skeleton_miss",
    "condition_miss",
    "ranking_miss",
    "normalization_or_reference_issue",
    "unresolved",
)


@dataclass(frozen=True)
class AttributionConfig:
    rank_col: str = "final_rank"
    top_rank: int = 1
    skeleton_hit_col: str = "precursor_exact_if_eval"
    condition_hit_col: str = "relaxed_condition_hit_if_eval"
    route_hit_col: str = "relaxed_route_hit_if_eval"
    reference_valid_col: str | None = None
    normalization_issue_col: str | None = None


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna(0).astype(str).str.lower().isin({"1", "1.0", "true", "yes"})


def attribute_candidates(df: pd.DataFrame, config: AttributionConfig = AttributionConfig()) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise ValueError("candidate table requires sample_id")
    rows = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        reference_valid = True
        if config.reference_valid_col and config.reference_valid_col in group.columns:
            reference_valid = bool(_truthy(group[config.reference_valid_col]).any())
        top = group[group[config.rank_col] <= config.top_rank] if config.rank_col in group.columns else group.head(1)
        skeleton_any = _truthy(group.get(config.skeleton_hit_col, pd.Series(0, index=group.index))).any()
        condition_any = _truthy(group.get(config.condition_hit_col, pd.Series(0, index=group.index))).any()
        route_any = _truthy(group.get(config.route_hit_col, pd.Series(0, index=group.index))).any()
        top_route = _truthy(top.get(config.route_hit_col, pd.Series(0, index=top.index))).any()
        norm_issue = False
        if config.normalization_issue_col and config.normalization_issue_col in group.columns:
            norm_issue = bool(_truthy(group[config.normalization_issue_col]).any())
        if not reference_valid:
            category = "reference_invalid"
        elif top_route:
            category = "success"
        elif not skeleton_any:
            category = "skeleton_miss"
        elif not condition_any:
            category = "condition_miss"
        elif route_any:
            category = "ranking_miss"
        elif norm_issue:
            category = "normalization_or_reference_issue"
        else:
            category = "unresolved"
        first_hit_rank = None
        if route_any and config.rank_col in group.columns:
            hit = group[_truthy(group.get(config.route_hit_col, pd.Series(0, index=group.index)))]
            if len(hit):
                first_hit_rank = int(pd.to_numeric(hit[config.rank_col], errors="coerce").min())
        rows.append(
            {
                "sample_id": sample_id,
                "attribution": category,
                "skeleton_oracle": bool(skeleton_any),
                "route_oracle": bool(route_any),
                "top1_success": bool(top_route),
                "first_hit_rank": first_hit_rank,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) and int(out["attribution"].notna().sum()) != int(out["sample_id"].nunique()):
        raise AssertionError("Attribution conservation failed")
    return out


def summarize_attribution(rows: pd.DataFrame) -> pd.DataFrame:
    total = max(len(rows), 1)
    return (
        rows["attribution"]
        .value_counts()
        .rename_axis("attribution")
        .reset_index(name="count")
        .assign(fraction=lambda x: x["count"] / total)
    )

