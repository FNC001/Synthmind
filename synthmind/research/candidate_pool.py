from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_CANDIDATE_COLUMNS = {
    "sample_id",
    "candidate_id",
    "target_id",
    "predicted_method",
    "canonical_precursor_set",
    "temperature",
    "time",
    "atmosphere",
    "solvent",
    "rsp_score",
    "cdg_score",
    "legacy_grv_score",
    "grv_score",
    "rsp_rank",
    "cdg_rank",
    "final_rank",
    "candidate_provenance",
    "chemistry_features",
    "canonicalization_version",
    "split_version",
    "candidate_budget_version",
}


@dataclass(frozen=True)
class BudgetConfig:
    budget_id: str
    route_budget: int
    skeleton_budget: int
    tie_break_columns: tuple[str, ...] = ("sample_id", "candidate_id")


def stable_file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_route_key(df: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for col in ["predicted_method", "canonical_precursor_set", "temperature", "time", "atmosphere", "solvent"]:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str).str.strip().str.lower())
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    key = parts[0]
    for p in parts[1:]:
        key = key + "||" + p
    return key


def dedupe_and_budget(df: pd.DataFrame, budget: BudgetConfig) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise ValueError("candidate pool requires sample_id")
    work = df.copy()
    if "candidate_id" not in work.columns:
        work["candidate_id"] = work["sample_id"].astype(str) + "::" + work.groupby("sample_id").cumcount().astype(str)
    work["_route_key"] = canonical_route_key(work)
    sort_cols = [c for c in ["sample_id", "final_rank", "grv_score", "legacy_grv_score", "candidate_id"] if c in work.columns]
    ascending = [True if c not in {"grv_score", "legacy_grv_score"} else False for c in sort_cols]
    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    work = work.drop_duplicates(["sample_id", "_route_key"], keep="first")
    work["_budget_rank"] = work.groupby("sample_id", sort=False).cumcount() + 1
    work = work[work["_budget_rank"] <= budget.route_budget].drop(columns=["_route_key", "_budget_rank"])
    return work


def validate_candidate_pool_columns(columns: Iterable[str]) -> set[str]:
    return REQUIRED_CANDIDATE_COLUMNS - set(columns)

