from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


GROUP_KEY_PRIORITY = (
    "target_material_id",
    "material_id",
    "target_id",
    "reduced_formula",
    "formula",
    "composition_family",
    "prototype",
    "composition_fingerprint_cluster",
)


def choose_group_key(columns: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for key in GROUP_KEY_PRIORITY:
        if key in lower:
            return lower[key]
    return None


def fingerprint_rows(df: pd.DataFrame, cols: list[str]) -> str:
    h = hashlib.sha256()
    use = [c for c in cols if c in df.columns]
    for row in df[use].fillna("").astype(str).itertuples(index=False, name=None):
        h.update(("||".join(row) + "\n").encode("utf-8"))
    return h.hexdigest()

