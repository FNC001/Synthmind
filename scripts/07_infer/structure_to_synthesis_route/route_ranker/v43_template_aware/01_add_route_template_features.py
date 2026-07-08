#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage35 v4.3 route-template-aware feature builder.

Input:
  A route candidate CSV, usually:
    synthesis_routes_stage35_v33_chemonly_reranked.csv
  or:
    synthesis_routes_stage35_v42_pairwise_foreignaware_chemonly_reranked.csv

Output:
  CSV/MD/summary JSON with route_template_* features.

This script is intentionally rule-based and conservative.
It does not change ranking. It only annotates route candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")


def clean_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def split_precursors(precursor_set: Any) -> list[str]:
    s = clean_str(precursor_set)
    if not s:
        return []
    return [x.strip() for x in s.split(";") if x.strip()]


def parse_elements_from_formula(formula: str) -> set[str]:
    return set(ELEMENT_RE.findall(clean_str(formula)))


def parse_semicolon_elements(x: Any) -> set[str]:
    s = clean_str(x)
    if not s:
        return set()
    return {item.strip() for item in s.split(";") if item.strip()}


def precursor_type(p: str) -> set[str]:
    """
    Conservative precursor-type classifier.
    A precursor can have multiple types, e.g. Co(NO3)2·6H2O = nitrate + hydrate.
    """
    s = clean_str(p)
    compact = s.replace(" ", "")

    types: set[str] = set()

    if not s:
        return types

    elems = parse_elements_from_formula(s)

    # Hydrate
    if "·" in compact or ".H2O" in compact or "H2O" in compact:
        # Avoid marking plain H2O2 as hydrate.
        if "·" in compact or re.search(r"[·\.]\d*H2O", compact):
            types.add("hydrate")

    # Common ions / groups
    if "NO3" in compact:
        types.add("nitrate")
    if "CO3" in compact:
        types.add("carbonate")
    if "PO4" in compact or "H3PO4" in compact or "P2O5" in compact:
        types.add("phosphate")
    if "SO4" in compact or "HSO4" in compact:
        types.add("sulfate")
    if "SO3" in compact:
        types.add("sulfite")
    if "OH" in compact:
        types.add("hydroxide")
    if "NH4" in compact:
        types.add("ammonium")

    # Chalcogen / halogen families
    if "SeO2" in compact or "SeO3" in compact or "SeO4" in compact:
        types.add("selenite_selenate")
    elif "Se" in elems:
        types.add("selenide_or_elemental_se")

    if "S" in elems and not ({"O", "S"} <= elems and ("SO4" in compact or "SO3" in compact)):
        # S, Na2S, metal sulfide, thiourea-like formulas
        types.add("sulfide_or_elemental_s")

    halogens = {"F", "Cl", "Br", "I"}
    if elems & halogens:
        types.add("halide_or_elemental_halogen")

    # Oxide: contains O but not dominated by oxyanion labels.
    oxyanion_types = {"nitrate", "carbonate", "phosphate", "sulfate", "sulfite", "selenite_selenate"}
    if "O" in elems and not (types & oxyanion_types):
        types.add("oxide_or_oxygen_source")

    # Organic-like
    if "C" in elems and "H" in elems and not ("carbonate" in types):
        types.add("organic_like")

    # Elemental precursor: exactly one element symbol and no obvious group syntax.
    # Examples: Fe, S, Se, I, Rb
    if re.fullmatch(r"[A-Z][a-z]?", compact):
        types.add("elemental")

    return types


def infer_primary_template(type_counts: Counter, target_elements: set[str]) -> tuple[str, str, float]:
    """
    Return:
      primary_template, secondary_templates, confidence
    """
    total_typed = sum(type_counts.values())
    if total_typed == 0:
        return "unknown_route", "", 0.0

    priority = [
        ("phosphate_route", "phosphate"),
        ("sulfate_route", "sulfate"),
        ("sulfite_route", "sulfite"),
        ("carbonate_route", "carbonate"),
        ("nitrate_route", "nitrate"),
        ("selenite_selenate_route", "selenite_selenate"),
        ("selenide_route", "selenide_or_elemental_se"),
        ("sulfide_route", "sulfide_or_elemental_s"),
        ("halide_route", "halide_or_elemental_halogen"),
        ("hydroxide_route", "hydroxide"),
        ("oxide_route", "oxide_or_oxygen_source"),
        ("elemental_route", "elemental"),
        ("organic_assisted_route", "organic_like"),
        ("hydrate_assisted_route", "hydrate"),
    ]

    present_templates = []
    for template_name, type_name in priority:
        if type_counts.get(type_name, 0) > 0:
            present_templates.append(template_name)

    if not present_templates:
        return "unknown_route", "", 0.0

    primary = present_templates[0]
    secondary = ";".join(present_templates[1:])

    # Conservative confidence: fraction of precursors supporting the primary type.
    primary_type = dict(priority)[primary]
    confidence = float(type_counts.get(primary_type, 0)) / max(total_typed, 1)
    confidence = max(0.0, min(1.0, confidence))

    return primary, secondary, confidence


def template_matches_target(primary_template: str, target_elements: set[str]) -> int:
    """
    Conservative target-template consistency.
    """
    if primary_template == "phosphate_route":
        return int("P" in target_elements)
    if primary_template in {"sulfate_route", "sulfite_route", "sulfide_route"}:
        return int("S" in target_elements)
    if primary_template in {"selenide_route", "selenite_selenate_route"}:
        return int("Se" in target_elements)
    if primary_template in {"oxide_route", "nitrate_route", "carbonate_route", "hydroxide_route"}:
        return int("O" in target_elements or len(target_elements) > 0)
    if primary_template == "halide_route":
        return int(bool(target_elements & {"F", "Cl", "Br", "I"}))
    if primary_template == "elemental_route":
        return 1
    return 0


def add_template_features(df: pd.DataFrame, precursor_col: str, target_elements_col: str) -> pd.DataFrame:
    out = df.copy()

    rows = []
    for _, row in out.iterrows():
        precursors = split_precursors(row.get(precursor_col, ""))
        target_elements = parse_semicolon_elements(row.get(target_elements_col, ""))

        all_types = []
        for p in precursors:
            all_types.extend(sorted(precursor_type(p)))

        type_counts = Counter(all_types)
        primary, secondary, conf = infer_primary_template(type_counts, target_elements)

        has = lambda name: int(type_counts.get(name, 0) > 0)

        n_precursors = len(precursors)
        n_template_types = len(type_counts)

        n_elemental = type_counts.get("elemental", 0)
        elemental_ratio = float(n_elemental) / max(n_precursors, 1)

        common_solid_state = int(
            has("oxide_or_oxygen_source")
            or has("nitrate")
            or has("carbonate")
            or has("phosphate")
            or has("sulfate")
            or has("sulfite")
            or has("selenite_selenate")
            or has("selenide_or_elemental_se")
            or has("sulfide_or_elemental_s")
            or has("halide_or_elemental_halogen")
            or has("hydroxide")
        )

        overly_elemental = int(n_precursors > 0 and elemental_ratio >= 0.5 and n_precursors >= 2)

        rows.append({
            "route_template_primary": primary,
            "route_template_secondary": secondary,
            "route_template_confidence": conf,
            "route_template_n_types": n_template_types,

            "route_has_oxide_template": has("oxide_or_oxygen_source"),
            "route_has_nitrate_template": has("nitrate"),
            "route_has_carbonate_template": has("carbonate"),
            "route_has_phosphate_template": has("phosphate"),
            "route_has_sulfate_template": has("sulfate"),
            "route_has_sulfite_template": has("sulfite"),
            "route_has_selenide_template": has("selenide_or_elemental_se"),
            "route_has_selenite_selenate_template": has("selenite_selenate"),
            "route_has_sulfide_template": has("sulfide_or_elemental_s"),
            "route_has_halide_template": has("halide_or_elemental_halogen"),
            "route_has_hydroxide_template": has("hydroxide"),
            "route_has_elemental_template": has("elemental"),
            "route_has_organic_template": has("organic_like"),
            "route_has_hydrate_template": has("hydrate"),
            "route_has_ammonium_template": has("ammonium"),

            "route_template_is_common_solid_state": common_solid_state,
            "route_template_is_overly_elemental": overly_elemental,
            "route_template_elemental_ratio": elemental_ratio,
            "route_template_matches_target_anion": template_matches_target(primary, target_elements),

            "route_template_type_signature": ";".join(sorted(type_counts.keys())),
        })

    feat = pd.DataFrame(rows)
    return pd.concat([out.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--output_md", default="")
    ap.add_argument("--summary_json", default="")
    ap.add_argument("--precursor_col", default="precursor_set")
    ap.add_argument("--target_elements_col", default="target_elements_v33")
    ap.add_argument("--top_n", type=int, default=30)
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    df = pd.read_csv(input_csv)
    out = add_template_features(df, args.precursor_col, args.target_elements_col)

    if args.top_n and args.top_n > 0:
        if "sample_id" in out.columns:
            out_to_write = out.groupby("sample_id", sort=False).head(args.top_n).reset_index(drop=True)
        else:
            out_to_write = out.head(args.top_n).copy()
    else:
        out_to_write = out

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_to_write.to_csv(output_csv, index=False)

    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        show_cols = [
            "precursor_set",
            "target_elements_v33",
            "route_template_primary",
            "route_template_secondary",
            "route_template_confidence",
            "route_template_matches_target_anion",
            "route_template_is_common_solid_state",
            "route_template_is_overly_elemental",
            "route_template_elemental_ratio",
            "route_template_type_signature",
        ]
        show_cols = [c for c in show_cols if c in out_to_write.columns]
        out_to_write[show_cols].to_markdown(md_path, index=False)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        primary_counts = out_to_write["route_template_primary"].fillna("").astype(str).value_counts().to_dict()
        summary = {
            "input_csv": str(input_csv.resolve()),
            "output_csv": str(output_csv.resolve()),
            "rows_input": int(len(df)),
            "rows_output": int(len(out_to_write)),
            "precursor_col": args.precursor_col,
            "target_elements_col": args.target_elements_col,
            "primary_template_counts": primary_counts,
            "common_solid_state_rate": float(out_to_write["route_template_is_common_solid_state"].mean()) if len(out_to_write) else 0.0,
            "overly_elemental_rate": float(out_to_write["route_template_is_overly_elemental"].mean()) if len(out_to_write) else 0.0,
            "target_anion_match_rate": float(out_to_write["route_template_matches_target_anion"].mean()) if len(out_to_write) else 0.0,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SAVE]", output_csv)
    if args.output_md:
        print("[SAVE]", args.output_md)
    if args.summary_json:
        print("[SAVE]", args.summary_json)


if __name__ == "__main__":
    main()
