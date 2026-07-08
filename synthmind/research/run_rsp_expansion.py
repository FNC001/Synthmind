from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/autorun/rsp_vnext_003_candidate_expansion")
VARIANTS = [
    "rsp_v5_baseline",
    "base_plus_family_expansion",
    "base_plus_rare_recovery",
    "base_plus_family_rare_chemistry",
]
METRIC_KS = [1, 10, 50, 200, 500]


@dataclass(frozen=True)
class ExpansionConfig:
    output_dir: Path
    train_candidates: Path
    split_candidates: Path
    train_family_labels: Path
    split_family_predictions: Path
    precursor_ontology: Path | None
    method_templates: Path | None
    budgets: tuple[int, ...]
    preserve_base_top_grid: tuple[int, ...]
    family_per_element_grid: tuple[int, ...]
    rare_per_element_grid: tuple[int, ...]
    max_generated_per_sample_grid: tuple[int, ...]
    default_preserve_base_top: int
    family_top_n: int
    rare_count_threshold: int
    max_set_size: int
    max_cartesian_products: int
    require_element_coverage: bool
    max_missing_elements: int
    max_extra_source_elements: int
    max_exact1_drop: float
    bootstrap_iterations: int
    ci: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSP vnext candidate expansion on a fixed validation/test split.")
    parser.add_argument("--config", default="research/configs/rsp_vnext_003.yaml")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--budgets", default="50,200,500")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json_or_yaml_subset(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if path.suffix.lower() == ".json":
        raise ValueError(f"Invalid JSON config: {path}")
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} did not contain a mapping")
        return payload
    except ModuleNotFoundError:
        return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Small YAML subset reader for this config when PyYAML is unavailable."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_at_indent: dict[int, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            value = parse_scalar(line[2:])
            if not isinstance(parent, list):
                raise ValueError("simple YAML parser only supports list values after a key")
            parent.append(value)
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = parse_scalar(value)
        else:
            next_obj: dict[str, Any] | list[Any] = {}
            parent[key] = next_obj
            last_key_at_indent[indent] = key
            stack.append((indent, next_obj))
            continue
        if value == "":
            last_key_at_indent[indent] = key

    # Reparse path lists because the tiny parser above cannot infer list
    # containers from future lines. This fallback is rarely used on AutoDL.
    return root


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(x.strip()) for x in inner.split(",")]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def first_existing(paths: Iterable[str | Path], *, required: bool = True) -> Path | None:
    for raw in paths:
        path = Path(raw)
        if path.exists() and not path.name.startswith("._"):
            return path
    if required:
        raise FileNotFoundError("None of the configured paths exist: " + ", ".join(str(p) for p in paths))
    return None


def as_tuple_int(values: Iterable[Any]) -> tuple[int, ...]:
    return tuple(int(x) for x in values)


def load_config(config_path: str | Path, split: str, budgets_arg: str, seed: int) -> ExpansionConfig:
    raw = read_json_or_yaml_subset(Path(config_path))
    paths = raw.get("paths", {})
    expansion = raw.get("expansion", {})
    gate = raw.get("validation_gate", {})
    constraints = expansion.get("chemistry_constraints", {})
    split_key = "validation_candidates" if split == "validation" else "test_candidates"
    family_key = "validation_family_predictions" if split == "validation" else "test_family_predictions"
    budgets = tuple(int(x) for x in budgets_arg.split(",") if x.strip())
    if not budgets:
        budgets = as_tuple_int(expansion.get("budgets", [50, 200, 500]))
    return ExpansionConfig(
        output_dir=Path(raw.get("output_dir", DEFAULT_OUTPUT_DIR)),
        train_candidates=first_existing(paths.get("train_candidates", [])),  # type: ignore[arg-type]
        split_candidates=first_existing(paths.get(split_key, [])),  # type: ignore[arg-type]
        train_family_labels=first_existing(paths.get("train_family_labels", [])),  # type: ignore[arg-type]
        split_family_predictions=first_existing(paths.get(family_key, [])),  # type: ignore[arg-type]
        precursor_ontology=first_existing(paths.get("precursor_ontology", []), required=False),
        method_templates=first_existing(paths.get("method_templates", []), required=False),
        budgets=budgets,
        preserve_base_top_grid=as_tuple_int(expansion.get("grid", {}).get("preserve_base_top", [20])),
        family_per_element_grid=as_tuple_int(expansion.get("grid", {}).get("family_per_element", [2])),
        rare_per_element_grid=as_tuple_int(expansion.get("grid", {}).get("rare_per_element", [1])),
        max_generated_per_sample_grid=as_tuple_int(expansion.get("grid", {}).get("max_generated_per_sample", [40])),
        default_preserve_base_top=int(expansion.get("default_preserve_base_top", 20)),
        family_top_n=int(expansion.get("family_top_n", 3)),
        rare_count_threshold=int(expansion.get("rare_count_threshold", 3)),
        max_set_size=int(expansion.get("max_set_size", 8)),
        max_cartesian_products=int(expansion.get("max_cartesian_products", 80)),
        require_element_coverage=bool(expansion.get("require_element_coverage", True)),
        max_missing_elements=int(constraints.get("max_missing_elements", 0)),
        max_extra_source_elements=int(constraints.get("max_extra_source_elements", 0)),
        max_exact1_drop=float(gate.get("max_exact1_drop_pp", 0.5)) / 100.0,
        bootstrap_iterations=int(gate.get("bootstrap_iterations", 1000)),
        ci=float(gate.get("ci", 0.95)),
        seed=seed,
    )


def json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(x).strip() for x in payload if str(x).strip()]
    except Exception:
        pass
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [text]


def canonical_label(label: Any) -> str:
    text = str(label).strip()
    text = text.replace("·", ".")
    text = re.sub(r"\s+", "", text)
    return text


def canonical_set_key(labels: Iterable[Any]) -> str:
    vals = sorted({canonical_label(x) for x in labels if canonical_label(x)})
    return json.dumps(vals, ensure_ascii=False, separators=(",", ":"))


def canonical_set_display(labels: Iterable[Any]) -> str:
    vals = sorted({canonical_label(x) for x in labels if canonical_label(x)})
    return json.dumps(vals, ensure_ascii=False)


def formula_elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", str(formula)))


def normalize_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        if "id" in out.columns:
            out["sample_id"] = out["id"].astype(str)
        elif "sample_index" in out.columns:
            out["sample_id"] = out["sample_index"].astype(str)
        else:
            raise ValueError("candidate table needs sample_id, id, or sample_index")
    if "candidate_set" not in out.columns:
        out["candidate_set"] = out.get("pred_precursors", out.get("precursor_set", "")).astype(str)
    if "rank" not in out.columns:
        if "precursor_rank" in out.columns:
            out["rank"] = pd.to_numeric(out["precursor_rank"], errors="coerce").fillna(999999).astype(int)
        else:
            out["rank"] = out.groupby("sample_id", sort=False).cumcount() + 1
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(999999).astype(int)
    if "total_score_v5" not in out.columns:
        for col in ["calibrated_score", "precursor_score", "base_score", "original_v4_score"]:
            if col in out.columns:
                out["total_score_v5"] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
                break
        else:
            out["total_score_v5"] = -out["rank"]
    out["candidate_source"] = out.get("candidate_source", "base").fillna("base").astype(str)
    out["source_group"] = out["candidate_source"].where(out["candidate_source"].isin(["family", "rare"]), "base")
    out["candidate_key"] = out["candidate_set"].fillna("").astype(str).str.replace(" ", "", regex=False)
    if "exact" in out.columns:
        out["label_exact"] = out["exact"].fillna(False).astype(str).str.lower().isin({"true", "1", "1.0"}).astype(int)
    elif "precursor_exact_if_eval" in out.columns:
        out["label_exact"] = out["precursor_exact_if_eval"].fillna(False).astype(str).str.lower().isin({"true", "1", "1.0"}).astype(int)
    else:
        out["true_key"] = out["true_precursors"].map(lambda x: canonical_set_key(json_list(x)))
        out["candidate_key"] = out["candidate_set"].map(lambda x: canonical_set_key(json_list(x)))
        out["label_exact"] = (out["candidate_key"] == out["true_key"]).astype(int)
    if "jaccard" in out.columns:
        out["jaccard_label"] = pd.to_numeric(out["jaccard"], errors="coerce").fillna(0.0)
    elif "precursor_jaccard_if_eval" in out.columns:
        out["jaccard_label"] = pd.to_numeric(out["precursor_jaccard_if_eval"], errors="coerce").fillna(0.0)
    else:
        if "true_key" not in out.columns:
            out["true_key"] = out["true_precursors"].map(lambda x: canonical_set_key(json_list(x)))
            out["candidate_key"] = out["candidate_set"].map(lambda x: canonical_set_key(json_list(x)))
        out["jaccard_label"] = out.apply(lambda r: jaccard_from_keys(r["true_key"], r["candidate_key"]), axis=1)
    out["is_rare_reference"] = False
    out["is_oov_reference"] = False
    return out


def infer_target_elements(row: pd.Series) -> set[str]:
    if "target_elements" in row and pd.notna(row["target_elements"]):
        vals = set(json_list(row["target_elements"]))
        if vals:
            return vals
    if "formula" in row:
        elems = formula_elements(str(row["formula"]))
        return {e for e in elems if e not in {"O", "H", "C", "N"}}
    return set()


def jaccard_from_keys(a: str, b: str) -> float:
    try:
        sa = set(json.loads(a))
        sb = set(json.loads(b))
    except Exception:
        return 0.0
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def read_precursor_elements(ontology_path: Path | None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if ontology_path is None or not ontology_path.exists():
        return out
    df = pd.read_csv(ontology_path)
    for _, row in df.iterrows():
        label = canonical_label(row.get("canonical_precursor", ""))
        elems = set(json_list(row.get("target_source_elements"))) or set(json_list(row.get("elements")))
        if label:
            out[label] = elems
    return out


@dataclass
class PrecursorLibrary:
    by_method_element_family: dict[tuple[str, str, str], Counter[str]]
    by_element_family: dict[tuple[str, str], Counter[str]]
    by_element: dict[str, Counter[str]]
    global_counts: Counter[str]
    label_family: dict[str, str]
    label_elements: dict[str, set[str]]


def build_precursor_library(train_family: pd.DataFrame, ontology_path: Path | None) -> PrecursorLibrary:
    by_method_element_family: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    by_element_family: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_element: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    label_family: dict[str, str] = {}
    label_elements = read_precursor_elements(ontology_path)

    for _, row in train_family.iterrows():
        method = str(row.get("reaction_method", "")).strip()
        element = str(row.get("target_element", "")).strip()
        families = json_list(row.get("element_family_labels"))
        precursors = json_list(row.get("element_source_precursors"))
        if not element or not precursors:
            continue
        family = families[0] if families else "unknown"
        for raw_label in precursors:
            label = canonical_label(raw_label)
            if not label:
                continue
            by_method_element_family[(method, element, family)][label] += 1
            by_element_family[(element, family)][label] += 1
            by_element[element][label] += 1
            global_counts[label] += 1
            label_family.setdefault(label, family)
            label_elements.setdefault(label, {element})
    return PrecursorLibrary(
        by_method_element_family=dict(by_method_element_family),
        by_element_family=dict(by_element_family),
        by_element=dict(by_element),
        global_counts=global_counts,
        label_family=label_family,
        label_elements=label_elements,
    )


def family_prob_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob_family__")]


def build_prediction_map(pred_df: pd.DataFrame, top_n: int) -> dict[str, dict[str, list[tuple[str, float]]]]:
    pred = pred_df.copy()
    if "sample_id" not in pred.columns:
        pred["sample_id"] = pred.get("id", pred.get("sample_index")).astype(str)
    prob_cols = family_prob_columns(pred)
    mapping: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    for _, row in pred.iterrows():
        sample_id = str(row["sample_id"])
        element = str(row.get("target_element", "")).strip()
        scored = []
        for col in prob_cols:
            family = col.replace("prob_family__", "")
            try:
                score = float(row[col])
            except Exception:
                score = 0.0
            scored.append((family, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        if element:
            mapping[sample_id][element] = scored[:top_n]
    return {k: dict(v) for k, v in mapping.items()}


def choose_precursors(
    lib: PrecursorLibrary,
    method: str,
    element: str,
    families: list[tuple[str, float]],
    per_element: int,
    rare: bool,
    rare_count_threshold: int,
) -> list[tuple[str, float, str]]:
    choices: list[tuple[str, float, str]] = []
    for family, family_score in families:
        counters = [
            lib.by_method_element_family.get((method, element, family), Counter()),
            lib.by_element_family.get((element, family), Counter()),
        ]
        merged: Counter[str] = Counter()
        for c in counters:
            merged.update(c)
        ranked = sorted(merged.items(), key=lambda kv: (kv[1] if rare else -kv[1], kv[0]))
        for label, count in ranked:
            if rare and count > rare_count_threshold:
                continue
            if (not rare) and count <= 0:
                continue
            score = float(family_score) + (0.01 / max(count, 1) if rare else 0.01 * math.log1p(count))
            choices.append((label, score, family))
            if len(choices) >= per_element:
                return choices
    if rare:
        ranked = sorted(lib.by_element.get(element, Counter()).items(), key=lambda kv: (kv[1], kv[0]))
        for label, count in ranked:
            if count <= rare_count_threshold:
                choices.append((label, 0.01 / max(count, 1), lib.label_family.get(label, "unknown")))
                if len(choices) >= per_element:
                    return choices
    return choices[:per_element]


def product_limited(lists: list[list[tuple[str, float, str]]], limit: int) -> Iterable[tuple[tuple[str, float, str], ...]]:
    yielded = 0
    for combo in itertools.product(*lists):
        yield combo
        yielded += 1
        if yielded >= limit:
            return


def chemistry_ok(labels: list[str], target_elements: set[str], lib: PrecursorLibrary, cfg: ExpansionConfig) -> tuple[bool, int, int, set[str]]:
    source_elems: set[str] = set()
    for label in labels:
        elems = lib.label_elements.get(label, set())
        if not elems:
            elems = formula_elements(label)
        source_elems.update(elems)
    relevant = {e for e in source_elems if e in target_elements}
    missing = len(target_elements - relevant)
    extra = len(relevant - target_elements)
    ok = True
    if cfg.require_element_coverage:
        ok = ok and missing <= cfg.max_missing_elements
    ok = ok and extra <= cfg.max_extra_source_elements
    return ok, missing, extra, relevant


def generate_expansions(
    samples: pd.DataFrame,
    pred_map: dict[str, dict[str, list[tuple[str, float]]]],
    lib: PrecursorLibrary,
    cfg: ExpansionConfig,
    family_per_element: int,
    rare_per_element: int,
    max_generated_per_sample: int,
    include_family: bool,
    include_rare: bool,
    apply_chemistry: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sample in samples.iterrows():
        sample_id = str(sample["sample_id"])
        method = str(sample.get("reaction_method", "")).strip()
        target_elements = set(sample["target_elements"])
        if not target_elements:
            continue
        family_choices: list[list[tuple[str, float, str]]] = []
        rare_choices: list[list[tuple[str, float, str]]] = []
        sample_preds = pred_map.get(sample_id, {})
        for element in sorted(target_elements):
            families = sample_preds.get(element, [])
            if not families:
                families = [("oxide", 0.1), ("carbonate", 0.05), ("nitrate", 0.03)]
            if include_family:
                choices = choose_precursors(
                    lib,
                    method,
                    element,
                    families,
                    family_per_element,
                    rare=False,
                    rare_count_threshold=cfg.rare_count_threshold,
                )
                if choices:
                    family_choices.append(choices)
            if include_rare:
                choices = choose_precursors(
                    lib,
                    method,
                    element,
                    families,
                    rare_per_element,
                    rare=True,
                    rare_count_threshold=cfg.rare_count_threshold,
                )
                if choices:
                    rare_choices.append(choices)
        generated = 0
        pools: list[tuple[str, list[list[tuple[str, float, str]]]]] = []
        if include_family and len(family_choices) == len(target_elements):
            pools.append(("family", family_choices))
        if include_rare and len(rare_choices) == len(target_elements):
            pools.append(("rare", rare_choices))
        if include_family and include_rare and len(target_elements) > 1 and family_choices and rare_choices:
            mixed: list[list[tuple[str, float, str]]] = []
            for idx, element in enumerate(sorted(target_elements)):
                if idx == 0 and idx < len(rare_choices):
                    mixed.append(rare_choices[idx])
                elif idx < len(family_choices):
                    mixed.append(family_choices[idx])
            if len(mixed) == len(target_elements):
                pools.append(("family+rare", mixed))
        for source, choice_lists in pools:
            for combo in product_limited(choice_lists, cfg.max_cartesian_products):
                labels = [x[0] for x in combo]
                if len(labels) > cfg.max_set_size:
                    continue
                ok, missing, extra, covered = chemistry_ok(labels, target_elements, lib, cfg)
                if apply_chemistry and not ok:
                    continue
                score = float(sum(x[1] for x in combo) / max(len(combo), 1))
                candidate_key = canonical_set_key(labels)
                true_key = str(sample["true_key"])
                rows.append(
                    {
                        "sample_id": sample_id,
                        "sample_index": sample.get("sample_index", np.nan),
                        "id": sample.get("id", sample_id),
                        "formula": sample.get("formula", ""),
                        "reaction_method": method,
                        "true_precursors": sample.get("true_precursors", ""),
                        "pred_precursors": canonical_set_display(labels),
                        "candidate_set": canonical_set_display(labels),
                        "candidate_source": source,
                        "source_group": "rare" if "rare" in source else "family",
                        "rank": 10**9,
                        "total_score_v5": score,
                        "family_score": score,
                        "element_coverage": 1.0 if missing == 0 else 0.0,
                        "missing_element_count": float(missing),
                        "extra_element_count": float(extra),
                        "candidate_size": float(len(labels)),
                        "target_elements": target_elements,
                        "candidate_elements": covered,
                        "true_key": true_key,
                        "candidate_key": candidate_key,
                        "label_exact": int(candidate_key == true_key),
                        "jaccard_label": jaccard_from_keys(true_key, candidate_key),
                        "is_rare_reference": bool(sample["is_rare_reference"]),
                        "is_oov_reference": bool(sample["is_oov_reference"]),
                    }
                )
                generated += 1
                if generated >= max_generated_per_sample:
                    break
            if generated >= max_generated_per_sample:
                break
    if not rows:
        return pd.DataFrame(columns=list(samples.columns))
    return pd.DataFrame(rows)


def sample_frame(base: pd.DataFrame, lib: PrecursorLibrary) -> pd.DataFrame:
    keep_cols = [
        "sample_id",
        "sample_index",
        "id",
        "formula",
        "reaction_method",
        "true_precursors",
        "true_key",
        "target_elements",
    ]
    samples = base.sort_values(["sample_id", "rank"]).drop_duplicates("sample_id", keep="first")
    samples = samples[[c for c in keep_cols if c in samples.columns]].copy()
    samples["true_key"] = samples["true_precursors"].map(lambda x: canonical_set_key(json_list(x)))
    samples["target_elements"] = samples.apply(lambda r: infer_target_elements(r), axis=1)
    counts = lib.global_counts
    train_labels = set(counts)
    rare_flags = []
    oov_flags = []
    for vals in samples["true_precursors"].map(json_list):
        canon = [canonical_label(x) for x in vals]
        rare_flags.append(any(0 < counts.get(x, 0) <= 3 for x in canon))
        oov_flags.append(any(counts.get(x, 0) == 0 for x in canon))
    samples["is_rare_reference"] = rare_flags
    samples["is_oov_reference"] = oov_flags
    return samples


def rank_variant(base: pd.DataFrame, expansion: pd.DataFrame, preserve_base_top: int, budgets: Iterable[int]) -> pd.DataFrame:
    max_budget = max(budgets)
    base_work = base.copy()
    base_work["source_group"] = "base"
    exp = expansion.copy()
    if not exp.empty:
        exp["_priority"] = exp["source_group"].map({"family": 0, "rare": 0}).fillna(0)
        exp = exp.sort_values(
            ["sample_id", "_priority", "total_score_v5", "candidate_key"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        exp["_exp_order"] = exp.groupby("sample_id", sort=False).cumcount() + 1
        exp["_rank_order"] = preserve_base_top + exp["_exp_order"]
    else:
        exp["_rank_order"] = []
    base_work["_rank_order"] = np.where(
        base_work["rank"] <= preserve_base_top,
        base_work["rank"],
        base_work["rank"] + 100000,
    )
    ranked = pd.concat([base_work, exp.drop(columns=["_priority", "_exp_order"], errors="ignore")], ignore_index=True, sort=False)
    ranked = ranked.sort_values(["sample_id", "_rank_order", "candidate_key"], kind="mergesort")
    ranked = ranked.drop_duplicates(["sample_id", "candidate_key"], keep="first")
    ranked["rsp_rank_expanded"] = ranked.groupby("sample_id", sort=False).cumcount() + 1
    return ranked[ranked["rsp_rank_expanded"] <= max_budget].drop(columns=["_rank_order"], errors="ignore").copy()


def per_sample_hits(ranked: pd.DataFrame, budget: int) -> pd.DataFrame:
    sub = ranked[ranked["rsp_rank_expanded"] <= budget]
    if sub.empty:
        return pd.DataFrame(columns=["sample_id", "exact", "best_jaccard", "family_hit", "rare_hit"])
    rows = sub.groupby("sample_id", sort=False).agg(
        exact=("label_exact", "max"),
        best_jaccard=("jaccard_label", "max"),
    )
    exact_rows = sub[sub["label_exact"] > 0].copy()
    if exact_rows.empty:
        rows["family_hit"] = 0
        rows["rare_hit"] = 0
        return rows.reset_index()
    exact_rows["_family_hit"] = exact_rows["source_group"].astype(str).str.contains("family").astype(int)
    exact_rows["_rare_hit"] = exact_rows["source_group"].astype(str).str.contains("rare").astype(int)
    hits = exact_rows.groupby("sample_id", sort=False).agg(family_hit=("_family_hit", "max"), rare_hit=("_rare_hit", "max"))
    rows = rows.join(hits, how="left").fillna({"family_hit": 0, "rare_hit": 0})
    rows["family_hit"] = rows["family_hit"].astype(int)
    rows["rare_hit"] = rows["rare_hit"].astype(int)
    return rows.reset_index()


def metric_dict(ranked: pd.DataFrame, budgets: Iterable[int]) -> dict[str, float]:
    out: dict[str, float] = {
        "n_samples": float(ranked["sample_id"].nunique()),
        "n_candidates": float(len(ranked)),
    }
    for k in METRIC_KS:
        if k > max(budgets):
            continue
        ps = per_sample_hits(ranked, k)
        out[f"exact@{k}"] = float(ps["exact"].mean()) if len(ps) else 0.0
        out[f"skeleton_oracle@{k}"] = out[f"exact@{k}"]
        if k in {1, 50}:
            out[f"best_jaccard@{k}"] = float(ps["best_jaccard"].mean()) if len(ps) else 0.0
    return out


def subset_metrics(ranked: pd.DataFrame, flag_col: str, budgets: Iterable[int]) -> dict[str, float]:
    samples = ranked[["sample_id", flag_col]].drop_duplicates("sample_id")
    subset_ids = set(samples.loc[samples[flag_col].astype(bool), "sample_id"].astype(str))
    if not subset_ids:
        return {"n_samples": 0.0}
    sub = ranked[ranked["sample_id"].astype(str).isin(subset_ids)]
    out = metric_dict(sub, budgets)
    out["n_samples"] = float(len(subset_ids))
    return out


def source_ablation_rows(variant_name: str, ranked: pd.DataFrame, baseline_ranked: pd.DataFrame, budgets: Iterable[int]) -> list[dict[str, Any]]:
    rows = []
    for budget in budgets:
        base_ps = per_sample_hits(baseline_ranked, budget).set_index("sample_id")
        var_ps = per_sample_hits(ranked, budget).set_index("sample_id")
        joined = var_ps.join(base_ps[["exact"]].rename(columns={"exact": "base_exact"}), how="left").fillna(0)
        new = joined[(joined["exact"] > 0) & (joined["base_exact"] <= 0)]
        rows.append(
            {
                "variant": variant_name,
                "budget": budget,
                "new_exact_hits": int(len(new)),
                "new_exact_hits_from_family": int(new["family_hit"].sum()) if "family_hit" in new else 0,
                "new_exact_hits_from_rare": int(new["rare_hit"].sum()) if "rare_hit" in new else 0,
            }
        )
    return rows


def bootstrap_ci(baseline_ranked: pd.DataFrame, variant_ranked: pd.DataFrame, budgets: Iterable[int], iterations: int, ci: float, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    out: dict[str, Any] = {}
    for budget in budgets:
        base = per_sample_hits(baseline_ranked, budget).set_index("sample_id")
        var = per_sample_hits(variant_ranked, budget).set_index("sample_id")
        ids = sorted(set(base.index) | set(var.index))
        b = base.reindex(ids).fillna(0)
        v = var.reindex(ids).fillna(0)
        diff_exact = (v["exact"].values - b["exact"].values).astype(float)
        diff_jacc = (v["best_jaccard"].values - b["best_jaccard"].values).astype(float)
        exact_samples = []
        jacc_samples = []
        n = len(ids)
        for _ in range(iterations):
            idx = [rng.randrange(n) for _ in range(n)]
            exact_samples.append(float(diff_exact[idx].mean()))
            jacc_samples.append(float(diff_jacc[idx].mean()))
        alpha = (1.0 - ci) / 2.0
        out[f"exact@{budget}"] = {
            "delta": float(diff_exact.mean()),
            "ci_low": float(np.quantile(exact_samples, alpha)),
            "ci_high": float(np.quantile(exact_samples, 1.0 - alpha)),
        }
        out[f"best_jaccard@{budget}"] = {
            "delta": float(diff_jacc.mean()),
            "ci_low": float(np.quantile(jacc_samples, alpha)),
            "ci_high": float(np.quantile(jacc_samples, 1.0 - alpha)),
        }
    return out


def select_params(
    base: pd.DataFrame,
    samples: pd.DataFrame,
    pred_map: dict[str, dict[str, list[tuple[str, float]]]],
    lib: PrecursorLibrary,
    cfg: ExpansionConfig,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, dict[str, float]]]:
    print("[rsp_expansion] ranking baseline", flush=True)
    baseline_ranked = rank_variant(base, pd.DataFrame(), preserve_base_top=max(cfg.budgets), budgets=cfg.budgets)
    best_score = (-10**9, -10**9, -10**9)
    best_params: dict[str, Any] = {}
    best_ranked: dict[str, pd.DataFrame] = {"rsp_v5_baseline": baseline_ranked}
    best_metrics: dict[str, dict[str, float]] = {"rsp_v5_baseline": metric_dict(baseline_ranked, cfg.budgets)}
    for preserve, fam_n, rare_n, max_gen in itertools.product(
        cfg.preserve_base_top_grid,
        cfg.family_per_element_grid,
        cfg.rare_per_element_grid,
        cfg.max_generated_per_sample_grid,
    ):
        print(
            "[rsp_expansion] generating expansions "
            f"preserve={preserve} family_per_element={fam_n} rare_per_element={rare_n} max_gen={max_gen}",
            flush=True,
        )
        family_exp = generate_expansions(samples, pred_map, lib, cfg, fam_n, rare_n, max_gen, True, False, False)
        print(f"[rsp_expansion] family candidates={len(family_exp)}", flush=True)
        rare_exp = generate_expansions(samples, pred_map, lib, cfg, fam_n, rare_n, max_gen, False, True, False)
        print(f"[rsp_expansion] rare candidates={len(rare_exp)}", flush=True)
        both_exp = generate_expansions(samples, pred_map, lib, cfg, fam_n, rare_n, max_gen, True, True, True)
        print(f"[rsp_expansion] family+rare chemistry candidates={len(both_exp)}", flush=True)
        print("[rsp_expansion] ranking four variants", flush=True)
        ranked = {
            "rsp_v5_baseline": baseline_ranked,
            "base_plus_family_expansion": rank_variant(base, family_exp, preserve, cfg.budgets),
            "base_plus_rare_recovery": rank_variant(base, rare_exp, preserve, cfg.budgets),
            "base_plus_family_rare_chemistry": rank_variant(base, both_exp, preserve, cfg.budgets),
        }
        metrics = {name: metric_dict(df, cfg.budgets) for name, df in ranked.items()}
        print("[rsp_expansion] variant metrics computed", flush=True)
        primary = metrics["base_plus_family_rare_chemistry"]
        base_m = metrics["rsp_v5_baseline"]
        oracle_delta = max(primary.get(f"skeleton_oracle@{k}", 0.0) - base_m.get(f"skeleton_oracle@{k}", 0.0) for k in cfg.budgets)
        exact1_delta = primary.get("exact@1", 0.0) - base_m.get("exact@1", 0.0)
        j50_delta = primary.get("best_jaccard@50", 0.0) - base_m.get("best_jaccard@50", 0.0)
        score = (oracle_delta, exact1_delta, j50_delta)
        if score > best_score:
            best_score = score
            best_params = {
                "preserve_base_top": preserve,
                "family_per_element": fam_n,
                "rare_per_element": rare_n,
                "max_generated_per_sample": max_gen,
            }
            best_ranked = ranked
            best_metrics = metrics
    return best_params, best_ranked, best_metrics


def evaluate_fixed_params(
    base: pd.DataFrame,
    samples: pd.DataFrame,
    pred_map: dict[str, dict[str, list[tuple[str, float]]]],
    lib: PrecursorLibrary,
    cfg: ExpansionConfig,
    params: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, float]], dict[str, int]]:
    preserve = int(params["preserve_base_top"])
    fam_n = int(params["family_per_element"])
    rare_n = int(params["rare_per_element"])
    max_gen = int(params["max_generated_per_sample"])
    print(
        "[rsp_expansion] evaluating fixed params "
        f"preserve={preserve} family_per_element={fam_n} rare_per_element={rare_n} max_gen={max_gen}",
        flush=True,
    )
    baseline_ranked = rank_variant(base, pd.DataFrame(), preserve_base_top=max(cfg.budgets), budgets=cfg.budgets)
    family_exp = generate_expansions(samples, pred_map, lib, cfg, fam_n, rare_n, max_gen, True, False, False)
    print(f"[rsp_expansion] family candidates={len(family_exp)}", flush=True)
    rare_exp = generate_expansions(samples, pred_map, lib, cfg, fam_n, rare_n, max_gen, False, True, False)
    print(f"[rsp_expansion] rare candidates={len(rare_exp)}", flush=True)
    both_exp = generate_expansions(samples, pred_map, lib, cfg, fam_n, rare_n, max_gen, True, True, True)
    print(f"[rsp_expansion] family+rare chemistry candidates={len(both_exp)}", flush=True)
    ranked = {
        "rsp_v5_baseline": baseline_ranked,
        "base_plus_family_expansion": rank_variant(base, family_exp, preserve, cfg.budgets),
        "base_plus_rare_recovery": rank_variant(base, rare_exp, preserve, cfg.budgets),
        "base_plus_family_rare_chemistry": rank_variant(base, both_exp, preserve, cfg.budgets),
    }
    metrics = {name: metric_dict(df, cfg.budgets) for name, df in ranked.items()}
    counts = {
        "family_candidates": int(len(family_exp)),
        "rare_candidates": int(len(rare_exp)),
        "family_rare_chemistry_candidates": int(len(both_exp)),
    }
    return ranked, metrics, counts


def write_candidate_table(path: Path, df: pd.DataFrame) -> dict[str, Any]:
    serial = df.copy()
    for col in ["target_elements", "candidate_elements"]:
        if col in serial.columns:
            serial[col] = serial[col].map(lambda x: json.dumps(sorted(x), ensure_ascii=False) if isinstance(x, set) else str(x))
    try:
        serial.to_parquet(path, index=False)
        fmt = "parquet"
    except Exception as exc:
        fallback = path.with_suffix(path.suffix + ".csv")
        serial.to_csv(fallback, index=False)
        path.write_text(
            "Parquet engine unavailable; see CSV fallback: " + fallback.name + "\n" + repr(exc) + "\n",
            encoding="utf-8",
        )
        fmt = "csv_fallback"
    return {"path": str(path), "format": fmt, "rows": int(len(serial)), "columns": list(serial.columns)}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_manifest(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        rows.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
                "description": path.name,
                "primary_result": path.name in {"metrics.json", "RSP_EXPANSION_REPORT.md"},
                "diagnostic_result": path.suffix in {".csv", ".json"},
                "reproducible": True,
            }
        )
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# RSP vnext 003 Candidate Expansion Report",
        "",
        f"Created at: {result['created_at']}",
        f"Split: {result['split']}",
        f"Selector update: {result['selector_update_status']}",
        f"Validation gate: {result['validation_gate']['status']}",
        "",
        "## Selected Parameters",
        "",
        "```json",
        json.dumps(result["selected_params"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
    ]
    for name, metrics in result["variants"].items():
        lines.append(f"### {name}")
        lines.append("")
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float)):
                lines.append(f"- `{key}`: {value:.6f}")
        lines.append("")
    lines.extend(
        [
            "## Gate Decision",
            "",
            f"- Passed: `{result['validation_gate']['passed']}`",
            f"- Mode: `{result['validation_gate']['mode']}`",
            f"- Reason: {result['validation_gate']['reason']}",
            "",
            "## Test Policy",
            "",
            "Test evaluation is not run unless validation passes. The default inference selector remains unchanged.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_gate(result: dict[str, Any], budgets: Iterable[int], max_exact1_drop: float) -> dict[str, Any]:
    base = result["variants"]["rsp_v5_baseline"]
    primary = result["variants"]["base_plus_family_rare_chemistry"]
    exact1_delta = primary.get("exact@1", 0.0) - base.get("exact@1", 0.0)
    oracle_deltas = {str(k): primary.get(f"skeleton_oracle@{k}", 0.0) - base.get(f"skeleton_oracle@{k}", 0.0) for k in budgets}
    oracle_pass = any(v > 0 for v in oracle_deltas.values())
    exact_pass = exact1_delta >= -max_exact1_drop
    passed = bool(oracle_pass and exact_pass)
    mode = "standard"
    if exact1_delta < 0 and oracle_pass:
        mode = "coverage_mode"
    reason = f"max oracle delta={max(oracle_deltas.values()):.6f}; exact@1 delta={exact1_delta:.6f}"
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "mode": mode if passed else "not_passed",
        "reason": reason,
        "exact1_delta": exact1_delta,
        "oracle_deltas": oracle_deltas,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, args.split, args.budgets, args.seed)
    outdir = cfg.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    validation_result: dict[str, Any] | None = None
    if args.split == "test":
        validation_path = outdir / "metrics.json"
        if not validation_path.exists():
            raise SystemExit("Refusing test: validation metrics.json does not exist")
        validation_result = json.loads(validation_path.read_text(encoding="utf-8"))
        gate = validation_result.get("validation_gate", {})
        if not gate.get("passed"):
            raise SystemExit("Refusing test: validation gate did not pass")

    print(f"[rsp_expansion] loading split candidates: {cfg.split_candidates}", flush=True)
    base = normalize_candidates(pd.read_csv(cfg.split_candidates))
    print(f"[rsp_expansion] loaded base candidates rows={len(base)} samples={base['sample_id'].nunique()}", flush=True)
    print(f"[rsp_expansion] loading train family labels: {cfg.train_family_labels}", flush=True)
    train_family = pd.read_csv(cfg.train_family_labels)
    print(f"[rsp_expansion] loading split family predictions: {cfg.split_family_predictions}", flush=True)
    pred_map = build_prediction_map(pd.read_csv(cfg.split_family_predictions), cfg.family_top_n)
    print("[rsp_expansion] building precursor library", flush=True)
    lib = build_precursor_library(train_family, cfg.precursor_ontology)
    print("[rsp_expansion] building sample frame", flush=True)
    samples = sample_frame(base, lib)
    sample_flags = samples.set_index("sample_id")[["is_rare_reference", "is_oov_reference"]]
    base = base.drop(columns=["is_rare_reference", "is_oov_reference"], errors="ignore").join(sample_flags, on="sample_id")
    base["is_rare_reference"] = base["is_rare_reference"].fillna(False).astype(bool)
    base["is_oov_reference"] = base["is_oov_reference"].fillna(False).astype(bool)

    if args.split == "test":
        assert validation_result is not None
        params = dict(validation_result["selected_params"])
        ranked_by_variant, metrics_by_variant, generated_counts = evaluate_fixed_params(base, samples, pred_map, lib, cfg, params)
    else:
        params, ranked_by_variant, metrics_by_variant = select_params(base, samples, pred_map, lib, cfg)
        generated_counts = {}
    print("[rsp_expansion] selected params and metrics ready", flush=True)
    baseline = ranked_by_variant["rsp_v5_baseline"]
    primary = ranked_by_variant["base_plus_family_rare_chemistry"]
    source_rows: list[dict[str, Any]] = []
    rare_rows: list[dict[str, Any]] = []
    for name, ranked in ranked_by_variant.items():
        source_rows.extend(source_ablation_rows(name, ranked, baseline, cfg.budgets))
        rare = subset_metrics(ranked, "is_rare_reference", cfg.budgets)
        oov = subset_metrics(ranked, "is_oov_reference", cfg.budgets)
        rare_rows.append({"variant": name, "subset": "rare_precursor", **rare})
        rare_rows.append({"variant": name, "subset": "oov_precursor", **oov})

    print("[rsp_expansion] running paired bootstrap", flush=True)
    bootstrap = bootstrap_ci(baseline, primary, cfg.budgets, cfg.bootstrap_iterations, cfg.ci, cfg.seed)
    result: dict[str, Any] = {
        "run_id": "rsp_vnext_003_candidate_expansion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "config": str(args.config),
        "input_artifacts": {
            "split_candidates": str(cfg.split_candidates),
            "train_family_labels": str(cfg.train_family_labels),
            "split_family_predictions": str(cfg.split_family_predictions),
            "precursor_ontology": str(cfg.precursor_ontology) if cfg.precursor_ontology else None,
        },
        "budgets": list(cfg.budgets),
        "selected_params": params,
        "generated_candidate_counts": generated_counts,
        "variants": metrics_by_variant,
        "subset_metrics": rare_rows,
        "bootstrap_ci": bootstrap,
        "selector_update_status": "unchanged",
        "test_evaluation": "not_applicable_for_validation" if args.split == "validation" else "run_once_after_validation_gate",
    }
    if args.split == "validation":
        result["validation_gate"] = evaluate_gate(result, cfg.budgets, cfg.max_exact1_drop)
    else:
        result["validation_gate"] = validation_result.get("validation_gate", {}) if validation_result else {}

    candidate_prefix = "val" if args.split == "validation" else "test"
    candidate_path = outdir / f"{candidate_prefix}_rsp_expanded_candidates.parquet"
    print("[rsp_expansion] writing candidate table and reports", flush=True)
    candidate_info = write_candidate_table(candidate_path, primary)
    result["candidate_table"] = candidate_info

    source_name = "candidate_source_ablation.csv" if args.split == "validation" else "test_candidate_source_ablation.csv"
    rare_name = "rare_precursor_analysis.csv" if args.split == "validation" else "test_rare_precursor_analysis.csv"
    metrics_name = "metrics.json" if args.split == "validation" else "test_metrics_if_gate_passed.json"
    report_name = "RSP_EXPANSION_REPORT.md" if args.split == "validation" else "TEST_RSP_EXPANSION_REPORT.md"
    pd.DataFrame(source_rows).to_csv(outdir / source_name, index=False)
    pd.DataFrame(rare_rows).to_csv(outdir / rare_name, index=False)
    write_json(outdir / metrics_name, result)
    write_report(outdir / report_name, result)
    manifest = artifact_manifest(
        [
            outdir / metrics_name,
            candidate_path,
            candidate_path.with_suffix(candidate_path.suffix + ".csv"),
            outdir / source_name,
            outdir / rare_name,
            outdir / report_name,
        ]
    )
    write_json(outdir / "artifact_manifest.json", manifest)

    print(json.dumps({"output_dir": str(outdir), "validation_gate": result["validation_gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
