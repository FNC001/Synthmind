#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition, Element


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthmind.chemistry.families import assign_cation_family, element_group_bucket  # noqa: E402


SetKey = Tuple[int, ...]
ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")


def key_from_row(row: np.ndarray) -> SetKey:
    return tuple(np.flatnonzero(row > 0.5).tolist())


def element_bucket(symbol: str) -> str:
    from pymatgen.core import Element

    return element_group_bucket(Element(symbol))


def mapping_variants(source_formula: str, target_formula: str, max_variants: int = 24) -> List[Dict[str, str]]:
    source = assign_cation_family(source_formula)
    target = assign_cation_family(target_formula)
    source_by_group: Dict[str, List[str]] = {}
    target_by_group: Dict[str, List[str]] = {}
    for symbol in source.target_cation_elements:
        source_by_group.setdefault(element_bucket(symbol), []).append(symbol)
    for symbol in target.target_cation_elements:
        target_by_group.setdefault(element_bucket(symbol), []).append(symbol)
    if set(source_by_group) != set(target_by_group):
        return []
    group_options: List[List[Dict[str, str]]] = []
    for group in sorted(source_by_group):
        source_elements = sorted(source_by_group[group])
        target_elements = sorted(target_by_group[group])
        if len(source_elements) != len(target_elements):
            return []
        options = []
        for permutation in itertools.permutations(target_elements):
            options.append(dict(zip(source_elements, permutation)))
        group_options.append(options)
    variants = []
    for combination in itertools.product(*group_options):
        merged: Dict[str, str] = {}
        for mapping in combination:
            merged.update(mapping)
        variants.append(merged)
        if len(variants) >= max_variants:
            break
    return variants


def all_group_mapping_variants(
    source_formula: str,
    target_formula: str,
    max_variants: int = 24,
) -> List[Dict[str, str]]:
    """Map cations and anions through periodic groups, ignoring stoichiometry."""
    try:
        source_elements = [str(value) for value in Composition(str(source_formula)).elements]
        target_elements = [str(value) for value in Composition(str(target_formula)).elements]
    except Exception:
        return []
    source_by_group: Dict[str, List[str]] = {}
    target_by_group: Dict[str, List[str]] = {}
    for symbol in source_elements:
        source_by_group.setdefault(element_bucket(symbol), []).append(symbol)
    for symbol in target_elements:
        target_by_group.setdefault(element_bucket(symbol), []).append(symbol)
    if set(source_by_group) != set(target_by_group):
        return []
    group_options: List[List[Dict[str, str]]] = []
    for group in sorted(source_by_group):
        source_values = sorted(source_by_group[group])
        target_values = sorted(target_by_group[group])
        if len(source_values) != len(target_values):
            return []
        group_options.append([
            dict(zip(source_values, permutation))
            for permutation in itertools.permutations(target_values)
        ])
    variants: List[Dict[str, str]] = []
    for combination in itertools.product(*group_options):
        merged: Dict[str, str] = {}
        for mapping in combination:
            merged.update(mapping)
        variants.append(merged)
        if len(variants) >= int(max_variants):
            break
    return variants


def substitute_formula(formula: str, mapping: Mapping[str, str]) -> str:
    return ELEMENT_PATTERN.sub(lambda match: mapping.get(match.group(0), match.group(0)), formula)


def cosine_neighbors(train_x: np.ndarray, query_x: np.ndarray, k: int, device: str) -> np.ndarray:
    resolved = torch.device(device if torch.cuda.is_available() else "cpu")
    train = torch.nn.functional.normalize(
        torch.from_numpy(np.nan_to_num(train_x).astype(np.float32)).to(resolved), dim=1
    )
    outputs = []
    for start in range(0, len(query_x), 256):
        query = torch.nn.functional.normalize(
            torch.from_numpy(np.nan_to_num(query_x[start : start + 256]).astype(np.float32)).to(resolved), dim=1
        )
        outputs.append(torch.topk(query @ train.T, k=min(k, len(train)), dim=1).indices.cpu().numpy())
    return np.vstack(outputs)


def periodic_group_features(formulas: Sequence[str], exact_scale: float = 0.0) -> np.ndarray:
    """Group-first composition features with an optional exact-element tie breaker."""
    output = np.zeros((len(formulas), 18 + 2 + 118), dtype=np.float32)
    for row, formula in enumerate(formulas):
        try:
            composition = Composition(str(formula))
        except Exception:
            continue
        total = max(float(composition.num_atoms), 1e-8)
        for symbol, amount in composition.get_el_amt_dict().items():
            element = Element(symbol)
            fraction = float(amount) / total
            if element.group is not None and 1 <= int(element.group) <= 18:
                output[row, int(element.group) - 1] += fraction
            output[row, 20 + int(element.Z) - 1] += float(exact_scale) * fraction
        base = 18
        output[row, base] = min(len(composition.elements), 10) / 10.0
        output[row, base + 1] = min(total, 40.0) / 40.0
    return output


def cosine_neighbors_by_family(
    train_x: np.ndarray,
    family_train: np.ndarray,
    query_x: np.ndarray,
    family_query: np.ndarray,
    k: int,
    device: str,
) -> List[np.ndarray]:
    """Retrieve within family before top-k truncation so analogs cannot be filtered out."""
    resolved = torch.device(device if torch.cuda.is_available() else "cpu")
    train_norm = torch.nn.functional.normalize(
        torch.from_numpy(np.nan_to_num(train_x).astype(np.float32)).to(resolved), dim=1
    )
    query_norm = torch.nn.functional.normalize(
        torch.from_numpy(np.nan_to_num(query_x).astype(np.float32)).to(resolved), dim=1
    )
    output: List[np.ndarray] = [np.zeros(0, dtype=np.int64) for _ in range(len(query_x))]
    family_train = np.asarray(family_train).astype(str)
    family_query = np.asarray(family_query).astype(str)
    for family in np.unique(family_query):
        train_indices = np.flatnonzero(family_train == family)
        query_indices = np.flatnonzero(family_query == family)
        if not len(train_indices):
            continue
        take = min(int(k), len(train_indices))
        train_tensor = train_norm[torch.from_numpy(train_indices).to(resolved)]
        for start in range(0, len(query_indices), 256):
            batch_indices = query_indices[start : start + 256]
            batch = query_norm[torch.from_numpy(batch_indices).to(resolved)]
            local = torch.topk(batch @ train_tensor.T, k=take, dim=1).indices.cpu().numpy()
            for query_index, local_indices in zip(batch_indices, local):
                output[int(query_index)] = train_indices[local_indices]
    return output


def topk_metrics(targets: Sequence[SetKey], candidates: Sequence[Sequence[SetKey]]) -> Dict[str, float]:
    result = {}
    for k in (1, 3, 5, 10, 20, 50, 100):
        result[f"exact_hit@{k}"] = float(
            np.mean([target in set(row[:k]) for target, row in zip(targets, candidates)])
        )
    result["mean_candidates"] = float(np.mean([len(row) for row in candidates]))
    result["nonempty_candidate_rate"] = float(np.mean([bool(row) for row in candidates]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate same-family element-substitution candidate generation.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", default="")
    parser.add_argument("--neighbors", type=int, default=2000)
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument(
        "--neighbor_features", choices=("raw", "periodic_group"), default="raw",
        help="Use periodic-group composition features for same-family analog retrieval.",
    )
    parser.add_argument(
        "--periodic_exact_scale", type=float, default=0.0,
        help="Optional exact-element tie breaker added after periodic-group normalization.",
    )
    parser.add_argument(
        "--substitute_all_groups", action="store_true",
        help="Also substitute same-group anions (for example O/S and Cl/Br).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    split_names = tuple(dict.fromkeys(("train", args.split)))
    packs = {split: np.load(input_dir / f"{split}.npz", allow_pickle=True) for split in split_names}
    meta = {split: pd.read_csv(input_dir / f"{split}_meta.csv", low_memory=False) for split in split_names}
    precursor_names = [str(value) for value in json.loads((input_dir / "precursor_names.json").read_text())]
    precursor_to_id = {value: index for index, value in enumerate(precursor_names)}
    train_keys = [key_from_row(row) for row in packs["train"]["y_multi_hot"]]
    query_keys = [key_from_row(row) for row in packs[args.split]["y_multi_hot"]]
    train_formulas = meta["train"]["formula"].fillna("").astype(str).tolist()
    query_formulas = meta[args.split]["formula"].fillna("").astype(str).tolist()
    train_families = meta["train"]["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    query_families = meta[args.split]["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    if args.neighbor_features == "periodic_group":
        neighbors = cosine_neighbors_by_family(
            periodic_group_features(train_formulas, args.periodic_exact_scale), train_families,
            periodic_group_features(query_formulas, args.periodic_exact_scale), query_families,
            int(args.neighbors), str(args.device),
        )
    else:
        neighbors = cosine_neighbors(
            np.asarray(packs["train"]["x"], dtype=np.float32),
            np.asarray(packs[args.split]["x"], dtype=np.float32),
            int(args.neighbors),
            str(args.device),
        )
    all_candidates: List[List[SetKey]] = []
    rows_for_output = []
    mapping_cache: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for query_index, (query_formula, query_family, neighbor_row) in enumerate(
        zip(query_formulas, query_families, neighbors)
    ):
        candidates: List[SetKey] = []
        seen = set()
        ordered_neighbors = [
            int(index) for index in neighbor_row if train_families[int(index)] == query_family
        ]
        for train_index in ordered_neighbors:
            cache_key = (train_formulas[train_index], query_formula)
            mappings = mapping_cache.get(cache_key)
            if mappings is None:
                try:
                    cation_mappings = mapping_variants(*cache_key)
                    if args.substitute_all_groups:
                        all_mappings = all_group_mapping_variants(*cache_key)
                        mappings = []
                        seen_mappings = set()
                        for mapping in [*all_mappings, *cation_mappings]:
                            key = tuple(sorted(mapping.items()))
                            if key not in seen_mappings:
                                seen_mappings.add(key)
                                mappings.append(mapping)
                    else:
                        mappings = cation_mappings
                except Exception:
                    mappings = []
                mapping_cache[cache_key] = mappings
            if not mappings:
                continue
            source_names = [precursor_names[label] for label in train_keys[train_index]]
            for mapping in mappings:
                substituted = [substitute_formula(name, mapping) for name in source_names]
                if any(name not in precursor_to_id for name in substituted):
                    continue
                candidate = tuple(sorted({precursor_to_id[name] for name in substituted}))
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
                if len(candidates) >= int(args.candidate_limit):
                    break
            if len(candidates) >= int(args.candidate_limit):
                break
        all_candidates.append(candidates)
        if args.output_candidates_jsonl:
            rows_for_output.append(
                {
                    "row_index": query_index,
                    "sample_id": str(meta[args.split].iloc[query_index].get("id", query_index)),
                    "formula": query_formula,
                    "candidate_label_ids": [list(value) for value in candidates],
                }
            )
    report = {
        "protocol": f"{args.split}_formula_disjoint_same_family_element_substitution",
        "split": args.split,
        "n_rows": len(query_keys),
        "neighbors": int(args.neighbors),
        "candidate_limit": int(args.candidate_limit),
        "neighbor_features": str(args.neighbor_features),
        "periodic_exact_scale": float(args.periodic_exact_scale),
        "substitute_all_groups": bool(args.substitute_all_groups),
        "metrics": topk_metrics(query_keys, all_candidates),
    }
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_candidates_jsonl:
        output_path = Path(args.output_candidates_jsonl).resolve()
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows_for_output:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
