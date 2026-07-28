#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthmind.chemistry.families import element_group_bucket  # noqa: E402


ELEMENT_PATTERN = re.compile(r"[A-Z][a-z]?")
TRANSITION_METALS = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
HALOGENS = {"F", "Cl", "Br", "I", "At"}
CHALCOGENS = {"O", "S", "Se", "Te", "Po"}
LANTHANOIDS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
ACTINOIDS = {"Ac", "Th", "Pa", "U", "Np", "Pu"}


@lru_cache(maxsize=None)
def group_for(element: str) -> str:
    return element_group_bucket(Element(element))


@lru_cache(maxsize=None)
def atomic_number(element: str) -> float:
    return float(Element(element).Z)


def json_set(value: Any) -> Set[str]:
    try:
        return {str(item) for item in json.loads(str(value))}
    except Exception:
        return set()


def substitute(text: str, mapping: Mapping[str, str]) -> str:
    return ELEMENT_PATTERN.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


def composition_features(formula: str, feature_cols: Sequence[str]) -> Dict[str, float]:
    try:
        composition = Composition(formula)
        values = {str(element): float(amount) for element, amount in composition.items()}
    except Exception:
        return {}
    total = sum(values.values())
    if total <= 0:
        return {}
    fractions = {element: amount / total for element, amount in values.items()}
    z_mean = sum(float(Element(element).Z) * fraction for element, fraction in fractions.items())
    z_var = sum((float(Element(element).Z) - z_mean) ** 2 * fraction for element, fraction in fractions.items())
    result = {
        "feat_n_elements_formula": float(len(values)),
        "feat_total_atoms_formula": float(total),
        "feat_stoich_entropy": float(-sum(value * math.log(value) for value in fractions.values() if value > 0)),
        "feat_z_mean": float(z_mean),
        "feat_z_std": float(math.sqrt(max(z_var, 0.0))),
        "feat_frac_tm": float(sum(fractions.get(element, 0.0) for element in TRANSITION_METALS)),
        "feat_frac_alkali": float(sum(fractions.get(element, 0.0) for element in ALKALI)),
        "feat_frac_alkaline": float(sum(fractions.get(element, 0.0) for element in ALKALINE)),
        "feat_frac_halogen": float(sum(fractions.get(element, 0.0) for element in HALOGENS)),
        "feat_frac_chalcogen": float(sum(fractions.get(element, 0.0) for element in CHALCOGENS)),
        "feat_frac_lanthanoid": float(sum(fractions.get(element, 0.0) for element in LANTHANOIDS)),
        "feat_frac_actinoid": float(sum(fractions.get(element, 0.0) for element in ACTINOIDS)),
    }
    for column in feature_cols:
        if column.startswith("feat_frac_el__"):
            result[column] = float(fractions.get(column.split("__", 1)[1], 0.0))
    return result


def substitute_raw_composition(
    raw: np.ndarray,
    mapping: Mapping[str, str],
    element_indices: Mapping[str, int],
    z_mean_index: int,
    z_std_index: int,
) -> np.ndarray:
    """Apply an isostoichiometric same-family symbol mapping without reparsing."""
    output = np.asarray(raw, dtype=np.float32).copy()
    original = {element: float(output[index]) for element, index in element_indices.items()}
    replaced = {element: 0.0 for element in element_indices}
    for element, fraction in original.items():
        replaced[mapping.get(element, element)] = replaced.get(mapping.get(element, element), 0.0) + fraction
    for element, index in element_indices.items():
        output[index] = replaced.get(element, 0.0)
    nonzero = [(element, fraction) for element, fraction in replaced.items() if fraction > 0]
    z_mean = sum(atomic_number(element) * fraction for element, fraction in nonzero)
    z_var = sum((atomic_number(element) - z_mean) ** 2 * fraction for element, fraction in nonzero)
    output[z_mean_index] = z_mean
    output[z_std_index] = math.sqrt(max(z_var, 0.0))
    return output


def replacement_pools(meta: pd.DataFrame) -> Dict[str, List[str]]:
    pools: Dict[str, Set[str]] = {}
    for value in meta["target_cation_elements"]:
        for element in json_set(value):
            group = group_for(element)
            pools.setdefault(group, set()).add(element)
    return {group: sorted(values) for group, values in pools.items()}


def sampled_mappings(
    elements: Sequence[str],
    pools: Mapping[str, Sequence[str]],
    count: int,
    rng: np.random.Generator,
) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    seen = set()
    by_group: Dict[str, List[str]] = {}
    for element in elements:
        by_group.setdefault(group_for(element), []).append(element)
    # Most single-family rows have only a handful of distinct valid mappings;
    # avoid spending hundreds of repeated draws after those are exhausted.
    attempts = max(10, count * 5)
    for _ in range(attempts):
        mapping: Dict[str, str] = {}
        valid = True
        for group, source_values in by_group.items():
            choices = list(pools.get(group, source_values))
            if len(choices) < len(source_values):
                valid = False
                break
            selected = rng.choice(choices, size=len(source_values), replace=False).tolist()
            mapping.update(dict(zip(sorted(source_values), selected)))
        key = tuple(sorted(mapping.items()))
        if not valid or key in seen or all(source == target for source, target in key):
            continue
        seen.add(key)
        output.append(mapping)
        if len(output) >= count:
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chemistry-valid same-family substitution augmentation for Stage2.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--augment_per_row", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = json.loads((input_dir / "feature_cols.json").read_text())
    precursor_names = json.loads((input_dir / "precursor_names.json").read_text())
    precursor_to_id = {str(value): index for index, value in enumerate(precursor_names)}
    action_to_id = json.loads((input_dir / "action_to_id.json").read_text())
    stop_id = int(action_to_id["<stop>"])
    scaler = json.loads((input_dir / "scaler.json").read_text())
    means = np.asarray(scaler["mean"], dtype=np.float32)
    stds = np.asarray(scaler["std"], dtype=np.float32)
    base_count = len(means)
    base_feature_cols = feature_cols[:base_count]
    element_indices = {
        column.split("__", 1)[1]: index
        for index, column in enumerate(base_feature_cols)
        if column.startswith("feat_frac_el__")
    }
    z_mean_index = base_feature_cols.index("feat_z_mean")
    z_std_index = base_feature_cols.index("feat_z_std")
    # Materialize the compressed archive once. Re-indexing an NpzFile inside
    # the row loop silently decompresses a whole array on every access.
    train = {
        key: value
        for key, value in np.load(input_dir / "train.npz", allow_pickle=True).items()
    }
    meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    meta_records = meta.to_dict(orient="records")
    pools = replacement_pools(meta)
    rng = np.random.default_rng(args.seed)

    augmented_raw = []
    augmented_x = []
    augmented_y = []
    augmented_actions = []
    augmented_masks = []
    augmented_lengths = []
    augmented_meta = []
    max_traj_len = int(train["traj_actions"].shape[1])
    accepted_by_group: Dict[str, int] = {}
    mapping_cache: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    attempted = 0
    for row in range(len(meta)):
        if row % 1000 == 0:
            print(
                json.dumps({"source_row": row, "accepted": len(augmented_meta), "attempted": attempted}),
                flush=True,
            )
        source_record = meta_records[row]
        elements = sorted(json_set(source_record.get("target_cation_elements", "[]")))
        if not elements:
            continue
        true_labels = tuple(np.flatnonzero(train["y_multi_hot"][row] > 0.5).tolist())
        source_names = [str(precursor_names[label]) for label in true_labels]
        element_key = tuple(elements)
        mappings = mapping_cache.get(element_key)
        if mappings is None:
            mappings = sampled_mappings(elements, pools, int(args.augment_per_row), rng)
            mapping_cache[element_key] = mappings
        for variant, mapping in enumerate(mappings):
            attempted += 1
            new_names = [substitute(name, mapping) for name in source_names]
            if any(name not in precursor_to_id for name in new_names):
                continue
            labels = tuple(sorted({precursor_to_id[name] for name in new_names}))
            if not labels or len(labels) + 1 > max_traj_len:
                continue
            new_formula = substitute(str(source_record.get("formula", "")), mapping)
            raw = substitute_raw_composition(
                np.asarray(train["x_raw"][row], dtype=np.float32),
                mapping,
                element_indices,
                z_mean_index,
                z_std_index,
            )
            scaled_base = (np.where(np.isfinite(raw[:base_count]), raw[:base_count], means) - means) / stds
            x = np.concatenate([scaled_base.astype(np.float32), np.asarray(train["x"][row, base_count:], dtype=np.float32)])
            y = np.zeros(len(precursor_names), dtype=np.float32)
            y[list(labels)] = 1.0
            actions = np.full(max_traj_len, stop_id, dtype=np.int64)
            mask = np.zeros(max_traj_len, dtype=np.float32)
            actions[: len(labels)] = np.asarray(labels, dtype=np.int64)
            actions[len(labels)] = stop_id
            mask[: len(labels) + 1] = 1.0
            record = dict(source_record)
            record["id"] = f"{record.get('id', row)}__family_aug_{variant}"
            record["formula"] = new_formula
            record["canonical_formula"] = new_formula
            new_elements = sorted({mapping.get(element, element) for element in elements})
            record["target_cation_elements"] = json.dumps(new_elements)
            record["augmentation_source_row"] = row
            record["augmentation_mapping"] = json.dumps(mapping, sort_keys=True)
            augmented_raw.append(raw)
            augmented_x.append(x)
            augmented_y.append(y)
            augmented_actions.append(actions)
            augmented_masks.append(mask)
            augmented_lengths.append(len(labels))
            augmented_meta.append(record)
            group = str(record.get("family_signature_primary", "UNK"))
            accepted_by_group[group] = accepted_by_group.get(group, 0) + 1

    arrays = {
        "x_raw": np.vstack([train["x_raw"], np.asarray(augmented_raw, dtype=np.float32)]),
        "x": np.vstack([train["x"], np.asarray(augmented_x, dtype=np.float32)]),
        "y_multi_hot": np.vstack([train["y_multi_hot"], np.asarray(augmented_y, dtype=np.float32)]),
        "traj_actions": np.vstack([train["traj_actions"], np.asarray(augmented_actions, dtype=np.int64)]),
        "traj_mask": np.vstack([train["traj_mask"], np.asarray(augmented_masks, dtype=np.float32)]),
        "set_len": np.concatenate([train["set_len"], np.asarray(augmented_lengths, dtype=train["set_len"].dtype)]),
    }
    np.savez_compressed(output_dir / "train.npz", **arrays)
    augmented_frame = pd.DataFrame(augmented_meta)
    pd.concat([meta, augmented_frame], ignore_index=True).to_csv(output_dir / "train_meta.csv", index=False)
    for filename in ("val.npz", "test.npz", "val_meta.csv", "test_meta.csv", "action_to_id.json", "action_vocab.json", "feature_cols.json", "label_cols.json", "precursor_names.json", "scaler.json", "split_manifest.json"):
        shutil.copy2(input_dir / filename, output_dir / filename)
    summary = json.loads((input_dir / "summary.json").read_text())
    summary["family_augmentation"] = {
        "source_rows": int(len(meta)),
        "attempted": int(attempted),
        "accepted": int(len(augmented_meta)),
        "train_rows_total": int(len(arrays["x"])),
        "augment_per_row": int(args.augment_per_row),
        "seed": int(args.seed),
        "accepted_by_family": dict(sorted(accepted_by_group.items())),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["family_augmentation"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
