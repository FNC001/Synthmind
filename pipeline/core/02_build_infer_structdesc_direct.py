#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Z_TABLE = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94,
}
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
}
ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
HALOGENS = {"F", "Cl", "Br", "I", "At"}
CHALCOGENS = {"O", "S", "Se", "Te", "Po"}
LANTHANOIDS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
ACTINOIDS = {"Ac", "Th", "Pa", "U", "Np", "Pu"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_formula_simple(formula: Optional[str]) -> Dict[str, float]:
    if not formula:
        return {}
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", str(formula))
    out: Dict[str, float] = {}
    for el, num in tokens:
        if el not in Z_TABLE:
            continue
        amt = float(num) if num else 1.0
        out[el] = out.get(el, 0.0) + amt
    return out


def composition_features(formula: Optional[str], element_vocab: List[str]) -> Dict[str, Any]:
    comp = parse_formula_simple(formula)
    feat: Dict[str, Any] = {}
    if not comp:
        feat["feat_n_elements_formula"] = 0.0
        feat["feat_total_atoms_formula"] = 0.0
        feat["feat_stoich_entropy"] = 0.0
        feat["feat_z_mean"] = 0.0
        feat["feat_z_std"] = 0.0
        feat["feat_frac_tm"] = 0.0
        feat["feat_frac_alkali"] = 0.0
        feat["feat_frac_alkaline"] = 0.0
        feat["feat_frac_halogen"] = 0.0
        feat["feat_frac_chalcogen"] = 0.0
        feat["feat_frac_lanthanoid"] = 0.0
        feat["feat_frac_actinoid"] = 0.0
        for el in element_vocab:
            feat[f"feat_frac_el__{el}"] = 0.0
        return feat

    total = sum(comp.values())
    fracs = {el: amt / total for el, amt in comp.items()}
    z_values = [Z_TABLE.get(el, 0) for el in comp.keys()]
    z_weights = [fracs[el] for el in comp.keys()]

    feat["feat_n_elements_formula"] = float(len(comp))
    feat["feat_total_atoms_formula"] = float(total)

    stoich_entropy = 0.0
    for p in fracs.values():
        if p > 0:
            stoich_entropy -= p * math.log(p)
    feat["feat_stoich_entropy"] = stoich_entropy

    z_mean = sum(z * w for z, w in zip(z_values, z_weights))
    z_var = sum(((z - z_mean) ** 2) * w for z, w in zip(z_values, z_weights))
    feat["feat_z_mean"] = float(z_mean)
    feat["feat_z_std"] = float(math.sqrt(max(z_var, 0.0)))

    feat["feat_frac_tm"] = sum(fracs.get(el, 0.0) for el in TRANSITION_METALS)
    feat["feat_frac_alkali"] = sum(fracs.get(el, 0.0) for el in ALKALI)
    feat["feat_frac_alkaline"] = sum(fracs.get(el, 0.0) for el in ALKALINE)
    feat["feat_frac_halogen"] = sum(fracs.get(el, 0.0) for el in HALOGENS)
    feat["feat_frac_chalcogen"] = sum(fracs.get(el, 0.0) for el in CHALCOGENS)
    feat["feat_frac_lanthanoid"] = sum(fracs.get(el, 0.0) for el in LANTHANOIDS)
    feat["feat_frac_actinoid"] = sum(fracs.get(el, 0.0) for el in ACTINOIDS)

    for el in element_vocab:
        feat[f"feat_frac_el__{el}"] = float(fracs.get(el, 0.0))
    return feat


def parse_poscar(poscar_path: Path) -> Optional[Dict[str, Any]]:
    try:
        lines = [x.rstrip("\n") for x in open(poscar_path, "r", encoding="utf-8").readlines()]
    except Exception:
        return None
    if len(lines) < 8:
        return None
    try:
        scale = float(lines[1].strip())
        lattice = np.array([
            [float(x) for x in lines[2].split()],
            [float(x) for x in lines[3].split()],
            [float(x) for x in lines[4].split()],
        ], dtype=float) * scale
    except Exception:
        return None
    try:
        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        idx = 7
    except Exception:
        return None
    if idx < len(lines) and lines[idx].strip().lower().startswith("s"):
        idx += 1
    if idx >= len(lines):
        return None
    coord_mode = lines[idx].strip().lower()
    idx += 1
    direct_mode = coord_mode.startswith("d")
    nsites = sum(counts)
    if idx + nsites > len(lines):
        return None
    coords = []
    for i in range(nsites):
        toks = lines[idx + i].split()
        if len(toks) < 3:
            return None
        coords.append([float(toks[0]), float(toks[1]), float(toks[2])])
    coords = np.array(coords, dtype=float)
    if direct_mode:
        frac = coords.copy()
        cart = frac @ lattice
    else:
        cart = coords.copy()
        try:
            inv_lattice = np.linalg.inv(lattice)
            frac = cart @ inv_lattice
        except Exception:
            return None
    return {"species": species, "counts": counts, "lattice": lattice, "frac": frac, "cart": cart, "nsites": nsites}


def lattice_lengths_angles(lattice: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    a_vec, b_vec, c_vec = lattice[0], lattice[1], lattice[2]
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))

    def ang(u: np.ndarray, v: np.ndarray) -> float:
        den = np.linalg.norm(u) * np.linalg.norm(v)
        if den <= 1e-12:
            return 0.0
        cosv = float(np.clip(np.dot(u, v) / den, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosv)))

    alpha = ang(b_vec, c_vec)
    beta = ang(a_vec, c_vec)
    gamma = ang(a_vec, b_vec)
    return a, b, c, alpha, beta, gamma


def min_image_distances(frac: np.ndarray, lattice: np.ndarray, max_sites: int = 256) -> Dict[str, Any]:
    feat = {
        "feat_poscar_has_geom": 0.0,
        "feat_poscar_nsites": 0.0,
        "feat_poscar_volume": 0.0,
        "feat_poscar_a": 0.0,
        "feat_poscar_b": 0.0,
        "feat_poscar_c": 0.0,
        "feat_poscar_alpha": 0.0,
        "feat_poscar_beta": 0.0,
        "feat_poscar_gamma": 0.0,
        "feat_pairdist_min": 0.0,
        "feat_pairdist_mean": 0.0,
        "feat_pairdist_std": 0.0,
        "feat_pairdist_q25": 0.0,
        "feat_pairdist_q50": 0.0,
        "feat_pairdist_q75": 0.0,
        "feat_nn_mean": 0.0,
        "feat_nn_std": 0.0,
        "feat_coord_3A_mean": 0.0,
        "feat_coord_3A_std": 0.0,
        "feat_coord_4A_mean": 0.0,
        "feat_coord_4A_std": 0.0,
        "feat_coord_5A_mean": 0.0,
        "feat_coord_5A_std": 0.0,
    }
    n = len(frac)
    if n == 0 or n > max_sites:
        return feat

    volume = abs(float(np.linalg.det(lattice)))
    a, b, c, alpha, beta, gamma = lattice_lengths_angles(lattice)
    pair_dists, nn_dists = [], []
    coord3, coord4, coord5 = [], [], []

    for i in range(n):
        dlist = []
        c3 = c4 = c5 = 0
        for j in range(n):
            if i == j:
                continue
            df = frac[j] - frac[i]
            df -= np.round(df)
            dc = df @ lattice
            dist = float(np.linalg.norm(dc))
            if dist <= 1e-8:
                continue
            dlist.append(dist)
            if dist <= 3.0:
                c3 += 1
            if dist <= 4.0:
                c4 += 1
            if dist <= 5.0:
                c5 += 1
            if j > i:
                pair_dists.append(dist)
        if dlist:
            nn_dists.append(min(dlist))
        coord3.append(c3)
        coord4.append(c4)
        coord5.append(c5)

    if pair_dists:
        arr = np.array(pair_dists, dtype=float)
        feat["feat_pairdist_min"] = float(np.min(arr))
        feat["feat_pairdist_mean"] = float(np.mean(arr))
        feat["feat_pairdist_std"] = float(np.std(arr))
        feat["feat_pairdist_q25"] = float(np.quantile(arr, 0.25))
        feat["feat_pairdist_q50"] = float(np.quantile(arr, 0.50))
        feat["feat_pairdist_q75"] = float(np.quantile(arr, 0.75))
    if nn_dists:
        arr = np.array(nn_dists, dtype=float)
        feat["feat_nn_mean"] = float(np.mean(arr))
        feat["feat_nn_std"] = float(np.std(arr))

    for name, vals in [("feat_coord_3A", coord3), ("feat_coord_4A", coord4), ("feat_coord_5A", coord5)]:
        arr = np.array(vals, dtype=float)
        if len(arr) > 0:
            feat[f"{name}_mean"] = float(np.mean(arr))
            feat[f"{name}_std"] = float(np.std(arr))

    feat["feat_poscar_has_geom"] = 1.0
    feat["feat_poscar_nsites"] = float(n)
    feat["feat_poscar_volume"] = volume
    feat["feat_poscar_a"] = a
    feat["feat_poscar_b"] = b
    feat["feat_poscar_c"] = c
    feat["feat_poscar_alpha"] = alpha
    feat["feat_poscar_beta"] = beta
    feat["feat_poscar_gamma"] = gamma
    return feat


def load_poscar_features(poscar_path: Optional[Path], max_sites: int = 256) -> Dict[str, Any]:
    defaults = min_image_distances(np.empty((0, 3)), np.eye(3), max_sites=max_sites)
    if poscar_path is None or not poscar_path.exists():
        return defaults
    parsed = parse_poscar(poscar_path)
    if parsed is None:
        return defaults
    return min_image_distances(parsed["frac"], parsed["lattice"], max_sites=max_sites)


def try_formula_from_poscar(path: Path) -> Optional[str]:
    parsed = parse_poscar(path)
    if parsed is None:
        return None
    species = parsed["species"]
    counts = parsed["counts"]
    if len(species) != len(counts):
        return None
    return "".join(f"{el}{cnt if cnt != 1 else ''}" for el, cnt in zip(species, counts))


def gather_element_vocab(rows: List[Dict[str, Any]]) -> List[str]:
    elems = set()
    for row in rows:
        formula = row.get("formula") or row.get("formula_guess")
        if not formula and row.get("poscar_path"):
            formula = try_formula_from_poscar(Path(row["poscar_path"]))
        comp = parse_formula_simple(formula)
        elems.update(comp.keys())
    return sorted(x for x in elems if x in Z_TABLE)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    cols = set()
    for row in rows:
        cols.update(row.keys())
    fieldnames = sorted(cols)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct inference-only structdesc builder from infer.jsonl")
    parser.add_argument("--infer_jsonl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--poscar_max_sites", type=int, default=256)
    args = parser.parse_args()

    infer_jsonl = Path(args.infer_jsonl).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    rows = read_jsonl(infer_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {infer_jsonl}")

    element_vocab = gather_element_vocab(rows)
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        poscar_path = Path(row["poscar_path"]).expanduser().resolve() if row.get("poscar_path") else None
        formula = row.get("formula") or row.get("formula_guess")
        if not formula and poscar_path is not None:
            formula = try_formula_from_poscar(poscar_path)

        feat: Dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "material_id": row.get("material_id", row.get("sample_id")),
            "formula": formula,
            "source_path": row.get("source_path"),
            "poscar_path": str(poscar_path) if poscar_path else None,
        }
        feat.update(composition_features(formula, element_vocab))
        feat.update(load_poscar_features(poscar_path, max_sites=int(args.poscar_max_sites)))
        out_rows.append(feat)

    write_csv(output_csv, out_rows)
    print(f"[DONE] Wrote {len(out_rows)} rows -> {output_csv}")


if __name__ == "__main__":
    main()
