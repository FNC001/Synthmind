#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Basic periodic table helpers
# -----------------------------
ELEMENTS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og"
]
Z_TABLE = {el: i + 1 for i, el in enumerate(ELEMENTS)}

ALKALI = {"Li", "Na", "K", "Rb", "Cs", "Fr"}
ALKALINE = {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}
HALOGENS = {"F", "Cl", "Br", "I", "At", "Ts"}
CHALCOGENS = {"O", "S", "Se", "Te", "Po", "Lv"}
P_BLOCK = {"B", "C", "N", "O", "F", "Ne", "Al", "Si", "P", "S", "Cl", "Ar",
           "Ga", "Ge", "As", "Se", "Br", "Kr", "In", "Sn", "Sb", "Te", "I", "Xe",
           "Tl", "Pb", "Bi", "Po", "At", "Rn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"}
LANTHANOIDS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
ACTINOIDS = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn"
}

CRYSTAL_SYSTEMS = [
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic"
]


# -----------------------------
# IO helpers
# -----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(maybe_rel_path: Optional[str], base_dir: str) -> Optional[Path]:
    if not maybe_rel_path:
        return None

    p = Path(maybe_rel_path)
    candidates = [
        p,
        Path(base_dir) / maybe_rel_path,
        Path(base_dir) / "raw" / maybe_rel_path,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def get_nested(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


# -----------------------------
# Formula features
# -----------------------------
def parse_formula_simple(formula: Optional[str]) -> Dict[str, float]:
    if not formula:
        return {}
    s = str(formula).strip().replace(" ", "")
    if not s:
        return {}

    out: Dict[str, float] = {}
    i = 0
    n = len(s)
    while i < n:
        if not s[i].isalpha() or not s[i].isupper():
            return {}
        el = s[i]
        i += 1
        if i < n and s[i].islower():
            el += s[i]
            i += 1

        num = []
        while i < n and (s[i].isdigit() or s[i] == "."):
            num.append(s[i])
            i += 1

        amt = float("".join(num)) if num else 1.0
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


# -----------------------------
# Summary JSON features
# -----------------------------
def load_summary_features(summary_path: Optional[Path]) -> Dict[str, Any]:
    feat: Dict[str, Any] = {}
    for cs in CRYSTAL_SYSTEMS:
        feat[f"feat_crystal_system__{cs}"] = 0.0

    default_keys = [
        "feat_density",
        "feat_volume",
        "feat_nsites",
        "feat_nelements",
        "feat_band_gap",
        "feat_energy_above_hull",
        "feat_spacegroup_number",
        "feat_lattice_a",
        "feat_lattice_b",
        "feat_lattice_c",
        "feat_lattice_alpha",
        "feat_lattice_beta",
        "feat_lattice_gamma",
    ]
    for k in default_keys:
        feat[k] = 0.0

    if summary_path is None or not summary_path.exists():
        feat["feat_has_summary"] = 0.0
        return feat

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        feat["feat_has_summary"] = 0.0
        return feat

    feat["feat_has_summary"] = 1.0

    density = get_nested(summary, "density", None)
    if density is None:
        density = get_nested(summary, "density_atomic", None)
    volume = get_nested(summary, "volume", None)
    if volume is None:
        volume = get_nested(summary, "structure.lattice.volume", None)

    feat["feat_density"] = float(safe_float(density) or 0.0)
    feat["feat_volume"] = float(safe_float(volume) or 0.0)
    feat["feat_nsites"] = float(safe_float(get_nested(summary, "nsites", None)) or 0.0)
    feat["feat_nelements"] = float(safe_float(get_nested(summary, "nelements", None)) or 0.0)
    feat["feat_band_gap"] = float(safe_float(get_nested(summary, "band_gap", None)) or 0.0)

    eah = get_nested(summary, "energy_above_hull", None)
    if eah is None:
        eah = get_nested(summary, "e_above_hull", None)
    feat["feat_energy_above_hull"] = float(safe_float(eah) or 0.0)

    spg_num = get_nested(summary, "symmetry.number", None)
    if spg_num is None:
        spg_num = get_nested(summary, "spacegroup_number", None)
    feat["feat_spacegroup_number"] = float(safe_float(spg_num) or 0.0)

    feat["feat_lattice_a"] = float(safe_float(get_nested(summary, "structure.lattice.a", None) or get_nested(summary, "lattice.a", None)) or 0.0)
    feat["feat_lattice_b"] = float(safe_float(get_nested(summary, "structure.lattice.b", None) or get_nested(summary, "lattice.b", None)) or 0.0)
    feat["feat_lattice_c"] = float(safe_float(get_nested(summary, "structure.lattice.c", None) or get_nested(summary, "lattice.c", None)) or 0.0)
    feat["feat_lattice_alpha"] = float(safe_float(get_nested(summary, "structure.lattice.alpha", None) or get_nested(summary, "lattice.alpha", None)) or 0.0)
    feat["feat_lattice_beta"] = float(safe_float(get_nested(summary, "structure.lattice.beta", None) or get_nested(summary, "lattice.beta", None)) or 0.0)
    feat["feat_lattice_gamma"] = float(safe_float(get_nested(summary, "structure.lattice.gamma", None) or get_nested(summary, "lattice.gamma", None)) or 0.0)

    crystal_system = get_nested(summary, "symmetry.crystal_system", None)
    if crystal_system is None:
        crystal_system = get_nested(summary, "crystal_system", None)
    if crystal_system:
        cs = str(crystal_system).strip().lower()
        if cs in CRYSTAL_SYSTEMS:
            feat[f"feat_crystal_system__{cs}"] = 1.0

    return feat


# -----------------------------
# POSCAR geometry features
# -----------------------------
def lattice_lengths_angles(latt: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    a_vec, b_vec, c_vec = latt[0], latt[1], latt[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    def angle(u: np.ndarray, v: np.ndarray) -> float:
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu <= 0 or nv <= 0:
            return 0.0
        cosang = np.dot(u, v) / (nu * nv)
        cosang = max(-1.0, min(1.0, float(cosang)))
        return float(np.degrees(np.arccos(cosang)))

    alpha = angle(b_vec, c_vec)
    beta = angle(a_vec, c_vec)
    gamma = angle(a_vec, b_vec)
    return float(a), float(b), float(c), alpha, beta, gamma


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

    # VASP5 format
    try:
        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        idx = 7
    except Exception:
        return None

    selective = False
    if idx < len(lines) and lines[idx].strip().lower().startswith("s"):
        selective = True
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

    return {
        "species": species,
        "counts": counts,
        "lattice": lattice,
        "frac": frac,
        "cart": cart,
        "nsites": nsites,
    }


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

    pair_dists: List[float] = []
    nn_dists: List[float] = []
    coord3: List[int] = []
    coord4: List[int] = []
    coord5: List[int] = []

    for i in range(n):
        dlist = []
        c3 = 0
        c4 = 0
        c5 = 0
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

    for name, vals in [
        ("feat_coord_3A", coord3),
        ("feat_coord_4A", coord4),
        ("feat_coord_5A", coord5),
    ]:
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


# -----------------------------
# Row featurization
# -----------------------------
def gather_element_vocab(rows: List[Dict[str, Any]]) -> List[str]:
    elems = set()
    for row in rows:
        comp = parse_formula_simple(row.get("formula"))
        elems.update(comp.keys())
    return sorted(x for x in elems if x in Z_TABLE)


def build_stage2_vocab(train_rows: List[Dict[str, Any]]) -> List[str]:
    vocab = set()
    for row in train_rows:
        for p in row.get("main_precursors", []) or []:
            if p:
                vocab.add(str(p))
    return sorted(vocab)


def build_stage3_class_vocab(train_rows: List[Dict[str, Any]], key: str) -> List[str]:
    vocab = set()
    for row in train_rows:
        val = row.get(key)
        if val is not None and str(val).strip():
            vocab.add(str(val).strip())
    return sorted(vocab)


def featurize_row(
    row: Dict[str, Any],
    base_dir: str,
    element_vocab: List[str],
    use_poscar_geom: bool,
    poscar_max_sites: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # metadata
    for k in [
        "id", "synth_uid", "source_dataset", "record_index", "material_id",
        "formula", "mp_formula", "synth_formula", "parent_formula",
        "doi", "split_group", "synthesis_type", "reaction_string",
    ]:
        out[k] = row.get(k)

    out["target_main_precursors"] = json.dumps(row.get("main_precursors", []) or [], ensure_ascii=False)
    out["target_aux_precursors"] = json.dumps(row.get("aux_precursors", []) or [], ensure_ascii=False)
    out["target_temperature_c"] = row.get("temperature_c")
    out["target_time_h"] = row.get("time_h")
    out["target_atmosphere"] = row.get("atmosphere")
    out["target_solvent"] = row.get("solvent")

    # formula/composition features
    out.update(composition_features(row.get("formula"), element_vocab))

    # summary features
    summary_path = resolve_path(row.get("summary_json_path"), base_dir)
    out.update(load_summary_features(summary_path))

    # poscar features
    if use_poscar_geom:
        poscar_path = resolve_path(row.get("poscar_path"), base_dir)
        out.update(load_poscar_features(poscar_path, max_sites=poscar_max_sites))
    else:
        out.update(load_poscar_features(None, max_sites=poscar_max_sites))

    return out


def raw_to_stage2_ml(df_raw: pd.DataFrame, precursor_vocab: List[str]) -> pd.DataFrame:
    df = df_raw.copy()
    prec_lists = df["target_main_precursors"].apply(lambda x: json.loads(x) if isinstance(x, str) and x else [])
    label_cols = [f"label_prec__{p}" for p in precursor_vocab]
    label_mat = np.zeros((len(df), len(precursor_vocab)), dtype=np.int8)
    prec_to_idx = {p: i for i, p in enumerate(precursor_vocab)}

    for row_idx, arr in enumerate(prec_lists):
        for precursor in arr or []:
            col_idx = prec_to_idx.get(str(precursor))
            if col_idx is not None:
                label_mat[row_idx, col_idx] = 1

    labels = pd.DataFrame(label_mat, columns=label_cols, index=df.index)
    return pd.concat([df, labels], axis=1)


def raw_to_stage3_ml(
    df_raw: pd.DataFrame,
    atmosphere_vocab: List[str],
    solvent_vocab: List[str],
    synthesis_type_vocab: List[str],
) -> pd.DataFrame:
    df = df_raw.copy()
    label_frames = []
    for source_col, prefix, vocab in [
        ("target_atmosphere", "label_atm__", atmosphere_vocab),
        ("target_solvent", "label_solv__", solvent_vocab),
        ("synthesis_type", "label_synth__", synthesis_type_vocab),
    ]:
        values = df[source_col].astype(str).to_numpy()
        cols = [f"{prefix}{v}" for v in vocab]
        mat = np.zeros((len(df), len(vocab)), dtype=np.int8)
        value_to_idx = {str(v): i for i, v in enumerate(vocab)}
        for row_idx, value in enumerate(values):
            col_idx = value_to_idx.get(str(value))
            if col_idx is not None:
                mat[row_idx, col_idx] = 1
        label_frames.append(pd.DataFrame(mat, columns=cols, index=df.index))

    return pd.concat([df] + label_frames, axis=1)


def count_columns(df: pd.DataFrame) -> int:
    return len(df.columns)


def sanitize_filename(prefix: str) -> str:
    return prefix.replace(".jsonl", "")


def process_stage2_split(
    split_name: str,
    rows: List[Dict[str, Any]],
    base_dir: str,
    out_dir: Path,
    element_vocab: List[str],
    precursor_vocab: List[str],
    use_poscar_geom: bool,
    poscar_max_sites: int,
) -> Dict[str, Any]:
    feat_rows = [
        featurize_row(
            row=row,
            base_dir=base_dir,
            element_vocab=element_vocab,
            use_poscar_geom=use_poscar_geom,
            poscar_max_sites=poscar_max_sites,
        )
        for row in rows
    ]

    df_raw = pd.DataFrame(feat_rows)
    df_ml = raw_to_stage2_ml(df_raw, precursor_vocab)

    raw_path = out_dir / f"{split_name}_raw.csv"
    ml_path = out_dir / f"{split_name}_ml.csv"
    df_raw.to_csv(raw_path, index=False)
    df_ml.to_csv(ml_path, index=False)

    return {
        "n_rows": len(df_raw),
        "n_features_raw": count_columns(df_raw),
        "n_features_ml": count_columns(df_ml),
    }


def process_stage3_split(
    split_name: str,
    rows: List[Dict[str, Any]],
    base_dir: str,
    out_dir: Path,
    element_vocab: List[str],
    atmosphere_vocab: List[str],
    solvent_vocab: List[str],
    synthesis_type_vocab: List[str],
    use_poscar_geom: bool,
    poscar_max_sites: int,
) -> Dict[str, Any]:
    feat_rows = [
        featurize_row(
            row=row,
            base_dir=base_dir,
            element_vocab=element_vocab,
            use_poscar_geom=use_poscar_geom,
            poscar_max_sites=poscar_max_sites,
        )
        for row in rows
    ]

    df_raw = pd.DataFrame(feat_rows)
    df_ml = raw_to_stage3_ml(df_raw, atmosphere_vocab, solvent_vocab, synthesis_type_vocab)

    raw_path = out_dir / f"{split_name}_raw.csv"
    ml_path = out_dir / f"{split_name}_ml.csv"
    df_raw.to_csv(raw_path, index=False)
    df_ml.to_csv(ml_path, index=False)

    return {
        "n_rows": len(df_raw),
        "n_features_raw": count_columns(df_raw),
        "n_features_ml": count_columns(df_ml),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structure descriptor features from split JSONL files.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/Users/wyc/SynPred/data",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/splits/structdesc_splits",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/features/structdesc_features",
    )
    parser.add_argument("--use_poscar_geom", action="store_true")
    parser.add_argument("--poscar_max_sites", type=int, default=256)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "meta")

    # load all splits
    stage2_splits = {}
    stage3_splits = {}
    for split_name in ["train", "val", "test", "gold_train_holdout"]:
        stage2_splits[split_name] = read_jsonl(str(input_dir / f"stage2_{split_name}.jsonl"))
        stage3_splits[split_name] = read_jsonl(str(input_dir / f"stage3_{split_name}.jsonl"))

    # vocab from train only
    stage2_train_rows = stage2_splits["train"]
    stage3_train_rows = stage3_splits["train"]

    # element vocab from all rows actually used in both tasks
    all_rows = []
    for rows in stage2_splits.values():
        all_rows.extend(rows)
    for rows in stage3_splits.values():
        all_rows.extend(rows)
    element_vocab = gather_element_vocab(all_rows)

    precursor_vocab = build_stage2_vocab(stage2_train_rows)
    atmosphere_vocab = build_stage3_class_vocab(stage3_train_rows, "atmosphere")
    solvent_vocab = build_stage3_class_vocab(stage3_train_rows, "solvent")
    synthesis_type_vocab = build_stage3_class_vocab(stage3_train_rows, "synthesis_type")

    write_json(output_dir / "meta" / "element_vocab.json", element_vocab)
    write_json(output_dir / "meta" / "precursor_vocab.json", precursor_vocab)
    write_json(output_dir / "meta" / "atmosphere_vocab.json", atmosphere_vocab)
    write_json(output_dir / "meta" / "solvent_vocab.json", solvent_vocab)
    write_json(output_dir / "meta" / "synthesis_type_vocab.json", synthesis_type_vocab)

    summary = {
        "config": {
            "base_dir": args.base_dir,
            "input_dir": args.input_dir,
            "output_dir": args.output_dir,
            "use_poscar_geom": args.use_poscar_geom,
            "poscar_max_sites": args.poscar_max_sites,
        },
        "vocab_sizes": {
            "element_vocab": len(element_vocab),
            "precursor_vocab": len(precursor_vocab),
            "atmosphere_vocab": len(atmosphere_vocab),
            "solvent_vocab": len(solvent_vocab),
            "synthesis_type_vocab": len(synthesis_type_vocab),
        },
        "files": {},
    }

    # stage2
    for split_name, rows in stage2_splits.items():
        tag = f"stage2_{split_name}"
        summary["files"][tag] = process_stage2_split(
            split_name=tag,
            rows=rows,
            base_dir=args.base_dir,
            out_dir=output_dir,
            element_vocab=element_vocab,
            precursor_vocab=precursor_vocab,
            use_poscar_geom=args.use_poscar_geom,
            poscar_max_sites=args.poscar_max_sites,
        )

    # stage3
    for split_name, rows in stage3_splits.items():
        tag = f"stage3_{split_name}"
        summary["files"][tag] = process_stage3_split(
            split_name=tag,
            rows=rows,
            base_dir=args.base_dir,
            out_dir=output_dir,
            element_vocab=element_vocab,
            atmosphere_vocab=atmosphere_vocab,
            solvent_vocab=solvent_vocab,
            synthesis_type_vocab=synthesis_type_vocab,
            use_poscar_geom=args.use_poscar_geom,
            poscar_max_sites=args.poscar_max_sites,
        )

    write_json(output_dir / "meta" / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
