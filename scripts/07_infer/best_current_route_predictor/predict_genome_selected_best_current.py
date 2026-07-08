#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pymatgen.core import Structure


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


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def parse_hidden_dims(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def chunked(seq: Sequence[Path], size: int) -> Iterable[Sequence[Path]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def lattice_lengths_angles(structure: Structure) -> Tuple[float, float, float, float, float, float]:
    lat = structure.lattice
    return float(lat.a), float(lat.b), float(lat.c), float(lat.alpha), float(lat.beta), float(lat.gamma)


def approx_crystal_system(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> str:
    eq = lambda x, y, tol=0.03: abs(x - y) <= tol * max(abs(x), abs(y), 1.0)
    ang = lambda x, y, tol=3.0: abs(x - y) <= tol
    all_90 = ang(alpha, 90) and ang(beta, 90) and ang(gamma, 90)
    if all_90 and eq(a, b) and eq(b, c):
        return "cubic"
    if all_90 and eq(a, b):
        return "tetragonal"
    if all_90:
        return "orthorhombic"
    if ang(alpha, 90) and ang(beta, 90) and ang(gamma, 120) and eq(a, b):
        return "hexagonal"
    if eq(a, b) and eq(b, c) and ang(alpha, beta) and ang(beta, gamma):
        return "trigonal"
    n90 = sum([ang(alpha, 90), ang(beta, 90), ang(gamma, 90)])
    if n90 >= 2:
        return "monoclinic"
    return "triclinic"


def empty_geom_features() -> Dict[str, float]:
    return {
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


def geom_features(structure: Structure, max_sites: int) -> Dict[str, float]:
    feat = empty_geom_features()
    n = len(structure)
    a, b, c, alpha, beta, gamma = lattice_lengths_angles(structure)
    feat.update({
        "feat_poscar_has_geom": 1.0 if 0 < n <= max_sites else 0.0,
        "feat_poscar_nsites": float(n),
        "feat_poscar_volume": float(structure.volume),
        "feat_poscar_a": a,
        "feat_poscar_b": b,
        "feat_poscar_c": c,
        "feat_poscar_alpha": alpha,
        "feat_poscar_beta": beta,
        "feat_poscar_gamma": gamma,
    })
    if n == 0 or n > max_sites:
        return feat
    frac = np.asarray(structure.frac_coords, dtype=float)
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    pair_dists: List[float] = []
    nn_dists: List[float] = []
    coord3: List[int] = []
    coord4: List[int] = []
    coord5: List[int] = []
    for i in range(n):
        dlist: List[float] = []
        c3 = c4 = c5 = 0
        for j in range(n):
            if i == j:
                continue
            df = frac[j] - frac[i]
            df -= np.round(df)
            dist = float(np.linalg.norm(df @ lattice))
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
        arr = np.asarray(pair_dists, dtype=float)
        feat["feat_pairdist_min"] = float(np.min(arr))
        feat["feat_pairdist_mean"] = float(np.mean(arr))
        feat["feat_pairdist_std"] = float(np.std(arr))
        feat["feat_pairdist_q25"] = float(np.quantile(arr, 0.25))
        feat["feat_pairdist_q50"] = float(np.quantile(arr, 0.50))
        feat["feat_pairdist_q75"] = float(np.quantile(arr, 0.75))
    if nn_dists:
        arr = np.asarray(nn_dists, dtype=float)
        feat["feat_nn_mean"] = float(np.mean(arr))
        feat["feat_nn_std"] = float(np.std(arr))
    for name, vals in [("feat_coord_3A", coord3), ("feat_coord_4A", coord4), ("feat_coord_5A", coord5)]:
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            feat[f"{name}_mean"] = float(np.mean(arr))
            feat[f"{name}_std"] = float(np.std(arr))
    return feat


def composition_feature_dict(structure: Structure, element_vocab: Sequence[str]) -> Tuple[Dict[str, float], str, List[str]]:
    comp = structure.composition.remove_charges()
    el_amt = {str(el): float(amt) for el, amt in comp.get_el_amt_dict().items() if str(el) in Z_TABLE}
    total = float(sum(el_amt.values()))
    formula = comp.reduced_formula
    feat: Dict[str, float] = {}
    if total <= 0:
        return {c: 0.0 for c in [
            "feat_n_elements_formula", "feat_total_atoms_formula", "feat_stoich_entropy", "feat_z_mean", "feat_z_std",
            "feat_frac_tm", "feat_frac_alkali", "feat_frac_alkaline", "feat_frac_halogen", "feat_frac_chalcogen",
            "feat_frac_lanthanoid", "feat_frac_actinoid",
        ]}, formula, []
    fracs = {el: amt / total for el, amt in el_amt.items()}
    z_mean = sum(Z_TABLE[el] * fracs[el] for el in fracs)
    z_var = sum(((Z_TABLE[el] - z_mean) ** 2) * fracs[el] for el in fracs)
    entropy = -sum(p * math.log(p) for p in fracs.values() if p > 0)
    feat.update({
        "feat_n_elements_formula": float(len(el_amt)),
        "feat_total_atoms_formula": total,
        "feat_stoich_entropy": float(entropy),
        "feat_z_mean": float(z_mean),
        "feat_z_std": float(math.sqrt(max(z_var, 0.0))),
        "feat_frac_tm": float(sum(fracs.get(el, 0.0) for el in TRANSITION_METALS)),
        "feat_frac_alkali": float(sum(fracs.get(el, 0.0) for el in ALKALI)),
        "feat_frac_alkaline": float(sum(fracs.get(el, 0.0) for el in ALKALINE)),
        "feat_frac_halogen": float(sum(fracs.get(el, 0.0) for el in HALOGENS)),
        "feat_frac_chalcogen": float(sum(fracs.get(el, 0.0) for el in CHALCOGENS)),
        "feat_frac_lanthanoid": float(sum(fracs.get(el, 0.0) for el in LANTHANOIDS)),
        "feat_frac_actinoid": float(sum(fracs.get(el, 0.0) for el in ACTINOIDS)),
    })
    for el in element_vocab:
        feat[f"feat_frac_el__{el}"] = float(fracs.get(el, 0.0))
    return feat, formula, sorted(el_amt)


def featurize_cif(path: Path, feature_cols: Sequence[str], max_sites: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    element_vocab = [c.replace("feat_frac_el__", "") for c in feature_cols if c.startswith("feat_frac_el__")]
    structure = Structure.from_file(str(path), primitive=False, merge_tol=0.0)
    comp_feat, formula, elements = composition_feature_dict(structure, element_vocab)
    a, b, c, alpha, beta, gamma = lattice_lengths_angles(structure)
    crystal = approx_crystal_system(a, b, c, alpha, beta, gamma)
    feat: Dict[str, float] = {col: 0.0 for col in feature_cols}
    feat.update(comp_feat)
    for sys_name in ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]:
        feat[f"feat_crystal_system__{sys_name}"] = 1.0 if crystal == sys_name else 0.0
    feat.update({
        "feat_density": float(structure.density),
        "feat_volume": float(structure.volume),
        "feat_nsites": float(len(structure)),
        "feat_nelements": float(len(elements)),
        "feat_band_gap": 0.0,
        "feat_energy_above_hull": 0.0,
        "feat_spacegroup_number": 0.0,
        "feat_lattice_a": a,
        "feat_lattice_b": b,
        "feat_lattice_c": c,
        "feat_lattice_alpha": alpha,
        "feat_lattice_beta": beta,
        "feat_lattice_gamma": gamma,
        "feat_has_summary": 0.0,
    })
    feat.update(geom_features(structure, max_sites=max_sites))
    x = np.asarray([feat.get(col, 0.0) for col in feature_cols], dtype=np.float32)
    meta = {
        "sample_id": path.stem,
        "cif_path": str(path),
        "formula": formula,
        "target_elements": elements,
        "n_sites": int(len(structure)),
        "crystal_system_approx": crystal,
        "parse_status": "ok",
        "parse_error": "",
    }
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), meta


def load_stage2_mlp(run_dir: Path, x_dim: int, n_labels: int) -> nn.Module:
    metrics = read_json(run_dir / "metrics.json")
    cfg = metrics.get("config", {})
    model = MLP(
        input_dim=x_dim,
        output_dim=n_labels,
        hidden_dims=parse_hidden_dims(cfg.get("hidden_dims", "512,256")),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_predictor(path: Path) -> Any:
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


@torch.no_grad()
def predict_stage2_probs(model: nn.Module, x_scaled: np.ndarray, batch_size: int) -> np.ndarray:
    probs: List[np.ndarray] = []
    for i in range(0, x_scaled.shape[0], batch_size):
        xb = torch.tensor(x_scaled[i:i + batch_size], dtype=torch.float32)
        probs.append(torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32))
    return np.vstack(probs) if probs else np.zeros((0, 0), dtype=np.float32)


def elemental_set(elements: Sequence[str], name_to_idx: Mapping[str, int]) -> List[str]:
    blocked = {"O", "H", "C", "N", "F", "Cl", "Br", "I", "S", "P", "Se", "Te"}
    out = [el for el in elements if el not in blocked and el in name_to_idx]
    return out


def choose_precursors(
    probs: np.ndarray,
    size_pred: int,
    names: Sequence[str],
    target_elements: Sequence[str],
    method: str,
    name_to_idx: Mapping[str, int],
) -> Tuple[List[str], str, float]:
    k = int(np.clip(int(round(size_pred)), 1, 6))
    if method == "melt_arc" and "O" not in target_elements:
        elems = elemental_set(target_elements, name_to_idx)
        if elems:
            score = float(np.mean([probs[name_to_idx[x]] for x in elems]))
            return elems, "elemental_template", score
    idx = np.argsort(-probs)[:k]
    labels = [str(names[j]) for j in idx]
    return labels, "mlp_set_size", float(np.mean(probs[idx])) if len(idx) else 0.0


def make_y_set(labels: Sequence[str], vocab_index: Mapping[str, int], dim: int) -> np.ndarray:
    arr = np.zeros(dim, dtype=np.float32)
    for label in labels:
        j = vocab_index.get(str(label))
        if j is not None and 0 <= j < dim:
            arr[j] = 1.0
    return arr


def predict_stage3(pack: Mapping[str, Any], X: np.ndarray, methods: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cont_names = list(pack["cont_names"])
    disc_names = list(pack["disc_names"])
    schema = pack["schema"]
    disc_schema = schema.get("discrete_schema", {}) or {}
    y_cont = np.zeros((X.shape[0], len(cont_names)), dtype=np.float32)
    y_disc = np.zeros((X.shape[0], len(disc_names)), dtype=np.int64)
    for method in sorted(set(methods.tolist())):
        idx = np.where(methods == method)[0]
        model_set = pack.get("experts", {}).get(method, {}) or {}
        for j, name in enumerate(cont_names):
            key = "target_time_h_log1p" if name == "target_time_h" else name
            model = model_set.get(key) or pack.get("global_models", {}).get(key)
            if model is None:
                continue
            pred = np.asarray(model.predict(X[idx], num_iteration=model.best_iteration), dtype=np.float32)
            y_cont[idx, j] = np.expm1(pred) if key.endswith("_log1p") else pred
        for j, name in enumerate(disc_names):
            model = model_set.get(name) or pack.get("global_models", {}).get(name)
            if model is None:
                y_disc[idx, j] = int((disc_schema.get(name, {}) or {}).get("missing_index", 0))
                continue
            prob = np.asarray(model.predict(X[idx], num_iteration=model.best_iteration))
            y_disc[idx, j] = np.asarray(np.argmax(prob, axis=1), dtype=np.int64)
    return y_cont, y_disc


def decode_disc(schema: Mapping[str, Any], name: str, idx: int) -> str:
    vocab = ((schema.get("discrete_schema", {}) or {}).get(name, {}) or {}).get("vocab", [])
    if 0 <= int(idx) < len(vocab):
        return str(vocab[int(idx)])
    return str(idx)


def method_prior_score(method: str, elements: Sequence[str], precursor_source: str) -> float:
    elems = set(elements)
    metals = elems - {"O", "H", "C", "N", "F", "Cl", "Br", "I", "S", "P", "Se", "Te"}
    if method == "solid_state":
        return 0.45 if "O" in elems and metals else 0.15
    if method == "solution":
        return 0.30 if len(elems) >= 2 else 0.12
    if method == "melt_arc":
        return 0.60 if "O" not in elems and precursor_source == "elemental_template" else 0.05
    return 0.0


def append_csv(path: Path, rows: List[Dict[str, Any]], header_written: bool) -> bool:
    if not rows:
        return header_written
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not header_written:
            writer.writeheader()
        writer.writerows(rows)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch predict current-best Stage2 precursors and Stage3 synthesis conditions for Genome/selected CIFs."
    )
    ap.add_argument("--input_dir", default="Genome/selected")
    ap.add_argument("--output_dir", default="outputs/inference/genome_selected_best_current_20260611")
    ap.add_argument("--stage2_dataset_dir", default="data/interim/generative/stage2_setpred_dataset/descriptor/core_methods_ss_solution_meltarc_20260610_relaxed_only")
    ap.add_argument("--stage2_mlp_run_dir", default="runs/stage2/mlp_core_methods_ss_solution_meltarc_20260610_descriptor")
    ap.add_argument("--stage2_set_size_model", default="runs/stage2/set_size_core_methods_ss_solution_meltarc_20260610_descriptor/set_size_predictor.joblib")
    ap.add_argument("--stage3_model", default="runs/stage3/lgbm_method_experts_core_methods_20260610/stage3_lgbm_method_experts.joblib")
    ap.add_argument("--methods", default="solid_state,solution,melt_arc")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--poscar_max_sites", type=int, default=96)
    ap.add_argument("--top_label_count", type=int, default=20)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    stage2_dataset_dir = Path(args.stage2_dataset_dir).expanduser().resolve()
    methods = [x.strip() for x in str(args.methods).split(",") if x.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [str(x) for x in read_json(stage2_dataset_dir / "feature_cols.json")]
    stage2_names = [str(x) for x in read_json(stage2_dataset_dir / "precursor_names.json")]
    stage2_name_to_idx = {p: i for i, p in enumerate(stage2_names)}
    feature_mean = np.load(stage2_dataset_dir / "feature_mean.npy").astype(np.float32)
    feature_std = np.load(stage2_dataset_dir / "feature_std.npy").astype(np.float32)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    stage2_model = load_stage2_mlp(Path(args.stage2_mlp_run_dir), len(feature_cols), len(stage2_names))
    set_size_model = load_predictor(Path(args.stage2_set_size_model))
    stage3_pack = joblib.load(Path(args.stage3_model))
    stage3_schema = stage3_pack["schema"]
    stage3_precursor_vocab = [str(x) for x in stage3_schema.get("precursor_vocab", [])]
    stage3_vocab_index = {p: i for i, p in enumerate(stage3_precursor_vocab)}
    y_set_dim = int((stage3_schema.get("data", {}) or {}).get("y_set_dim", len(stage3_precursor_vocab)))

    files = sorted(input_dir.glob("*.cif"))
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"No CIF files found in {input_dir}")

    pred_path = output_dir / "genome_selected_predictions.csv"
    rec_path = output_dir / "genome_selected_recommended_top1.csv"
    fail_path = output_dir / "failed_cifs.csv"
    for p in [pred_path, rec_path, fail_path]:
        if p.exists():
            p.unlink()

    n_ok = n_failed = 0
    pred_header = rec_header = fail_header = False
    for batch_no, paths in enumerate(chunked(files, int(args.batch_size)), start=1):
        raw_xs: List[np.ndarray] = []
        metas: List[Dict[str, Any]] = []
        failed_rows: List[Dict[str, Any]] = []
        for path in paths:
            try:
                x_raw, meta = featurize_cif(path, feature_cols, max_sites=int(args.poscar_max_sites))
                raw_xs.append(x_raw)
                metas.append(meta)
            except Exception as exc:
                n_failed += 1
                failed_rows.append({
                    "sample_id": path.stem,
                    "cif_path": str(path),
                    "parse_status": "failed",
                    "parse_error": repr(exc),
                })
        fail_header = append_csv(fail_path, failed_rows, fail_header)
        if not raw_xs:
            print(f"[BATCH {batch_no}] parsed=0 failed={len(failed_rows)}")
            continue

        X_raw = np.vstack(raw_xs).astype(np.float32)
        X_scaled = ((X_raw - feature_mean) / feature_std).astype(np.float32)
        probs = predict_stage2_probs(stage2_model, X_scaled, batch_size=int(args.batch_size))
        size_pred = np.asarray(set_size_model.predict(X_scaled)).astype(int)

        route_rows: List[Dict[str, Any]] = []
        stage3_inputs: List[np.ndarray] = []
        stage3_methods: List[str] = []
        route_context: List[Dict[str, Any]] = []

        for i, meta in enumerate(metas):
            top_idx = np.argsort(-probs[i])[: int(args.top_label_count)]
            top_labels = [
                {"precursor": stage2_names[j], "probability": float(probs[i, j])}
                for j in top_idx
            ]
            for method in methods:
                labels, source, prec_score = choose_precursors(
                    probs[i], int(size_pred[i]), stage2_names, meta["target_elements"], method, stage2_name_to_idx
                )
                y_set = make_y_set(labels, stage3_vocab_index, y_set_dim)
                stage3_inputs.append(np.concatenate([X_raw[i], y_set]).astype(np.float32))
                stage3_methods.append(method)
                route_context.append({
                    "meta": meta,
                    "method": method,
                    "precursors": labels,
                    "precursor_source": source,
                    "precursor_score": prec_score,
                    "predicted_set_size": int(np.clip(int(round(size_pred[i])), 1, 6)),
                    "top_labels": top_labels,
                    "mapped_precursor_count": int(sum(1 for x in labels if x in stage3_vocab_index)),
                })

        X_stage3 = np.vstack(stage3_inputs).astype(np.float32)
        y_cont, y_disc = predict_stage3(stage3_pack, X_stage3, np.asarray(stage3_methods, dtype=object))

        cont_names = list(stage3_pack["cont_names"])
        disc_names = list(stage3_pack["disc_names"])
        for r, ctx in enumerate(route_context):
            meta = ctx["meta"]
            temp = float(y_cont[r, cont_names.index("target_temperature_c")]) if "target_temperature_c" in cont_names else 0.0
            time_h = float(y_cont[r, cont_names.index("target_time_h")]) if "target_time_h" in cont_names else 0.0
            disc_decoded = {
                name: decode_disc(stage3_schema, name, int(y_disc[r, j]))
                for j, name in enumerate(disc_names)
            }
            prior = method_prior_score(ctx["method"], meta["target_elements"], ctx["precursor_source"])
            route_score = float(ctx["precursor_score"] + prior)
            route_rows.append({
                "sample_id": meta["sample_id"],
                "cif_path": meta["cif_path"],
                "formula": meta["formula"],
                "target_elements": json.dumps(meta["target_elements"], ensure_ascii=False),
                "n_sites": meta["n_sites"],
                "crystal_system_approx": meta["crystal_system_approx"],
                "reaction_method": ctx["method"],
                "route_rank_score": route_score,
                "method_prior_score": prior,
                "precursor_score": ctx["precursor_score"],
                "predicted_set_size": ctx["predicted_set_size"],
                "precursor_source": ctx["precursor_source"],
                "mapped_precursor_count": ctx["mapped_precursor_count"],
                "predicted_precursors": json.dumps(ctx["precursors"], ensure_ascii=False),
                "top20_precursor_labels": json.dumps(ctx["top_labels"], ensure_ascii=False),
                "pred_temperature_c": temp,
                "pred_time_h": max(time_h, 0.0),
                "pred_atmosphere": disc_decoded.get("target_atmosphere", ""),
                "pred_solvent": disc_decoded.get("target_solvent", ""),
                "pred_synthesis_type": disc_decoded.get("synthesis_type", ""),
                "stage2_model": str(Path(args.stage2_mlp_run_dir)),
                "stage3_model": str(Path(args.stage3_model)),
                "parse_status": meta["parse_status"],
                "parse_error": meta["parse_error"],
            })

        pred_header = append_csv(pred_path, route_rows, pred_header)
        rec_df = pd.DataFrame(route_rows)
        rec_rows = []
        if not rec_df.empty:
            for _, sub in rec_df.sort_values(["sample_id", "route_rank_score"], ascending=[True, False]).groupby("sample_id", sort=False):
                best = sub.iloc[0].to_dict()
                best["route_rank"] = 1
                rec_rows.append(best)
        rec_header = append_csv(rec_path, rec_rows, rec_header)
        n_ok += len(metas)
        if batch_no % 10 == 0 or n_ok + n_failed == len(files):
            print(f"[PROGRESS] batch={batch_no} ok={n_ok} failed={n_failed} total={len(files)}")

    if not fail_path.exists():
        fail_header = append_csv(fail_path, [{
            "sample_id": "",
            "cif_path": "",
            "parse_status": "",
            "parse_error": "",
        }], False)
        pd.read_csv(fail_path).iloc[0:0].to_csv(fail_path, index=False)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_cif_total": len(files),
        "n_parsed_ok": n_ok,
        "n_failed": n_failed,
        "n_routes": n_ok * len(methods),
        "methods": methods,
        "stage2_model": str(Path(args.stage2_mlp_run_dir)),
        "stage2_set_size_model": str(Path(args.stage2_set_size_model)),
        "stage3_model": str(Path(args.stage3_model)),
        "stage2_precursor_labels": len(stage2_names),
        "stage3_precursor_vocab": len(stage3_precursor_vocab),
        "stage2_labels_mapped_to_stage3": int(sum(1 for x in stage2_names if x in stage3_vocab_index)),
        "outputs": {
            "all_routes_csv": str(pred_path),
            "recommended_top1_csv": str(rec_path),
            "failed_cifs_csv": str(fail_path),
        },
        "notes": [
            "Stage2 uses the current deployable core-method MLP plus set-size predictor.",
            "Stage3 predicts each requested reaction method with the current core-method LightGBM expert model.",
            "route_rank_score is a deployment heuristic combining precursor confidence and a method prior; all method rows are preserved.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
