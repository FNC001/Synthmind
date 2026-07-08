#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SHIFT_VECS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float32,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_list_like(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, float) and np.isnan(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    return [s]


def select_split_paths(input_dir: Path, train_mode: str) -> Dict[str, Path]:
    train_path = input_dir / "stage3_train.jsonl"
    val_path = input_dir / "stage3_val.jsonl"
    test_path = input_dir / "stage3_test.jsonl"
    gold_path = input_dir / "stage3_gold_train_holdout.jsonl"

    if train_mode == "relaxed_only":
        chosen_train = train_path
    elif train_mode == "gold_only":
        chosen_train = gold_path
    elif train_mode == "curriculum_phase1":
        chosen_train = train_path
    elif train_mode == "curriculum_phase2":
        chosen_train = gold_path
    else:
        raise ValueError(f"Unsupported train_mode: {train_mode}")

    return {
        "train": chosen_train,
        "val": val_path,
        "test": test_path,
        "gold_train_holdout": gold_path,
    }


def symbol_to_atomic_number(sym: str) -> int:
    periodic = [
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
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ]
    mapping = {s: i + 1 for i, s in enumerate(periodic)}
    if sym not in mapping:
        raise ValueError(f"Unknown chemical symbol: {sym}")
    return mapping[sym]


def parse_poscar(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    if len(lines) < 8:
        raise ValueError(f"POSCAR too short: {path}")

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=np.float32)
    lattice = lattice * scale

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if len(species) != len(counts):
        raise ValueError(f"POSCAR species/count mismatch: {path}")

    coord_mode_idx = 7
    selective = lines[coord_mode_idx].lower().startswith("s")
    if selective:
        coord_mode_idx += 1

    coord_mode = lines[coord_mode_idx].strip().lower()
    is_direct = coord_mode.startswith("d")
    coord_start = coord_mode_idx + 1

    nsites = sum(counts)
    coord_lines = lines[coord_start: coord_start + nsites]
    if len(coord_lines) < nsites:
        raise ValueError(f"POSCAR missing coordinates: {path}")

    frac = []
    expanded_species = []
    for sp, c in zip(species, counts):
        expanded_species.extend([sp] * c)

    for line in coord_lines:
        toks = line.split()
        if len(toks) < 3:
            raise ValueError(f"Bad coordinate line in {path}: {line}")
        frac.append([float(toks[0]), float(toks[1]), float(toks[2])])

    frac = np.array(frac, dtype=np.float32)
    if not is_direct:
        inv_lat = np.linalg.inv(lattice.T)
        frac = frac @ inv_lat

    cart = frac @ lattice
    atomic_numbers = np.array([symbol_to_atomic_number(sp) for sp in expanded_species], dtype=np.int64)

    return {
        "lattice": lattice.astype(np.float32),
        "frac_coords": frac.astype(np.float32),
        "cart_coords": cart.astype(np.float32),
        "species": expanded_species,
        "atomic_numbers": atomic_numbers,
        "nsites": int(nsites),
    }


def build_neighbor_graph(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    radius: float = 8.0,
    max_num_nbr: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = frac_coords.shape[0]
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_dist: List[float] = []

    for i in range(n):
        nbrs: List[Tuple[float, int]] = []
        fi = frac_coords[i]

        for j in range(n):
            fj = frac_coords[j]
            deltas = (fj[None, :] + SHIFT_VECS) - fi[None, :]
            carts = deltas @ lattice
            dists = np.linalg.norm(carts, axis=1)

            for d in dists.tolist():
                if i == j and d < 1e-8:
                    continue
                if d <= radius:
                    nbrs.append((float(d), j))

        nbrs.sort(key=lambda x: x[0])
        nbrs = nbrs[:max_num_nbr]

        for d, j in nbrs:
            edge_src.append(int(j))
            edge_dst.append(int(i))
            edge_dist.append(float(d))

    return (
        np.asarray(edge_src, dtype=np.int64),
        np.asarray(edge_dst, dtype=np.int64),
        np.asarray(edge_dist, dtype=np.float32),
    )


def first_present(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def build_stage3_targets(row: Dict[str, Any]) -> Dict[str, Any]:
    target_temperature = first_present(
        row,
        [
            "target_temperature_c_clean",
            "target_temperature_c",
            "temperature_c_clean",
            "temperature_c",
        ],
    )
    target_time = first_present(
        row,
        [
            "target_time_h_clean",
            "target_time_h",
            "time_h_clean",
            "time_h",
        ],
    )
    target_time_bucket = first_present(
        row,
        [
            "target_time_bucket",
            "time_bucket",
        ],
    )
    target_atmosphere = first_present(
        row,
        [
            "target_atmosphere_coarse",
            "target_atmosphere",
            "atmosphere_coarse",
            "atmosphere",
        ],
    )
    target_solvent = first_present(
        row,
        [
            "target_solvent_clean",
            "target_solvent",
            "solvent_clean",
            "solvent",
        ],
    )
    synthesis_type = first_present(
        row,
        [
            "synthesis_type",
            "target_synthesis_type",
        ],
    )

    return {
        "target_temperature_c": target_temperature,
        "target_time_h": target_time,
        "target_time_bucket": target_time_bucket,
        "target_atmosphere": target_atmosphere,
        "target_solvent": target_solvent,
        "synthesis_type": synthesis_type,
    }


def process_split(
    split_name: str,
    rows: List[Dict[str, Any]],
    out_path: Path,
    radius: float,
    max_num_nbr: int,
    max_sites: int,
) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    dropped_missing_poscar = 0
    dropped_bad_poscar = 0
    dropped_too_many_sites = 0

    n_has_temperature = 0
    n_has_time = 0
    n_has_time_bucket = 0
    n_has_atmosphere = 0
    n_has_solvent = 0

    for row in rows:
        poscar_path = row.get("poscar_path")
        if poscar_path is None or not Path(str(poscar_path)).exists():
            dropped_missing_poscar += 1
            continue

        try:
            s = parse_poscar(Path(str(poscar_path)))
        except Exception:
            dropped_bad_poscar += 1
            continue

        if s["nsites"] > max_sites:
            dropped_too_many_sites += 1
            continue

        edge_src, edge_dst, edge_dist = build_neighbor_graph(
            lattice=s["lattice"],
            frac_coords=s["frac_coords"],
            radius=radius,
            max_num_nbr=max_num_nbr,
        )

        targets = build_stage3_targets(row)

        if targets["target_temperature_c"] is not None:
            n_has_temperature += 1
        if targets["target_time_h"] is not None:
            n_has_time += 1
        if targets["target_time_bucket"] is not None:
            n_has_time_bucket += 1
        if targets["target_atmosphere"] is not None:
            n_has_atmosphere += 1
        if targets["target_solvent"] is not None:
            n_has_solvent += 1

        kept.append(
            {
                "id": row.get("id"),
                "material_id": row.get("material_id"),
                "formula": row.get("formula"),
                "doi": row.get("doi"),
                "split_group": row.get("split_group"),
                "main_precursors": normalize_list_like(row.get("main_precursors")),
                "atomic_numbers": s["atomic_numbers"],
                "frac_coords": s["frac_coords"],
                "lattice": s["lattice"],
                "edge_src": edge_src,
                "edge_dst": edge_dst,
                "edge_dist": edge_dist,
                **targets,
            }
        )

    ensure_dir(out_path.parent)
    with open(out_path, "wb") as f:
        pickle.dump(kept, f)

    return {
        "input_rows": int(len(rows)),
        "kept": int(len(kept)),
        "dropped_missing_poscar": int(dropped_missing_poscar),
        "dropped_bad_poscar": int(dropped_bad_poscar),
        "dropped_too_many_sites": int(dropped_too_many_sites),
        "n_has_temperature": int(n_has_temperature),
        "n_has_time": int(n_has_time),
        "n_has_time_bucket": int(n_has_time_bucket),
        "n_has_atmosphere": int(n_has_atmosphere),
        "n_has_solvent": int(n_has_solvent),
        "output_pkl": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CGCNN-style stage3 graph cache with train_mode-aware split selection.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/Users/wyc/SynPred/data",
        help="Base data directory.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/splits/structdesc_splits",
        help="Directory containing stage3 split jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional explicit output directory. If omitted, defaults to base_dir/interim/graph_cache/cgcnn_stage3/<train_mode>.",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="gold_only",
        choices=["relaxed_only", "gold_only", "curriculum_phase1", "curriculum_phase2"],
        help="Which split definition to use for the train cache.",
    )
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--max_num_nbr", type=int, default=12)
    parser.add_argument("--max_sites", type=int, default=256)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    input_dir = Path(args.input_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "interim" / "graph_cache" / "cgcnn_stage3" / args.train_mode

    ensure_dir(output_dir)

    split_paths = select_split_paths(input_dir, args.train_mode)
    split_rows = {k: read_jsonl(v) for k, v in split_paths.items()}

    summary: Dict[str, Any] = {
        "config": {
            "base_dir": str(base_dir),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "train_mode": args.train_mode,
            "radius": args.radius,
            "max_num_nbr": args.max_num_nbr,
            "max_sites": args.max_sites,
        },
        "split_paths": {k: str(v) for k, v in split_paths.items()},
        "splits": {},
    }

    for split_name, rows in split_rows.items():
        out_path = output_dir / f"{split_name}.pkl"
        summary["splits"][split_name] = process_split(
            split_name=split_name,
            rows=rows,
            out_path=out_path,
            radius=args.radius,
            max_num_nbr=args.max_num_nbr,
            max_sites=args.max_sites,
        )

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
