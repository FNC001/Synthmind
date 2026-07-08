#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


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


def load_precursor_vocab(path: Optional[Path], train_rows: List[Dict[str, Any]]) -> List[str]:
    if path is not None and path.exists():
        vocab = read_json(path)
        if not isinstance(vocab, list) or not vocab:
            raise ValueError(f"Invalid precursor vocab JSON: {path}")
        return [str(x) for x in vocab]

    vocab_set = set()
    for row in train_rows:
        for p in normalize_list_like(row.get("main_precursors")):
            vocab_set.add(p)
    vocab = sorted(vocab_set)
    if not vocab:
        raise ValueError("Failed to build precursor vocab from training rows.")
    return vocab


def build_multihot(main_precursors: List[str], precursor_to_idx: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(precursor_to_idx), dtype=np.float32)
    for p in main_precursors:
        idx = precursor_to_idx.get(p)
        if idx is not None:
            y[idx] = 1.0
    return y


def select_split_paths(input_dir: Path, train_mode: str) -> Dict[str, Path]:
    train_path = input_dir / "stage2_train.jsonl"
    val_path = input_dir / "stage2_val.jsonl"
    test_path = input_dir / "stage2_test.jsonl"
    gold_path = input_dir / "stage2_gold_train_holdout.jsonl"

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
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
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


def lattice_to_lengths_angles(lattice: np.ndarray) -> Dict[str, np.ndarray]:
    a_vec, b_vec, c_vec = lattice[0], lattice[1], lattice[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    def angle(u: np.ndarray, v: np.ndarray) -> float:
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom < 1e-12:
            return 0.0
        cosang = np.clip(np.dot(u, v) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    alpha = angle(b_vec, c_vec)
    beta = angle(a_vec, c_vec)
    gamma = angle(a_vec, b_vec)

    return {
        "lengths": np.array([a, b, c], dtype=np.float32),
        "angles": np.array([alpha, beta, gamma], dtype=np.float32),
    }


def build_chgnet_ready_payload(struct: Dict[str, Any]) -> Dict[str, Any]:
    la = lattice_to_lengths_angles(struct["lattice"])
    volume = float(abs(np.linalg.det(struct["lattice"])))
    return {
        "atomic_numbers": struct["atomic_numbers"],
        "frac_coords": struct["frac_coords"],
        "cart_coords": struct["cart_coords"],
        "lattice": struct["lattice"],
        "lattice_lengths": la["lengths"],
        "lattice_angles": la["angles"],
        "volume": np.float32(volume),
        "nsites": struct["nsites"],
    }


def get_main_precursors(row: Dict[str, Any]) -> List[str]:
    # 训练数据兼容：优先 main_precursors，若没有则回退到 target_main_precursors
    main_precursors = normalize_list_like(row.get("main_precursors"))
    if main_precursors:
        return main_precursors
    return normalize_list_like(row.get("target_main_precursors"))


def process_split(
    split_name: str,
    rows: List[Dict[str, Any]],
    out_path: Path,
    precursor_to_idx: Dict[str, int],
    max_sites: int,
    base_dir: str,
    allow_empty_label: bool = False,
) -> Dict[str, Any]:
    kept: List[Dict[str, Any]] = []
    dropped_missing_poscar = 0
    dropped_bad_poscar = 0
    dropped_too_many_sites = 0
    dropped_empty_label = 0

    for row in rows:
        poscar_path = resolve_path(row.get("poscar_path"), base_dir)
        if poscar_path is None or not poscar_path.exists():
            dropped_missing_poscar += 1
            continue

        main_precursors = get_main_precursors(row)
        y = build_multihot(main_precursors, precursor_to_idx)

        if float(y.sum()) <= 0 and not allow_empty_label:
            dropped_empty_label += 1
            continue

        try:
            struct = parse_poscar(poscar_path)
        except Exception:
            dropped_bad_poscar += 1
            continue

        if struct["nsites"] > max_sites:
            dropped_too_many_sites += 1
            continue

        payload = build_chgnet_ready_payload(struct)

        kept.append(
            {
                "id": row.get("id", row.get("sample_id")),
                "sample_id": row.get("sample_id", row.get("id")),
                "material_id": row.get("material_id", row.get("sample_id")),
                "formula": row.get("formula", row.get("formula_guess")),
                "doi": row.get("doi"),
                "split_group": row.get("split_group", row.get("sample_id")),
                "main_precursors": main_precursors,
                "y": y,
                "poscar_path": str(poscar_path),
                **payload,
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
        "dropped_empty_label": int(dropped_empty_label),
        "allow_empty_label": bool(allow_empty_label),
        "output_pkl": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CHGNet-ready stage2 cache with train_mode-aware split selection.")
    parser.add_argument("--base_dir", type=str, default="/Users/wyc/SynPred/data", help="Base data directory.")
    parser.add_argument("--input_dir", type=str, default="/Users/wyc/SynPred/data/interim/splits/structdesc_splits", help="Directory containing stage2 split jsonl files.")
    parser.add_argument("--output_dir", type=str, default="", help="Optional explicit output directory. If omitted, defaults to base_dir/interim/graph_cache/chgnet_stage2/<train_mode>.")
    parser.add_argument("--train_mode", type=str, default="relaxed_only", choices=["relaxed_only", "gold_only", "curriculum_phase1", "curriculum_phase2"], help="Which split definition to use for the train cache.")
    parser.add_argument("--precursor_vocab_json", type=str, default="/Users/wyc/SynPred/data/interim/features/structdesc_features/meta/precursor_vocab.json", help="Fixed precursor vocab json. If missing, falls back to building from the selected train split.")
    parser.add_argument("--max_sites", type=int, default=256)
    parser.add_argument(
        "--inference_mode",
        action="store_true",
        help="Keep unlabeled structures when building cache for inference."
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    input_dir = Path(args.input_dir)
    vocab_json = Path(args.precursor_vocab_json) if args.precursor_vocab_json else None

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "interim" / "graph_cache" / "chgnet_stage2" / args.train_mode

    ensure_dir(output_dir)

    split_paths = select_split_paths(input_dir, args.train_mode)
    split_rows = {k: read_jsonl(v) for k, v in split_paths.items()}

    precursor_vocab = load_precursor_vocab(vocab_json, split_rows["train"])
    precursor_to_idx = {p: i for i, p in enumerate(precursor_vocab)}

    summary: Dict[str, Any] = {
        "config": {
            "base_dir": str(base_dir),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "train_mode": args.train_mode,
            "precursor_vocab_json": str(vocab_json) if vocab_json is not None else "",
            "max_sites": args.max_sites,
            "inference_mode": bool(args.inference_mode),
        },
        "split_paths": {k: str(v) for k, v in split_paths.items()},
        "precursor_vocab_size": int(len(precursor_vocab)),
        "precursor_vocab_path": str(output_dir / "precursor_vocab.json"),
        "splits": {},
    }

    write_json(output_dir / "precursor_vocab.json", precursor_vocab)

    for split_name, rows in split_rows.items():
        out_path = output_dir / f"{split_name}.pkl"
        summary["splits"][split_name] = process_split(
            split_name=split_name,
            rows=rows,
            out_path=out_path,
            precursor_to_idx=precursor_to_idx,
            max_sites=args.max_sites,
            base_dir=str(base_dir),
            allow_empty_label=bool(args.inference_mode),
        )

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
