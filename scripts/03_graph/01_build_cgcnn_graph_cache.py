#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged version of 01_build_cgcnn_graph_cache.py

支持两种工作模式：
1) 全量构图（保持原始行为）
   - 从 input_dir 下的 stage2_{train,val,test,gold_train_holdout}.jsonl 读取样本
   - 解析 POSCAR 并构建 CGCNN 风格邻接图缓存
2) mode 子缓存快速构建（新增）
   - 从 base_cache_dir 下已存在的 train/val/test.pkl 读取全量图缓存
   - 根据 mode_input_root/train_mode/{train,val,test}/*.csv 里的样本 ID 过滤出子缓存

典型用法：
A. 全量构图
python 01_build_cgcnn_graph_cache.py \
  --input_dir /Users/wyc/MP_exp_doi/data/interim/splits/structdesc_splits \
  --output_dir /Users/wyc/MP_exp_doi/data/interim/graph_cache/cgcnn_stage2

B. 从 base cache 快速生成 relaxed_only 子缓存
python 01_build_cgcnn_graph_cache.py \
  --base_cache_dir /Users/wyc/MP_exp_doi/data/interim/graph_cache/cgcnn_stage2 \
  --mode_input_root /Users/wyc/MP_exp_doi/data/interim/training_modes/stage2_hybrid_cgcnn \
  --train_mode relaxed_only \
  --output_dir /Users/wyc/MP_exp_doi/data/interim/graph_cache/cgcnn_stage2/relaxed_only
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


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
    "Lv", "Ts", "Og",
]
Z_TABLE = {el: i + 1 for i, el in enumerate(ELEMENTS)}
SHIFT_VECS = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=float)
JOIN_KEYS = [
    "row_id",
    "sample_id",
    "material_id",
    "entry_id",
    "reaction_id",
    "id",
    "synth_uid",
    "record_index",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
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


def parse_poscar(poscar_path: Path) -> Optional[Dict[str, Any]]:
    try:
        lines = [x.rstrip("\n") for x in open(poscar_path, "r", encoding="utf-8").readlines()]
    except Exception:
        return None

    if len(lines) < 8:
        return None

    try:
        scale = float(lines[1].strip())
        lattice = np.array(
            [
                [float(x) for x in lines[2].split()],
                [float(x) for x in lines[3].split()],
                [float(x) for x in lines[4].split()],
            ],
            dtype=float,
        ) * scale
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

    atom_species = []
    for sp, cnt in zip(species, counts):
        atom_species.extend([sp] * cnt)

    try:
        atomic_numbers = [Z_TABLE[sp] for sp in atom_species]
    except KeyError:
        return None

    return {
        "lattice": lattice,
        "frac": frac,
        "cart": cart,
        "species": atom_species,
        "atomic_numbers": np.array(atomic_numbers, dtype=np.int64),
        "nsites": nsites,
    }


def build_neighbor_graph(
    frac: np.ndarray,
    lattice: np.ndarray,
    max_num_nbr: int = 12,
    radius: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = frac.shape[0]
    src_list: List[int] = []
    dst_list: List[int] = []
    dist_list: List[float] = []

    for i in range(n):
        diff = frac[None, :, :] - frac[i][None, None, :] + SHIFT_VECS[:, None, :]
        cart = diff @ lattice
        dmat = np.linalg.norm(cart, axis=2)  # (27, n)

        zero_shift_idx = 13
        dmat[zero_shift_idx, i] = np.inf

        flat_d = dmat.reshape(-1)
        flat_j = np.tile(np.arange(n), len(SHIFT_VECS))

        within = np.where(flat_d <= radius)[0]
        if len(within) == 0:
            order = np.argsort(flat_d)[:max_num_nbr]
        else:
            order = within[np.argsort(flat_d[within])[:max_num_nbr]]

        for idx in order:
            j = int(flat_j[idx])
            d = float(flat_d[idx])
            if not np.isfinite(d):
                continue
            src_list.append(j)  # message from j -> i
            dst_list.append(i)
            dist_list.append(d)

    return (
        np.array(src_list, dtype=np.int64),
        np.array(dst_list, dtype=np.int64),
        np.array(dist_list, dtype=np.float32),
    )


def build_precursor_vocab(rows: List[Dict[str, Any]]) -> List[str]:
    vocab = set()
    for row in rows:
        for p in row.get("main_precursors", []) or []:
            if p:
                vocab.add(str(p))
    return sorted(vocab)


def make_multihot(row: Dict[str, Any], precursor_to_idx: Dict[str, int]) -> np.ndarray:
    y = np.zeros(len(precursor_to_idx), dtype=np.uint8)
    for p in row.get("main_precursors", []) or []:
        if p in precursor_to_idx:
            y[precursor_to_idx[p]] = 1
    return y


def process_split(
    rows: List[Dict[str, Any]],
    precursor_to_idx: Dict[str, int],
    base_dir: str,
    max_sites: int,
    max_num_nbr: int,
    radius: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    data_list = []
    stats = {
        "input_rows": len(rows),
        "kept": 0,
        "dropped_missing_poscar": 0,
        "dropped_bad_poscar": 0,
        "dropped_too_many_sites": 0,
        "dropped_empty_label": 0,
    }

    for row in rows:
        poscar_path = resolve_path(row.get("poscar_path"), base_dir)
        if poscar_path is None or not poscar_path.exists():
            stats["dropped_missing_poscar"] += 1
            continue

        parsed = parse_poscar(poscar_path)
        if parsed is None:
            stats["dropped_bad_poscar"] += 1
            continue

        if parsed["nsites"] > max_sites:
            stats["dropped_too_many_sites"] += 1
            continue

        y = make_multihot(row, precursor_to_idx)
        if int(y.sum()) == 0:
            stats["dropped_empty_label"] += 1
            continue

        edge_src, edge_dst, edge_dist = build_neighbor_graph(
            frac=parsed["frac"],
            lattice=parsed["lattice"],
            max_num_nbr=max_num_nbr,
            radius=radius,
        )

        data = {
            "id": row.get("id"),
            "material_id": row.get("material_id"),
            "formula": row.get("formula"),
            "doi": row.get("doi"),
            "split_group": row.get("split_group"),
            "atomic_numbers": parsed["atomic_numbers"],
            "frac_coords": parsed["frac"].astype(np.float32),
            "lattice": parsed["lattice"].astype(np.float32),
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_dist": edge_dist,
            "y": y,
            "main_precursors": row.get("main_precursors", []),
        }
        data_list.append(data)
        stats["kept"] += 1

    return data_list, stats


# ---------------------- mode 子缓存快速构建（新增） ----------------------

def norm_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def extract_uid_from_mapping(d: Dict[str, Any]) -> Optional[str]:
    for k in JOIN_KEYS:
        if k in d:
            uid = norm_value(d.get(k))
            if uid is not None:
                return uid

    for mk in ["meta", "metadata", "record", "sample", "row"]:
        sub = d.get(mk)
        if isinstance(sub, dict):
            uid = extract_uid_from_mapping(sub)
            if uid is not None:
                return uid

    return None


def extract_uid(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return extract_uid_from_mapping(obj)
    if isinstance(obj, (list, tuple)):
        for x in obj:
            uid = extract_uid(x)
            if uid is not None:
                return uid
        return None
    for k in JOIN_KEYS:
        if hasattr(obj, k):
            uid = norm_value(getattr(obj, k))
            if uid is not None:
                return uid
    if hasattr(obj, "__dict__"):
        return extract_uid_from_mapping(vars(obj))
    return None


def find_single_csv(split_dir: Path) -> Path:
    cands = sorted(split_dir.glob("*.csv"))
    if not cands:
        raise FileNotFoundError(f"{split_dir} 下没有找到 CSV 文件")
    if len(cands) > 1:
        print(f"[Warn] {split_dir} 下找到多个 CSV，默认使用: {cands[0].name}")
    return cands[0]


def resolve_mode_split_root(mode_input_root: Path, train_mode: str) -> Path:
    # 常规布局: root/train_mode/{train,val,test}
    cand1 = mode_input_root / train_mode
    if cand1.exists() and cand1.is_dir():
        return cand1
    # 少数情况下 root 本身已经是 mode 目录
    if (mode_input_root / "train").exists() or (mode_input_root / "val").exists() or (mode_input_root / "test").exists():
        return mode_input_root
    raise FileNotFoundError(
        f"未找到 mode split 根目录。已尝试:\n- {cand1}\n- {mode_input_root} (as direct mode dir)"
    )


def load_split_ids(csv_path: Path) -> Set[str]:
    df = pd.read_csv(csv_path)
    ids: Set[str] = set()
    miss = 0
    for rec in df.to_dict(orient="records"):
        uid = extract_uid_from_mapping(rec)
        if uid is None:
            miss += 1
            continue
        ids.add(uid)
    print(f"[Info] load split csv={csv_path.name} rows={len(df)} extracted_ids={len(ids)} rows_without_uid={miss}")
    return ids


def filter_graph_items(items: List[Any], keep_ids: Set[str], split_name: str) -> Tuple[List[Any], Dict[str, int]]:
    out = []
    miss_uid = 0
    for x in items:
        uid = extract_uid(x)
        if uid is None:
            miss_uid += 1
            continue
        if uid in keep_ids:
            out.append(x)
    stats = {
        "input_items": len(items),
        "mode_ids": len(keep_ids),
        "matched_items": len(out),
        "items_without_uid": miss_uid,
    }
    print(f"[Info] split={split_name} input={len(items)} mode_ids={len(keep_ids)} matched={len(out)} miss_uid={miss_uid}")
    return out, stats


def build_mode_cache_from_base(args: argparse.Namespace) -> Dict[str, Any]:
    base_cache_dir = Path(args.base_cache_dir).expanduser().resolve()
    mode_input_root = Path(args.mode_input_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not base_cache_dir.exists():
        raise FileNotFoundError(f"base_cache_dir 不存在: {base_cache_dir}")
    if not mode_input_root.exists():
        raise FileNotFoundError(f"mode_input_root 不存在: {mode_input_root}")

    split_root = resolve_mode_split_root(mode_input_root, args.train_mode)
    ensure_dir(out_dir)

    base_split_files = {
        "train": base_cache_dir / "train.pkl",
        "val": base_cache_dir / "val.pkl",
        "test": base_cache_dir / "test.pkl",
    }
    for name, p in base_split_files.items():
        if not p.exists():
            raise FileNotFoundError(f"base cache 缺少 {name}.pkl: {p}")

    base_items = {k: load_pickle(v) for k, v in base_split_files.items()}

    split_csvs = {
        "train": find_single_csv(split_root / "train"),
        "val": find_single_csv(split_root / "val"),
        "test": find_single_csv(split_root / "test"),
    }
    split_ids = {k: load_split_ids(v) for k, v in split_csvs.items()}

    out_items: Dict[str, List[Any]] = {}
    stats: Dict[str, Dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        out_items[split_name], stats[split_name] = filter_graph_items(
            base_items[split_name], split_ids[split_name], split_name
        )

    # gold_train_holdout 若上游存在就一起处理
    gold_dir = split_root / "gold_train_holdout"
    if gold_dir.exists() and gold_dir.is_dir() and (base_cache_dir / "gold_train_holdout.pkl").exists():
        gold_csv = find_single_csv(gold_dir)
        gold_ids = load_split_ids(gold_csv)
        gold_base = load_pickle(base_cache_dir / "gold_train_holdout.pkl")
        out_items["gold_train_holdout"], stats["gold_train_holdout"] = filter_graph_items(gold_base, gold_ids, "gold_train_holdout")
        split_csvs["gold_train_holdout"] = gold_csv

    if sum(len(v) for v in out_items.values()) == 0:
        raise RuntimeError(
            "mode cache 全部匹配为 0。高概率是 join key 没对上，或 mode CSV 与 base cache 样本 ID 体系不一致。"
        )
    if len(out_items.get("train", [])) == 0:
        raise RuntimeError("train 子缓存为空，停止写出。请检查 join key 或 mode split。")

    for split_name, items in out_items.items():
        dump_pickle(items, out_dir / f"{split_name}.pkl")

    vocab_src = base_cache_dir / "precursor_vocab.json"
    if vocab_src.exists():
        with open(vocab_src, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        write_json(out_dir / "precursor_vocab.json", vocab)

    summary = {
        "mode_build": True,
        "config": {
            "base_cache_dir": str(base_cache_dir),
            "mode_input_root": str(mode_input_root),
            "resolved_split_root": str(split_root),
            "train_mode": args.train_mode,
            "output_dir": str(out_dir),
        },
        "join_keys": JOIN_KEYS,
        "split_csvs": {k: str(v) for k, v in split_csvs.items()},
        "base_counts": {k: len(v) for k, v in base_items.items()},
        "output_counts": {k: len(v) for k, v in out_items.items()},
        "splits": stats,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------- 全量构图（保留原始行为） ----------------------

def build_full_cache_from_raw(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    ensure_dir(out_dir)

    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split_name in ["train", "val", "test", "gold_train_holdout"]:
        path = input_dir / f"stage2_{split_name}.jsonl"
        if path.exists():
            split_rows[split_name] = read_jsonl(str(path))
        else:
            if split_name == "gold_train_holdout":
                continue
            raise FileNotFoundError(f"缺少 split 文件: {path}")

    precursor_vocab = build_precursor_vocab(split_rows["train"])
    precursor_to_idx = {p: i for i, p in enumerate(precursor_vocab)}

    summary = {
        "mode_build": False,
        "config": {
            "base_dir": args.base_dir,
            "input_dir": str(input_dir),
            "output_dir": str(out_dir),
            "max_sites": args.max_sites,
            "max_num_nbr": args.max_num_nbr,
            "radius": args.radius,
        },
        "precursor_vocab_size": len(precursor_vocab),
        "splits": {},
    }

    for split_name, rows in split_rows.items():
        data_list, stats = process_split(
            rows=rows,
            precursor_to_idx=precursor_to_idx,
            base_dir=args.base_dir,
            max_sites=args.max_sites,
            max_num_nbr=args.max_num_nbr,
            radius=args.radius,
        )

        dump_pickle(data_list, out_dir / f"{split_name}.pkl")
        summary["splits"][split_name] = stats

    write_json(out_dir / "precursor_vocab.json", precursor_vocab)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CGCNN-style graph cache for stage2 (merged full-build + mode-cache builder).")

    # 原始全量构图参数（保留）
    parser.add_argument("--base_dir", type=str, default="/Users/wyc/SynPred/data")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/splits/structdesc_splits",
        help="全量构图模式下的原始 split 输入目录，期望含 stage2_train.jsonl 等文件。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/wyc/SynPred/data/interim/graph_cache/cgcnn_stage2",
        help="输出 graph cache 目录。",
    )
    parser.add_argument("--max_sites", type=int, default=256)
    parser.add_argument("--max_num_nbr", type=int, default=12)
    parser.add_argument("--radius", type=float, default=8.0)

    # 新增：mode 子缓存快速构建参数
    parser.add_argument(
        "--base_cache_dir",
        type=str,
        default="",
        help="若提供，则从已有 base cache 快速筛选 mode 子缓存；例如 data/interim/graph_cache/cgcnn_stage2",
    )
    parser.add_argument(
        "--mode_input_root",
        type=str,
        default="",
        help="训练模式根目录，例如 data/interim/training_modes/stage2_hybrid_cgcnn",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="gold_only",
        help="训练模式名称，例如 relaxed_only / gold_only / curriculum",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="即使给了 base_cache_dir / mode_input_root，也强制走原始全量构图分支。",
    )

    args = parser.parse_args()

    use_mode_builder = (
        bool(str(args.base_cache_dir).strip())
        and bool(str(args.mode_input_root).strip())
        and bool(str(args.train_mode).strip())
        and not args.force_rebuild
    )

    if use_mode_builder:
        build_mode_cache_from_base(args)
    else:
        build_full_cache_from_raw(args)


if __name__ == "__main__":
    main()
