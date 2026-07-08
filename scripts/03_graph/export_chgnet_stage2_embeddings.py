#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from chgnet.model.model import CHGNet
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_structure(item: Dict[str, Any]) -> Structure:
    species = [Element.from_Z(int(z)) for z in item["atomic_numbers"]]
    return Structure(
        lattice=item["lattice"],
        species=species,
        coords=item["frac_coords"],
        coords_are_cartesian=False,
    )


def normalize_prediction(pred: Any) -> Dict[str, Any]:
    if isinstance(pred, list):
        if len(pred) != 1:
            raise ValueError(f"Expected single prediction, got list of length {len(pred)}")
        pred = pred[0]
    if not isinstance(pred, dict):
        raise TypeError(f"Expected dict prediction, got {type(pred)}")
    return pred


def extract_embeddings(model: CHGNet, data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for item in tqdm(data_list):
        try:
            struct = to_structure(item)
            graph = model.graph_converter(struct)

            pred = model.predict_graph(
                graph,
                task="e",
                return_crystal_feas=True,
                batch_size=1,
            )
            pred = normalize_prediction(pred)

            if "crystal_fea" not in pred:
                raise KeyError(f"Prediction keys do not contain 'crystal_fea': {list(pred.keys())}")

            emb = pred["crystal_fea"]
            if hasattr(emb, "tolist"):
                emb = emb.tolist()

            row = {
                "id": item.get("id"),
                "material_id": item.get("material_id"),
                "formula": item.get("formula"),
                "doi": item.get("doi"),
                "split_group": item.get("split_group"),
            }
            for i, v in enumerate(emb):
                row[f"graph_emb_{i}"] = float(v)
            rows.append(row)

        except Exception as e:
            print(f"[WARN] id={item.get('id')} failed: {repr(e)}")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print("[INFO] Loading CHGNet...")
    model = CHGNet.load()
    model.eval()

    splits = ["train", "val", "test", "gold_train_holdout"]

    for split in splits:
        pkl_path = cache_dir / f"{split}.pkl"
        if not pkl_path.exists():
            print(f"[WARN] missing {pkl_path}")
            continue

        print(f"\n[INFO] Processing {split}...")
        data_list = load_pickle(pkl_path)
        df = extract_embeddings(model, data_list)

        out_csv = output_dir / f"stage2_{split}_graph_embed.csv"
        df.to_csv(out_csv, index=False)

        print(f"[OK] saved: {out_csv}  rows={len(df)}")


if __name__ == "__main__":
    main()
