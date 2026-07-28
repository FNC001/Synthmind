#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


def load_json(path: Path) -> Any:
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
    return obj


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def classify_reaction_method(row: Mapping[str, Any]) -> str:
    text = " ".join([
        norm_text(row.get("synthesis_type")),
        norm_text(row.get("synthesis_text")),
        norm_text(row.get("reaction_string")),
    ])
    solvent = norm_text(row.get("solvent"))

    def has(*patterns: str) -> bool:
        return any(p in text for p in patterns)

    if has("hydrothermal", "solvothermal", "teflon-lined autoclave", "teflon lined autoclave", "autoclave"):
        return "hydro_solvothermal"
    if has("sol-gel", "sol gel", "pechini", "citrate gel", "gel combustion"):
        return "sol_gel"
    if has("co-precip", "coprecip", "precipitat"):
        return "precipitation"
    if has("combustion"):
        return "combustion"
    if has("molten salt", "flux"):
        return "flux_molten_salt"
    if has("arc-melting", "arc melting", "arc-melt", "melted", "melting"):
        return "melt_arc"
    if has("mechanochemical", "ball mill", "ball-mill", "milling"):
        return "mechanochemical"
    if has("thermal decomposition", "decomposed", "decomposition"):
        return "thermal_decomposition"
    if has(
        "solid-state",
        "solid state",
        "sinter",
        "calcined",
        "calcination",
        "anneal",
        "fired",
        "pellet",
        "ground and heated",
        "heated at",
    ):
        return "solid_state"
    if solvent or has("aqueous", "solution", "dissolved", "stirred", "reflux"):
        return "solution"
    return "other"


def build_method_map(refined_dir: Path) -> Dict[str, str]:
    id_to_method: Dict[str, str] = {}
    for name in ["stage3_train_relaxed.jsonl", "stage3_gold.jsonl"]:
        path = refined_dir / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = str(row.get("id") or "")
                if sid:
                    id_to_method[sid] = classify_reaction_method(row)
    return id_to_method


def combine_stage2(stage2_dir: Path, splits: Sequence[str]) -> tuple[Dict[str, np.ndarray], pd.DataFrame]:
    packs: List[Dict[str, np.ndarray]] = []
    metas: List[pd.DataFrame] = []
    for split in splits:
        pack = load_npz(stage2_dir / f"{split}.npz")
        meta = pd.read_csv(stage2_dir / f"{split}_meta.csv")
        meta["_old_split"] = split
        packs.append(pack)
        metas.append(meta)
    keys = packs[0].keys()
    combined = {k: np.concatenate([p[k] for p in packs], axis=0) for k in keys}
    return combined, pd.concat(metas, ignore_index=True)


def combine_stage3(stage3_dir: Path, splits: Sequence[str]) -> Dict[str, np.ndarray]:
    packs = [load_npz(stage3_dir / f"{split}.npz") for split in splits]
    keys = packs[0].keys()
    return {k: np.concatenate([p[k] for p in packs], axis=0) for k in keys}


def fit_standardizer(x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_raw.mean(axis=0)
    std = x_raw.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_slots(y_multi_hot: np.ndarray, n_slots: int) -> Dict[str, np.ndarray]:
    n = y_multi_hot.shape[0]
    slot_targets = np.zeros((n, n_slots), dtype=np.int64)
    slot_mask = np.zeros((n, n_slots), dtype=np.int64)
    set_len = np.zeros(n, dtype=np.int64)
    overflow = np.zeros(n, dtype=np.int64)
    for i in range(n):
        active = np.where(y_multi_hot[i] > 0)[0].tolist()
        set_len[i] = len(active)
        if len(active) > n_slots:
            overflow[i] = 1
            active = active[:n_slots]
        if active:
            slot_targets[i, : len(active)] = np.asarray([j + 1 for j in active], dtype=np.int64)
            slot_mask[i, : len(active)] = 1
    return {
        "slot_targets": slot_targets,
        "slot_mask": slot_mask,
        "set_len": set_len,
        "overflow": overflow,
    }


def split_groups(
    row_df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed)
    group_rows: Dict[str, List[int]] = defaultdict(list)
    for i, group in enumerate(row_df["split_group"].fillna(row_df["id"]).astype(str).tolist()):
        group_rows[group].append(i)

    group_records = []
    for group, rows in group_rows.items():
        methods = row_df.iloc[rows]["reaction_method"].astype(str).tolist()
        method = Counter(methods).most_common(1)[0][0]
        group_records.append({"group": group, "rows": rows, "n_rows": len(rows), "method": method})

    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in group_records:
        by_method[rec["method"]].append(rec)

    out = {"train": [], "val": [], "test": []}
    for method, records in by_method.items():
        records = records[:]
        rng.shuffle(records)
        n_total = sum(int(r["n_rows"]) for r in records)
        n_val_target = int(round(n_total * val_frac))
        n_test_target = int(round(n_total * test_frac))
        method_split = {"train": [], "val": [], "test": []}
        counts = {"val": 0, "test": 0}
        for rec in sorted(records, key=lambda r: int(r["n_rows"]), reverse=True):
            if counts["test"] < n_test_target:
                target = "test"
                counts["test"] += int(rec["n_rows"])
            elif counts["val"] < n_val_target:
                target = "val"
                counts["val"] += int(rec["n_rows"])
            else:
                target = "train"
            method_split[target].extend(rec["rows"])
        for split in out:
            out[split].extend(method_split[split])

    for split in out:
        out[split] = sorted(out[split])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reaction-method-stratified paired Stage2/Stage3 datasets.")
    ap.add_argument("--stage2_dir", required=True)
    ap.add_argument("--stage3_dir", required=True)
    ap.add_argument("--refined_dir", required=True)
    ap.add_argument("--out_stage2_dir", required=True)
    ap.add_argument("--out_stage3_dir", required=True)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_slots", type=int, default=7)
    args = ap.parse_args()

    stage2_dir = Path(args.stage2_dir).resolve()
    stage3_dir = Path(args.stage3_dir).resolve()
    out_stage2 = Path(args.out_stage2_dir).resolve()
    out_stage3 = Path(args.out_stage3_dir).resolve()
    out_stage2.mkdir(parents=True, exist_ok=True)
    out_stage3.mkdir(parents=True, exist_ok=True)

    method_map = build_method_map(Path(args.refined_dir))
    s2_pack, s2_meta = combine_stage2(stage2_dir, ["train", "val", "test"])
    s3_pack = combine_stage3(stage3_dir, ["train", "val", "test"])

    s2_ids = s2_meta["id"].astype(str).tolist()
    s3_ids = [str(x) for x in s3_pack["sample_id"].tolist()]
    s2_index = {sid: i for i, sid in enumerate(s2_ids)}
    s3_index = {sid: i for i, sid in enumerate(s3_ids)}
    common_ids = [sid for sid in s3_ids if sid in s2_index]
    if not common_ids:
        raise RuntimeError("No common ids between Stage2 and Stage3 datasets.")

    s2_common_idx = np.asarray([s2_index[sid] for sid in common_ids], dtype=np.int64)
    s3_common_idx = np.asarray([s3_index[sid] for sid in common_ids], dtype=np.int64)
    row_df = s2_meta.iloc[s2_common_idx].reset_index(drop=True).copy()
    row_df["reaction_method"] = [method_map.get(sid, "other") for sid in common_ids]
    row_df["id"] = common_ids

    split_to_rows = split_groups(row_df, args.val_frac, args.test_frac, args.seed)

    feature_cols = load_json(stage2_dir / "feature_cols.json")
    label_cols = load_json(stage2_dir / "label_cols.json")
    precursor_names = load_json(stage2_dir / "precursor_names.json")
    slot_vocab = ["<pad>"] + [str(x) for x in precursor_names]
    slot_to_id = {tok: i for i, tok in enumerate(slot_vocab)}
    write_json(out_stage2 / "feature_cols.json", feature_cols)
    write_json(out_stage2 / "label_cols.json", label_cols)
    write_json(out_stage2 / "precursor_names.json", precursor_names)
    write_json(out_stage2 / "slot_vocab.json", slot_vocab)
    write_json(out_stage2 / "slot_to_id.json", slot_to_id)

    train_s2_idx = s2_common_idx[np.asarray(split_to_rows["train"], dtype=np.int64)]
    mean, std = fit_standardizer(s2_pack["x_raw"][train_s2_idx].astype(np.float32))
    np.save(out_stage2 / "feature_mean.npy", mean)
    np.save(out_stage2 / "feature_std.npy", std)

    summary: Dict[str, Any] = {
        "config": vars(args),
        "n_common_ids": int(len(common_ids)),
        "method_counts_all": dict(Counter(row_df["reaction_method"].tolist())),
        "splits": {},
    }

    for split, rows in split_to_rows.items():
        rows_arr = np.asarray(rows, dtype=np.int64)
        s2_idx = s2_common_idx[rows_arr]
        s3_idx = s3_common_idx[rows_arr]
        meta = row_df.iloc[rows_arr].copy()

        x_raw = s2_pack["x_raw"][s2_idx].astype(np.float32)
        y_multi_hot = s2_pack["y_multi_hot"][s2_idx].astype(np.float32)
        slots = build_slots(y_multi_hot, args.n_slots)
        x = ((x_raw - mean) / std).astype(np.float32)
        np.savez_compressed(
            out_stage2 / f"{split}.npz",
            x_raw=x_raw,
            x=x,
            y_multi_hot=y_multi_hot,
            slot_targets=slots["slot_targets"],
            slot_mask=slots["slot_mask"],
            set_len=slots["set_len"],
            overflow=slots["overflow"],
        )
        meta.to_csv(out_stage2 / f"{split}_meta.csv", index=False)

        np.savez_compressed(out_stage3 / f"{split}.npz", **{k: v[s3_idx] for k, v in s3_pack.items()})

        summary["splits"][split] = {
            "n_rows": int(len(rows_arr)),
            "method_counts": dict(Counter(meta["reaction_method"].tolist())),
            "stage2_npz": str(out_stage2 / f"{split}.npz"),
            "stage3_npz": str(out_stage3 / f"{split}.npz"),
        }

    for name in ["schema.json", "condition_schema.json"]:
        src = stage3_dir / name
        if src.exists():
            (out_stage3 / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(out_stage2 / "summary.json", summary)
    write_json(out_stage3 / "export_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
