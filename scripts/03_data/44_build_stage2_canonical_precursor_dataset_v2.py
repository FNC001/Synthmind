#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Composition
except Exception:  # pragma: no cover - optional at import time
    Composition = None  # type: ignore


PAD_TOKEN = "<pad>"
PAD_ID = 0
SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
DOT_CHARS = ["▪", "•", "·", "∙", "⋅", "･", "．", "*", "·"]

PHRASE_ALIASES = {
    "aluminum nitrate nonahydrate": "Al(NO3)3·9H2O",
    "aluminium nitrate nonahydrate": "Al(NO3)3·9H2O",
    "cobalt acetate tetrahydrate": "Co(CH3COO)2·4H2O",
    "cobalt(ii) acetate tetrahydrate": "Co(CH3COO)2·4H2O",
    "cobalt acetate": "Co(CH3COO)2",
    "lithium carbonate": "Li2CO3",
    "lithium hydroxide": "LiOH",
    "lithium hydroxide monohydrate": "LiOH·H2O",
    "sodium carbonate": "Na2CO3",
    "potassium carbonate": "K2CO3",
    "calcium carbonate": "CaCO3",
    "barium carbonate": "BaCO3",
    "strontium carbonate": "SrCO3",
    "titanium dioxide": "TiO2",
    "silicon dioxide": "SiO2",
    "iron oxide": "Fe2O3",
    "ferric oxide": "Fe2O3",
    "ferrous oxide": "FeO",
    "ammonium metavanadate": "NH4VO3",
    "ammonium dihydrogen phosphate": "NH4H2PO4",
    "diammonium hydrogen phosphate": "(NH4)2HPO4",
    "ammonium hydroxide": "NH4OH",
    "water": "H2O",
    "deionized water": "H2O",
    "distilled water": "H2O",
    "ethanol": "C2H5OH",
    "methanol": "CH3OH",
    "isopropanol": "(CH3)2CHOH",
    "ammonia": "NH3",
}

TOKEN_ALIASES = {
    "DIwater": "H2O",
    "deionizedH2O": "H2O",
    "EtOH": "C2H5OH",
    "MeOH": "CH3OH",
    "IPA": "(CH3)2CHOH",
}


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def compact_phrase(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def formula_key(part: str) -> str:
    part = part.strip()
    if not part or Composition is None:
        return part
    try:
        comp = Composition(part)
        return comp.alphabetical_formula.replace(" ", "")
    except Exception:
        return part


def canonical_formula_part(part: str) -> str:
    part = part.strip()
    if not part:
        return part
    # Do not replace the precursor label by composition alone: that would merge
    # chemically distinct isomers/salts with the same elemental formula. Pymatgen
    # is used for validation/reporting via formula_key, while the label keeps the
    # explicit precursor spelling after safe text normalization.
    _ = formula_key(part)
    return part


def normalize_hydrate_text(s: str) -> str:
    s = re.sub(r"(?i)H20", "H2O", s)
    s = re.sub(r"(?i)H₂O", "H2O", s)
    s = re.sub(r"(?i)(mono|one)hydrate$", "·H2O", s)
    word_to_n = {
        "di": 2,
        "tri": 3,
        "tetra": 4,
        "penta": 5,
        "hexa": 6,
        "hepta": 7,
        "octa": 8,
        "nona": 9,
        "deca": 10,
    }
    for word, n in word_to_n.items():
        s = re.sub(fr"(?i){word}hydrate$", f"·{n}H2O", s)
    s = re.sub(r"(?i)(?:hydrate|hydrated)$", "", s)
    s = re.sub(r"(?i)(?:mono)?hydrate$", "·H2O", s)
    s = re.sub(r"(?i)[\.-]([0-9]+)H2O$", r"·\1H2O", s)
    s = re.sub(r"(?i)[\.-]H2O$", r"·H2O", s)
    s = re.sub(r"(?i)·1H2O$", r"·H2O", s)
    return s


def strip_suffix_tokens(s: str) -> str:
    s = re.sub(r"(?i)\((?:s|l|g|aq)\)$", "", s)
    s = re.sub(r"(?i)\[(?:s|l|g|aq)\]$", "", s)
    for suffix in ["powder", "anhydrous", "solution", "soln", "aq"]:
        s = re.sub(fr"(?i)(?:[-_,;:]?{suffix})$", "", s)
    return s


def canonical_precursor(name: str) -> str:
    original = str(name or "").strip()
    s = unicodedata.normalize("NFKC", original).strip().translate(SUBSCRIPT)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    phrase = compact_phrase(s)
    if phrase in PHRASE_ALIASES:
        return PHRASE_ALIASES[phrase]

    for ch in DOT_CHARS:
        s = s.replace(ch, "·")
    s = strip_suffix_tokens(s)
    s = normalize_hydrate_text(s)
    s = re.sub(r"\s+", "", s)
    s = TOKEN_ALIASES.get(s, s)
    s = normalize_hydrate_text(s)
    s = strip_suffix_tokens(s)
    s = s.replace("h2o", "H2O").replace("H₂O", "H2O")

    if "·" in s:
        parts = [p for p in s.split("·") if p]
        if not parts:
            return s
        head = canonical_formula_part(parts[0])
        hydrate_parts = []
        for p in parts[1:]:
            p = normalize_hydrate_text(p)
            if re.fullmatch(r"(?i)[0-9]*H2O", p):
                n = re.match(r"([0-9]*)", p).group(1)  # type: ignore[union-attr]
                hydrate_parts.append(("H2O", int(n) if n else 1, f"{n}H2O" if n else "H2O"))
            else:
                hydrate_parts.append((p, 1, canonical_formula_part(p)))
        rendered = []
        for _, n, text in hydrate_parts:
            rendered.append("H2O" if n == 1 and text.upper().endswith("H2O") else text)
        return "·".join([head] + rendered)

    return canonical_formula_part(s)


def build_mapping(precursors: List[str]) -> Tuple[List[str], Dict[int, int], Dict[str, List[str]], pd.DataFrame]:
    canon_to_originals: Dict[str, List[str]] = defaultdict(list)
    for p in precursors:
        canon_to_originals[canonical_precursor(p)].append(str(p))
    canonical_names = sorted(canon_to_originals)
    canon_idx = {p: i for i, p in enumerate(canonical_names)}
    old_to_new = {i: canon_idx[canonical_precursor(p)] for i, p in enumerate(precursors)}
    rows = []
    for canonical, originals in sorted(canon_to_originals.items()):
        for original in sorted(set(originals)):
            rows.append({
                "canonical_precursor": canonical,
                "original_precursor": original,
                "canonical_formula_key": formula_key(canonical.split("·", 1)[0]),
                "original_formula_key": formula_key(str(original).split("·", 1)[0]),
                "n_original_variants_in_group": len(set(originals)),
                "was_merged": len(set(originals)) > 1,
            })
    return canonical_names, old_to_new, dict(canon_to_originals), pd.DataFrame(rows)


def collapse_y(y_old: np.ndarray, old_to_new: Dict[int, int], n_new: int) -> np.ndarray:
    y_new = np.zeros((y_old.shape[0], n_new), dtype=np.float32)
    old_pos = np.where(y_old > 0)
    if len(old_pos[0]):
        new_cols = np.asarray([old_to_new[int(j)] for j in old_pos[1]], dtype=np.int64)
        y_new[old_pos[0], new_cols] = 1.0
    return y_new


def build_slots(y_multi_hot: np.ndarray, n_slots: int) -> Dict[str, np.ndarray]:
    n = y_multi_hot.shape[0]
    slot_targets = np.full((n, n_slots), PAD_ID, dtype=np.int64)
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


def copy_feature_files(input_dir: Path, output_dir: Path) -> None:
    for fname in ["feature_cols.json", "feature_mean.npy", "feature_std.npy"]:
        src = input_dir / fname
        if not src.exists():
            continue
        dst = output_dir / fname
        if src.suffix == ".npy":
            np.save(dst, np.load(src))
        else:
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a stronger canonicalized Stage2 precursor dataset.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_slots", type=int, default=7)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    precursor_names = [str(x) for x in load_json(input_dir / "precursor_names.json")]
    canonical_names, old_to_new, canon_to_originals, alias_df = build_mapping(precursor_names)
    copy_feature_files(input_dir, output_dir)

    label_cols = [f"label_prec__{p}" for p in canonical_names]
    write_json(output_dir / "label_cols.json", label_cols)
    write_json(output_dir / "precursor_names.json", canonical_names)
    write_json(output_dir / "slot_vocab.json", [PAD_TOKEN] + canonical_names)
    write_json(output_dir / "slot_to_id.json", {tok: i for i, tok in enumerate([PAD_TOKEN] + canonical_names)})
    write_json(output_dir / "canonical_to_originals.json", canon_to_originals)
    alias_df.to_csv(output_dir / "precursor_alias_report.csv", index=False)

    summary: Dict[str, Any] = {
        "config": vars(args),
        "n_original_precursors": len(precursor_names),
        "n_canonical_precursors": len(canonical_names),
        "n_merged_labels": len(precursor_names) - len(canonical_names),
        "n_alias_report_rows": int(alias_df.shape[0]),
        "largest_merge_groups": [
            {"canonical": k, "n": len(set(v)), "examples": sorted(set(v))[:25]}
            for k, v in sorted(canon_to_originals.items(), key=lambda kv: len(set(kv[1])), reverse=True)[:40]
            if len(set(v)) > 1
        ],
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        pack = load_npz(input_dir / f"{split}.npz")
        y_old = np.asarray(pack["y_multi_hot"], dtype=np.float32)
        y_new = collapse_y(y_old, old_to_new, len(canonical_names))
        slots = build_slots(y_new, int(args.n_slots))
        out_pack = {
            "x_raw": np.asarray(pack["x_raw"], dtype=np.float32),
            "x": np.asarray(pack["x"], dtype=np.float32),
            "y_multi_hot": y_new,
            "slot_targets": slots["slot_targets"],
            "slot_mask": slots["slot_mask"],
            "set_len": slots["set_len"],
            "overflow": slots["overflow"],
        }
        np.savez_compressed(output_dir / f"{split}.npz", **out_pack)
        meta_path = input_dir / f"{split}_meta.csv"
        if meta_path.exists():
            pd.read_csv(meta_path).to_csv(output_dir / f"{split}_meta.csv", index=False)
        old_lens = y_old.sum(axis=1)
        new_lens = y_new.sum(axis=1)
        summary["splits"][split] = {
            "n_rows": int(y_new.shape[0]),
            "mean_old_set_len": float(old_lens.mean()),
            "mean_new_set_len": float(new_lens.mean()),
            "rows_changed_set_len": int(np.sum(old_lens != new_lens)),
            "max_new_set_len": int(new_lens.max()) if len(new_lens) else 0,
            "overflow_rows": int(slots["overflow"].sum()),
            "label_positive_counts_top": {
                canonical_names[int(i)]: int(v)
                for i, v in sorted(enumerate(y_new.sum(axis=0).astype(int)), key=lambda kv: kv[1], reverse=True)[:25]
            },
        }

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
