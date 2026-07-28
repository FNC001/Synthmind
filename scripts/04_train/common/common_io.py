#!/usr/bin/env python3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# A required/optional spec can be:
# - list[str]: each filename required as-is
# - dict[str, str]: key -> filename
# - dict[str, list[str]]: key -> alternative filenames, any one is acceptable
RequiredSpec = Union[Sequence[str], Mapping[str, Union[str, Sequence[str]]]]
OptionalSpec = Union[Sequence[str], Mapping[str, Union[str, Sequence[str]]], None]


@dataclass
class ResolvedInput:
    resolved_mode: str
    resolved_root: str
    resolved_input_dir: str
    files: Dict[str, str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_numpy_item(arr: np.ndarray) -> Any:
    try:
        return arr.item()
    except Exception:
        return arr


def _normalize_npz_value(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim == 0:
            return _safe_numpy_item(value)
        return value
    return value


def load_npz(path: Union[str, Path], allow_pickle: bool = True) -> Dict[str, Any]:
    with np.load(path, allow_pickle=allow_pickle) as arr:
        out: Dict[str, Any] = {}
        for k in arr.files:
            out[k] = _normalize_npz_value(arr[k])
        return out


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_hidden_dims(s: Union[str, Sequence[int], None]) -> List[int]:
    if s is None:
        return []
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    s = str(s).strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _first_existing(candidates: Sequence[Path], what: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"未找到 {what}，候选路径为：\n" + "\n".join(str(x) for x in candidates))


def resolve_mode_dir(root: Path, train_mode: str, mapping: Optional[Mapping[str, Sequence[Path]]] = None) -> Path:
    if mapping is None:
        mapping = {
            "curriculum_phase1": [root / "curriculum_phase1", root / "curriculum" / "phase1", root / "curriculum" / "phase1_train"],
            "curriculum_phase2": [root / "curriculum_phase2", root / "curriculum" / "phase2", root / "curriculum" / "phase2_train"],
            "curriculum": [root / "curriculum"],
            "relaxed_only": [root / "relaxed_only"],
            "gold_only": [root / "gold_only"],
        }
    candidates = list(mapping.get(train_mode, [root / train_mode]))
    return _first_existing(candidates, f"train_mode={train_mode} 对应的数据目录")


def _filename_to_key(name: str) -> str:
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    key = stem
    if suffix == ".npz":
        key = f"{stem}_npz"
    elif suffix == ".csv":
        key = f"{stem}_csv"
    elif suffix == ".json":
        key = stem
    return key


def _normalize_required_spec(spec: RequiredSpec) -> Dict[str, List[str]]:
    if isinstance(spec, Mapping):
        out: Dict[str, List[str]] = {}
        for k, v in spec.items():
            if isinstance(v, (list, tuple)):
                out[str(k)] = [str(x) for x in v]
            else:
                out[str(k)] = [str(v)]
        return out

    if isinstance(spec, (list, tuple)):
        out: Dict[str, List[str]] = {}
        for name in spec:
            name = str(name)
            out[_filename_to_key(name)] = [name]
        return out

    raise TypeError(f"Unsupported required spec type: {type(spec)}")


def _normalize_optional_spec(spec: OptionalSpec) -> Dict[str, List[str]]:
    if spec is None:
        return {}
    if isinstance(spec, Mapping):
        out: Dict[str, List[str]] = {}
        for k, v in spec.items():
            if isinstance(v, (list, tuple)):
                out[str(k)] = [str(x) for x in v]
            else:
                out[str(k)] = [str(v)]
        return out

    if isinstance(spec, (list, tuple)):
        out: Dict[str, List[str]] = {}
        for name in spec:
            name = str(name)
            out[_filename_to_key(name)] = [name]
        return out

    raise TypeError(f"Unsupported optional spec type: {type(spec)}")


def validate_input_dir(input_dir: Path, required: RequiredSpec, optional: OptionalSpec = None) -> Dict[str, str]:
    required_norm = _normalize_required_spec(required)
    optional_norm = _normalize_optional_spec(optional)
    files: Dict[str, str] = {}
    missing: List[str] = []

    for k, rel_candidates in required_norm.items():
        found = None
        for rel in rel_candidates:
            p = input_dir / rel
            if p.exists():
                found = p
                break
        if found is not None:
            files[k] = str(found)
        else:
            missing.append(str(input_dir / str(rel_candidates)))

    if missing:
        raise FileNotFoundError("输入目录缺少必需文件：\n" + "\n".join(missing))

    for k, rel_candidates in optional_norm.items():
        for rel in rel_candidates:
            p = input_dir / rel
            if p.exists():
                files[k] = str(p)
                break

    return files


def resolve_input_paths(args: Any, required: RequiredSpec, optional: OptionalSpec = None, mode_mapping: Optional[Mapping[str, Sequence[Path]]] = None) -> ResolvedInput:
    if getattr(args, "input_dir", ""):
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists():
            raise FileNotFoundError(f"--input_dir 不存在: {input_dir}")
        files = validate_input_dir(input_dir, required=required, optional=optional)
        return ResolvedInput("legacy_input_dir", str(input_dir), str(input_dir), files)

    if not getattr(args, "mode_input_root", ""):
        raise ValueError("必须提供 --input_dir 或 --mode_input_root")

    root = Path(args.mode_input_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"--mode_input_root 不存在: {root}")

    train_mode = getattr(args, "train_mode", "relaxed_only")

    try:
        mode_dir = resolve_mode_dir(root, train_mode, mapping=mode_mapping)
        files = validate_input_dir(mode_dir, required=required, optional=optional)
        return ResolvedInput(train_mode, str(root), str(mode_dir), files)
    except FileNotFoundError:
        files = validate_input_dir(root, required=required, optional=optional)
        return ResolvedInput(f"{train_mode} (fallback_root)", str(root), str(root), files)


def load_summary_schema(files: Mapping[str, str]) -> Dict[str, Any]:
    if "summary" in files:
        obj = load_json(files["summary"])
        schema = obj.get("schema", obj)
        return {"summary": obj, "schema": schema}
    if "schema" in files:
        obj = load_json(files["schema"])
        return {"summary": obj, "schema": obj}
    raise KeyError("files 中未找到 summary 或 schema")


STAGE2_BUNDLE_REQUIRED = {
    "train_npz": "train.npz",
    "val_npz": "val.npz",
    "test_npz": "test.npz",
    "train_meta_csv": "train_meta.csv",
    "val_meta_csv": "val_meta.csv",
    "test_meta_csv": "test_meta.csv",
    "summary": "summary.json",
}

STAGE2_BUNDLE_OPTIONAL = {
    "action_vocab": "action_vocab.json",
    "action_to_id": "action_to_id.json",
    "precursor_names": "precursor_names.json",
    "label_cols": "label_cols.json",
    "label_names": "label_names.json",
    "schema": "schema.json",
}


STAGE3_TABLE_REQUIRED_MINIMAL = {
    "train_npz": "train.npz",
    "val_npz": "val.npz",
    "test_npz": "test.npz",
}

STAGE3_TABLE_OPTIONAL = {
    "train_meta_csv": "train_meta.csv",
    "val_meta_csv": "val_meta.csv",
    "test_meta_csv": "test_meta.csv",
    "summary": "summary.json",
    "schema": "schema.json",
    "condition_schema": "condition_schema.json",
}


def load_stage2_bundle(files: Mapping[str, str]) -> Dict[str, Any]:
    summary_pack = load_summary_schema(files)
    return {
        "train_pack": load_npz(files["train_npz"]),
        "val_pack": load_npz(files["val_npz"]),
        "test_pack": load_npz(files["test_npz"]),
        "train_meta": load_csv(files["train_meta_csv"]),
        "val_meta": load_csv(files["val_meta_csv"]),
        "test_meta": load_csv(files["test_meta_csv"]),
        "summary": summary_pack["summary"],
        "schema": summary_pack["schema"],
        "action_vocab": load_json(files["action_vocab"]) if "action_vocab" in files else None,
        "action_to_id": load_json(files["action_to_id"]) if "action_to_id" in files else None,
        "precursor_names": load_json(files["precursor_names"]) if "precursor_names" in files else None,
        "label_cols": load_json(files["label_cols"]) if "label_cols" in files else None,
        "label_names": load_json(files["label_names"]) if "label_names" in files else None,
    }


def resolve_match_cols(meta_df: pd.DataFrame, preferred_cols: Optional[Sequence[str]] = None) -> List[str]:
    default_cols = ["sample_index", "row_id", "sample_id", "material_id", "entry_id", "reaction_id", "id", "synth_uid", "record_index"]
    cols = list(preferred_cols) if preferred_cols is not None else default_cols
    return [c for c in cols if c in meta_df.columns]


def attach_sample_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_index" not in out.columns:
        out["sample_index"] = np.arange(len(out))
    return out
