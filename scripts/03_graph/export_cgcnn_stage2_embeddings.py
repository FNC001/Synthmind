#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def to_tensor(x: Any, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def build_model(
    checkpoint_path: Path,
    device: torch.device,
    model_py: str = "",
    model_class: str = "",
) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        model.eval()
        return model

    if not model_py or not model_class:
        raise ValueError(
            "checkpoint 不是完整模型对象时，请提供 --model_py 和 --model_class。"
        )

    model_py_path = Path(model_py)
    if not model_py_path.exists():
        raise FileNotFoundError(f"model_py 不存在: {model_py_path}")

    spec = importlib.util.spec_from_file_location("user_cgcnn_module", model_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模型文件: {model_py_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, model_class):
        raise AttributeError(f"{model_py_path} 中找不到类: {model_class}")

    ModelClass = getattr(module, model_class)

    model_kwargs = {}
    if isinstance(ckpt, dict) and "model_kwargs" in ckpt and isinstance(ckpt["model_kwargs"], dict):
        model_kwargs = ckpt["model_kwargs"]

    model = ModelClass(**model_kwargs).to(device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        raise RuntimeError("无法从 checkpoint 中解析 state_dict。")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warn] unexpected keys: {len(unexpected)}")

    model.eval()
    return model


def sample_to_model_inputs(sample: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "atomic_numbers": to_tensor(sample["atomic_numbers"], dtype=torch.long, device=device),
        "edge_src": to_tensor(sample["edge_src"], dtype=torch.long, device=device),
        "edge_dst": to_tensor(sample["edge_dst"], dtype=torch.long, device=device),
        "edge_dist": to_tensor(sample["edge_dist"], dtype=torch.float32, device=device),
        "graph_node_slices": [(0, int(np.asarray(sample["atomic_numbers"]).shape[0]))],
    }


def extract_embedding(
    model: nn.Module,
    model_inputs: Dict[str, Any],
    embedding_key: str = "embedding",
) -> np.ndarray:
    with torch.no_grad():
        if hasattr(model, "extract_embedding") and callable(getattr(model, "extract_embedding")):
            emb = model.extract_embedding(**model_inputs)
        else:
            out = model(**model_inputs)
            if isinstance(out, dict):
                if embedding_key not in out:
                    raise KeyError(
                        f"模型返回 dict，但不包含 embedding_key='{embedding_key}'。可用键: {list(out.keys())}"
                    )
                emb = out[embedding_key]
            elif isinstance(out, (tuple, list)):
                if len(out) == 0:
                    raise ValueError("模型输出为空 tuple/list。")
                emb = out[0]
            else:
                emb = out

    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)

    return emb.detach().float().cpu().view(-1).numpy()


def export_one_split(
    split_name: str,
    cache_path: Path,
    output_csv: Path,
    model: nn.Module,
    device: torch.device,
    embedding_key: str,
) -> Dict[str, Any]:
    if not cache_path.exists():
        raise FileNotFoundError(f"找不到 cache 文件: {cache_path}")

    samples = load_pickle(cache_path)
    if not isinstance(samples, list):
        raise ValueError(f"cache 文件内容不是 list: {cache_path}")

    rows: List[Dict[str, Any]] = []
    failed = 0
    emb_dim: Optional[int] = None

    for i, sample in enumerate(samples):
        try:
            model_inputs = sample_to_model_inputs(sample, device=device)
            emb = extract_embedding(model, model_inputs, embedding_key=embedding_key)

            if emb_dim is None:
                emb_dim = int(len(emb))

            row = {
                "id": sample.get("id"),
                "material_id": sample.get("material_id"),
                "formula": sample.get("formula"),
                "doi": sample.get("doi"),
                "split_group": sample.get("split_group"),
            }
            for j, v in enumerate(emb.tolist()):
                row[f"graph_emb_{j}"] = float(v)
            rows.append(row)
        except Exception as e:
            failed += 1
            print(f"[Warn] split={split_name} sample_idx={i} id={sample.get('id')} failed: {repr(e)}")

    df = pd.DataFrame(rows)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)

    return {
        "input_cache": str(cache_path),
        "output_csv": str(output_csv),
        "input_samples": int(len(samples)),
        "exported_rows": int(len(df)),
        "failed_rows": int(failed),
        "embedding_dim": int(emb_dim or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export stage2 CGCNN graph embeddings from cached pkl splits.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory containing train.pkl / val.pkl / test.pkl / gold_train_holdout.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save stage2_*_graph_embed.csv")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained CGCNN checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu / cuda / cuda:0 / mps")
    parser.add_argument("--task_prefix", type=str, default="stage2", help="Output prefix, default stage2")
    parser.add_argument("--embedding_key", type=str, default="embedding", help="If model returns dict, which key to use as embedding")
    parser.add_argument("--model_py", type=str, default="", help="Path to model definition file when checkpoint stores only state_dict")
    parser.add_argument("--model_class", type=str, default="", help="Model class name when checkpoint stores only state_dict")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    checkpoint = Path(args.checkpoint)

    if not cache_dir.exists():
        raise FileNotFoundError(f"cache_dir 不存在: {cache_dir}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")

    ensure_dir(output_dir)
    device = torch.device(args.device)

    model = build_model(
        checkpoint_path=checkpoint,
        device=device,
        model_py=args.model_py,
        model_class=args.model_class,
    )

    summary: Dict[str, Any] = {
        "config": {
            "cache_dir": str(cache_dir),
            "output_dir": str(output_dir),
            "checkpoint": str(checkpoint),
            "device": str(device),
            "task_prefix": args.task_prefix,
            "embedding_key": args.embedding_key,
            "model_py": args.model_py,
            "model_class": args.model_class,
        },
        "splits": {},
    }

    for split_name in ["train", "val", "test", "gold_train_holdout"]:
        cache_path = cache_dir / f"{split_name}.pkl"
        output_csv = output_dir / f"{args.task_prefix}_{split_name}_graph_embed.csv"
        summary["splits"][split_name] = export_one_split(
            split_name=split_name,
            cache_path=cache_path,
            output_csv=output_csv,
            model=model,
            device=device,
            embedding_key=args.embedding_key,
        )

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
