#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(obj), f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_builtin(row), ensure_ascii=False) + "\n")


def choose_device(device_arg: str) -> torch.device:
    s = str(device_arg or "").strip()
    if s:
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def import_module_from_path(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_hidden_dims(s: Any) -> List[int]:
    if isinstance(s, (list, tuple)):
        return [int(x) for x in s]
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_list_cell(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None or pd.isna(v):
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
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    if ";" in s:
        return [x.strip() for x in s.split(";") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def strip_label_prefix(x: str) -> str:
    s = str(x).strip()
    if s.startswith("label_prec__"):
        return s[len("label_prec__"):]
    return s


def encode_y_set(labels: Sequence[str], vocab: Sequence[str]) -> np.ndarray:
    index = {strip_label_prefix(v): i for i, v in enumerate(vocab)}
    y = np.zeros((len(vocab),), dtype=np.float32)
    for raw in labels:
        s = strip_label_prefix(str(raw).strip())
        if s in index:
            y[index[s]] = 1.0
    return y


def safe_float_array(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    tmp = df.reindex(columns=list(cols), fill_value=0.0)
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
    return tmp[list(cols)].to_numpy(dtype=np.float32)


def unpack_baseline_continuous(out: Any) -> torch.Tensor:
    """
    Return one continuous prediction tensor [B, C].
    Compatible with dict / tuple / tensor MDN-like outputs.
    """
    if isinstance(out, dict):
        mu = out.get("mu", None)
        if mu is None:
            mu = out.get("means", None)
        if mu is None:
            mu = out.get("cont_mu", None)
        if mu is None:
            mu = out.get("y_cont", None)
        if mu is None:
            mu = out.get("cont_pred", None)
        if mu is None:
            mu = out.get("continuous", None)
        if mu is None:
            raise RuntimeError(f"Cannot find continuous baseline output in keys={list(out.keys())}")

        pi = out.get("pi_logits", None)
        if pi is None:
            pi = out.get("logits", None)
        if pi is None:
            pi = out.get("pi", None)

        if mu.ndim == 3:
            if pi is not None and pi.ndim == 2 and pi.shape[1] == mu.shape[1]:
                idx = torch.argmax(pi, dim=1)
            else:
                idx = torch.zeros((mu.shape[0],), dtype=torch.long, device=mu.device)
            return mu[torch.arange(mu.shape[0], device=mu.device), idx, :]
        if mu.ndim == 2:
            return mu
        raise RuntimeError(f"Unsupported baseline mu shape={tuple(mu.shape)}")

    if isinstance(out, (list, tuple)):
        tensors = [x for x in out if torch.is_tensor(x)]
        if not tensors:
            raise RuntimeError("No tensor in baseline output tuple.")
        mu = None
        for t in tensors:
            if t.ndim == 3:
                mu = t
                break
        if mu is None:
            mu = tensors[0]
        if mu.ndim == 3:
            return mu[:, 0, :]
        if mu.ndim == 2:
            return mu
        raise RuntimeError(f"Unsupported baseline tuple tensor shape={tuple(mu.shape)}")

    if torch.is_tensor(out):
        if out.ndim == 3:
            return out[:, 0, :]
        if out.ndim == 2:
            return out
        raise RuntimeError(f"Unsupported baseline tensor shape={tuple(out.shape)}")

    raise RuntimeError(f"Unsupported baseline output type: {type(out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run precursor-conditioned Stage3 Mixture Flow inference.")
    ap.add_argument("--conditioned_x_csv", required=True)
    ap.add_argument("--schema_json", required=True)
    ap.add_argument("--flow_ckpt", required=True)
    ap.add_argument("--flow_script", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_k_conditions", type=int, default=5)
    ap.add_argument("--n_flow_samples", type=int, default=64)
    ap.add_argument("--device", default="")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    conditioned_x_csv = Path(args.conditioned_x_csv).expanduser().resolve()
    schema_json = Path(args.schema_json).expanduser().resolve()
    flow_ckpt = Path(args.flow_ckpt).expanduser().resolve()
    flow_script = Path(args.flow_script).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    schema = json.load(open(schema_json, "r", encoding="utf-8"))
    feature_cols = list(schema["feature_cols"])
    precursor_vocab = list(schema["precursor_vocab"])

    df = pd.read_csv(conditioned_x_csv)

    x_raw = safe_float_array(df, feature_cols)

    y_rows = []
    for _, row in df.iterrows():
        labels = parse_list_cell(row.get("parent_precursor_set", "[]"))
        y_rows.append(encode_y_set(labels, precursor_vocab))
    y_set = np.stack(y_rows, axis=0).astype(np.float32)

    device = choose_device(args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    ckpt = torch.load(flow_ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    flow_schema = ckpt.get("schema", {}) or {}

    cont_cols = list(flow_schema.get("cont_col_names", [])) or list(schema.get("continuous_schema", {}).keys()) or ["temperature_c", "time_h"]
    disc_class_sizes = list(flow_schema.get("disc_class_sizes", []))

    mod = import_module_from_path(flow_script)

    FlowCls = getattr(mod, "MixtureResidualConditionFlowMixed")
    build_baseline = getattr(mod, "build_baseline_from_checkpoint")

    model = FlowCls(
        x_dim=int(x_raw.shape[1]),
        y_set_dim=int(y_set.shape[1]),
        disc_class_sizes=disc_class_sizes,
        y_cont_dim=len(cont_cols),
        hidden_dims=parse_hidden_dims(cfg.get("hidden_dims", "512,256")),
        flow_hidden_dim=int(cfg.get("flow_hidden_dim", 256)),
        n_flow_layers=int(cfg.get("n_flow_layers", 4)),
        n_components=int(cfg.get("n_components", 5)),
        gating_hidden_dim=int(cfg.get("gating_hidden_dim", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_layernorm=bool(cfg.get("use_layernorm", True)),
        set_proj_dim=int(cfg.get("set_proj_dim", 128)),
        fuse_mode=str(cfg.get("fuse_mode", "concat")),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    baseline_ckpt = Path(ckpt.get("baseline_ckpt") or cfg.get("baseline_ckpt") or "").expanduser()
    if not baseline_ckpt.is_absolute():
        baseline_ckpt = (flow_ckpt.parent / baseline_ckpt).resolve()
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Missing baseline_ckpt: {baseline_ckpt}")

    baseline = build_baseline(
        ckpt_path=baseline_ckpt,
        x_dim=int(x_raw.shape[1]),
        y_set_dim=int(y_set.shape[1]),
        disc_class_sizes=disc_class_sizes,
        y_cont_dim=len(cont_cols),
        device=device,
    )
    baseline.eval()

    xt = torch.tensor(x_raw, dtype=torch.float32, device=device)
    yt = torch.tensor(y_set, dtype=torch.float32, device=device)

    rows = []
    debug_rows = []

    with torch.no_grad():
        base_out = baseline(xt, yt)
        baseline_cont = unpack_baseline_continuous(base_out)  # [B, C]

        out = model(xt, yt)
        context = out["context"]
        gating_logits = out["gating_logits"]
        gating_probs = F.softmax(gating_logits, dim=-1)
        top_comp_prob, top_comp_idx = torch.max(gating_probs, dim=1)

        top1_resid = out["top1_residual"]
        top1_cont = baseline_cont + top1_resid

        # Random residual samples: [S, B, C]
        samples_resid = model.sample_residual(context, int(args.n_flow_samples))
        samples_cont = baseline_cont.unsqueeze(0) + samples_resid

        k = min(int(args.top_k_conditions), int(args.n_flow_samples) + 1)

        for i in range(len(df)):
            parent_set = parse_list_cell(df.iloc[i].get("parent_precursor_set", "[]"))
            parent_rank = int(df.iloc[i].get("parent_precursor_rank", i))
            sample_id = str(df.iloc[i].get("sample_id", ""))
            material_id = str(df.iloc[i].get("material_id", sample_id))
            parent_key = str(df.iloc[i].get("parent_precursor_set_key", ""))

            debug_rows.append({
                "sample_id": sample_id,
                "material_id": material_id,
                "parent_precursor_rank": parent_rank,
                "parent_precursor_set_key": parent_key,
                "parent_precursor_set": json.dumps(parent_set, ensure_ascii=False),
                "baseline_cont": baseline_cont[i].detach().cpu().numpy().tolist(),
                "top_component_index": int(top_comp_idx[i].detach().cpu().item()),
                "top_component_prob": float(top_comp_prob[i].detach().cpu().item()),
                "n_flow_samples": int(args.n_flow_samples),
                "n_conditions_exported": k,
            })

            # First row: deterministic top component mean residual.
            candidate_values = [top1_cont[i].detach().cpu().numpy()]
            candidate_scores = [float(top_comp_prob[i].detach().cpu().item())]
            candidate_src = ["top_component_mean"]

            # Then sampled candidates.
            for s in range(int(args.n_flow_samples)):
                candidate_values.append(samples_cont[s, i].detach().cpu().numpy())
                candidate_scores.append(float(top_comp_prob[i].detach().cpu().item()))
                candidate_src.append("flow_sample")

            # Deduplicate roughly by rounded temperature/time, keep first k.
            seen = set()
            exported = 0
            for vals, score, src in zip(candidate_values, candidate_scores, candidate_src):
                vals = np.asarray(vals, dtype=float)
                key = tuple(np.round(vals, 4).tolist())
                if key in seen:
                    continue
                seen.add(key)

                cont_conditions = {}
                for c_i, c_name in enumerate(cont_cols):
                    if c_i < len(vals):
                        val = float(vals[c_i])
                        if c_name == "temperature_c":
                            val = max(0.0, min(2000.0, val))
                        if c_name == "time_h":
                            val = max(0.0, min(5000.0, val))
                        cont_conditions[c_name] = val

                rows.append({
                    "sample_id": sample_id,
                    "material_id": material_id,
                    "parent_precursor_rank": parent_rank,
                    "parent_precursor_set_key": parent_key,
                    "parent_precursor_set": parent_set,
                    "condition_rank": int(exported),
                    "mixture_index": int(top_comp_idx[i].detach().cpu().item()),
                    "stage3_score": score,
                    "condition_source": src,
                    "cont_conditions": cont_conditions,
                    "stage3_model": "condition_mixture_flow_conditioned_v1",
                })

                exported += 1
                if exported >= k:
                    break

    out_jsonl = output_dir / "test_candidates.jsonl"
    out_csv = output_dir / "test_candidates_flat.csv"
    debug_csv = output_dir / "debug_parent_candidates.csv"
    summary_json = output_dir / "candidate_summary.json"

    write_jsonl(out_jsonl, rows)
    pd.DataFrame(debug_rows).to_csv(debug_csv, index=False)

    flat_rows = []
    for r in rows:
        rr = {k: v for k, v in r.items() if k not in ["cont_conditions", "parent_precursor_set"]}
        rr["parent_precursor_set"] = json.dumps(r["parent_precursor_set"], ensure_ascii=False)
        for k2, v2 in r["cont_conditions"].items():
            rr[k2] = v2
        flat_rows.append(rr)
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    summary = {
        "mode": "stage3_conditioned_mixture_flow_infer",
        "conditioned_x_csv": str(conditioned_x_csv),
        "schema_json": str(schema_json),
        "flow_ckpt": str(flow_ckpt),
        "flow_script": str(flow_script),
        "baseline_ckpt": str(baseline_ckpt),
        "output_dir": str(output_dir),
        "n_input_parent_candidates": int(len(df)),
        "n_output_rows": int(len(rows)),
        "x_dim": int(x_raw.shape[1]),
        "y_set_dim": int(y_set.shape[1]),
        "cont_cols": cont_cols,
        "n_flow_samples": int(args.n_flow_samples),
        "top_k_conditions": int(args.top_k_conditions),
        "device": str(device),
        "artifacts": {
            "test_candidates_jsonl": str(out_jsonl),
            "test_candidates_flat_csv": str(out_csv),
            "debug_parent_candidates_csv": str(debug_csv),
        },
    }
    write_json(summary_json, summary)

    print(f"[DONE] summary -> {summary_json}")
    print(f"[DONE] jsonl   -> {out_jsonl}")
    print(f"[DONE] flat    -> {out_csv}")
    print(f"[DONE] debug   -> {debug_csv}")


if __name__ == "__main__":
    main()
