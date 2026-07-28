#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "base_score",
    "family_score",
    "method_template_score",
    "cooccurrence_score",
    "method_prior_score",
    "mlp_score",
    "retrieval_score",
    "element_coverage",
    "missing_element_count",
    "extra_element_count",
    "candidate_size",
    "open_vocab_score",
    "oov_risk_score",
    "assembly_score",
    "set_size_score",
    "oof_exact_probability",
    "oof_f1_prediction",
    "contains_open_generated_precursor",
    "contains_repair_precursor",
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(obj), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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


def bool_series(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.lower().isin({"true", "1", "1.0", "yes"})


def zscore(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sd = float(vals.std())
    if not math.isfinite(sd) or sd <= 1e-12:
        return vals * 0.0
    return (vals - float(vals.mean())) / sd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        if "id" in out.columns:
            out["sample_id"] = out["id"].astype(str)
        elif "sample_index" in out.columns:
            out["sample_id"] = out["sample_index"].astype(str)
        else:
            raise ValueError("candidate table needs sample_id, id, or sample_index")
    if "candidate_set" not in out.columns:
        out["candidate_set"] = out.get("pred_precursors", out.get("precursor_set", "")).astype(str)
    if "base_score" not in out.columns:
        for col in ["total_score_v5", "calibrated_score", "precursor_score", "rank"]:
            if col in out.columns:
                out["base_score"] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
                if col == "rank":
                    out["base_score"] = -out["base_score"]
                break
        else:
            out["base_score"] = 0.0
    if "label_exact" not in out.columns:
        for col in ["exact", "precursor_exact_if_eval"]:
            if col in out.columns:
                out["label_exact"] = bool_series(out[col]).astype(int)
                break
        else:
            raise ValueError("candidate table needs exact or precursor_exact_if_eval")
    if "jaccard_label" not in out.columns:
        out["jaccard_label"] = pd.to_numeric(
            out.get("jaccard", out.get("precursor_jaccard_if_eval", 0.0)), errors="coerce"
        ).fillna(0.0)
    if "rank" not in out.columns:
        rank_source = pd.to_numeric(out.get("precursor_rank", np.nan), errors="coerce")
        if rank_source.notna().any():
            out["rank"] = rank_source.fillna(999).astype(int)
        else:
            out["rank"] = out.groupby("sample_id").cumcount() + 1
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["candidate_key"] = out["sample_id"].astype(str) + "||" + out["candidate_set"].astype(str)
    return out


def rank_by_score(df: pd.DataFrame, score_col: str, budget: int) -> pd.DataFrame:
    work = df.copy()
    work["_candidate_norm"] = work["candidate_set"].fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
    work = work.sort_values(["sample_id", score_col, "candidate_set"], ascending=[True, False, True], kind="mergesort")
    work = work.drop_duplicates(["sample_id", "_candidate_norm"], keep="first")
    work["rsp_rank_vnext"] = work.groupby("sample_id", sort=False).cumcount() + 1
    return work[work["rsp_rank_vnext"] <= budget].drop(columns=["_candidate_norm"])


def metrics(df: pd.DataFrame, score_col: str, budget: int) -> dict[str, float]:
    ranked = rank_by_score(df, score_col, budget)
    out: dict[str, float] = {
        "n_samples": float(ranked["sample_id"].nunique()),
        "n_candidates": float(len(ranked)),
        "candidate_budget": float(budget),
    }
    for k in [1, 5, 10, 50]:
        sub = ranked[ranked["rsp_rank_vnext"] <= min(k, budget)]
        g = sub.groupby("sample_id", sort=False)
        out[f"precursor_exact@{k}"] = float(g["label_exact"].max().mean()) if len(g) else 0.0
        out[f"precursor_recall@{k}"] = out[f"precursor_exact@{k}"]
        out[f"best_jaccard@{k}"] = float(g["jaccard_label"].max().mean()) if len(g) else 0.0
    top1 = ranked[ranked["rsp_rank_vnext"] <= 1]
    out["precursor_jaccard@1"] = float(top1.groupby("sample_id")["jaccard_label"].max().mean()) if len(top1) else 0.0
    out["skeleton_oracle"] = out[f"precursor_exact@{min(50, budget)}"]
    return out


def build_feature_matrix(train: pd.DataFrame, val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    cols = [c for c in FEATURE_COLUMNS if c in train.columns]
    train_x = train[cols].copy()
    val_x = val[cols].copy()
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).replace(0, 1.0)
    return ((train_x - mean) / std).values.astype("float32"), ((val_x - mean) / std).values.astype("float32"), cols, {"mean": mean.to_dict(), "std": std.to_dict()}


def train_torch_mlp(train_x: np.ndarray, y: np.ndarray, val_x: np.ndarray, seed: int, device_name: str) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    pos = float(y.sum())
    neg = float(len(y) - y.sum())
    pos_weight = torch.tensor([max(1.0, min(50.0, neg / max(pos, 1.0)))], device=device)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 64),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x = torch.tensor(train_x, device=device)
    yt = torch.tensor(y.reshape(-1, 1).astype("float32"), device=device)
    n = x.shape[0]
    batch = 8192
    history: list[float] = []
    for epoch in range(1, 61):
        order = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[idx]), yt[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    with torch.no_grad():
        scores = model(torch.tensor(val_x, device=device)).detach().cpu().numpy().reshape(-1)
    state = {k: v.detach().cpu().numpy().tolist() for k, v in model.state_dict().items()}
    return scores, {"device": str(device), "loss_last": history[-1], "loss_first": history[0], "epochs": len(history), "state_dict_json": state}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train rsp_vnext family-prior reranker on fixed Stage2 candidate pools.")
    parser.add_argument("--train_csv", default="outputs/evaluation/stage2_train_oof_top20_candidates_v4_20260612/train_oof_top20_precursor_candidates.csv")
    parser.add_argument("--val_csv", default="outputs/evaluation/stage2_candidate_pool_v5_20260610/val_candidate_sets_repaired.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow_test_eval", action="store_true", default=False)
    args = parser.parse_args()

    if args.allow_test_eval:
        raise SystemExit("test evaluation is disabled for rsp_vnext validation training")
    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output_dir: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    train = normalize(pd.read_csv(args.train_csv))
    val = normalize(pd.read_csv(args.val_csv))
    val["score_base"] = val["base_score"]
    train["score_base"] = train["base_score"]

    results: dict[str, Any] = {
        "run_id": outdir.name,
        "model_id": "rsp_vnext_family_prior",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "candidate_budget": args.candidate_budget,
        "allow_test_eval": False,
        "selector_update_status": "unchanged",
        "variants": {},
    }
    results["variants"]["base_rsp_v5_score"] = metrics(val, "score_base", args.candidate_budget)

    best = {"name": "base_rsp_v5_score", "metric": results["variants"]["base_rsp_v5_score"].get("precursor_exact@1", 0.0)}
    grid_rows = []
    for wf in np.linspace(-0.5, 2.0, 11):
        for wm in np.linspace(-0.3, 1.2, 7):
            score_col = f"score_family_{wf:.2f}_{wm:.2f}"
            val[score_col] = zscore(val["base_score"]) + wf * zscore(val["family_score"]) + wm * zscore(val["method_prior_score"])
            m = metrics(val, score_col, args.candidate_budget)
            rec = {"variant": score_col, "family_weight": float(wf), "method_prior_weight": float(wm), **m}
            grid_rows.append(rec)
            if (m["precursor_exact@1"], m["precursor_recall@10"], m["skeleton_oracle"]) > (
                best["metric"],
                results["variants"].get(best["name"], {}).get("precursor_recall@10", 0.0),
                results["variants"].get(best["name"], {}).get("skeleton_oracle", 0.0),
            ):
                best = {"name": score_col, "metric": m["precursor_exact@1"]}
                results["variants"]["best_family_prior_blend"] = m | {"family_weight": float(wf), "method_prior_weight": float(wm)}
                val["score_best_family_prior_blend"] = val[score_col]
    pd.DataFrame(grid_rows).to_csv(outdir / "family_prior_grid_val.csv", index=False)
    if "best_family_prior_blend" not in results["variants"]:
        results["variants"]["best_family_prior_blend"] = results["variants"]["base_rsp_v5_score"] | {
            "family_weight": 0.0,
            "method_prior_weight": 0.0,
            "note": "no validation improvement over base",
        }
        val["score_best_family_prior_blend"] = val["score_base"]

    try:
        train_x, val_x, feature_cols, stats = build_feature_matrix(train, val)
        y = train["label_exact"].values.astype("float32")
        mlp_scores, mlp_info = train_torch_mlp(train_x, y, val_x, args.seed, args.device)
        val["score_mlp"] = mlp_scores
        val["score_mlp_blend"] = zscore(val["base_score"]) + 0.75 * zscore(pd.Series(mlp_scores, index=val.index)) + 0.30 * zscore(val["family_score"])
        results["variants"]["mlp_reranker"] = metrics(val, "score_mlp", args.candidate_budget)
        results["variants"]["mlp_base_family_blend"] = metrics(val, "score_mlp_blend", args.candidate_budget)
        write_json(outdir / "feature_standardization.json", {"features": feature_cols, **stats})
        write_json(outdir / "mlp_model_state.json", mlp_info)
    except Exception as exc:
        results["variants"]["mlp_reranker"] = {"status": "failed", "error": repr(exc)}

    ranked = rank_by_score(val, "score_best_family_prior_blend", args.candidate_budget)
    ranked.to_csv(outdir / "val_rsp_vnext_candidates.csv", index=False)
    results["selected_by_validation"] = max(
        (
            (name, vals)
            for name, vals in results["variants"].items()
            if isinstance(vals, dict) and "precursor_exact@1" in vals
        ),
        key=lambda item: (item[1].get("precursor_exact@1", 0.0), item[1].get("precursor_recall@10", 0.0), item[1].get("skeleton_oracle", 0.0)),
    )[0]
    write_json(outdir / "metrics.json", results)
    lines = [
        "# RSP vnext Family-Prior Validation Run",
        "",
        f"- Run id: `{outdir.name}`",
        f"- Train CSV: `{args.train_csv}`",
        f"- Validation CSV: `{args.val_csv}`",
        f"- Candidate budget: {args.candidate_budget}",
        "- Test evaluation: disabled",
        "- Default selector: unchanged",
        "",
        "## Validation Metrics",
        "",
        "| variant | exact@1 | recall@10 | recall@50/oracle | jaccard@1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, vals in results["variants"].items():
        if isinstance(vals, dict) and "precursor_exact@1" in vals:
            lines.append(
                f"| {name} | {vals.get('precursor_exact@1', 0):.4f} | {vals.get('precursor_recall@10', 0):.4f} | {vals.get('skeleton_oracle', 0):.4f} | {vals.get('precursor_jaccard@1', 0):.4f} |"
            )
    lines.extend(["", f"Selected by validation: `{results['selected_by_validation']}`"])
    (outdir / "RSP_VNEXT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(outdir), "selected": results["selected_by_validation"], "variants": results["variants"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
