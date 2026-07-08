#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


CORE_METHODS = {"solid_state", "solution", "melt_arc"}
METHOD_TEMPLATE_LIMIT = {"solid_state": 420, "solution": 420, "melt_arc": 520}
METHOD_ASSEMBLY_LIMIT = {"solid_state": 450, "solution": 380, "melt_arc": 260}


def load_v5_module(repo_root: Path):
    path = repo_root / "scripts/06_eval/generate_stage2_candidate_pool_v5.py"
    spec = importlib.util.spec_from_file_location("stage2_v5_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Core model should not inherit weak-method expansion bonus.
    mod.WEAK_METHODS = set()
    return mod


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(x) for x in obj]
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


def source_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out = {}
    for src, sub in df.groupby("candidate_source"):
        out[str(src)] = {
            "n_candidates": int(len(sub)),
            "exact_rows": int(sub["exact"].sum()),
            "unique_samples_with_exact": int(sub[sub["exact"]]["sample_index"].nunique()),
        }
    return out


def by_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("reaction_method", dropna=False):
        m = {}
        for k in [1, 3, 5, 10, 50, 100, 200, 500]:
            s = sub[sub["rank"] <= k]
            g = s.groupby("sample_index", sort=False)
            m[f"top{k}_exact"] = float(g["exact"].any().mean()) if len(g) else 0.0
        rows.append({"reaction_method": method, "n_samples": int(sub["sample_index"].nunique()), **m})
    return pd.DataFrame(rows).sort_values("n_samples", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Stage2 core-method candidate pool for solid_state, solution, and melt_arc.")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--template_csv", required=True)
    ap.add_argument("--base_candidate_csv", required=True, help="All-method v5 candidate CSV to keep as base candidates.")
    ap.add_argument("--patch_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--top_n", type=int, default=500)
    args = ap.parse_args()

    repo_root = Path.cwd().resolve()
    v5 = load_v5_module(repo_root)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    patch = v5.load_patch(args.patch_csv)
    names = [str(x) for x in v5.load_json(dataset_dir / "precursor_names.json")]
    meta = pd.read_csv(dataset_dir / f"{args.split}_meta.csv")
    bad_methods = sorted(set(meta["reaction_method"].astype(str)) - CORE_METHODS)
    if bad_methods:
        raise ValueError(f"Non-core methods found in core dataset: {bad_methods}")
    true_sets = v5.y_true_sets(dataset_dir, args.split, names, patch)
    ont = pd.read_csv(args.ontology_csv)
    family_lookup = dict(zip(ont["canonical_precursor"].astype(str), ont["precursor_family"].astype(str)))
    templates_by_method = v5.load_templates(Path(args.template_csv), patch)
    label_index = v5.build_label_index(dataset_dir, names, family_lookup)
    base = pd.read_csv(args.base_candidate_csv)
    base_groups = {int(i): g.copy() for i, g in base.groupby("sample_index", sort=False)}
    all_rows = []

    for i in range(len(meta)):
        formula = str(meta.loc[i, "formula"])
        method = str(meta.loc[i, "reaction_method"])
        original_i = int(meta.loc[i, "original_sample_index"]) if "original_sample_index" in meta.columns else i
        true = true_sets[i]
        store: Dict[frozenset, Dict[str, Any]] = {}

        group = base_groups.get(original_i, pd.DataFrame())
        for _, r in group.iterrows():
            labels = v5.patch_set(v5.parse_list(r["pred_precursors"]), patch)
            orig = float(r.get("total_score_v5", r.get("calibrated_score_v5", r.get("calibrated_score", 0.0))))
            feats = {
                "original_v4_score": orig,
                "family_score": float(r.get("family_score", v5.family_score(labels, method, family_lookup)) if not pd.isna(r.get("family_score", 0.0)) else v5.family_score(labels, method, family_lookup)),
                "open_vocab_score": 0.05 * float(r.get("open_vocab_count", 0.0) if "open_vocab_count" in r else 0.0),
                "oov_risk_score": 0.05,
                "assembly_score": float(r.get("assembly_score", 0.10) if not pd.isna(r.get("assembly_score", 0.10)) else 0.10),
                "set_size_score": float(r.get("set_size_score", 1.0 / (1.0 + abs(len(labels) - max(1, len(v5.target_elements(formula))))))),
                "cooccurrence_score": float(r.get("cooccurrence_score", r.get("retrieval_score", 0.0)) if not pd.isna(r.get("cooccurrence_score", 0.0)) else 0.0),
                "method_prior_score": v5.method_prior_score(method, labels, family_lookup),
                "mlp_score": float(r.get("mlp_score", 0.0) if not pd.isna(r.get("mlp_score", 0.0)) else 0.0),
                "retrieval_score": float(r.get("retrieval_score", 0.0) if not pd.isna(r.get("retrieval_score", 0.0)) else 0.0),
            }
            src = str(r.get("candidate_source", "v5_base"))
            v5.add_candidate(store, labels, formula, method, f"core_base_{src}", str(r.get("method_template_id", "")), feats)

        scored_templates = []
        for tpl in templates_by_method.get(method, []):
            sc = v5.template_score(tpl, formula, method)
            if method == "melt_arc" and set(tpl["precursor_families"]) & {"elemental"}:
                sc += 0.35
            if method == "solid_state" and set(tpl["precursor_families"]) & {"oxide", "carbonate", "hydroxide"}:
                sc += 0.20
            if method == "solution" and set(tpl["precursor_families"]) & {"nitrate", "acetate", "halide", "organic"}:
                sc += 0.20
            if sc > 1.15:
                scored_templates.append((sc, tpl))
        scored_templates.sort(key=lambda x: x[0], reverse=True)
        for sc, tpl in scored_templates[: METHOD_TEMPLATE_LIMIT.get(method, 300)]:
            labels = tpl["labels"]
            feats = {
                "method_template_score": sc,
                "family_score": v5.family_score(labels, method, family_lookup),
                "open_vocab_score": 0.0,
                "oov_risk_score": 0.05,
                "assembly_score": 0.38,
                "set_size_score": 1.0 / (1.0 + abs(len(labels) - tpl.get("average_set_size", len(labels)))),
                "cooccurrence_score": float(tpl.get("cooccurrence_score", 0.0)),
                "method_prior_score": v5.method_prior_score(method, labels, family_lookup),
            }
            v5.add_candidate(store, labels, formula, method, "core_method_template", tpl["template_id"], feats)

        preferred = list(v5.METHOD_FAMILY_PRIORS.get(method, {"oxide", "carbonate", "nitrate", "halide"}))
        per_elem: List[List[tuple[str, float]]] = []
        for e in v5.target_elements(formula):
            opts: List[tuple[str, float]] = []
            for fam in preferred:
                n_keep = 5 if method != "melt_arc" else 4
                for lab, cnt in label_index.get(e, {}).get(fam, [])[:n_keep]:
                    opts.append((lab, math.log1p(cnt) + 0.35))
            opts = sorted(dict(opts).items(), key=lambda x: x[1], reverse=True)[:5]
            if opts:
                per_elem.append(opts)
        combo_count = 0
        for combo in __import__("itertools").product(*per_elem) if per_elem else []:
            labels = [x[0] for x in combo]
            score = float(np.mean([x[1] for x in combo])) if combo else 0.0
            feats = {
                "method_template_score": 0.25,
                "family_score": v5.family_score(labels, method, family_lookup),
                "open_vocab_score": 0.0,
                "oov_risk_score": 0.05,
                "assembly_score": 0.45,
                "set_size_score": 1.0 / (1.0 + abs(len(labels) - len(v5.target_elements(formula)))),
                "cooccurrence_score": score,
                "method_prior_score": v5.method_prior_score(method, labels, family_lookup),
            }
            v5.add_candidate(store, labels, formula, method, "core_train_label_assembly", "", feats)
            combo_count += 1
            if combo_count >= METHOD_ASSEMBLY_LIMIT.get(method, 300):
                break

        rows = list(store.values())
        rows.sort(key=lambda x: x["total_score_v5"], reverse=True)
        for rank, cand in enumerate(rows[: int(args.top_n)], start=1):
            labels = cand.pop("labels")
            sm = v5.set_metrics(true, set(labels))
            all_rows.append({
                "sample_index": i,
                "original_sample_index": original_i,
                "rank": rank,
                "id": meta.loc[i, "id"],
                "formula": formula,
                "reaction_method": method,
                "true_precursors": json.dumps(sorted(true), ensure_ascii=False),
                "pred_precursors": json.dumps(labels, ensure_ascii=False),
                "candidate_set": json.dumps(labels, ensure_ascii=False),
                **cand,
                **sm,
            })
        if (i + 1) % 500 == 0:
            print(f"[Progress] {args.split} {i+1}/{len(meta)}", flush=True)

    df = pd.DataFrame(all_rows).sort_values(["sample_index", "total_score_v5"], ascending=[True, False], kind="mergesort")
    df["rank"] = df.groupby("sample_index", sort=False).cumcount() + 1
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("none").replace("", "none")
    out_csv = out_dir / f"{args.split}_core_candidate_sets.csv"
    df.to_csv(out_csv, index=False)
    method_df = by_method_metrics(df)
    method_df.to_csv(out_dir / f"{args.split}_core_by_reaction_method.csv", index=False)
    summary = {
        "config": vars(args),
        "data": {"n_rows": int(len(meta)), "n_candidates": int(len(df))},
        "metrics": v5.metrics_by_k(df),
        "by_reaction_method": method_df.to_dict(orient="records"),
        "candidate_source_contribution": source_metrics(df),
        "artifacts": {"candidate_csv": str(out_csv), "summary": str(out_dir / f"{args.split}_core_candidate_pool_summary.json")},
    }
    write_json(out_dir / f"{args.split}_core_candidate_pool_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
