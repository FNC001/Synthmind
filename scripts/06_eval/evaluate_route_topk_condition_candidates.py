#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

import joblib
import numpy as np
import pandas as pd


def load_base_module(project_root: Path) -> Any:
    path = project_root / "scripts/06_eval/evaluate_route_top10_with_setsize_rerank.py"
    spec = importlib.util.spec_from_file_location("route_top10_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base evaluator from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def raw_continuous(values_norm: np.ndarray, schema: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    out = values_norm.astype(np.float32).copy()
    cont_schema = schema.get("continuous_schema", {}) or {}
    for j, name in enumerate(names):
        stats = cont_schema.get(name, {}) or {}
        out[:, j] = out[:, j] * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))
    return out


def parse_json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def make_yset(label_sets: Sequence[Set[str]], vocab: Sequence[str]) -> np.ndarray:
    idx = {str(v): i for i, v in enumerate(vocab)}
    y = np.zeros((len(label_sets), len(vocab)), dtype=np.float32)
    for i, labels in enumerate(label_sets):
        for lab in labels:
            j = idx.get(str(lab))
            if j is not None:
                y[i, j] = 1.0
    return y


def build_stage2_candidates(
    base: Any,
    stage2_dir: Path,
    stage2_run_dir: Path,
    set_size_model_path: Path,
    split: str,
    top_n_precursors: int,
    max_set_size: int,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    s2_pack = base.load_npz(stage2_dir / f"{split}.npz")
    s2_meta = pd.read_csv(stage2_dir / f"{split}_meta.csv")
    x2 = base.get_x(s2_pack)
    y2 = base.get_y(s2_pack)

    stage2_model, precursor_names = base.load_stage2_model(stage2_run_dir, x2.shape[1])
    probs = base.predict_probs(stage2_model, x2, batch_size)
    true_sets = [{precursor_names[j] for j in np.where(y2[i] > 0)[0]} for i in range(y2.shape[0])]

    size_pack = joblib.load(set_size_model_path)
    size_model = size_pack["model"] if isinstance(size_pack, dict) and "model" in size_pack else size_pack
    size_pred = size_model.predict(x2).astype(int)
    size_proba = size_model.predict_proba(x2)
    size_classes = [int(x) for x in size_model.classes_.tolist()]

    rows = []
    for i in range(len(x2)):
        prob_by_k = {k: float(size_proba[i, j]) for j, k in enumerate(size_classes)}
        cands = base.generate_candidates(
            probs=probs[i],
            names=precursor_names,
            formula=str(s2_meta.loc[i, "formula"]),
            pred_size=int(size_pred[i]),
            size_prob_by_k=prob_by_k,
            max_size=int(max_set_size),
            keep=int(top_n_precursors),
        )
        true = true_sets[i]
        for rank, cand in enumerate(cands, start=1):
            sm = base.set_metrics(true, cand["label_set"])
            rows.append({
                "sample_index": i,
                "precursor_rank": rank,
                "id": s2_meta.loc[i, "id"],
                "material_id": s2_meta.loc[i, "material_id"],
                "formula": s2_meta.loc[i, "formula"],
                "true_precursors": json.dumps(sorted(true), ensure_ascii=False),
                "pred_precursors": json.dumps(cand["labels"], ensure_ascii=False),
                "precursor_score": cand["score"],
                "prob_score": cand["prob_score"],
                "coverage": cand["coverage"],
                "extra": cand["extra"],
                "predicted_size": int(size_pred[i]),
                "candidate_size": int(cand["n_labels"]),
                **sm,
            })
    return pd.DataFrame(rows), x2, y2


def predict_moe_with_disc_proba(model_path: Path, X_stage3: np.ndarray, sample_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    pack = joblib.load(model_path)
    schema = pack["schema"]
    cont_names = [str(x) for x in pack.get("cont_names", schema["continuous_cols"])]
    disc_names = [str(x) for x in pack.get("disc_names", schema["discrete_cols"])]
    disc_schema = schema.get("discrete_schema", {}) or {}
    global_models = pack["global_models"]
    experts = pack.get("experts", {}) or {}
    id_to_method = pack.get("id_to_method", {}) or {}
    methods = np.asarray([str(id_to_method.get(str(sid), "other")) for sid in sample_ids])

    y_cont = np.zeros((X_stage3.shape[0], len(cont_names)), dtype=np.float32)
    y_disc = np.zeros((X_stage3.shape[0], len(disc_names)), dtype=np.int64)
    prob_by_name: Dict[str, np.ndarray] = {}

    for method in sorted(set(methods.tolist())):
        idx = np.where(methods == method)[0]
        model_set = experts.get(method, {}) or {}
        for j, name in enumerate(cont_names):
            key = "target_time_h_log1p" if name == "target_time_h" else name
            model = model_set.get(key) or global_models.get(key)
            if model is None:
                continue
            pred = np.asarray(model.predict(X_stage3[idx], num_iteration=model.best_iteration), dtype=np.float32)
            y_cont[idx, j] = np.expm1(pred) if key.endswith("_log1p") else pred
        for j, name in enumerate(disc_names):
            model = model_set.get(name) or global_models.get(name)
            if model is None:
                missing = int((disc_schema.get(name, {}) or {}).get("missing_index", 0))
                y_disc[idx, j] = missing
                continue
            prob = np.asarray(model.predict(X_stage3[idx], num_iteration=model.best_iteration), dtype=np.float32)
            if name not in prob_by_name:
                prob_by_name[name] = np.zeros((X_stage3.shape[0], prob.shape[1]), dtype=np.float32)
            prob_by_name[name][idx] = prob
            y_disc[idx, j] = np.asarray(np.argmax(prob, axis=1), dtype=np.int64)
    return y_cont, y_disc, prob_by_name


def train_retrieval_cache(stage3_dir: Path, schema: Mapping[str, Any], model_path: Path) -> Dict[str, Any]:
    pack = load_npz(stage3_dir / "train.npz")
    model_pack = joblib.load(model_path)
    id_to_method = model_pack.get("id_to_method", {}) or {}
    ids = np.asarray([str(x) for x in pack["sample_id"]])
    methods = np.asarray([str(id_to_method.get(str(x), "other")) for x in ids])
    cont_names = [str(x) for x in schema["continuous_cols"]]
    y_cont_raw = raw_continuous(np.asarray(pack["y_cond_continuous"], dtype=np.float32), schema, cont_names)
    y_cont_mask = np.asarray(pack["y_cond_continuous_mask"], dtype=np.float32)
    y_disc = np.asarray(pack["y_cond_discrete"])
    y_disc_mask = np.asarray(pack["y_cond_discrete_mask"], dtype=np.float32)
    y_set = np.asarray(pack["y_set"], dtype=np.float32)
    set_size = y_set.sum(axis=1).astype(np.float32)
    valid = (y_cont_mask[:, 0] > 0.5) & (y_cont_mask[:, 1] > 0.5) & (y_disc_mask[:, 0] > 0.5)
    method_to_idx: Dict[str, np.ndarray] = {}
    for method in sorted(set(methods.tolist())):
        idx = np.where((methods == method) & valid)[0]
        if idx.size:
            method_to_idx[method] = idx
    inverted: Dict[str, Dict[int, List[int]]] = {}
    global_inverted: Dict[int, List[int]] = {}
    for idx in np.where(valid)[0].tolist():
        method = str(methods[idx])
        active = np.where(y_set[idx] > 0.5)[0].tolist()
        method_map = inverted.setdefault(method, {})
        for j in active:
            method_map.setdefault(int(j), []).append(int(idx))
            global_inverted.setdefault(int(j), []).append(int(idx))
    all_valid = np.where(valid)[0]
    return {
        "ids": ids,
        "methods": methods,
        "y_set": y_set,
        "set_size": set_size,
        "y_cont_raw": y_cont_raw,
        "y_disc": y_disc,
        "method_to_idx": method_to_idx,
        "inverted": inverted,
        "global_inverted": global_inverted,
        "all_valid": all_valid,
    }


def retrieve_templates(
    cache: Mapping[str, Any],
    method: str,
    yset_row: np.ndarray,
    k: int,
    min_similarity: float,
) -> List[Dict[str, Any]]:
    active = np.where(yset_row > 0.5)[0]
    cand_size = float(len(active))
    if cand_size <= 0:
        return []

    counts: Dict[int, int] = {}
    inv = cache["inverted"].get(method, {})
    if not inv:
        inv = cache["global_inverted"]
    for j in active.tolist():
        for idx in inv.get(int(j), []):
            counts[idx] = counts.get(idx, 0) + 1

    if not counts:
        pool = cache["method_to_idx"].get(method)
        if pool is None or len(pool) == 0:
            pool = cache["all_valid"]
        rows = np.asarray(pool[: max(k * 3, k)], dtype=np.int64)
        scores = np.zeros(len(rows), dtype=np.float32)
    else:
        rows = np.asarray(list(counts.keys()), dtype=np.int64)
        overlap = np.asarray([counts[int(idx)] for idx in rows], dtype=np.float32)
        denom = cand_size + cache["set_size"][rows] - overlap
        scores = np.divide(overlap, np.clip(denom, 1.0, None)).astype(np.float32)
    order = np.argsort(-scores)
    out = []
    seen = set()
    for pos in order:
        score = float(scores[pos])
        if score < min_similarity and out:
            break
        idx = int(rows[pos])
        key = (
            round(float(cache["y_cont_raw"][idx, 0]), 3),
            round(float(cache["y_cont_raw"][idx, 1]), 3),
            int(cache["y_disc"][idx, 0]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": "retrieval_template",
            "condition_score": score,
            "template_id": str(cache["ids"][idx]),
            "pred_cont": cache["y_cont_raw"][idx].astype(np.float32),
            "pred_disc": cache["y_disc"][idx].astype(np.int64),
        })
        if len(out) >= k:
            break
    return out


def condition_success(
    true_cont_raw: np.ndarray,
    pred_cont_raw: np.ndarray,
    cont_mask: np.ndarray,
    true_disc: np.ndarray,
    pred_disc: np.ndarray,
    disc_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    temp_valid = cont_mask[:, 0] > 0.5
    time_valid = cont_mask[:, 1] > 0.5
    atm_valid = disc_mask[:, 0] > 0.5
    evaluable = temp_valid & time_valid & atm_valid
    temp_err = np.abs(pred_cont_raw[:, 0] - true_cont_raw[:, 0])
    time_err = np.abs(pred_cont_raw[:, 1] - true_cont_raw[:, 1])
    atm_ok = pred_disc[:, 0] == true_disc[:, 0]
    return {
        "temp_err": temp_err,
        "time_err": time_err,
        "atm_ok": atm_ok,
        "cond100_24_atm": evaluable & (temp_err <= 100.0) & (time_err <= 24.0) & atm_ok,
        "cond200_48_atm": evaluable & (temp_err <= 200.0) & (time_err <= 48.0) & atm_ok,
    }


def compute_metrics(df: pd.DataFrame, route_top_k: int) -> Dict[str, float]:
    ordered = df.sort_values(["sample_index", "route_score"], ascending=[True, False]).copy()
    ordered["route_rank"] = ordered.groupby("sample_index").cumcount() + 1
    top1 = ordered[ordered["route_rank"] == 1]
    topk = ordered[ordered["route_rank"] <= int(route_top_k)]
    grouped_topk = topk.groupby("sample_index", sort=False)
    precursor_grouped = df.groupby("sample_index", sort=False)
    return {
        "top1_precursor_exact": float(top1["exact"].mean()),
        "top1_precursor_mean_f1": float(top1["f1"].mean()),
        "top1_precursor_mean_jaccard": float(top1["jaccard"].mean()),
        "top1_condition_100c_24h_atm": float(top1["condition_100c_24h_atm"].mean()),
        "top1_condition_200c_48h_atm": float(top1["condition_200c_48h_atm"].mean()),
        "top1_strict_route_success": float(top1["strict_route_success"].mean()),
        "top1_relaxed_route_success": float(top1["relaxed_route_success"].mean()),
        "topk_precursor_exact_recall": float(grouped_topk["exact"].any().mean()),
        "topk_precursor_jaccard50_recall": float(grouped_topk["jaccard"].max().ge(0.5).mean()),
        "topk_strict_route_oracle_success": float(grouped_topk["strict_route_success"].any().mean()),
        "topk_relaxed_route_oracle_success": float(grouped_topk["relaxed_route_success"].any().mean()),
        "all_precursor_exact_recall": float(precursor_grouped["exact"].any().mean()),
        "all_strict_route_oracle_success": float(precursor_grouped["strict_route_success"].any().mean()),
        "all_relaxed_route_oracle_success": float(precursor_grouped["relaxed_route_success"].any().mean()),
        "n": int(top1.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate route candidates with multiple Stage3 condition candidates per precursor set.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--stage2_dataset_dir", required=True)
    ap.add_argument("--stage2_run_dir", required=True)
    ap.add_argument("--set_size_model", required=True)
    ap.add_argument("--stage3_dataset_dir", required=True)
    ap.add_argument("--stage3_lgbm_moe_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top_n_precursors", type=int, default=10)
    ap.add_argument("--route_top_k", type=int, default=10)
    ap.add_argument("--retrieval_k", type=int, default=4)
    ap.add_argument("--min_retrieval_similarity", type=float, default=0.0)
    ap.add_argument("--condition_weight", type=float, default=0.25)
    ap.add_argument("--max_set_size", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    base = load_base_module(root)
    stage2_dir = root / args.stage2_dataset_dir
    stage3_dir = root / args.stage3_dataset_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_df, _, y2 = build_stage2_candidates(
        base=base,
        stage2_dir=stage2_dir,
        stage2_run_dir=root / args.stage2_run_dir,
        set_size_model_path=root / args.set_size_model,
        split=args.split,
        top_n_precursors=args.top_n_precursors,
        max_set_size=args.max_set_size,
        batch_size=args.batch_size,
    )

    schema = load_json(stage3_dir / "schema.json")
    cont_names = [str(x) for x in schema["continuous_cols"]]
    stage3_vocab = [str(x) for x in schema["precursor_vocab"]]
    s3_pack = load_npz(stage3_dir / f"{args.split}.npz")
    x3 = np.asarray(s3_pack["x"], dtype=np.float32)
    sample_ids_all = np.asarray([str(x) for x in s3_pack["sample_id"]])
    if len(x3) != cand_df["sample_index"].max() + 1:
        raise ValueError("Stage2/Stage3 split length mismatch.")

    flat_sets = [set(parse_json_list(x)) for x in cand_df["pred_precursors"].tolist()]
    flat_sample_idx = cand_df["sample_index"].to_numpy(dtype=int)
    yset = make_yset(flat_sets, stage3_vocab)
    X_stage3 = np.concatenate([x3[flat_sample_idx], yset], axis=1).astype(np.float32)
    sample_ids = sample_ids_all[flat_sample_idx].tolist()

    pred_cont_model, pred_disc_model, disc_probs = predict_moe_with_disc_proba(root / args.stage3_lgbm_moe_model, X_stage3, sample_ids)
    model_pack = joblib.load(root / args.stage3_lgbm_moe_model)
    id_to_method = model_pack.get("id_to_method", {}) or {}
    methods = [str(id_to_method.get(str(sid), "other")) for sid in sample_ids]
    retrieval_cache = train_retrieval_cache(stage3_dir, schema, root / args.stage3_lgbm_moe_model)

    expanded_rows: List[Dict[str, Any]] = []
    pred_cont_rows: List[np.ndarray] = []
    pred_disc_rows: List[np.ndarray] = []
    for i, row in cand_df.reset_index(drop=True).iterrows():
        conds: List[Dict[str, Any]] = [{
            "source": "model_point",
            "condition_score": 0.35,
            "template_id": "",
            "pred_cont": pred_cont_model[i],
            "pred_disc": pred_disc_model[i],
        }]
        conds.extend(retrieve_templates(retrieval_cache, methods[i], yset[i], args.retrieval_k, args.min_retrieval_similarity))
        for c_rank, cond in enumerate(conds, start=1):
            out = row.to_dict()
            out["condition_rank"] = c_rank
            out["condition_source"] = cond["source"]
            out["condition_score"] = float(cond["condition_score"])
            out["template_id"] = cond["template_id"]
            out["score"] = float(row["precursor_score"]) + float(args.condition_weight) * float(cond["condition_score"]) - 0.02 * (c_rank - 1)
            out["rank"] = int(row["precursor_rank"])
            expanded_rows.append(out)
            pred_cont_rows.append(np.asarray(cond["pred_cont"], dtype=np.float32))
            pred_disc_rows.append(np.asarray(cond["pred_disc"], dtype=np.int64))

    out_df = pd.DataFrame(expanded_rows)
    pred_cont = np.vstack(pred_cont_rows).astype(np.float32)
    pred_disc = np.vstack(pred_disc_rows).astype(np.int64)
    expanded_sample_idx = out_df["sample_index"].to_numpy(dtype=int)
    y_cont_true_raw = raw_continuous(np.asarray(s3_pack["y_cond_continuous"], dtype=np.float32), schema, cont_names)
    y_cont_mask = np.asarray(s3_pack["y_cond_continuous_mask"], dtype=np.float32)
    y_disc_true = np.asarray(s3_pack["y_cond_discrete"])
    y_disc_mask = np.asarray(s3_pack["y_cond_discrete_mask"], dtype=np.float32)
    cond = condition_success(
        y_cont_true_raw[expanded_sample_idx],
        pred_cont,
        y_cont_mask[expanded_sample_idx],
        y_disc_true[expanded_sample_idx],
        pred_disc,
        y_disc_mask[expanded_sample_idx],
    )
    out_df["pred_temperature_c"] = pred_cont[:, 0]
    out_df["pred_time_h"] = pred_cont[:, 1]
    out_df["pred_atmosphere_idx"] = pred_disc[:, 0]
    out_df["temperature_abs_error_c"] = cond["temp_err"]
    out_df["time_abs_error_h"] = cond["time_err"]
    out_df["atmosphere_correct"] = cond["atm_ok"]
    out_df["condition_100c_24h_atm"] = cond["cond100_24_atm"]
    out_df["condition_200c_48h_atm"] = cond["cond200_48_atm"]
    out_df["strict_route_success"] = out_df["exact"].to_numpy(dtype=bool) & cond["cond100_24_atm"]
    out_df["relaxed_route_success"] = (out_df["jaccard"].to_numpy(dtype=float) >= 0.5) & cond["cond200_48_atm"]

    ordered = out_df.sort_values(["sample_index", "score"], ascending=[True, False]).copy()
    ordered["route_score"] = ordered["score"]
    ordered["route_rank"] = ordered.groupby("sample_index").cumcount() + 1
    out_df = ordered.sort_index()
    metrics = compute_metrics(out_df, args.route_top_k)
    summary = {
        "config": vars(args),
        "data": {
            "n_rows": int(len(x3)),
            "n_precursor_candidates": int(len(cand_df)),
            "n_route_candidates": int(len(out_df)),
            "mean_conditions_per_precursor": float(len(out_df) / max(len(cand_df), 1)),
            "stage3_vocab_size": int(len(stage3_vocab)),
            "mean_true_size": float(np.mean(y2.sum(axis=1))),
        },
        "metrics": metrics,
        "artifacts": {
            "candidate_csv": str((out_dir / f"{args.split}_topk_condition_route_candidates.csv").resolve()),
            "summary_json": str((out_dir / f"{args.split}_topk_condition_route_eval_summary.json").resolve()),
        },
    }
    out_df.to_csv(out_dir / f"{args.split}_topk_condition_route_candidates.csv", index=False)
    write_json(out_dir / f"{args.split}_topk_condition_route_eval_summary.json", summary)
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
