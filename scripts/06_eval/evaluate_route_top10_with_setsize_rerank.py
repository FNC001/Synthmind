#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int], dropout: float):
        super().__init__()
        dims = [int(input_dim)] + [int(x) for x in hidden_dims] + [int(output_dim)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
IGNORED_EXTRA_ELEMENTS = {"H", "O", "C", "N"}


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


def parse_hidden_dims(value: Any) -> List[int]:
    s = str(value or "").strip()
    return [int(x.strip()) for x in s.split(",") if x.strip()] if s else []


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


def element_set(text: Any) -> Set[str]:
    return set(ELEMENT_RE.findall(str(text or "")))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def get_x(pack: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in ["x", "features", "X"]:
        if key in pack:
            return np.asarray(pack[key], dtype=np.float32)
    raise KeyError(f"Missing x/features/X in npz keys={list(pack)}")


def get_y(pack: Mapping[str, np.ndarray]) -> np.ndarray:
    for key in ["y_multi_hot", "y", "labels", "targets"]:
        if key in pack:
            return (np.asarray(pack[key]) > 0).astype(np.int8)
    raise KeyError(f"Missing y labels in npz keys={list(pack)}")


def raw_continuous(values_norm: np.ndarray, schema: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    out = values_norm.astype(np.float32).copy()
    cont_schema = schema.get("continuous_schema", {}) or {}
    for j, name in enumerate(names):
        stats = cont_schema.get(name, {}) or {}
        out[:, j] = out[:, j] * float(stats.get("std", 1.0)) + float(stats.get("mean", 0.0))
    return out


def set_metrics(true_set: Set[str], pred_set: Set[str]) -> Dict[str, Any]:
    inter = len(true_set & pred_set)
    union = len(true_set | pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = inter / union if union else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact": pred_set == true_set,
        "any_overlap": inter > 0,
    }


def load_stage2_model(run_dir: Path, input_dim: int) -> Tuple[nn.Module, List[str]]:
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu")
    cfg = ckpt.get("config", {})
    names = [str(x) for x in ckpt["precursor_names"]]
    model = MLP(
        input_dim=input_dim,
        output_dim=int(ckpt.get("n_labels", len(names))),
        hidden_dims=parse_hidden_dims(cfg.get("hidden_dims", "512,256")),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, names


@torch.no_grad()
def predict_probs(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32)
        out.append(torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32))
    return np.vstack(out)


def load_stage3_baseline(ckpt_path: Path, project_root: Path) -> Any:
    sys.path.insert(0, str((project_root / "scripts/04_train/stage3").resolve()))
    import __main__  # type: ignore
    import train_baseline_linear  # type: ignore

    __main__.Stage3BaselineModel = train_baseline_linear.Stage3BaselineModel
    pack = joblib.load(ckpt_path)
    return pack["model"] if isinstance(pack, dict) and "model" in pack else pack


def predict_with_stage3_lgbm(model_path: Path, X_stage3: np.ndarray, schema: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pack = joblib.load(model_path)
    models = pack["models"]
    cont_names = [str(x) for x in schema["continuous_cols"]]
    disc_names = [str(x) for x in schema["discrete_cols"]]
    disc_schema = schema.get("discrete_schema", {}) or {}

    y_cont = np.zeros((X_stage3.shape[0], len(cont_names)), dtype=np.float32)
    if "target_temperature_c" in models and "target_temperature_c" in cont_names:
        j = cont_names.index("target_temperature_c")
        m = models["target_temperature_c"]
        y_cont[:, j] = np.asarray(m.predict(X_stage3, num_iteration=m.best_iteration), dtype=np.float32)
    if "target_time_h_log1p" in models and "target_time_h" in cont_names:
        j = cont_names.index("target_time_h")
        m = models["target_time_h_log1p"]
        y_cont[:, j] = np.expm1(np.asarray(m.predict(X_stage3, num_iteration=m.best_iteration), dtype=np.float32))

    y_disc = np.zeros((X_stage3.shape[0], len(disc_names)), dtype=np.int64)
    for j, name in enumerate(disc_names):
        if name not in models:
            missing = int((disc_schema.get(name, {}) or {}).get("missing_index", 0))
            y_disc[:, j] = missing
            continue
        m = models[name]
        prob = m.predict(X_stage3, num_iteration=m.best_iteration)
        y_disc[:, j] = np.asarray(np.argmax(prob, axis=1), dtype=np.int64)
    return y_cont, y_disc


def predict_with_stage3_lgbm_moe(model_path: Path, X_stage3: np.ndarray, sample_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
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
            prob = model.predict(X_stage3[idx], num_iteration=model.best_iteration)
            y_disc[idx, j] = np.asarray(np.argmax(prob, axis=1), dtype=np.int64)
    return y_cont, y_disc


def chemistry_score(labels: Sequence[str], target_formula: str) -> Dict[str, float]:
    target = element_set(target_formula) - {"H", "O"}
    prec = set()
    for label in labels:
        prec |= element_set(label)
    prec_core = prec - IGNORED_EXTRA_ELEMENTS
    coverage = len(target & prec_core) / len(target) if target else 1.0
    extra = len(prec_core - target)
    return {"coverage": float(coverage), "extra": float(extra)}


def candidate_score(
    label_indices: Sequence[int],
    probs: np.ndarray,
    names: Sequence[str],
    formula: str,
    pred_size: int,
    size_prob: float,
) -> Dict[str, float]:
    labels = [names[i] for i in label_indices]
    selected = np.clip(probs[list(label_indices)], 1e-6, 1.0 - 1e-6) if label_indices else np.asarray([1e-6])
    chem = chemistry_score(labels, formula)
    prob_score = float(np.mean(np.log(selected)))
    score = (
        prob_score
        + 2.0 * chem["coverage"]
        - 0.35 * chem["extra"]
        - 0.15 * abs(len(label_indices) - int(pred_size))
        + 0.5 * math.log(max(float(size_prob), 1e-6))
    )
    return {
        "score": float(score),
        "prob_score": prob_score,
        "coverage": chem["coverage"],
        "extra": chem["extra"],
        "size_prob": float(size_prob),
    }


def generate_candidates(
    probs: np.ndarray,
    names: Sequence[str],
    formula: str,
    pred_size: int,
    size_prob_by_k: Mapping[int, float],
    max_size: int,
    keep: int,
) -> List[Dict[str, Any]]:
    order = np.argsort(-probs)
    pool = order[: min(120, len(order))].tolist()
    size_options = set(int(k) for k, _ in sorted(size_prob_by_k.items(), key=lambda kv: -kv[1])[:5])
    size_options.update(range(max(1, pred_size - 2), min(max_size, pred_size + 2) + 1))
    size_options = {int(k) for k in size_options if 1 <= int(k) <= int(max_size)}

    target_core = element_set(formula) - {"H", "O"}
    label_core_cache: Dict[int, Set[str]] = {
        int(i): element_set(names[int(i)]) - IGNORED_EXTRA_ELEMENTS
        for i in pool
    }
    rank_map = {int(idx): pos for pos, idx in enumerate(order.tolist())}

    raw: List[Tuple[int, ...]] = []
    raw_by_size: Dict[int, int] = {}
    forced_by_size: Dict[int, int] = {}
    per_size_cap = min(max(int(keep) * 4, 120), 500)
    forced_per_size_cap = min(max(int(keep) * 2, 80), 300)

    def add_combo(items: Iterable[int], force: bool = False) -> None:
        combo = tuple(sorted(set(int(x) for x in items)))
        k = len(combo)
        if not combo or k > max_size:
            return
        if force:
            if forced_by_size.get(k, 0) >= forced_per_size_cap:
                return
            raw.append(combo)
            forced_by_size[k] = forced_by_size.get(k, 0) + 1
            return
        if raw_by_size.get(k, 0) < per_size_cap:
            raw.append(combo)
            raw_by_size[k] = raw_by_size.get(k, 0) + 1

    for k in sorted(size_options):
        if k <= 0:
            continue
        add_combo(pool[:k])

    for k in sorted(size_options):
        if k <= 0 or k > max_size:
            continue
        base = pool[:k]
        replacement_pool = pool[k:k + 12]
        for pos in range(min(k, 4)):
            for repl in replacement_pool[:8]:
                cand = list(base)
                cand[pos] = repl
                add_combo(cand)

    def label_weight(idx: int) -> float:
        p = float(np.clip(probs[int(idx)], 1e-6, 1.0 - 1e-6))
        w = math.log(p / (1.0 - p))
        core = label_core_cache.get(int(idx), set())
        if target_core and core:
            if core & target_core:
                w += 0.25 * len(core & target_core)
            w -= 0.18 * len(core - target_core)
        return float(w)

    def add_beam_pool(candidate_pool: Sequence[int], k: int, limit: int) -> None:
        ranked = sorted(set(int(x) for x in candidate_pool), key=label_weight, reverse=True)
        n = len(ranked)
        if k <= 0 or k > n:
            return
        start = tuple(range(k))
        weights = np.asarray([label_weight(x) for x in ranked], dtype=np.float32)
        heap: List[Tuple[float, Tuple[int, ...]]] = [(-float(weights[list(start)].sum()), start)]
        seen_pos = {start}
        popped = 0
        while heap and popped < int(limit):
            _, pos = heapq.heappop(heap)
            add_combo(ranked[j] for j in pos)
            popped += 1
            for t in range(k - 1, -1, -1):
                if pos[t] >= n - k + t:
                    continue
                nxt = list(pos)
                nxt[t] += 1
                for u in range(t + 1, k):
                    nxt[u] = max(nxt[u], nxt[u - 1] + 1)
                nxt_tuple = tuple(nxt)
                if nxt_tuple in seen_pos or nxt_tuple[-1] >= n:
                    continue
                seen_pos.add(nxt_tuple)
                heapq.heappush(heap, (-float(weights[list(nxt_tuple)].sum()), nxt_tuple))

    # The MLP often ranks each true precursor reasonably high while the old
    # prefix/replacement decoder misses the exact combination. Add bounded
    # beam search over high-probability and element-focused pools.
    beam_pool = pool[:80]
    for k in sorted(size_options):
        if k <= 0 or k > max_size:
            continue
        add_beam_pool(beam_pool, int(k), per_size_cap)

    focused: List[int] = pool[:18]
    if target_core:
        for elem in sorted(target_core):
            hits = [
                int(i) for i in pool
                if elem in label_core_cache.get(int(i), set())
            ][:14]
            focused.extend(hits)
        # Prefer labels that cover target elements without introducing many
        # additional non-CHON elements.
        chemically_close = []
        for i in pool:
            core = label_core_cache.get(int(i), set())
            if core & target_core:
                extra = len(core - target_core)
                chemically_close.append((extra, rank_map[int(i)], int(i)))
        chemically_close.sort()
        focused.extend([i for _, _, i in chemically_close[:35]])
    focused = sorted(set(focused), key=lambda i: rank_map.get(int(i), 10**9))[:34]

    focus_limits = {1: 34, 2: 34, 3: 28, 4: 18, 5: 13, 6: 10, 7: 9}
    for k in sorted(size_options):
        if k <= 0 or k > max_size:
            continue
        focus_pool = focused[: min(focus_limits.get(int(k), 10), len(focused))]
        add_beam_pool(focus_pool, int(k), max(80, per_size_cap // 2))

    if 1 <= len(target_core) <= 5:
        per_elem: List[List[int]] = []
        for elem in sorted(target_core):
            hits = [
                int(i) for i in pool
                if elem in label_core_cache.get(int(i), set())
            ][:15]
            if hits:
                per_elem.append(hits)
        if per_elem:
            for n_product, combo in enumerate(itertools.product(*per_elem), start=1):
                add_combo(combo, force=True)
                if n_product >= 5000:
                    break

    seen = set()
    scored = []
    for cand in raw:
        if not cand or len(cand) > max_size or cand in seen:
            continue
        seen.add(cand)
        sp = float(size_prob_by_k.get(len(cand), 1e-6))
        sc = candidate_score(cand, probs, names, formula, pred_size, sp)
        labels = [names[i] for i in cand]
        scored.append({
            "indices": list(cand),
            "labels": labels,
            "label_set": set(labels),
            "n_labels": len(labels),
            **sc,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:keep]


def make_yset(label_sets: Sequence[Set[str]], vocab: Sequence[str]) -> np.ndarray:
    idx = {str(v): i for i, v in enumerate(vocab)}
    y = np.zeros((len(label_sets), len(vocab)), dtype=np.float32)
    for i, labels in enumerate(label_sets):
        for lab in labels:
            j = idx.get(str(lab))
            if j is not None:
                y[i, j] = 1.0
    return y


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
        "evaluable": evaluable,
        "temp_err": temp_err,
        "time_err": time_err,
        "atm_ok": atm_ok,
        "cond100_24_atm": evaluable & (temp_err <= 100.0) & (time_err <= 24.0) & atm_ok,
        "cond200_48_atm": evaluable & (temp_err <= 200.0) & (time_err <= 48.0) & atm_ok,
    }


def markdown_report(summary: Mapping[str, Any]) -> str:
    m = summary["metrics"]
    lines = ["# Stage2 Top-10 Route Evaluation", ""]
    lines.append(f"- split: {summary['config']['split']}")
    lines.append(f"- n_rows: {summary['data']['n_rows']}")
    lines.append(f"- top_n candidates per sample: {summary['config']['top_n']}")
    lines.append("")
    lines.append("## Top-1")
    lines.append(f"- precursor exact: {m['top1_precursor_exact']:.4f}")
    lines.append(f"- precursor mean F1: {m['top1_precursor_mean_f1']:.4f}")
    lines.append(f"- precursor mean Jaccard: {m['top1_precursor_mean_jaccard']:.4f}")
    lines.append(f"- relaxed route success: {m['top1_relaxed_route_success']:.4f}")
    lines.append(f"- strict route success: {m['top1_strict_route_success']:.4f}")
    lines.append("")
    lines.append("## In Top-10")
    lines.append(f"- precursor exact in top10: {m['top10_precursor_exact_recall']:.4f}")
    lines.append(f"- precursor Jaccard>=0.5 in top10: {m['top10_precursor_jaccard50_recall']:.4f}")
    lines.append(f"- relaxed route success in top10: {m['top10_relaxed_route_oracle_success']:.4f}")
    lines.append(f"- strict route success in top10: {m['top10_strict_route_oracle_success']:.4f}")
    lines.append("")
    lines.append("Top-10 metrics are oracle-in-list: they answer whether a good route appears somewhere in the first ten candidates, not whether the current ranker already placed it first.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Stage2 set-size + chemistry reranked top-N candidates with Stage3 baseline.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--stage2_dataset_dir", default="data/interim/generative/stage2_setpred_dataset/descriptor/route_unified_20260609_relaxed_only")
    ap.add_argument("--stage2_run_dir", default="runs/stage2/mlp_route_unified_20260609_descriptor")
    ap.add_argument("--set_size_model", default="runs/stage2/set_size_route_unified_20260609_descriptor/set_size_predictor.joblib")
    ap.add_argument("--stage3_dataset_dir", default="data/interim/generative/stage3_condition_dataset_mixed/new_stage2_all_stage3_20260609_poscar_geom1024")
    ap.add_argument("--stage3_baseline_ckpt", default="runs/stage3/baseline_linear_new_stage2_all_stage3_20260609/best_model.pkl")
    ap.add_argument("--stage3_lgbm_model", default="")
    ap.add_argument("--stage3_lgbm_moe_model", default="")
    ap.add_argument("--output_dir", default="outputs/evaluation/route_top10_setsize_rerank_20260609")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "gold_train_holdout"])
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--max_set_size", type=int, default=7)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    stage2_dir = root / args.stage2_dataset_dir
    stage3_dir = root / args.stage3_dataset_dir
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split = str(args.split)
    s2_pack = load_npz(stage2_dir / f"{split}.npz")
    s2_meta = pd.read_csv(stage2_dir / f"{split}_meta.csv")
    x2 = get_x(s2_pack)
    y2 = get_y(s2_pack)
    true_sets = []

    stage2_model, precursor_names = load_stage2_model(root / args.stage2_run_dir, x2.shape[1])
    probs = predict_probs(stage2_model, x2, args.batch_size)
    name_sets = [{precursor_names[j] for j in np.where(y2[i] > 0)[0]} for i in range(y2.shape[0])]
    true_sets = name_sets

    size_pack = joblib.load(root / args.set_size_model)
    size_model = size_pack["model"] if isinstance(size_pack, dict) and "model" in size_pack else size_pack
    size_pred = size_model.predict(x2).astype(int)
    size_proba = size_model.predict_proba(x2)
    size_classes = [int(x) for x in size_model.classes_.tolist()]

    all_candidates: List[List[Dict[str, Any]]] = []
    rows = []
    for i in range(len(x2)):
        prob_by_k = {k: float(size_proba[i, j]) for j, k in enumerate(size_classes)}
        cands = generate_candidates(
            probs=probs[i],
            names=precursor_names,
            formula=str(s2_meta.loc[i, "formula"]),
            pred_size=int(size_pred[i]),
            size_prob_by_k=prob_by_k,
            max_size=int(args.max_set_size),
            keep=int(args.top_n),
        )
        all_candidates.append(cands)
        true = true_sets[i]
        for rank, cand in enumerate(cands, start=1):
            sm = set_metrics(true, cand["label_set"])
            rows.append({
                "sample_index": i,
                "rank": rank,
                "id": s2_meta.loc[i, "id"],
                "material_id": s2_meta.loc[i, "material_id"],
                "formula": s2_meta.loc[i, "formula"],
                "true_precursors": json.dumps(sorted(true), ensure_ascii=False),
                "pred_precursors": json.dumps(cand["labels"], ensure_ascii=False),
                "score": cand["score"],
                "prob_score": cand["prob_score"],
                "coverage": cand["coverage"],
                "extra": cand["extra"],
                "predicted_size": int(size_pred[i]),
                "candidate_size": int(cand["n_labels"]),
                **sm,
            })

    cand_df = pd.DataFrame(rows)

    stage3_schema = load_json(stage3_dir / "schema.json")
    cont_names = [str(x) for x in stage3_schema["continuous_cols"]]
    stage3_vocab = [str(x) for x in stage3_schema["precursor_vocab"]]
    s3_pack = load_npz(stage3_dir / f"{split}.npz")
    x3 = np.asarray(s3_pack["x"], dtype=np.float32)
    y_cont_true_norm = np.asarray(s3_pack["y_cond_continuous"], dtype=np.float32)
    y_cont_mask = np.asarray(s3_pack["y_cond_continuous_mask"], dtype=np.float32)
    y_disc_true = np.asarray(s3_pack["y_cond_discrete"])
    y_disc_mask = np.asarray(s3_pack["y_cond_discrete_mask"], dtype=np.float32)
    y_cont_true_raw = raw_continuous(y_cont_true_norm, stage3_schema, cont_names)

    if len(x3) != len(x2):
        raise ValueError(f"Stage2/Stage3 {split} length mismatch: {len(x2)} vs {len(x3)}")

    flat_sets = [set(parse_json_list(x)) for x in cand_df["pred_precursors"].tolist()]
    flat_sample_idx = cand_df["sample_index"].to_numpy(dtype=int)
    yset = make_yset(flat_sets, stage3_vocab)
    x3_flat = x3[flat_sample_idx]
    X_stage3 = np.concatenate([x3_flat, yset], axis=1).astype(np.float32)

    if args.stage3_lgbm_moe_model:
        sample_ids = [str(x) for x in np.asarray(s3_pack["sample_id"])[flat_sample_idx]]
        pred_cont_raw, pred_disc = predict_with_stage3_lgbm_moe(root / args.stage3_lgbm_moe_model, X_stage3, sample_ids)
    elif args.stage3_lgbm_model:
        pred_cont_raw, pred_disc = predict_with_stage3_lgbm(root / args.stage3_lgbm_model, X_stage3, stage3_schema)
    else:
        stage3_model = load_stage3_baseline(root / args.stage3_baseline_ckpt, root)
        pred = stage3_model.predict(X_stage3)
        pred_cont_raw = raw_continuous(np.asarray(pred["y_cont_pred"], dtype=np.float32), stage3_schema, cont_names)
        pred_disc = np.asarray(pred["y_disc_pred"])

    true_cont_flat = y_cont_true_raw[flat_sample_idx]
    cont_mask_flat = y_cont_mask[flat_sample_idx]
    true_disc_flat = y_disc_true[flat_sample_idx]
    disc_mask_flat = y_disc_mask[flat_sample_idx]
    cond = condition_success(true_cont_flat, pred_cont_raw, cont_mask_flat, true_disc_flat, pred_disc, disc_mask_flat)

    cand_df["temperature_abs_error_c"] = cond["temp_err"]
    cand_df["time_abs_error_h"] = cond["time_err"]
    cand_df["atmosphere_correct"] = cond["atm_ok"]
    cand_df["condition_100c_24h_atm"] = cond["cond100_24_atm"]
    cand_df["condition_200c_48h_atm"] = cond["cond200_48_atm"]
    cand_df["strict_route_success"] = cand_df["exact"].to_numpy(dtype=bool) & cond["cond100_24_atm"]
    cand_df["relaxed_route_success"] = (cand_df["jaccard"].to_numpy(dtype=float) >= 0.5) & cond["cond200_48_atm"]

    top1 = cand_df[cand_df["rank"] == 1].copy()
    grouped = cand_df.groupby("sample_index", sort=False)
    evaluable_top1 = top1["condition_200c_48h_atm"].notna()
    n = len(top1)

    metrics = {
        "top1_precursor_exact": float(top1["exact"].mean()),
        "top1_precursor_any_overlap": float(top1["any_overlap"].mean()),
        "top1_precursor_mean_f1": float(top1["f1"].mean()),
        "top1_precursor_mean_jaccard": float(top1["jaccard"].mean()),
        "top1_condition_100c_24h_atm": float(top1["condition_100c_24h_atm"].mean()),
        "top1_condition_200c_48h_atm": float(top1["condition_200c_48h_atm"].mean()),
        "top1_strict_route_success": float(top1["strict_route_success"].mean()),
        "top1_relaxed_route_success": float(top1["relaxed_route_success"].mean()),
        "top10_precursor_exact_recall": float(grouped["exact"].any().mean()),
        "top10_precursor_jaccard50_recall": float(grouped["jaccard"].max().ge(0.5).mean()),
        "top10_strict_route_oracle_success": float(grouped["strict_route_success"].any().mean()),
        "top10_relaxed_route_oracle_success": float(grouped["relaxed_route_success"].any().mean()),
        "mean_predicted_size_top1": float(top1["candidate_size"].mean()),
        "mean_true_size": float(np.mean(y2.sum(axis=1))),
        "n": int(n),
    }

    summary = {
        "config": vars(args),
        "data": {
            "n_rows": int(len(x2)),
            "split": split,
            "n_candidates": int(len(cand_df)),
            "stage2_vocab_size": int(len(precursor_names)),
            "stage3_vocab_size": int(len(stage3_vocab)),
        },
        "metrics": metrics,
        "artifacts": {
            "candidate_csv": str((out_dir / f"{split}_top10_route_candidates.csv").resolve()),
            "summary_json": str((out_dir / f"{split}_top10_route_eval_summary.json").resolve()),
            "report_md": str((out_dir / f"{split}_top10_route_eval_report.md").resolve()),
        },
    }

    cand_df.to_csv(out_dir / f"{split}_top10_route_candidates.csv", index=False)
    write_json(out_dir / f"{split}_top10_route_eval_summary.json", summary)
    (out_dir / f"{split}_top10_route_eval_report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
