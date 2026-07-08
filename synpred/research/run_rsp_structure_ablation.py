from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from synpred.research.run_rsp_expansion import (
    build_precursor_library,
    build_prediction_map,
    evaluate_fixed_params,
    load_config,
    normalize_candidates,
    sample_frame,
    subset_metrics,
    source_ablation_rows,
    write_json,
)


MODES = ("composition_only", "structure_only", "structure_plus_composition")
COMPOSITION_PREFIXES = (
    "feat_n_elements_formula",
    "feat_total_atoms_formula",
    "feat_stoich_entropy",
    "feat_z_",
    "feat_frac_",
)
STRUCTURE_PREFIXES = (
    "feat_crystal_system__",
    "feat_density",
    "feat_volume",
    "feat_nsites",
    "feat_nelements",
    "feat_band_gap",
    "feat_energy_above_hull",
    "feat_spacegroup_number",
    "feat_lattice_",
    "feat_has_summary",
    "feat_poscar_",
    "feat_pairdist_",
    "feat_nn_",
    "feat_coord_",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run RSP structure/composition ablation via family-guided expansion.")
    ap.add_argument("--config", default="research/configs/rsp_vnext_003.yaml")
    ap.add_argument("--split", choices=["validation"], default="validation")
    ap.add_argument("--budgets", default="50,200,500")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="outputs/autorun/rsp_structure_ablation_20260623")
    ap.add_argument("--n_estimators", type=int, default=260)
    ap.add_argument("--learning_rate", type=float, default=0.055)
    ap.add_argument("--num_leaves", type=int, default=63)
    return ap.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_labels(value: Any) -> list[str]:
    try:
        obj = json.loads(str(value))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def load_split(family_dir: Path, split: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(family_dir / f"{split}_family_labels.csv")
    x = np.load(family_dir / f"{split}_features.npz", allow_pickle=True)["x"].astype(np.float32)
    return df, x


def feature_cols_for_family_dir(family_dir: Path) -> list[str]:
    summary = load_json(family_dir / "family_dataset_summary.json")
    dataset_dir = Path(summary["config"]["dataset_dir"])
    if not dataset_dir.exists():
        alt = Path(str(dataset_dir).replace("data/interim/", "data/interim.local_bak_20260612/"))
        dataset_dir = alt if alt.exists() else dataset_dir
    return [str(x) for x in load_json(dataset_dir / "feature_cols.json")]


def mode_indices(feature_cols: Sequence[str], mode: str) -> list[int]:
    comp = [i for i, c in enumerate(feature_cols) if c.startswith(COMPOSITION_PREFIXES)]
    struct = [i for i, c in enumerate(feature_cols) if c.startswith(STRUCTURE_PREFIXES)]
    if mode == "composition_only":
        return comp
    if mode == "structure_only":
        return struct
    if mode == "structure_plus_composition":
        return sorted(set(comp) | set(struct))
    raise ValueError(mode)


def build_x(
    df: pd.DataFrame,
    sample_x: np.ndarray,
    sample_indices: Sequence[int],
    element_vocab: Sequence[str],
    method_vocab: Sequence[str],
) -> np.ndarray:
    x_rows = sample_x[df["x_row_index"].to_numpy(dtype=int)][:, list(sample_indices)]
    elem_idx = {e: i for i, e in enumerate(element_vocab)}
    method_idx = {m: i for i, m in enumerate(method_vocab)}
    elem_one = np.zeros((len(df), len(element_vocab)), dtype=np.float32)
    method_one = np.zeros((len(df), len(method_vocab)), dtype=np.float32)
    for r, elem in enumerate(df["target_element"].astype(str)):
        j = elem_idx.get(elem)
        if j is not None:
            elem_one[r, j] = 1.0
    for r, method in enumerate(df["reaction_method"].fillna("other").astype(str)):
        j = method_idx.get(method)
        if j is not None:
            method_one[r, j] = 1.0
    return np.concatenate([x_rows, elem_one, method_one], axis=1).astype(np.float32)


def binarize(df: pd.DataFrame, families: Sequence[str]) -> np.ndarray:
    idx = {f: i for i, f in enumerate(families)}
    y = np.zeros((len(df), len(families)), dtype=np.int8)
    for i, labs in enumerate(df["element_family_labels"].map(parse_labels)):
        for lab in labs:
            j = idx.get(lab)
            if j is not None:
                y[i, j] = 1
    return y


def predict_matrix(models: dict[str, Any], x: np.ndarray, families: Sequence[str]) -> np.ndarray:
    out = np.zeros((x.shape[0], len(families)), dtype=np.float32)
    for j, fam in enumerate(families):
        model = models.get(fam)
        if model is not None:
            out[:, j] = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    return out


def family_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred_top1 = np.zeros_like(y_true)
    pred_top1[np.arange(len(y_true)), np.argmax(prob, axis=1)] = 1
    top3 = np.argsort(-prob, axis=1)[:, :3]
    pred_top3 = np.zeros_like(y_true)
    for i in range(len(y_true)):
        pred_top3[i, top3[i]] = 1
    nonempty = y_true.sum(axis=1) > 0
    exact = np.all(pred_top1 == y_true, axis=1)
    any1 = (pred_top1 & y_true).sum(axis=1) > 0
    any3 = (pred_top3 & y_true).sum(axis=1) > 0
    return {
        "family_top1_exact": float(exact[nonempty].mean()) if np.any(nonempty) else 0.0,
        "family_top1_recall": float(any1[nonempty].mean()) if np.any(nonempty) else 0.0,
        "family_top3_recall": float(any3[nonempty].mean()) if np.any(nonempty) else 0.0,
    }


def save_predictions(path: Path, df: pd.DataFrame, prob: np.ndarray, families: Sequence[str]) -> None:
    out = df.copy()
    for j, fam in enumerate(families):
        out[f"prob_family__{fam}"] = prob[:, j]
    out["pred_top1_family"] = [families[int(j)] for j in np.argmax(prob, axis=1)]
    out["pred_top3_families"] = [
        json.dumps([families[int(j)] for j in row], ensure_ascii=False)
        for row in np.argsort(-prob, axis=1)[:, :3]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def train_family_mode(
    mode: str,
    family_dir: Path,
    outdir: Path,
    seed: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
) -> dict[str, Any]:
    import lightgbm as lgb

    families = [f for f in load_json(family_dir / "family_vocab.json") if f != "unknown"]
    elements = load_json(family_dir / "element_vocab.json")
    train_df, train_x = load_split(family_dir, "train")
    val_df, val_x = load_split(family_dir, "val")
    methods = sorted(set(train_df["reaction_method"].fillna("other").astype(str)) | set(val_df["reaction_method"].fillna("other").astype(str)))
    feature_cols = feature_cols_for_family_dir(family_dir)
    sample_idx = mode_indices(feature_cols, mode)
    x_train = build_x(train_df, train_x, sample_idx, elements, methods)
    x_val = build_x(val_df, val_x, sample_idx, elements, methods)
    y_train = binarize(train_df, families)
    y_val = binarize(val_df, families)
    models: dict[str, Any] = {}
    for j, fam in enumerate(families):
        if y_train[:, j].sum() == 0:
            continue
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=seed + j,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            x_train,
            y_train[:, j],
            eval_set=[(x_val, y_val[:, j])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        models[fam] = model
    prob = predict_matrix(models, x_val, families)
    pred_path = outdir / mode / "val_family_predictions.csv"
    save_predictions(pred_path, val_df, prob, families)
    metrics = family_metrics(y_val, prob)
    return {
        "mode": mode,
        "selected_sample_feature_count": len(sample_idx),
        "total_feature_count_with_element_method": int(x_train.shape[1]),
        "prediction_path": str(pred_path),
        "family_metrics": metrics,
    }


def rsp_eval_for_predictions(pred_path: Path, cfg: Any, mode: str, outdir: Path, params: dict[str, Any]) -> dict[str, Any]:
    base = normalize_candidates(pd.read_csv(cfg.split_candidates))
    train_family = pd.read_csv(cfg.train_family_labels)
    pred_map = build_prediction_map(pd.read_csv(pred_path), cfg.family_top_n)
    lib = build_precursor_library(train_family, cfg.precursor_ontology)
    samples = sample_frame(base, lib)
    flags = samples.set_index("sample_id")[["is_rare_reference", "is_oov_reference"]]
    base = base.drop(columns=["is_rare_reference", "is_oov_reference"], errors="ignore").join(flags, on="sample_id")
    base["is_rare_reference"] = base["is_rare_reference"].fillna(False).astype(bool)
    base["is_oov_reference"] = base["is_oov_reference"].fillna(False).astype(bool)
    ranked, metrics_by_variant, counts = evaluate_fixed_params(base, samples, pred_map, lib, cfg, params)
    primary = ranked["base_plus_family_rare_chemistry"]
    baseline = ranked["rsp_v5_baseline"]
    source_rows = source_ablation_rows("base_plus_family_rare_chemistry", primary, baseline, cfg.budgets)
    rare = subset_metrics(primary, "is_rare_reference", cfg.budgets)
    oov = subset_metrics(primary, "is_oov_reference", cfg.budgets)
    pd.DataFrame(source_rows).to_csv(outdir / mode / "candidate_source_ablation.csv", index=False)
    return {
        "mode": mode,
        "generated_candidate_counts": counts,
        "variants": metrics_by_variant,
        "primary_rare_subset": rare,
        "primary_oov_subset": oov,
        "source_ablation": source_rows,
    }


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, args.split, args.budgets, args.seed)
    family_dir = cfg.train_family_labels.parent
    validation_metrics = Path(cfg.output_dir) / "metrics.json"
    if validation_metrics.exists():
        selected_params = json.loads(validation_metrics.read_text(encoding="utf-8"))["selected_params"]
    else:
        selected_params = {
            "preserve_base_top": 20,
            "family_per_element": 2,
            "rare_per_element": 1,
            "max_generated_per_sample": 40,
        }
    results = {
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "family_dir": str(family_dir),
        "selected_rsp_expansion_params": selected_params,
        "modes": {},
    }
    for mode in MODES:
        print(f"[rsp_structure_ablation] training mode={mode}", flush=True)
        fam_result = train_family_mode(
            mode,
            family_dir,
            outdir,
            args.seed,
            args.n_estimators,
            args.learning_rate,
            args.num_leaves,
        )
        print(f"[rsp_structure_ablation] evaluating RSP expansion mode={mode}", flush=True)
        rsp_result = rsp_eval_for_predictions(Path(fam_result["prediction_path"]), cfg, mode, outdir, selected_params)
        results["modes"][mode] = {"family": fam_result, "rsp": rsp_result}
        write_json(outdir / "structure_ablation_metrics.json", results)
    write_report(outdir / "RSP_STRUCTURE_ABLATION_REPORT.md", results)
    print(json.dumps(results, ensure_ascii=False, indent=2)[:5000])
    return 0


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines = ["# RSP Structure/Composition Ablation", ""]
    for mode, payload in results["modes"].items():
        lines.append(f"## {mode}")
        lines.append("")
        lines.append("Family metrics:")
        for k, v in payload["family"]["family_metrics"].items():
            lines.append(f"- `{k}`: {v:.6f}")
        primary = payload["rsp"]["variants"]["base_plus_family_rare_chemistry"]
        lines.append("")
        lines.append("RSP primary metrics:")
        for k in ["exact@1", "exact@10", "exact@50", "exact@200", "exact@500", "best_jaccard@1", "best_jaccard@50"]:
            lines.append(f"- `{k}`: {primary.get(k, 0.0):.6f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
