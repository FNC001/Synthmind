#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


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
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def parse_hidden_dims(value: Any) -> List[int]:
    s = str(value or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_precursor_list(value: Any) -> List[str]:
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


def set_prf(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float, float, bool, bool]:
    inter = len(true_set & pred_set)
    union = len(true_set | pred_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    jaccard = inter / union if union else 1.0
    return precision, recall, f1, jaccard, pred_set == true_set, inter > 0


def topk_predict(probs: np.ndarray, ks: np.ndarray) -> np.ndarray:
    n, d = probs.shape
    out = np.zeros((n, d), dtype=np.int8)
    for i, k0 in enumerate(ks.astype(int)):
        k = int(np.clip(k0, 1, d))
        idx = np.argpartition(-probs[i], kth=k - 1)[:k]
        out[i, idx] = 1
    return out


def threshold_predict(probs: np.ndarray, threshold: float) -> np.ndarray:
    out = (probs >= float(threshold)).astype(np.int8)
    empty = out.sum(axis=1) == 0
    if np.any(empty):
        best = np.argmax(probs[empty], axis=1)
        out[empty, best] = 1
    return out


def binary_to_sets(y: np.ndarray, names: Sequence[str]) -> List[Set[str]]:
    out: List[Set[str]] = []
    for i in range(y.shape[0]):
        idx = np.where(y[i] > 0)[0].tolist()
        out.append({str(names[j]) for j in idx if 0 <= j < len(names)})
    return out


def build_stage2_x(stage3_raw_csv: Path, feature_cols: Sequence[str], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    df = pd.read_csv(stage3_raw_csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"stage3 raw CSV is missing {len(missing)} Stage2 feature columns, e.g. {missing[:5]}")
    x_raw = df[list(feature_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    return ((x_raw - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)


def load_stage2_model(run_dir: Path, input_dim: int) -> Tuple[nn.Module, List[str]]:
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu")
    cfg = ckpt.get("config", {})
    names = [str(x) for x in ckpt.get("precursor_names", [])]
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
def predict_stage2_probs(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32)
        chunks.append(torch.sigmoid(model(xb)).cpu().numpy().astype(np.float32))
    return np.vstack(chunks)


def make_stage3_yset(pred_sets: Sequence[Set[str]], stage3_vocab: Sequence[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    idx = {str(v): i for i, v in enumerate(stage3_vocab)}
    y = np.zeros((len(pred_sets), len(stage3_vocab)), dtype=np.float32)
    dropped = 0
    total = 0
    for i, labels in enumerate(pred_sets):
        for lab in labels:
            total += 1
            j = idx.get(str(lab))
            if j is None:
                dropped += 1
                continue
            y[i, j] = 1.0
    return y, {
        "predicted_precursor_tokens": int(total),
        "tokens_missing_from_stage3_vocab": int(dropped),
        "missing_from_stage3_vocab_rate": float(dropped / total) if total else 0.0,
    }


def load_stage3_baseline(ckpt_path: Path, project_root: Path) -> Any:
    sys.path.insert(0, str((project_root / "scripts/04_train/stage3").resolve()))
    import __main__  # type: ignore
    import train_baseline_linear  # type: ignore

    __main__.Stage3BaselineModel = train_baseline_linear.Stage3BaselineModel
    pack = joblib.load(ckpt_path)
    return pack["model"] if isinstance(pack, dict) and "model" in pack else pack


def raw_continuous(values_norm: np.ndarray, schema: Mapping[str, Any], names: Sequence[str]) -> np.ndarray:
    out = values_norm.astype(np.float32).copy()
    cont_schema = schema.get("continuous_schema", {}) or {}
    for j, name in enumerate(names):
        stats = cont_schema.get(name, {}) or {}
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0))
        out[:, j] = out[:, j] * std + mean
    return out


def condition_metrics(
    y_cont_true_raw: np.ndarray,
    y_cont_pred_raw: np.ndarray,
    y_cont_mask: np.ndarray,
    y_disc_true: np.ndarray,
    y_disc_pred: np.ndarray,
    y_disc_mask: np.ndarray,
    cont_names: Sequence[str],
    disc_names: Sequence[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"continuous": {}, "discrete": {}}
    maes = []
    rmses = []
    for j, name in enumerate(cont_names):
        valid = y_cont_mask[:, j] > 0.5
        err = y_cont_pred_raw[valid, j] - y_cont_true_raw[valid, j]
        mae = float(np.mean(np.abs(err))) if err.size else float("nan")
        rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")
        out["continuous"][name] = {"n": int(valid.sum()), "mae": mae, "rmse": rmse}
        if err.size:
            maes.append(mae)
            rmses.append(rmse)
    out["continuous"]["mean_mae"] = float(np.mean(maes)) if maes else float("nan")
    out["continuous"]["mean_rmse"] = float(np.mean(rmses)) if rmses else float("nan")

    accs = []
    macro_f1s = []
    for j, name in enumerate(disc_names):
        valid = y_disc_mask[:, j] > 0.5
        yt = y_disc_true[valid, j]
        yp = y_disc_pred[valid, j]
        acc = float(np.mean(yt == yp)) if yt.size else float("nan")
        mf1 = float(f1_score(yt, yp, average="macro", zero_division=0)) if yt.size else float("nan")
        out["discrete"][name] = {"n": int(valid.sum()), "accuracy": acc, "macro_f1": mf1}
        if yt.size:
            accs.append(acc)
            macro_f1s.append(mf1)
    out["discrete"]["mean_accuracy"] = float(np.mean(accs)) if accs else float("nan")
    out["discrete"]["mean_macro_f1"] = float(np.mean(macro_f1s)) if macro_f1s else float("nan")
    return out


def precursor_metrics(true_sets: Sequence[Set[str]], pred_sets: Sequence[Set[str]], stage2_vocab: Set[str]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    for i, (t, p) in enumerate(zip(true_sets, pred_sets)):
        precision, recall, f1, jaccard, exact, any_overlap = set_prf(t, p)
        oov = sorted(x for x in t if x not in stage2_vocab)
        rows.append({
            "row_index": i,
            "true_precursors": json.dumps(sorted(t), ensure_ascii=False),
            "pred_precursors": json.dumps(sorted(p), ensure_ascii=False),
            "n_true": len(t),
            "n_pred": len(p),
            "n_true_oov_stage2": len(oov),
            "all_true_in_stage2_vocab": len(oov) == 0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "exact_match": exact,
            "any_overlap": any_overlap,
        })
    df = pd.DataFrame(rows)
    known = df["all_true_in_stage2_vocab"]
    metrics = {
        "n": int(len(df)),
        "true_token_oov_stage2_rate": float(df["n_true_oov_stage2"].sum() / max(df["n_true"].sum(), 1)),
        "rows_all_true_in_stage2_vocab_rate": float(known.mean()) if len(df) else 0.0,
        "mean_precision": float(df["precision"].mean()),
        "mean_recall": float(df["recall"].mean()),
        "mean_f1": float(df["f1"].mean()),
        "mean_jaccard": float(df["jaccard"].mean()),
        "exact_match_rate": float(df["exact_match"].mean()),
        "any_overlap_rate": float(df["any_overlap"].mean()),
    }
    if known.any():
        sub = df.loc[known]
        metrics["known_vocab_subset"] = {
            "n": int(len(sub)),
            "mean_precision": float(sub["precision"].mean()),
            "mean_recall": float(sub["recall"].mean()),
            "mean_f1": float(sub["f1"].mean()),
            "mean_jaccard": float(sub["jaccard"].mean()),
            "exact_match_rate": float(sub["exact_match"].mean()),
            "any_overlap_rate": float(sub["any_overlap"].mean()),
        }
    return metrics, df


def route_success_metrics(
    row_df: pd.DataFrame,
    y_cont_true_raw: np.ndarray,
    y_cont_pred_raw: np.ndarray,
    y_cont_mask: np.ndarray,
    y_disc_true: np.ndarray,
    y_disc_pred: np.ndarray,
    y_disc_mask: np.ndarray,
) -> Dict[str, float]:
    temp_valid = y_cont_mask[:, 0] > 0.5
    time_valid = y_cont_mask[:, 1] > 0.5
    atm_valid = y_disc_mask[:, 0] > 0.5
    temp_err = np.abs(y_cont_pred_raw[:, 0] - y_cont_true_raw[:, 0])
    time_err = np.abs(y_cont_pred_raw[:, 1] - y_cont_true_raw[:, 1])
    atm_ok = y_disc_pred[:, 0] == y_disc_true[:, 0]

    evaluable = temp_valid & time_valid & atm_valid
    exact_prec = row_df["exact_match"].to_numpy(dtype=bool)
    jacc = row_df["jaccard"].to_numpy(dtype=float)
    any_overlap = row_df["any_overlap"].to_numpy(dtype=bool)

    def rate(mask: np.ndarray) -> float:
        return float(np.mean(mask[evaluable])) if np.any(evaluable) else float("nan")

    return {
        "n_evaluable_temp_time_atmosphere": int(evaluable.sum()),
        "strict_route_success_exact_prec_temp100_time24_atm": rate(
            exact_prec & (temp_err <= 100.0) & (time_err <= 24.0) & atm_ok
        ),
        "relaxed_route_success_jaccard50_temp200_time48_atm": rate(
            (jacc >= 0.5) & (temp_err <= 200.0) & (time_err <= 48.0) & atm_ok
        ),
        "precursor_any_overlap_temp200_time48_atm": rate(
            any_overlap & (temp_err <= 200.0) & (time_err <= 48.0) & atm_ok
        ),
        "condition_only_temp100_time24_atm": rate((temp_err <= 100.0) & (time_err <= 24.0) & atm_ok),
        "condition_only_temp200_time48_atm": rate((temp_err <= 200.0) & (time_err <= 48.0) & atm_ok),
    }


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = ["# Route Stack Evaluation", ""]
    lines.append(f"- Evaluation split: Stage3 test, n={summary['data']['n_stage3_test']}")
    lines.append(f"- Stage2 model: `{summary['artifacts']['stage2_run_dir']}`")
    lines.append(f"- Stage3 closed-loop model: `{summary['artifacts']['stage3_baseline_ckpt']}`")
    lines.append(f"- Residual-flow metrics are teacher-forced references from `{summary['artifacts']['stage3_residual_run_dir']}`")
    lines.append("")
    lines.append("## Closed-loop route metrics")
    for name, block in summary["closed_loop"].items():
        lines.append(f"### {name}")
        pm = block["precursor_metrics"]
        cm = block["condition_metrics"]
        rm = block["route_success"]
        lines.append(f"- precursor exact={pm['exact_match_rate']:.4f}, mean F1={pm['mean_f1']:.4f}, mean Jaccard={pm['mean_jaccard']:.4f}, any-overlap={pm['any_overlap_rate']:.4f}")
        lines.append(f"- condition mean MAE={cm['continuous']['mean_mae']:.2f}, temp MAE={cm['continuous']['target_temperature_c']['mae']:.2f} C, time MAE={cm['continuous']['target_time_h']['mae']:.2f} h")
        lines.append(f"- atmosphere acc={cm['discrete']['target_atmosphere']['accuracy']:.4f}, synthesis-type acc={cm['discrete']['synthesis_type']['accuracy']:.4f}")
        lines.append(f"- strict route success={rm['strict_route_success_exact_prec_temp100_time24_atm']:.4f}; relaxed route success={rm['relaxed_route_success_jaccard50_temp200_time48_atm']:.4f}")
        lines.append("")
    lines.append("## Teacher-forced references")
    tf = summary["teacher_forced_stage3_baseline"]
    lines.append(f"- Stage3 baseline with true precursors: mean MAE={tf['continuous']['mean_mae']:.2f}, temp MAE={tf['continuous']['target_temperature_c']['mae']:.2f} C, time MAE={tf['continuous']['target_time_h']['mae']:.2f} h")
    rf = summary.get("teacher_forced_stage3_residual_flow", {})
    if rf:
        lines.append(f"- Stage3 residual flow with true precursors: test mean MAE={rf.get('top1_continuous_mean_mae_raw', float('nan')):.2f}, temp MAE={rf.get('target_temperature_c_mae', float('nan')):.2f} C, time MAE={rf.get('target_time_h_mae', float('nan')):.2f} h")
    lines.append("")
    lines.append("Note: closed-loop route metrics use Stage2-predicted precursor sets followed by the Stage3 linear baseline. Residual-flow numbers are not closed-loop here because the saved residual test predictions were generated with true precursor sets.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the new Stage2 + all-data Stage3 route stack on Stage3 test rows.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--feature_dir", default="data/interim/features/structdesc_features_task_new_stage2_all_stage3_20260609_poscar_geom1024")
    ap.add_argument("--stage2_dataset_dir", default="data/interim/generative/stage2_setpred_dataset/descriptor/new_structured_20260609_relaxed_only")
    ap.add_argument("--stage3_dataset_dir", default="data/interim/generative/stage3_condition_dataset_mixed/new_stage2_all_stage3_20260609_poscar_geom1024")
    ap.add_argument("--stage2_run_dir", default="runs/stage2/mlp_new_structured_20260609_descriptor")
    ap.add_argument("--stage3_baseline_ckpt", default="runs/stage3/baseline_linear_new_stage2_all_stage3_20260609/best_model.pkl")
    ap.add_argument("--stage3_residual_run_dir", default="runs/stage3/residual_flow_mixed_new_stage2_all_stage3_20260609")
    ap.add_argument("--output_dir", default="outputs/evaluation/route_stack_new_stage2_all_stage3_20260609")
    ap.add_argument("--stage2_threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    feature_dir = root / args.feature_dir
    stage2_dataset_dir = root / args.stage2_dataset_dir
    stage3_dataset_dir = root / args.stage3_dataset_dir
    stage2_run_dir = root / args.stage2_run_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stage3_raw = pd.read_csv(feature_dir / "stage3_test_raw.csv")
    stage3_npz = np.load(stage3_dataset_dir / "test.npz", allow_pickle=True)
    schema = load_json(stage3_dataset_dir / "schema.json")
    cont_names = [str(x) for x in schema["continuous_cols"]]
    disc_names = [str(x) for x in schema["discrete_cols"]]
    stage3_vocab = [str(x) for x in schema["precursor_vocab"]]

    feature_cols = [str(x) for x in load_json(stage2_dataset_dir / "feature_cols.json")]
    mean = np.load(stage2_dataset_dir / "feature_mean.npy")
    std = np.load(stage2_dataset_dir / "feature_std.npy")
    stage2_x = build_stage2_x(feature_dir / "stage3_test_ml.csv", feature_cols, mean, std)
    stage2_model, stage2_names = load_stage2_model(stage2_run_dir, stage2_x.shape[1])
    stage2_probs = predict_stage2_probs(stage2_model, stage2_x, args.batch_size)
    stage2_vocab = set(stage2_names)

    true_sets = [set(parse_precursor_list(x)) for x in stage3_raw["target_main_precursors"].tolist()]
    true_counts = np.asarray([len(x) for x in true_sets], dtype=int)

    threshold_y = threshold_predict(stage2_probs, args.stage2_threshold)
    toptrue_y = topk_predict(stage2_probs, np.maximum(true_counts, 1))

    pred_variants = {
        "threshold_0.5": binary_to_sets(threshold_y, stage2_names),
        "diagnostic_top_true_count": binary_to_sets(toptrue_y, stage2_names),
    }

    x_stage3 = np.asarray(stage3_npz["x"], dtype=np.float32)
    y_set_true = np.asarray(stage3_npz["y_set"], dtype=np.float32)
    y_cont_true_norm = np.asarray(stage3_npz["y_cond_continuous"], dtype=np.float32)
    y_cont_mask = np.asarray(stage3_npz["y_cond_continuous_mask"], dtype=np.float32)
    y_disc_true = np.asarray(stage3_npz["y_cond_discrete"])
    y_disc_mask = np.asarray(stage3_npz["y_cond_discrete_mask"], dtype=np.float32)
    y_cont_true_raw = raw_continuous(y_cont_true_norm, schema, cont_names)

    stage3_model = load_stage3_baseline(root / args.stage3_baseline_ckpt, root)

    def predict_conditions(y_set: np.ndarray) -> Dict[str, Any]:
        X = np.concatenate([x_stage3, y_set], axis=1).astype(np.float32)
        pred = stage3_model.predict(X)
        y_cont_pred_norm = np.asarray(pred["y_cont_pred"], dtype=np.float32)
        y_cont_pred_raw = raw_continuous(y_cont_pred_norm, schema, cont_names)
        y_disc_pred = np.asarray(pred["y_disc_pred"])
        return {
            "y_cont_pred_raw": y_cont_pred_raw,
            "y_disc_pred": y_disc_pred,
            "metrics": condition_metrics(
                y_cont_true_raw, y_cont_pred_raw, y_cont_mask,
                y_disc_true, y_disc_pred, y_disc_mask,
                cont_names, disc_names,
            ),
        }

    teacher = predict_conditions(y_set_true)

    closed_loop: Dict[str, Any] = {}
    row_tables = []
    for name, pred_sets in pred_variants.items():
        pm, row_df = precursor_metrics(true_sets, pred_sets, stage2_vocab)
        y_set_pred, vocab_map_stats = make_stage3_yset(pred_sets, stage3_vocab)
        cond = predict_conditions(y_set_pred)
        route_m = route_success_metrics(
            row_df, y_cont_true_raw, cond["y_cont_pred_raw"], y_cont_mask,
            y_disc_true, cond["y_disc_pred"], y_disc_mask,
        )
        row_df.insert(0, "variant", name)
        row_df["sample_id"] = stage3_npz["sample_id"].astype(str)
        row_df["material_id"] = stage3_raw["material_id"].astype(str).to_numpy()
        row_df["formula"] = stage3_raw["formula"].astype(str).to_numpy()
        row_tables.append(row_df)
        closed_loop[name] = {
            "precursor_metrics": pm,
            "stage3_vocab_mapping": vocab_map_stats,
            "condition_metrics": cond["metrics"],
            "route_success": route_m,
        }

    residual_metrics_path = root / args.stage3_residual_run_dir / "metrics.json"
    residual_ref: Dict[str, Any] = {}
    if residual_metrics_path.exists():
        residual_summary = load_json(residual_metrics_path)
        residual_ref = residual_summary.get("test_metrics", {})

    summary = {
        "data": {
            "n_stage3_test": int(len(stage3_raw)),
            "stage3_test_source_counts": stage3_raw["source_dataset"].value_counts(dropna=False).to_dict(),
            "stage2_vocab_size": int(len(stage2_names)),
            "stage3_vocab_size": int(len(stage3_vocab)),
        },
        "closed_loop": closed_loop,
        "teacher_forced_stage3_baseline": teacher["metrics"],
        "teacher_forced_stage3_residual_flow": residual_ref,
        "artifacts": {
            "stage2_run_dir": str(stage2_run_dir),
            "stage3_baseline_ckpt": str((root / args.stage3_baseline_ckpt).resolve()),
            "stage3_residual_run_dir": str((root / args.stage3_residual_run_dir).resolve()),
            "summary_json": str((output_dir / "route_stack_eval_summary.json").resolve()),
            "row_metrics_csv": str((output_dir / "route_stack_eval_rows.csv").resolve()),
            "report_md": str((output_dir / "route_stack_eval_report.md").resolve()),
        },
    }

    rows = pd.concat(row_tables, ignore_index=True)
    rows.to_csv(output_dir / "route_stack_eval_rows.csv", index=False)
    write_json(output_dir / "route_stack_eval_summary.json", summary)
    (output_dir / "route_stack_eval_report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
