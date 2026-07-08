#!/usr/bin/env python3
"""Summarize inference results into a compact Markdown + JSON report.

Pulls the target structure info from data/interim/infer/<infer_name>/infer_structdesc.csv
and the recommended routes from outputs/inference/<infer_name>/routes_*/final_recommended_routes.csv,
then writes:
  outputs/inference/<infer_name>/summary.md
  outputs/inference/<infer_name>/summary.json

Usage:
  python summarize_inference.py --infer_name case_000001_batch_001_poscars_POSCAR
  python summarize_inference.py --infer_name <name> --top_k 5 --project_root /path/to/SynPred
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_ROUTE_COLS = [
    "final_recommendation_rank",
    "final_recommendation_score",
    "final_recommendation_status",
    "final_recommendation_penalty_reason",
    "precursor_set",
    "temperature_c",
    "time_h",
    "pred_atmosphere",
    "pred_atmosphere_proba",
    "pred_time_bucket",
    "route_template_primary",
    "route_template_secondary",
    "route_confidence_score",
    "route_confidence_level",
    "route_warning_level",
    "precursor_qc_level",
    "real_stage3_condition_reference_support_score",
    "condition_distribution_support_score",
    "stage35_v43_safe_strict_score",
    "condition_temperature_center_c",
    "condition_temperature_width_c",
    "real_stage3_condition_reference_temperature_center_c",
    "real_stage3_condition_reference_temperature_width_c",
]


def find_route_dir(out_dir: Path) -> Path:
    candidates = sorted(out_dir.glob("routes_*"))
    if not candidates:
        raise FileNotFoundError(f"No routes_* directory under {out_dir}")
    for c in candidates:
        if (c / "final_recommended_routes.csv").exists():
            return c
    return candidates[0]


def load_structures(structdesc_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(structdesc_csv)
    keep = [
        "sample_id", "material_id", "formula",
        "feat_poscar_a", "feat_poscar_b", "feat_poscar_c",
        "feat_poscar_alpha", "feat_poscar_beta", "feat_poscar_gamma",
        "feat_poscar_volume", "feat_poscar_nsites",
        "feat_n_elements_formula", "feat_total_atoms_formula",
        "feat_stoich_entropy",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    el_cols = [c for c in pd.read_csv(structdesc_csv, nrows=0).columns
               if c.startswith("feat_frac_el__")]
    if el_cols:
        full = pd.read_csv(structdesc_csv, usecols=["sample_id"] + el_cols)
        full["element_fractions"] = full[el_cols].apply(
            lambda r: {c.replace("feat_frac_el__", ""): float(r[c])
                       for c in el_cols if r[c] > 0},
            axis=1,
        )
        df = df.merge(full[["sample_id", "element_fractions"]], on="sample_id", how="left")
    return df


def fmt_lattice(row: pd.Series) -> str:
    a = row.get("feat_poscar_a")
    b = row.get("feat_poscar_b")
    c = row.get("feat_poscar_c")
    al = row.get("feat_poscar_alpha")
    be = row.get("feat_poscar_beta")
    ga = row.get("feat_poscar_gamma")
    if pd.isna(a):
        return "n/a"
    return (f"a={a:.3f} Å, b={b:.3f} Å, c={c:.3f} Å, "
            f"α={al:.2f}°, β={be:.2f}°, γ={ga:.2f}°")


def fmt_elements(d) -> str:
    if not isinstance(d, dict) or not d:
        return ""
    return ", ".join(f"{k}: {v:.3f}" for k, v in sorted(d.items(), key=lambda x: -x[1]))


def make_topk_table(df: pd.DataFrame, top_k: int) -> str:
    cols = [
        ("final_recommendation_rank", "Rank"),
        ("precursor_set", "Precursors"),
        ("temperature_c", "T (°C)"),
        ("time_h", "t (h)"),
        ("pred_atmosphere", "Atmosphere"),
        ("final_recommendation_score", "Score"),
        ("final_recommendation_status", "Status"),
        ("route_confidence_level", "Confidence"),
        ("route_warning_level", "Warning"),
        ("route_template_primary", "Template"),
    ]
    cols = [(k, v) for k, v in cols if k in df.columns]
    head = df.head(top_k)
    lines = ["| " + " | ".join(v for _, v in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in head.iterrows():
        cells = []
        for k, _ in cols:
            v = r[k]
            if pd.isna(v):
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize_metrics(df: pd.DataFrame) -> dict:
    out = {
        "n_routes": int(len(df)),
        "status_counts": df["final_recommendation_status"].value_counts().to_dict()
        if "final_recommendation_status" in df.columns else {},
        "confidence_level_counts": df["route_confidence_level"].value_counts().to_dict()
        if "route_confidence_level" in df.columns else {},
        "warning_level_counts": df["route_warning_level"].value_counts().to_dict()
        if "route_warning_level" in df.columns else {},
        "precursor_qc_level_counts": df["precursor_qc_level"].value_counts().to_dict()
        if "precursor_qc_level" in df.columns else {},
        "atmosphere_counts": df["pred_atmosphere"].value_counts().to_dict()
        if "pred_atmosphere" in df.columns else {},
    }
    if "final_recommendation_score" in df.columns:
        s = df["final_recommendation_score"].dropna()
        out["score_stats"] = {
            "mean": float(s.mean()), "min": float(s.min()),
            "max": float(s.max()), "top1": float(s.iloc[0]) if len(s) else None,
        }
    if "temperature_c" in df.columns:
        t = df["temperature_c"].dropna()
        out["temperature_c_stats"] = {
            "mean": float(t.mean()), "min": float(t.min()), "max": float(t.max()),
        }
    if "time_h" in df.columns:
        h = df["time_h"].dropna()
        out["time_h_stats"] = {
            "mean": float(h.mean()), "min": float(h.min()), "max": float(h.max()),
        }
    return out


def render_markdown(infer_name: str, manifest: dict, structures: pd.DataFrame,
                    routes: pd.DataFrame, top_k: int, route_dir: Path) -> str:
    out = [f"# Inference Summary — {infer_name}", ""]
    out.append(f"- Routes dir: `{route_dir.relative_to(route_dir.parents[1])}`")
    if manifest:
        degraded = manifest.get("degraded_steps") or []
        if degraded:
            out.append(f"- Degraded steps: {len(degraded)}")
            for d in degraded:
                out.append(f"  - `{d.get('step')}`: {d.get('reason')}")
    out.append("")

    out.append("## Target structure(s)")
    out.append("")
    for _, s in structures.iterrows():
        out.append(f"### `{s.get('sample_id', '')}`")
        out.append(f"- Formula: **{s.get('formula', '')}**")
        out.append(f"- Material id: `{s.get('material_id', '')}`")
        out.append(f"- Lattice: {fmt_lattice(s)}")
        if "feat_poscar_volume" in s and not pd.isna(s["feat_poscar_volume"]):
            out.append(f"- Volume: {s['feat_poscar_volume']:.2f} Å³, "
                       f"sites: {int(s.get('feat_poscar_nsites', 0))}, "
                       f"elements: {int(s.get('feat_n_elements_formula', 0))}")
        if "element_fractions" in s:
            out.append(f"- Element fractions: {fmt_elements(s['element_fractions'])}")
        out.append("")

    out.append(f"## Top-{top_k} recommended routes")
    out.append("")
    if "sample_id" in routes.columns and routes["sample_id"].nunique() > 1:
        for sid, sub in routes.groupby("sample_id", sort=False):
            out.append(f"### `{sid}`")
            out.append("")
            out.append(make_topk_table(sub, top_k))
            out.append("")
    else:
        out.append(make_topk_table(routes, top_k))
        out.append("")

    metrics = summarize_metrics(routes)
    out.append("## Evaluation metrics")
    out.append("")
    out.append(f"- Total routes: **{metrics['n_routes']}**")
    if metrics.get("score_stats"):
        s = metrics["score_stats"]
        out.append(f"- Score: top1={s['top1']:.3f}, mean={s['mean']:.3f}, "
                   f"range=[{s['min']:.3f}, {s['max']:.3f}]")
    if metrics.get("temperature_c_stats"):
        t = metrics["temperature_c_stats"]
        out.append(f"- Temperature (°C): mean={t['mean']:.1f}, "
                   f"range=[{t['min']:.1f}, {t['max']:.1f}]")
    if metrics.get("time_h_stats"):
        h = metrics["time_h_stats"]
        out.append(f"- Time (h): mean={h['mean']:.1f}, "
                   f"range=[{h['min']:.1f}, {h['max']:.1f}]")
    out.append("")
    for label, key in [
        ("Status", "status_counts"),
        ("Confidence level", "confidence_level_counts"),
        ("Warning level", "warning_level_counts"),
        ("Precursor QC", "precursor_qc_level_counts"),
        ("Atmosphere", "atmosphere_counts"),
    ]:
        d = metrics.get(key) or {}
        if d:
            parts = ", ".join(f"{k}={v}" for k, v in d.items())
            out.append(f"- {label}: {parts}")
    out.append("")
    out.append("> Score is an internal ranking score, not experimental validation.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infer_name", required=True)
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--route_dir", default=None,
                    help="Override routes_* dir (default: auto-detect)")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = root / "outputs" / "inference" / args.infer_name
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)

    route_dir = Path(args.route_dir) if args.route_dir else find_route_dir(out_dir)
    routes_csv = route_dir / "final_recommended_routes.csv"
    if not routes_csv.exists():
        raise FileNotFoundError(routes_csv)
    routes = pd.read_csv(routes_csv)
    keep = [c for c in KEY_ROUTE_COLS if c in routes.columns]
    extra = [c for c in routes.columns if c in {"sample_id", "material_id"}]
    routes_view = routes[keep + [c for c in extra if c not in keep]].copy()

    structdesc_csv = root / "data" / "interim" / "infer" / args.infer_name / "infer_structdesc.csv"
    structures = load_structures(structdesc_csv) if structdesc_csv.exists() else pd.DataFrame()

    manifest_path = out_dir / "pipeline_v3_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    md = render_markdown(args.infer_name, manifest, structures, routes_view,
                         args.top_k, route_dir)
    md_path = out_dir / "summary.md"
    md_path.write_text(md)

    payload = {
        "infer_name": args.infer_name,
        "route_dir": str(route_dir),
        "n_structures": int(len(structures)),
        "structures": structures.assign(
            element_fractions=structures["element_fractions"]
            if "element_fractions" in structures.columns else None
        ).to_dict(orient="records") if len(structures) else [],
        "top_routes": routes_view.head(args.top_k).to_dict(orient="records"),
        "metrics": summarize_metrics(routes_view),
        "degraded_steps": manifest.get("degraded_steps", []),
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
