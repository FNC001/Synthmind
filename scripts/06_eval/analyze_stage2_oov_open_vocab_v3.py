#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    if isinstance(obj, set):
        return sorted(str(x) for x in obj)
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


def parse_list(s: Any) -> List[str]:
    try:
        obj = json.loads(str(s))
        if isinstance(obj, list):
            return [str(x) for x in obj]
    except Exception:
        pass
    return []


def markdown(summary: Dict[str, Any]) -> str:
    m = summary["metrics"]
    lines = ["# Stage2 v3 OOV/Open-Vocab Analysis", ""]
    for k, v in m.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## OOV by Reaction Method")
    for k, v in summary["oov_by_reaction_method"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## OOV by Family")
    for k, v in summary["oov_by_family"].items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze OOV and open-vocab contribution for Stage2 v3 candidate pool.")
    ap.add_argument("--v2_oov_rows", required=True)
    ap.add_argument("--v3_candidate_csv", required=True)
    ap.add_argument("--ontology_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    oov = pd.read_csv(args.v2_oov_rows)
    cand = pd.read_csv(args.v3_candidate_csv)
    ont = pd.read_csv(args.ontology_csv).set_index("canonical_precursor")
    fam_lookup = ont["precursor_family"].astype(str).to_dict()

    rows = []
    method_counter: Counter = Counter()
    family_counter: Counter = Counter()
    generated_oov_hits = 0
    ontology_open_oov_hits = 0
    exact_oov_by_k = {1: 0, 10: 0, 50: 0, 100: 0, 200: 0, 500: 0}
    oov_rows = oov[oov["has_oov"].astype(bool)].copy()
    cand_groups = {int(i): g.copy() for i, g in cand.groupby("sample_index", sort=False)}
    for _, row in oov_rows.iterrows():
        i = int(row["sample_index"])
        oov_labels = set(parse_list(row["oov_precursors"]))
        method_counter[str(row["reaction_method"])] += len(oov_labels)
        for lab in oov_labels:
            family_counter[fam_lookup.get(lab, "unknown")] += 1
        group = cand_groups.get(i, pd.DataFrame())
        hit_oov = False
        hit_oov_generated = False
        hit_oov_open = False
        for _, c in group.iterrows():
            pred = set(parse_list(c["pred_precursors"]))
            if oov_labels & pred:
                hit_oov = True
                mix = str(c.get("candidate_source_mix", ""))
                if "generated" in mix:
                    hit_oov_generated = True
                if "ontology_open" in mix:
                    hit_oov_open = True
            if bool(c.get("exact", False)):
                for k in exact_oov_by_k:
                    if int(c["rank"]) <= k:
                        exact_oov_by_k[k] += 1
                break
        generated_oov_hits += int(hit_oov_generated)
        ontology_open_oov_hits += int(hit_oov_open)
        rows.append({
            "sample_index": i,
            "id": row["id"],
            "formula": row["formula"],
            "reaction_method": row["reaction_method"],
            "oov_precursors": row["oov_precursors"],
            "n_oov": int(row["n_oov"]),
            "any_oov_label_in_v3_pool": bool(hit_oov),
            "oov_label_from_generated_candidate": bool(hit_oov_generated),
            "oov_label_from_ontology_open_candidate": bool(hit_oov_open),
            **{f"exact_in_top{k}": bool(group[group["rank"] <= k]["exact"].any()) if not group.empty else False for k in exact_oov_by_k},
        })
    row_df = pd.DataFrame(rows)
    n_oov_rows = max(len(row_df), 1)
    summary = {
        "config": vars(args),
        "metrics": {
            "n_oov_rows": int(len(row_df)),
            "oov_rows_with_any_oov_label_in_pool": float(row_df["any_oov_label_in_v3_pool"].mean()) if len(row_df) else 0.0,
            "oov_rows_with_generated_oov_label": float(row_df["oov_label_from_generated_candidate"].mean()) if len(row_df) else 0.0,
            "oov_rows_with_ontology_open_oov_label": float(row_df["oov_label_from_ontology_open_candidate"].mean()) if len(row_df) else 0.0,
            **{f"oov_exact_top{k}": float(row_df[f"exact_in_top{k}"].mean()) if len(row_df) else 0.0 for k in exact_oov_by_k},
        },
        "oov_by_reaction_method": dict(method_counter.most_common()),
        "oov_by_family": dict(family_counter.most_common()),
        "artifacts": {
            "csv": str((out_dir / "oov_analysis.csv").resolve()),
            "report": str((out_dir / "oov_analysis_report.md").resolve()),
            "summary": str((out_dir / "oov_analysis_summary.json").resolve()),
        },
    }
    row_df.to_csv(out_dir / "oov_analysis.csv", index=False)
    write_json(out_dir / "oov_analysis_summary.json", summary)
    (out_dir / "oov_analysis_report.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
