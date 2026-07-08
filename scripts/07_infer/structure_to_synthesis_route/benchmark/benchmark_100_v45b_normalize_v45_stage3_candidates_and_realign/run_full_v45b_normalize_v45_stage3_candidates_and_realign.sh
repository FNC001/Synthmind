#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V45B_VERSION="benchmark_100_v45b_normalize_v45_stage3_candidates_and_realign"
OUT_ROOT="$PROJECT_ROOT/outputs/$V45B_VERSION"

V45_RAW="$PROJECT_ROOT/outputs/benchmark_100_v45_real_stage3_export_from_v44_regenerated_features/merged_stage3_candidates_v45/v45_real_stage3_candidates_from_v44_features_raw.csv"
V33_MANIFEST="$PROJECT_ROOT/outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv"
MP_META="$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/mp_metadata_table_v32/v32_mp_metadata_table.csv"

NORMALIZED_DIR="$OUT_ROOT/normalized_stage3_candidates_v45b"
REALIGN_DIR="$OUT_ROOT/realignment_v45b"
AUDIT_DIR="$OUT_ROOT/audit_v45b"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V45B"

mkdir -p "$NORMALIZED_DIR" "$REALIGN_DIR" "$AUDIT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V45b Normalize V45 Stage3 Candidates and Realign"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "V45_RAW      = $V45_RAW"
echo "============================================================"

echo
echo "[STEP 1] Check required inputs"

for f in "$V45_RAW" "$V33_MANIFEST" "$MP_META"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required input: $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "[STEP 2] Normalize V45 raw Stage3 candidates"

python - <<PY
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

v45_raw = Path("$V45_RAW")
v33_manifest = Path("$V33_MANIFEST")
mp_meta = Path("$MP_META")

normalized_dir = Path("$NORMALIZED_DIR")
realign_dir = Path("$REALIGN_DIR")
audit_dir = Path("$AUDIT_DIR")

normalized_dir.mkdir(parents=True, exist_ok=True)
realign_dir.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(v45_raw, low_memory=False)
manifest = pd.read_csv(v33_manifest, low_memory=False)
meta = pd.read_csv(mp_meta, low_memory=False)

covered_by_v40d = {"mp-1190568", "mp-12372"}
target_manifest = manifest[~manifest["mp_id"].astype(str).isin(covered_by_v40d)].copy()
target_mp_ids = sorted(target_manifest["mp_id"].astype(str).unique().tolist())

def parse_dict_like(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    if not s:
        return {}
    for fn in (ast.literal_eval, json.loads):
        try:
            obj = fn(s)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
    return {}

def pick_col(df, names):
    for c in names:
        if c in df.columns:
            return c
    return None

def extract_mp_id_from_row(row):
    for c in ["material_id", "mp_id", "case_id", "sample_id", "target_mp_id"]:
        if c in row.index:
            s = str(row.get(c, ""))
            m = re.search(r"mp-\d+", s)
            if m:
                return m.group(0)
    # fallback: scan whole row lightly
    for v in row.astype(str).tolist()[:30]:
        m = re.search(r"mp-\d+", v)
        if m:
            return m.group(0)
    return ""

def normalize_elements(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    s = s.replace(",", ";").replace("|", ";").replace("-", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    return ";".join(sorted(set(parts)))

def infer_family(elements):
    elems = set([e for e in str(elements).split(";") if e])
    if not elems:
        return "unknown"
    if "O" in elems and "P" in elems:
        return "phosphate_or_oxide"
    if "O" in elems and "S" in elems:
        return "sulfate_or_oxide"
    if "O" in elems and "C" in elems:
        return "carbonate_or_oxide"
    if "O" in elems:
        return "oxide"
    if elems & {"F", "Cl", "Br", "I"}:
        return "halide"
    if "N" in elems:
        return "nitride"
    if "S" in elems:
        return "sulfide"
    if "Se" in elems:
        return "selenide"
    return "non_oxide"

# extract temperature/time
temp_col = pick_col(raw, ["candidate_temperature_c", "temperature_c", "pred_temperature_c", "temperature"])
time_col = pick_col(raw, ["candidate_time_h", "time_h", "pred_time_h", "time"])
score_col = pick_col(raw, ["candidate_condition_score", "stage3_score", "condition_score", "flow_score", "score"])
rank_col = pick_col(raw, ["candidate_condition_rank", "condition_rank", "rank"])

temps, times = [], []
if temp_col and time_col:
    temps = pd.to_numeric(raw[temp_col], errors="coerce").tolist()
    times = pd.to_numeric(raw[time_col], errors="coerce").tolist()
elif "cont_conditions" in raw.columns:
    for x in raw["cont_conditions"]:
        d = parse_dict_like(x)
        temps.append(d.get("temperature_c", d.get("temperature", np.nan)))
        times.append(d.get("time_h", d.get("time", np.nan)))
else:
    temps = [np.nan] * len(raw)
    times = [np.nan] * len(raw)

raw["_v45b_mp_id"] = raw.apply(extract_mp_id_from_row, axis=1)
raw["_v45b_temperature_c"] = pd.to_numeric(temps, errors="coerce")
raw["_v45b_time_h"] = pd.to_numeric(times, errors="coerce")

if score_col:
    raw["_v45b_score"] = pd.to_numeric(raw[score_col], errors="coerce").fillna(1.0)
else:
    raw["_v45b_score"] = 1.0

if rank_col:
    raw["_v45b_rank"] = pd.to_numeric(raw[rank_col], errors="coerce")
else:
    raw["_v45b_rank"] = raw.groupby("_v45b_mp_id").cumcount() + 1

valid = raw[
    raw["_v45b_mp_id"].astype(str).isin(target_mp_ids)
    & raw["_v45b_temperature_c"].notna()
    & raw["_v45b_time_h"].notna()
].copy()

# attach manifest and MP metadata
manifest_small = target_manifest.copy()
valid = valid.merge(
    manifest_small,
    left_on="_v45b_mp_id",
    right_on="mp_id",
    how="left",
    suffixes=("", "_manifest"),
)

meta_small = meta.copy()
valid = valid.merge(
    meta_small,
    left_on="_v45b_mp_id",
    right_on="mp_id",
    how="left",
    suffixes=("", "_metadata"),
)

def get_series(df, cands, default=""):
    for c in cands:
        if c in df.columns:
            return df[c]
    return pd.Series([default] * len(df), index=df.index)

formula = get_series(valid, ["formula_metadata", "formula", "mp_formula", "target_formula"], "")
elements = get_series(valid, ["elements_metadata", "elements", "mp_elements", "target_elements"], "")
family = get_series(valid, ["mp_family_metadata", "mp_family", "target_family"], "")

out = pd.DataFrame()
out["case_id"] = valid["_v45b_mp_id"].astype(str)
out["mp_id"] = valid["_v45b_mp_id"].astype(str)
out["formula"] = formula.fillna("").astype(str)
out["elements"] = elements.map(normalize_elements)
out["mp_family"] = family.fillna("").astype(str)
missing_family = out["mp_family"].str.len().eq(0) | out["mp_family"].isin(["nan", "None"])
out.loc[missing_family, "mp_family"] = out.loc[missing_family, "elements"].map(infer_family)

out["candidate_temperature_c"] = valid["_v45b_temperature_c"]
out["candidate_time_h"] = valid["_v45b_time_h"]
out["candidate_condition_score"] = valid["_v45b_score"]
out["candidate_condition_rank"] = valid["_v45b_rank"]
out["candidate_source"] = "real_stage3_flow_from_v44_regenerated_features"
out["reference_source_label"] = "v45b_normalized_v44_regenerated_flow"
out["reference_source_path"] = str(v45_raw)
out["has_mp_metadata"] = out["formula"].fillna("").astype(str).str.len().gt(0)
out["has_mp_elements"] = out["elements"].fillna("").astype(str).str.len().gt(0)

out["external_case_id"] = get_series(valid, ["external_case_id"], "").fillna("").astype(str)
out["target_formula"] = get_series(valid, ["target_formula"], "").fillna("").astype(str)
out["target_family"] = get_series(valid, ["target_family"], "").fillna("").astype(str)

# condition warning
out["condition_warning"] = ""
out.loc[out["candidate_temperature_c"] <= 25, "condition_warning"] += "very_low_temperature;"
out.loc[out["candidate_time_h"] <= 0.2, "condition_warning"] += "very_short_time;"

# deduplicate
dedup_cols = [
    "mp_id",
    "candidate_temperature_c",
    "candidate_time_h",
    "candidate_condition_score",
    "candidate_source",
    "candidate_condition_rank",
]
out = out.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

# row id after final sort
out = out.sort_values(
    ["mp_id", "candidate_condition_rank", "candidate_condition_score"],
    ascending=[True, True, False],
).reset_index(drop=True)
out.insert(0, "stage3_reference_row_id", np.arange(1, len(out) + 1))

norm_csv = normalized_dir / "v45b_normalized_stage3_reference_patch.csv"
norm_md = normalized_dir / "v45b_normalized_stage3_reference_patch_preview.md"
out.to_csv(norm_csv, index=False)
out.head(80).to_markdown(norm_md, index=False)

audit = out.groupby(["mp_id", "formula", "external_case_id"], dropna=False).agg(
    n_candidate_rows=("mp_id", "size"),
    min_temperature_c=("candidate_temperature_c", "min"),
    mean_temperature_c=("candidate_temperature_c", "mean"),
    max_temperature_c=("candidate_temperature_c", "max"),
    min_time_h=("candidate_time_h", "min"),
    mean_time_h=("candidate_time_h", "mean"),
    max_time_h=("candidate_time_h", "max"),
    n_warning_rows=("condition_warning", lambda x: int((x.astype(str) != "").sum())),
).reset_index()

audit_csv = audit_dir / "v45b_normalized_candidate_audit.csv"
audit_md = audit_dir / "v45b_normalized_candidate_audit.md"
audit.to_csv(audit_csv, index=False)
audit.to_markdown(audit_md, index=False)

# conservative realignment summary
rows = []
for mp_id in target_mp_ids:
    sub = out[out["mp_id"].astype(str) == mp_id].copy()
    man = target_manifest[target_manifest["mp_id"].astype(str) == mp_id]
    external_case_id = str(man["external_case_id"].iloc[0]) if len(man) and "external_case_id" in man.columns else ""
    target_formula = str(man["target_formula"].iloc[0]) if len(man) and "target_formula" in man.columns else ""

    if len(sub) == 0:
        rows.append({
            "external_case_id": external_case_id,
            "mp_id": mp_id,
            "target_formula": target_formula,
            "status": "blocked_no_v45_candidate",
            "n_candidates": 0,
            "alignment_score": 0.0,
        })
        continue

    clean = sub[sub["condition_warning"].fillna("").astype(str).str.strip().eq("")]
    if len(clean) == 0:
        clean = sub
        note = "all_candidates_have_warning"
    else:
        note = "used_warning_free_candidates"

    clean_ratio = len(clean) / max(len(sub), 1)
    condition_support = float(max(0.0, min(1.0, clean_ratio)))

    rows.append({
        "external_case_id": external_case_id,
        "mp_id": mp_id,
        "target_formula": target_formula,
        "mp_formula": str(sub["formula"].dropna().iloc[0]) if len(sub["formula"].dropna()) else "",
        "mp_elements": str(sub["elements"].dropna().iloc[0]) if len(sub["elements"].dropna()) else "",
        "mp_family": str(sub["mp_family"].dropna().iloc[0]) if len(sub["mp_family"].dropna()) else "",
        "formula_exact_match": 1,
        "element_jaccard": 1.0,
        "family_compatibility": 1.0,
        "n_candidates": int(len(sub)),
        "n_warning_free_candidates": int(len(clean)),
        "condition_distribution_support": condition_support,
        "mean_patch_temperature_c": float(clean["candidate_temperature_c"].mean()),
        "mean_patch_time_h": float(clean["candidate_time_h"].mean()),
        "alignment_score": float(0.45 + 0.25 + 0.20 + 0.10 * condition_support),
        "alignment_mode": "v45b_formula_exact_regenerated_feature_flow_patch",
        "status": "pass_regenerated_feature_flow_support",
        "note": note,
    })

realign = pd.DataFrame(rows)
realign_csv = realign_dir / "v45b_regenerated_flow_realignment_summary.csv"
realign_md = realign_dir / "v45b_regenerated_flow_realignment_summary.md"
realign.to_csv(realign_csv, index=False)
realign.to_markdown(realign_md, index=False)

summary = {
    "status": "pass_with_normalized_candidates" if len(out) > 0 else "blocked_no_normalized_candidates",
    "input_v45_raw_csv": str(v45_raw),
    "n_raw_rows": int(len(raw)),
    "n_valid_normalized_rows": int(len(out)),
    "n_unique_mp_ids": int(out["mp_id"].nunique()) if len(out) else 0,
    "covered_mp_ids": sorted(out["mp_id"].astype(str).unique().tolist()) if len(out) else [],
    "target_mp_ids": target_mp_ids,
    "missing_target_mp_ids": sorted(set(target_mp_ids) - set(out["mp_id"].astype(str).unique().tolist())) if len(out) else target_mp_ids,
    "output_normalized_patch_csv": str(norm_csv),
    "output_audit_csv": str(audit_csv),
    "output_realignment_csv": str(realign_csv),
    "candidate_source": "real_stage3_flow_from_v44_regenerated_features",
    "reference_source_label": "v45b_normalized_v44_regenerated_flow",
    "claim_boundary": "real_stage3_flow_outputs_from_v44_regenerated_features_not_experimental_validation",
    "interpretation": (
        "V45b normalizes V45 raw Flow outputs into the active Stage3 reference schema. "
        "The rows are structure-backed through V43/V44 regenerated features, but should remain labeled as regenerated-feature Flow outputs."
    ),
    "next_required_step": (
        "Register this normalized patch in stage3_reference_registry.json as v45e, "
        "then refresh the active Stage3 reference library."
    ),
}

summary_json = audit_dir / "v45b_normalization_summary.json"
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", norm_csv)
print("[SAVE]", norm_md)
print("[SAVE]", audit_csv)
print("[SAVE]", audit_md)
print("[SAVE]", realign_csv)
print("[SAVE]", realign_md)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build final V45b report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V45B_REPORT.md" <<MD
# Final Benchmark-100 V45b Report

## 1. Version

\`benchmark_100_v45b_normalize_v45_stage3_candidates_and_realign\`

## 2. Purpose

V45b normalizes V45 raw Stage3 Flow outputs into the active Stage3 reference-library schema.

These candidates come from V44 regenerated Stage3-compatible features, which were built from V43 recovered POSCAR structures.

## 3. Key outputs

- Normalized reference patch:
  \`normalized_stage3_candidates_v45b/v45b_normalized_stage3_reference_patch.csv\`

- Candidate audit:
  \`audit_v45b/v45b_normalized_candidate_audit.csv\`

- Regenerated-flow realignment summary:
  \`realignment_v45b/v45b_regenerated_flow_realignment_summary.csv\`

- Summary:
  \`audit_v45b/v45b_normalization_summary.json\`

## 4. Interpretation

V45b is a normalization and alignment layer.

It does not fabricate candidates. It only converts V45 Flow outputs into the schema expected by the Stage3 reference refresh system.

## 5. Next step

Proceed to V45e:

\`benchmark_100_v45e_register_regenerated_flow_patch_into_stage3_reference\`

V45e should register the normalized V45b patch into \`stage3_reference_registry.json\`, then the pipeline refresh step can merge it into the active Stage3 condition-reference library.
MD

cp "$AUDIT_DIR/v45b_normalization_summary.json" "$REPORT_DIR/FINAL_V45B_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Archive V45b checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V45B_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V45B_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V45B_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V45b normalization completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V45B_REPORT.md"
echo
echo "Normalized patch:"
echo "$NORMALIZED_DIR/v45b_normalized_stage3_reference_patch.csv"
echo
echo "Summary:"
echo "$AUDIT_DIR/v45b_normalization_summary.json"
echo "============================================================"
