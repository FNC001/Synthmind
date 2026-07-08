#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V45B_VERSION="benchmark_100_v45b_normalize_v45_stage3_candidates_and_register_patch"
OUT_ROOT="$PROJECT_ROOT/outputs/$V45B_VERSION"

V45_RAW="$PROJECT_ROOT/outputs/benchmark_100_v45_real_stage3_export_from_v44_regenerated_features/merged_stage3_candidates_v45/v45_real_stage3_candidates_from_v44_features_raw.csv"
V33_MANIFEST="$PROJECT_ROOT/outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv"
MP_META="$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/mp_metadata_table_v32/v32_mp_metadata_table.csv"

PATCH_DIR="$OUT_ROOT/registered_reference_patch_v45b"
AUDIT_DIR="$OUT_ROOT/audit_v45b"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V45B"

mkdir -p "$PATCH_DIR" "$AUDIT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V45b Normalize V45 Stage3 Candidates and Register Patch"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "V45_RAW      = $V45_RAW"
echo "============================================================"

echo
echo "[STEP 1] Check inputs"

for f in "$V45_RAW" "$V33_MANIFEST" "$MP_META"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required input: $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "[STEP 2] Normalize V45 raw candidates"

python - <<PY
import ast
import json
import pandas as pd
from pathlib import Path

v45_raw = Path("$V45_RAW")
v33_manifest = Path("$V33_MANIFEST")
mp_meta = Path("$MP_META")

patch_dir = Path("$PATCH_DIR")
audit_dir = Path("$AUDIT_DIR")
patch_dir.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(v45_raw, low_memory=False)
manifest = pd.read_csv(v33_manifest, low_memory=False)
meta = pd.read_csv(mp_meta, low_memory=False)

def parse_cont_conditions(x):
    if pd.isna(x):
        return {}
    if isinstance(x, dict):
        return x
    s = str(x).strip()
    if not s:
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return {}

def first_col(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

id_col = first_col(df, ["material_id", "mp_id", "case_id", "sample_id"])
if id_col is None:
    raise RuntimeError(f"Cannot find material/mp id column in V45 raw columns: {list(df.columns)}")

# Parse condition columns.
if "cont_conditions" in df.columns:
    conds = df["cont_conditions"].map(parse_cont_conditions)
    df["candidate_temperature_c"] = pd.to_numeric(
        [d.get("temperature_c", d.get("temperature", None)) for d in conds],
        errors="coerce",
    )
    df["candidate_time_h"] = pd.to_numeric(
        [d.get("time_h", d.get("time", None)) for d in conds],
        errors="coerce",
    )
else:
    temp_col = first_col(df, ["candidate_temperature_c", "temperature_c", "pred_temperature_c", "temperature"])
    time_col = first_col(df, ["candidate_time_h", "time_h", "pred_time_h", "time"])
    if temp_col is None or time_col is None:
        raise RuntimeError("Cannot find temperature/time columns or cont_conditions in V45 raw candidates.")
    df["candidate_temperature_c"] = pd.to_numeric(df[temp_col], errors="coerce")
    df["candidate_time_h"] = pd.to_numeric(df[time_col], errors="coerce")

score_col = first_col(df, ["candidate_condition_score", "stage3_score", "flow_score", "score"])
rank_col = first_col(df, ["candidate_condition_rank", "condition_rank", "rank"])

df["mp_id"] = df[id_col].astype(str)
df = df[
    df["mp_id"].str.startswith("mp-")
    & df["candidate_temperature_c"].notna()
    & df["candidate_time_h"].notna()
].copy()

# Only V44 seven targets should be in V45b.
v40d_covered = {"mp-1190568", "mp-12372"}
all_v33 = set(manifest["mp_id"].astype(str))
v44_targets = sorted(all_v33 - v40d_covered)
df = df[df["mp_id"].isin(v44_targets)].copy()

# Attach V33 manifest.
manifest_small = manifest.copy()
df = df.merge(
    manifest_small,
    on="mp_id",
    how="left",
    suffixes=("", "_v33"),
)

# Attach MP metadata.
meta_small = meta.copy()
df = df.merge(
    meta_small,
    on="mp_id",
    how="left",
    suffixes=("", "_metadata"),
)

def pick(row, cols, default=""):
    for c in cols:
        if c in row.index:
            v = row[c]
            if pd.notna(v) and str(v).strip() not in {"", "nan", "None"}:
                return v
    return default

rows = []
for _, r in df.iterrows():
    mp_id = str(r["mp_id"])

    formula = pick(r, ["formula_metadata", "formula", "mp_formula", "target_formula"], "")
    elements = pick(r, ["elements_metadata", "elements", "mp_elements", "target_elements"], "")
    family = pick(r, ["mp_family_metadata", "mp_family", "target_family"], "")

    candidate_source = "real_stage3_flow_from_v44_regenerated_features"
    if "stage3_model" in r.index and pd.notna(r["stage3_model"]):
        model_name = str(r["stage3_model"])
    else:
        model_name = "stage3_flow"

    if score_col is not None:
        score = pd.to_numeric(r.get(score_col), errors="coerce")
        if pd.isna(score):
            score = 1.0
    else:
        score = 1.0

    if rank_col is not None:
        rank = pd.to_numeric(r.get(rank_col), errors="coerce")
    else:
        rank = None

    rows.append({
        "case_id": mp_id,
        "mp_id": mp_id,
        "formula": str(formula),
        "elements": str(elements),
        "mp_family": str(family),
        "candidate_temperature_c": float(r["candidate_temperature_c"]),
        "candidate_time_h": float(r["candidate_time_h"]),
        "candidate_condition_score": float(score),
        "candidate_condition_rank": rank,
        "candidate_source": candidate_source,
        "stage3_model": model_name,
        "source_split": str(r.get("source_split", "")),
        "external_case_id": str(r.get("external_case_id", "")),
        "target_formula": str(r.get("target_formula", "")),
        "target_family": str(r.get("target_family", "")),
        "reference_source_label": "v45b_registered_v44_regenerated_flow_patch",
        "reference_source_path": str(v45_raw),
        "claim_boundary": "real_stage3_flow_from_regenerated_v44_features_not_experimental_validation",
    })

out = pd.DataFrame(rows)

# Remove obvious duplicate rows.
dedup_cols = [
    "mp_id",
    "candidate_temperature_c",
    "candidate_time_h",
    "candidate_source",
    "source_split",
    "candidate_condition_rank",
]
dedup_cols = [c for c in dedup_cols if c in out.columns]
out = out.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)

# Condition warning, do not delete.
out["condition_warning"] = ""
out.loc[out["candidate_temperature_c"] <= 25, "condition_warning"] += "very_low_temperature;"
out.loc[out["candidate_time_h"] <= 0.2, "condition_warning"] += "very_short_time;"

patch_csv = patch_dir / "v45b_registered_stage3_reference_patch.csv"
patch_md = patch_dir / "v45b_registered_stage3_reference_patch_preview.md"
audit_csv = audit_dir / "v45b_registered_stage3_reference_patch_audit.csv"
audit_md = audit_dir / "v45b_registered_stage3_reference_patch_audit.md"
summary_json = audit_dir / "v45b_registered_stage3_reference_patch_summary.json"

out.to_csv(patch_csv, index=False)
out.head(80).to_markdown(patch_md, index=False)

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

audit.to_csv(audit_csv, index=False)
audit.to_markdown(audit_md, index=False)

summary = {
    "status": "pass_registered_patch" if len(out) > 0 else "blocked_no_normalized_candidates",
    "input_v45_raw": str(v45_raw),
    "n_input_rows": int(len(df)),
    "n_registered_patch_rows": int(len(out)),
    "n_unique_mp_ids": int(out["mp_id"].nunique()) if len(out) else 0,
    "mp_ids": sorted(out["mp_id"].astype(str).unique().tolist()) if len(out) else [],
    "candidate_source_counts": out["candidate_source"].value_counts(dropna=False).to_dict() if len(out) else {},
    "reference_source_label": "v45b_registered_v44_regenerated_flow_patch",
    "output_patch_csv": str(patch_csv),
    "output_audit_csv": str(audit_csv),
    "claim_boundary": "v45b_registers_real_stage3_flow_outputs_from_v44_regenerated_features_not_experimental_validation",
    "interpretation": (
        "V45b normalizes V45 raw Flow outputs into the active Stage3 reference schema. "
        "These rows are real Stage3 Flow outputs from V44 regenerated structure/formula features, "
        "but should remain labeled separately from original-distribution Stage3 candidates."
    ),
    "next_required_step": (
        "Register this patch in configs/stage3_reference_registry.json and rerun refresh_stage3_reference_library.py "
        "with registered patch priority above existing_current_reference."
    ),
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", patch_csv)
print("[SAVE]", patch_md)
print("[SAVE]", audit_csv)
print("[SAVE]", audit_md)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build V45b final report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V45B_REPORT.md" <<MD
# Final Benchmark-100 V45b Report

## 1. Version

\`benchmark_100_v45b_normalize_v45_stage3_candidates_and_register_patch\`

## 2. Purpose

V45b normalizes V45 raw Stage3 Flow candidates into the active Stage3 reference-library schema.

The resulting patch should be registered as:

\`v45b_registered_v44_regenerated_flow_patch\`

## 3. Key outputs

- Registered reference patch:
  \`registered_reference_patch_v45b/v45b_registered_stage3_reference_patch.csv\`

- Patch preview:
  \`registered_reference_patch_v45b/v45b_registered_stage3_reference_patch_preview.md\`

- Patch audit:
  \`audit_v45b/v45b_registered_stage3_reference_patch_audit.csv\`

- Summary:
  \`audit_v45b/v45b_registered_stage3_reference_patch_summary.json\`

## 4. Interpretation

V45b does not fabricate Stage3 candidates.

It converts real V45 Flow outputs generated from V44 regenerated Stage3 features into the common reference schema.

These candidates should be labeled conservatively as:

\`real_stage3_flow_from_v44_regenerated_features\`

## 5. Next required step

Register this patch in:

\`pipeline_v3/configs/stage3_reference_registry.json\`

Then rerun:

\`refresh_stage3_reference_library.py\`

so that these rows become part of the active Stage3 condition-reference library used by pipeline_v3 reliability scoring.
MD

cp "$AUDIT_DIR/v45b_registered_stage3_reference_patch_summary.json" "$REPORT_DIR/FINAL_V45B_METRIC_SUMMARY.json"

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
echo "[DONE] V45b normalization and registered patch completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V45B_REPORT.md"
echo
echo "Patch:"
echo "$PATCH_DIR/v45b_registered_stage3_reference_patch.csv"
echo
echo "Summary:"
echo "$AUDIT_DIR/v45b_registered_stage3_reference_patch_summary.json"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
