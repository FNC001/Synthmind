#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V48_VERSION="benchmark_100_v48_flow_export_with_global_stage3_clip_range"
OUT_ROOT="$PROJECT_ROOT/outputs/$V48_VERSION"
AUDIT_DIR="$OUT_ROOT/audit_v48"
RAW_DIR="$OUT_ROOT/raw_flow_candidates_v48"
NORM_DIR="$OUT_ROOT/normalized_stage3_candidates_v48"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V48"

NO_CLIP_CSV="$PROJECT_ROOT/outputs/benchmark_100_v47c_no_clip_flow_export_confirms_clipping_collapse/normalized_no_clip_candidates_v47c/v47c_no_clip_normalized_stage3_candidates.csv"
V47C_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v47c_no_clip_flow_export_confirms_clipping_collapse/audit_v47c/v47c_no_clip_flow_export_summary.json"
ORIG_COND_SCHEMA="$PROJECT_ROOT/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/condition_schema.json"

mkdir -p "$AUDIT_DIR" "$RAW_DIR" "$NORM_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V48 Flow Export with Global Stage3 Clip Range"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

echo
echo "[STEP 1] Check required inputs"

for p in "$NO_CLIP_CSV" "$V47C_SUMMARY" "$ORIG_COND_SCHEMA"
do
  if [[ ! -e "$p" ]]; then
    echo "[ERROR] Missing required input: $p"
    exit 1
  fi
  echo "[OK] $p"
done

echo
echo "[STEP 2] Apply global original Stage3 condition bounds"

python - <<PY
import json
from pathlib import Path
import pandas as pd
import numpy as np

no_clip_csv = Path("$NO_CLIP_CSV")
schema_path = Path("$ORIG_COND_SCHEMA")
audit_dir = Path("$AUDIT_DIR")
raw_dir = Path("$RAW_DIR")
norm_dir = Path("$NORM_DIR")

audit_dir.mkdir(parents=True, exist_ok=True)
raw_dir.mkdir(parents=True, exist_ok=True)
norm_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(no_clip_csv)
schema = json.loads(schema_path.read_text(encoding="utf-8"))

cont_schema = schema.get("continuous_schema", {})
temp_lo = float(cont_schema.get("temperature_c", {}).get("clip_lo", 4.0))
temp_hi = float(cont_schema.get("temperature_c", {}).get("clip_hi", 1525.0))
time_lo = float(cont_schema.get("time_h", {}).get("clip_lo", 0.1667))
time_hi = float(cont_schema.get("time_h", {}).get("clip_hi", 126.0))

df["raw_temperature_c"] = pd.to_numeric(df["candidate_temperature_c"], errors="coerce")
df["raw_time_h"] = pd.to_numeric(df["candidate_time_h"], errors="coerce")

df["candidate_temperature_c"] = df["raw_temperature_c"].clip(lower=temp_lo, upper=temp_hi)
df["candidate_time_h"] = df["raw_time_h"].clip(lower=time_lo, upper=time_hi)

df["v48_clip_temperature_changed"] = ~np.isclose(
    df["raw_temperature_c"], df["candidate_temperature_c"], equal_nan=True
)
df["v48_clip_time_changed"] = ~np.isclose(
    df["raw_time_h"], df["candidate_time_h"], equal_nan=True
)

df["candidate_source"] = "real_stage3_flow_v48_global_clip_from_v44_regenerated_features"
df["v48_clip_mode"] = "global_original_stage3_condition_schema_bounds"
df["v48_temperature_clip_lo"] = temp_lo
df["v48_temperature_clip_hi"] = temp_hi
df["v48_time_clip_lo"] = time_lo
df["v48_time_clip_hi"] = time_hi

def warning(row):
    w = []
    if bool(row["v48_clip_temperature_changed"]):
        w.append("temperature_clipped_to_global_stage3_bounds")
    if bool(row["v48_clip_time_changed"]):
        w.append("time_clipped_to_global_stage3_bounds")
    if float(row["candidate_time_h"]) <= time_lo + 1e-8:
        w.append("time_at_lower_bound")
    if float(row["candidate_temperature_c"]) <= temp_lo + 1e-8:
        w.append("temperature_at_lower_bound")
    return ";".join(w)

df["condition_warning"] = df.apply(warning, axis=1)

norm_csv = norm_dir / "v48_global_clip_normalized_stage3_candidates.csv"
norm_md = norm_dir / "v48_global_clip_normalized_stage3_candidates_preview.md"
df.to_csv(norm_csv, index=False)
df.head(80).to_markdown(norm_md, index=False)

audit_rows = []
for mp_id, g in df.groupby("mp_id"):
    temps = pd.to_numeric(g["candidate_temperature_c"], errors="coerce")
    times = pd.to_numeric(g["candidate_time_h"], errors="coerce")
    raw_temps = pd.to_numeric(g["raw_temperature_c"], errors="coerce")
    raw_times = pd.to_numeric(g["raw_time_h"], errors="coerce")

    n_ut = temps.round(6).nunique(dropna=True)
    n_uh = times.round(6).nunique(dropna=True)

    if n_ut > 1 and n_uh > 1:
        diversity_status = "condition_diversity_present_after_global_clip"
    elif n_ut > 1 or n_uh > 1:
        diversity_status = "partial_condition_diversity_after_global_clip"
    else:
        diversity_status = "no_condition_diversity_after_global_clip"

    audit_rows.append({
        "mp_id": mp_id,
        "external_case_id": g["external_case_id"].dropna().astype(str).iloc[0] if len(g) else "",
        "formula": g["formula"].dropna().astype(str).iloc[0] if len(g) else "",
        "n_candidate_rows": int(len(g)),
        "n_unique_temperature": int(n_ut),
        "n_unique_time": int(n_uh),
        "min_temperature_c": float(temps.min()) if temps.notna().any() else None,
        "mean_temperature_c": float(temps.mean()) if temps.notna().any() else None,
        "max_temperature_c": float(temps.max()) if temps.notna().any() else None,
        "min_time_h": float(times.min()) if times.notna().any() else None,
        "mean_time_h": float(times.mean()) if times.notna().any() else None,
        "max_time_h": float(times.max()) if times.notna().any() else None,
        "raw_min_temperature_c": float(raw_temps.min()) if raw_temps.notna().any() else None,
        "raw_max_temperature_c": float(raw_temps.max()) if raw_temps.notna().any() else None,
        "raw_min_time_h": float(raw_times.min()) if raw_times.notna().any() else None,
        "raw_max_time_h": float(raw_times.max()) if raw_times.notna().any() else None,
        "n_temperature_clipped": int(g["v48_clip_temperature_changed"].sum()),
        "n_time_clipped": int(g["v48_clip_time_changed"].sum()),
        "n_warning_rows": int(g["condition_warning"].astype(str).str.len().gt(0).sum()),
        "diversity_status": diversity_status,
    })

audit = pd.DataFrame(audit_rows).sort_values("mp_id")

audit_csv = audit_dir / "v48_global_clip_condition_diversity_audit.csv"
audit_md = audit_dir / "v48_global_clip_condition_diversity_audit.md"
summary_json = audit_dir / "v48_global_clip_flow_export_summary.json"

audit.to_csv(audit_csv, index=False)
audit.to_markdown(audit_md, index=False)

n_diverse = int((audit["diversity_status"] == "condition_diversity_present_after_global_clip").sum())
n_total = int(len(audit))

summary = {
    "status": "pass_global_clip_preserves_condition_diversity" if n_diverse == n_total else "review_global_clip_partial_diversity",
    "n_candidate_rows": int(len(df)),
    "n_unique_mp_ids": int(df["mp_id"].nunique()),
    "n_targets_with_condition_diversity_after_global_clip": n_diverse,
    "n_targets_total": n_total,
    "temperature_clip_bounds": [temp_lo, temp_hi],
    "time_clip_bounds": [time_lo, time_hi],
    "n_temperature_clipped_rows": int(df["v48_clip_temperature_changed"].sum()),
    "n_time_clipped_rows": int(df["v48_clip_time_changed"].sum()),
    "normalized_global_clip_candidates_csv": str(norm_csv),
    "condition_diversity_audit_csv": str(audit_csv),
    "interpretation": (
        "V48 applies the original global Stage3 condition bounds from condition_schema.json to the V47c no-clip Flow candidates. "
        "This avoids the artificial V44 single-point train-range clipping while still enforcing physically/data-supported bounds."
    ),
    "important_caution": (
        "Candidates remain generated from V44 regenerated sparse/zero-filled features. "
        "They are improved over V45 seed-collapsed outputs, but should still be labeled separately from original-distribution V30 Stage3 candidates."
    ),
    "next_required_step": (
        "Proceed to V49: merge V40c partial real Flow candidates and V48 global-clipped candidates into the metadata-aware Stage3 library, "
        "then rerun alignment for all nine V33 gap cases."
    )
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", norm_csv)
print("[SAVE]", norm_md)
print("[SAVE]", audit_csv)
print("[SAVE]", audit_md)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build V48 final report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V48_REPORT.md" <<MD
# Final Benchmark-100 V48 Report

## 1. Version

\`benchmark_100_v48_flow_export_with_global_stage3_clip_range\`

## 2. Purpose

V48 converts the V47c no-clip Flow candidates into a bounded candidate set using the original global Stage3 condition bounds from:

\`data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/condition_schema.json\`

## 3. Why V48 is needed

V45/V46 collapsed to:

- \`temperature_c = 830.0\`
- \`time_h = 14.0\`

because \`--clip_to_train_range\` used the V44 regenerated extension train range:

- \`train_min_raw = [830.0, 14.0]\`
- \`train_max_raw = [830.0, 14.0]\`

V47c showed that no-clip Flow export produces diverse candidates.

V48 keeps that diversity while applying the original global Stage3 bounds:

- temperature: \`4.0–1525.0\`
- time: \`0.1667–126.0\`

## 4. Key outputs

- Global-clipped normalized candidates:
  \`normalized_stage3_candidates_v48/v48_global_clip_normalized_stage3_candidates.csv\`

- Condition-diversity audit:
  \`audit_v48/v48_global_clip_condition_diversity_audit.csv\`

- Summary:
  \`audit_v48/v48_global_clip_flow_export_summary.json\`

## 5. Interpretation

V48 should be interpreted as the corrected Flow export layer for the seven V44-regenerated targets.

It removes the artificial single-point clipping error while preserving global condition bounds from the original Stage3 dataset.

## 6. Caution

The V48 candidates are still generated from V44 regenerated sparse/zero-filled features.

They should be labeled as:

\`real_stage3_flow_v48_global_clip_from_v44_regenerated_features\`

and not mixed silently with original-distribution V30 Stage3 candidates.

## 7. Next required step

Proceed to V49:

\`benchmark_100_v49_merge_v40c_v48_and_realign_all_v33_gap_cases\`

V49 should merge:

- V40c partial Flow patch for BaSO4 and CaSO4;
- V48 global-clipped Flow candidates for the remaining seven targets;

then rerun metadata-aware alignment for all nine V33 gap cases.
MD

cp "$AUDIT_DIR/v48_global_clip_flow_export_summary.json" "$REPORT_DIR/FINAL_V48_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Archive V48 checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V48_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V48_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V48_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V48 global-clipped Flow candidate construction completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V48_REPORT.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v48_global_clip_flow_export_summary.json"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
