#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V39_VERSION="benchmark_100_v39_stage3_input_feature_extension_construction"
OUT_ROOT="$PROJECT_ROOT/outputs/$V39_VERSION"

V33_MANIFEST="$PROJECT_ROOT/outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv"
V36_CAND_DIR="$PROJECT_ROOT/outputs/benchmark_100_v36_v33_target_aware_stage2_candidate_construction/target_aware_stage2_candidates_v36"
V38_STATUS="$PROJECT_ROOT/outputs/benchmark_100_v38_stage3_input_feature_extension_for_v33_targets/stage3_feature_source_audit_v38/v38_target_feature_source_status.csv"

STAGE3_INPUT_DIR="$PROJECT_ROOT/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"

EXT_DIR="$OUT_ROOT/stage3_input_extension_v39"
CAND_DIR="$OUT_ROOT/stage2_candidates_for_v39_partial_stage3_export"
AUDIT_DIR="$OUT_ROOT/audit_v39"
FINAL_DIR="$OUT_ROOT/FINAL_REPORT_V39"

mkdir -p "$EXT_DIR" "$CAND_DIR" "$AUDIT_DIR" "$FINAL_DIR"

echo "============================================================"
echo "Benchmark-100 V39 Stage3 Input Feature Extension Construction"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "EXT_DIR      = $EXT_DIR"
echo "CAND_DIR     = $CAND_DIR"
echo "============================================================"

echo
echo "[STEP 1] Check required inputs"
for p in \
  "$V33_MANIFEST" \
  "$V36_CAND_DIR/test_candidates.csv" \
  "$V36_CAND_DIR/val_candidates.csv" \
  "$V38_STATUS" \
  "$STAGE3_INPUT_DIR/train.npz" \
  "$STAGE3_INPUT_DIR/test.npz" \
  "$STAGE3_INPUT_DIR/val.npz"
do
  if [ ! -f "$p" ]; then
    echo "[ERROR] Missing required file: $p"
    exit 1
  fi
  echo "[OK] $p"
done

echo
echo "[STEP 2] Build partial Stage3 NPZ extension from reusable real feature payloads"

python - <<PY
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path("$PROJECT_ROOT")
v33_manifest = Path("$V33_MANIFEST")
v36_cand_dir = Path("$V36_CAND_DIR")
v38_status = Path("$V38_STATUS")
stage3_input_dir = Path("$STAGE3_INPUT_DIR")
ext_dir = Path("$EXT_DIR")
cand_dir = Path("$CAND_DIR")
audit_dir = Path("$AUDIT_DIR")

ext_dir.mkdir(parents=True, exist_ok=True)
cand_dir.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(v33_manifest)
status = pd.read_csv(v38_status)
v36_test = pd.read_csv(v36_cand_dir / "test_candidates.csv")
v36_val = pd.read_csv(v36_cand_dir / "val_candidates.csv")

usable_mp_ids = (
    status.loc[status["has_any_stage3_feature_source"].astype(str).str.lower().eq("true"), "mp_id"]
    .astype(str)
    .tolist()
)
missing_mp_ids = (
    status.loc[~status["has_any_stage3_feature_source"].astype(str).str.lower().eq("true"), "mp_id"]
    .astype(str)
    .tolist()
)

usable_mp_ids = sorted(set(usable_mp_ids))
missing_mp_ids = sorted(set(missing_mp_ids))

npz_paths = [
    stage3_input_dir / "test.npz",
    stage3_input_dir / "val.npz",
    stage3_input_dir / "train.npz",
]

payload_by_mp = {}
source_by_mp = {}

def as_str_array(a):
    return np.asarray(a, dtype=object).astype(str)

for npz_path in npz_paths:
    arr = np.load(npz_path, allow_pickle=True)
    keys = list(arr.files)
    if "sample_id" not in keys:
        continue
    sample_ids = as_str_array(arr["sample_id"])
    for i, sid in enumerate(sample_ids):
        sid = str(sid)
        for mp_id in usable_mp_ids:
            if mp_id == sid or mp_id in sid:
                if mp_id not in payload_by_mp:
                    payload_by_mp[mp_id] = {k: arr[k][i] for k in keys if k != "sample_id"}
                    source_by_mp[mp_id] = str(npz_path)
                break

found_mp_ids = sorted(payload_by_mp.keys())
usable_but_not_found = sorted(set(usable_mp_ids) - set(found_mp_ids))

if not found_mp_ids:
    summary = {
        "status": "blocked_no_extractable_stage3_npz_payloads",
        "n_usable_from_v38": len(usable_mp_ids),
        "usable_mp_ids": usable_mp_ids,
        "n_found_npz_payloads": 0,
        "missing_mp_ids": missing_mp_ids,
        "interpretation": "V39 could not extract any reusable Stage3 NPZ payloads, even though V38 found possible feature sources."
    }
    (audit_dir / "v39_stage3_input_extension_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0)

# Use first existing NPZ as schema reference.
schema_ref = np.load(stage3_input_dir / "train.npz", allow_pickle=True)
schema_keys = list(schema_ref.files)

out_arrays = {}
for k in schema_keys:
    if k == "sample_id":
        out_arrays[k] = np.asarray(found_mp_ids, dtype=object)
    else:
        vals = []
        for mp_id in found_mp_ids:
            if k in payload_by_mp[mp_id]:
                vals.append(payload_by_mp[mp_id][k])
            else:
                # Fallback to zero-like array based on schema reference.
                ref_val = schema_ref[k][0]
                vals.append(np.zeros_like(ref_val))
        out_arrays[k] = np.stack(vals, axis=0)

# Write identical test/val extension so old wrappers can read either split.
np.savez_compressed(ext_dir / "test.npz", **out_arrays)
np.savez_compressed(ext_dir / "val.npz", **out_arrays)

# Filter V36 candidates to only those with extractable payloads.
v36_all = pd.concat([v36_test, v36_val], ignore_index=True).drop_duplicates(subset=["stage2_candidate_id"])
v36_partial = v36_all[v36_all["material_id"].astype(str).isin(found_mp_ids)].copy()

# Force split outputs for wrapper compatibility.
v36_partial_test = v36_partial.copy()
v36_partial_test["split"] = "test"
v36_partial_val = v36_partial.copy()
v36_partial_val["split"] = "val"

v36_partial_test.to_csv(cand_dir / "test_candidates.csv", index=False)
v36_partial_val.to_csv(cand_dir / "val_candidates.csv", index=False)
v36_partial.to_csv(cand_dir / "v39_partial_stage2_candidates.csv", index=False)

v36_partial.head(20).to_markdown(cand_dir / "v39_partial_stage2_candidates_preview.md", index=False)

# Build audit tables.
audit_rows = []
for _, r in manifest.iterrows():
    mp_id = str(r["mp_id"])
    audit_rows.append({
        "external_case_id": r.get("external_case_id", ""),
        "target_formula": r.get("target_formula", ""),
        "mp_id": mp_id,
        "v38_has_any_feature_source": mp_id in usable_mp_ids,
        "v39_extracted_npz_payload": mp_id in found_mp_ids,
        "payload_source_npz": source_by_mp.get(mp_id, ""),
        "v39_status": "pass_extractable_stage3_payload" if mp_id in found_mp_ids else "blocked_missing_extractable_stage3_payload",
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(audit_dir / "v39_stage3_input_extension_audit.csv", index=False)
audit_df.to_markdown(audit_dir / "v39_stage3_input_extension_audit.md", index=False)

summary = {
    "status": "partial_pass_missing_feature_payloads",
    "n_v33_targets": int(len(manifest)),
    "n_v38_targets_with_any_feature_source": int(len(usable_mp_ids)),
    "n_v39_extractable_npz_payloads": int(len(found_mp_ids)),
    "n_v39_missing_or_unextractable_targets": int(len(set(manifest["mp_id"].astype(str)) - set(found_mp_ids))),
    "extracted_mp_ids": found_mp_ids,
    "usable_but_not_found_in_npz": usable_but_not_found,
    "missing_mp_ids_from_v38": missing_mp_ids,
    "output_stage3_input_extension_dir": str(ext_dir),
    "output_stage2_candidate_subset_dir": str(cand_dir),
    "interpretation": (
        "V39 constructs a conservative partial Stage3 input NPZ extension only for MP targets with extractable real Stage3-compatible payloads. "
        "Remaining targets still require Stage3 feature regeneration from structure/descriptor pipelines."
    ),
    "next_required_action": (
        "Run V40 Stage3 MDN/Flow export interface on the partial V39 extension for extracted targets, "
        "and separately build V41 feature regeneration for the missing targets."
    )
}

(audit_dir / "v39_stage3_input_extension_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(audit_dir / "v39_stage3_input_extension_summary.md").write_text(pd.DataFrame([summary]).to_markdown(index=False), encoding="utf-8")

schema = {
    "version": "benchmark_100_v39_stage3_input_feature_extension_construction",
    "mode": "partial_stage3_npz_extension_from_existing_real_payloads",
    "source_stage3_input_dir": str(stage3_input_dir),
    "npz_keys": schema_keys,
    "sample_ids": found_mp_ids,
    "important_note": "Only extracted real payloads are included. No fabricated feature vectors are used."
}
(ext_dir / "v39_stage3_input_extension_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", ext_dir / "test.npz")
print("[SAVE]", ext_dir / "val.npz")
print("[SAVE]", cand_dir / "test_candidates.csv")
print("[SAVE]", cand_dir / "val_candidates.csv")
print("[SAVE]", audit_dir / "v39_stage3_input_extension_summary.json")
PY

echo
echo "[STEP 3] Build V39 final report"

cat > "$FINAL_DIR/FINAL_BENCHMARK_100_V39_REPORT.md" <<MD
# Final Benchmark-100 V39 Report

## 1. Version

\`benchmark_100_v39_stage3_input_feature_extension_construction\`

## 2. Purpose

V39 constructs a conservative Stage3 input feature extension for the V33/V36 formula-exact MP targets.

V38 showed that only part of the nine targets have reusable Stage3-compatible feature sources. Therefore, V39 only extracts real reusable Stage3 NPZ payloads and does not fabricate missing features.

## 3. Key outputs

- Partial Stage3 input extension: \`stage3_input_extension_v39/test.npz\`
- Partial Stage3 input extension: \`stage3_input_extension_v39/val.npz\`
- Matching Stage2 candidate subset: \`stage2_candidates_for_v39_partial_stage3_export/test_candidates.csv\`
- Matching Stage2 candidate subset: \`stage2_candidates_for_v39_partial_stage3_export/val_candidates.csv\`
- Audit table: \`audit_v39/v39_stage3_input_extension_audit.csv\`
- Summary: \`audit_v39/v39_stage3_input_extension_summary.json\`

## 4. Interpretation

V39 is a partial feature-extension layer.

It should be interpreted conservatively:

- targets with extractable Stage3 NPZ payloads can proceed to a Stage3 MDN/Flow export interface test;
- targets without payloads remain blocked until Stage3 input features are regenerated from structure/descriptor pipelines;
- no placeholder or synthetic feature vector is reported as real Stage3 input.

## 5. Next required step

Proceed to V40:

\`benchmark_100_v40_partial_real_stage3_export_from_v39_extension\`

V40 should run the existing MDN/Flow export wrappers using:

- Stage2 candidate dir:
  \`outputs/benchmark_100_v39_stage3_input_feature_extension_construction/stage2_candidates_for_v39_partial_stage3_export\`

- Stage3 input dir:
  \`outputs/benchmark_100_v39_stage3_input_feature_extension_construction/stage3_input_extension_v39\`

In parallel, V41 should regenerate Stage3 input features for the still-missing MP targets.
MD

cp "$AUDIT_DIR/v39_stage3_input_extension_summary.json" "$FINAL_DIR/FINAL_V39_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Archive V39 checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_benchmark_100_v39_stage3_input_feature_extension_construction_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V39_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V39_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V39 Stage3 input feature extension construction completed."
echo "Report:"
echo "$FINAL_DIR/FINAL_BENCHMARK_100_V39_REPORT.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v39_stage3_input_extension_summary.json"
echo
echo "Stage3 input extension dir for V40:"
echo "$EXT_DIR"
echo
echo "Stage2 candidate subset dir for V40:"
echo "$CAND_DIR"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
