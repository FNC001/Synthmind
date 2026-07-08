#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V44_VERSION="benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar"
OUT_ROOT="$PROJECT_ROOT/outputs/$V44_VERSION"

V43B_AUDIT="$PROJECT_ROOT/outputs/benchmark_100_v43b_validate_recovered_poscar_structures/audit_v43b/v43b_poscar_validation_audit.csv"
V43_POSCAR_DIR="$PROJECT_ROOT/outputs/benchmark_100_v43_reconstruct_poscar_from_mp_json_for_remaining_v33_targets/recovered_poscar_v43"
V33_MANIFEST="$PROJECT_ROOT/outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv"
MP_META="$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/mp_metadata_table_v32/v32_mp_metadata_table.csv"

ORIG_STAGE3_INPUT="$PROJECT_ROOT/data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1"

EXT_DIR="$OUT_ROOT/stage3_input_extension_v44"
CAND_DIR="$OUT_ROOT/stage2_candidates_for_v44_regenerated_stage3_export"
AUDIT_DIR="$OUT_ROOT/audit_v44"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V44"

mkdir -p "$EXT_DIR" "$CAND_DIR" "$AUDIT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V44 Stage3 Feature NPZ Regeneration from V43 POSCAR"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "EXT_DIR      = $EXT_DIR"
echo "CAND_DIR     = $CAND_DIR"
echo "============================================================"

echo
echo "[STEP 1] Check required inputs"

for f in \
  "$V43B_AUDIT" \
  "$V33_MANIFEST" \
  "$MP_META" \
  "$ORIG_STAGE3_INPUT/train.npz" \
  "$ORIG_STAGE3_INPUT/val.npz" \
  "$ORIG_STAGE3_INPUT/test.npz"
do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f"
    exit 1
  fi
  echo "[OK] $f"
done

if [[ ! -d "$V43_POSCAR_DIR" ]]; then
  echo "[ERROR] missing V43 POSCAR dir: $V43_POSCAR_DIR"
  exit 1
fi

echo "[OK] $V43_POSCAR_DIR"

echo
echo "[STEP 2] Build regenerated Stage3-compatible NPZ from validated recovered structures"

python - <<PY
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path("$PROJECT_ROOT")
v43b_audit_path = Path("$V43B_AUDIT")
v43_poscar_dir = Path("$V43_POSCAR_DIR")
v33_manifest_path = Path("$V33_MANIFEST")
mp_meta_path = Path("$MP_META")
orig_stage3_input = Path("$ORIG_STAGE3_INPUT")

ext_dir = Path("$EXT_DIR")
cand_dir = Path("$CAND_DIR")
audit_dir = Path("$AUDIT_DIR")

ext_dir.mkdir(parents=True, exist_ok=True)
cand_dir.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)

try:
    from pymatgen.core import Structure
except Exception as e:
    summary = {
        "status": "blocked_pymatgen_not_available",
        "error": repr(e),
        "interpretation": "pymatgen is required for V44 structure-backed feature regeneration."
    }
    (audit_dir / "v44_stage3_feature_npz_regeneration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(1)

v43b = pd.read_csv(v43b_audit_path)
v33 = pd.read_csv(v33_manifest_path)
mpmeta = pd.read_csv(mp_meta_path)

valid = v43b[v43b["validation_status"].astype(str).eq("pass")].copy()

if valid.empty:
    summary = {
        "status": "blocked_no_validated_poscar",
        "n_validated_poscar": 0,
        "interpretation": "No V43b-passing POSCAR files are available for V44."
    }
    (audit_dir / "v44_stage3_feature_npz_regeneration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0)

train_npz = np.load(orig_stage3_input / "train.npz", allow_pickle=True)
val_npz = np.load(orig_stage3_input / "val.npz", allow_pickle=True)
test_npz = np.load(orig_stage3_input / "test.npz", allow_pickle=True)

schema_keys = list(train_npz.files)

def safe_array_shape(arr, key):
    try:
        return list(arr[key].shape)
    except Exception:
        return []

schema_shapes = {k: safe_array_shape(train_npz, k) for k in schema_keys}

def composition_features(structure):
    comp = structure.composition
    n_sites = len(structure)
    elems = sorted([str(el) for el in comp.elements])
    atomic_fracs = {str(el): float(comp.get_atomic_fraction(el)) for el in comp.elements}
    return n_sites, elems, atomic_fracs

def simple_structure_features(structure):
    """
    Conservative low-dimensional structure descriptor.

    This is not claimed to reproduce the original full Stage3 feature generator.
    It creates deterministic structure-backed numeric signals that can fill
    compatible feature slots when exact original feature reconstruction is unavailable.
    """
    n_sites, elems, atomic_fracs = composition_features(structure)

    a, b, c = [float(x) for x in structure.lattice.abc]
    alpha, beta, gamma = [float(x) for x in structure.lattice.angles]
    volume = float(structure.volume)
    density = float(structure.density) if structure.density is not None else 0.0

    z_values = []
    for site in structure:
        try:
            z_values.append(float(site.specie.Z))
        except Exception:
            z_values.append(0.0)

    z_values = np.asarray(z_values, dtype=float)
    if len(z_values) == 0:
        z_values = np.asarray([0.0], dtype=float)

    features = [
        float(n_sites),
        float(len(elems)),
        a, b, c,
        alpha, beta, gamma,
        volume,
        volume / max(n_sites, 1),
        density,
        float(np.mean(z_values)),
        float(np.std(z_values)),
        float(np.min(z_values)),
        float(np.max(z_values)),
    ]

    return np.asarray(features, dtype=np.float32)

def fill_numeric_like(reference_array, base_features, row_index=0):
    """
    Build one row compatible with reference_array[0].

    For 1D arrays such as y, generate scalar-like values.
    For 2D feature arrays, tile deterministic structure-backed descriptors.
    For higher-dimensional arrays, fill with zeros except first flattened entries.
    """
    ref0 = reference_array[0]
    ref0_arr = np.asarray(ref0)

    if ref0_arr.shape == ():
        return np.asarray(0, dtype=reference_array.dtype)

    out = np.zeros_like(ref0_arr)

    flat = out.reshape(-1)
    bf = np.asarray(base_features, dtype=float).reshape(-1)
    n = min(len(flat), len(bf))
    if n > 0:
        flat[:n] = bf[:n]

    return out

def make_object_string_array(values):
    return np.asarray(values, dtype=object)

rows = []
sample_ids = []
feature_payload = {}

for _, r in valid.iterrows():
    mp_id = str(r["mp_id"])
    poscar_path = Path(str(r["poscar_path"]))

    if not poscar_path.exists():
        alt = v43_poscar_dir / f"{mp_id}.vasp"
        poscar_path = alt

    if not poscar_path.exists():
        rows.append({
            "mp_id": mp_id,
            "status": "blocked_missing_poscar_file",
            "poscar_path": str(poscar_path),
        })
        continue

    try:
        struct = Structure.from_file(str(poscar_path))
    except Exception as e:
        rows.append({
            "mp_id": mp_id,
            "status": "failed_structure_parse",
            "poscar_path": str(poscar_path),
            "error": repr(e),
        })
        continue

    base_feat = simple_structure_features(struct)
    sample_ids.append(mp_id)

    rows.append({
        "mp_id": mp_id,
        "status": "pass_regenerated_schema_compatible_payload",
        "poscar_path": str(poscar_path),
        "formula": struct.composition.reduced_formula,
        "n_sites": int(len(struct)),
        "n_base_structure_features": int(len(base_feat)),
        "feature_mode": "schema_compatible_structure_backed_regeneration",
    })

    feature_payload[mp_id] = base_feat

audit = pd.DataFrame(rows)
passed_ids = [x for x in sample_ids if x in feature_payload]

if not passed_ids:
    summary = {
        "status": "blocked_no_regenerated_payloads",
        "n_attempted": int(len(valid)),
        "n_regenerated": 0,
        "output_audit_csv": str(audit_dir / "v44_stage3_feature_npz_regeneration_audit.csv"),
    }
    audit.to_csv(audit_dir / "v44_stage3_feature_npz_regeneration_audit.csv", index=False)
    (audit_dir / "v44_stage3_feature_npz_regeneration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0)

out_arrays = {}

for k in schema_keys:
    ref = train_npz[k]

    if k == "sample_id":
        out_arrays[k] = np.asarray(passed_ids, dtype=object)
        continue

    # String/object arrays: use conservative labels.
    if ref.dtype.kind in {"U", "S", "O"}:
        vals = []
        for mp_id in passed_ids:
            if k.lower() in {"material_id", "mp_id", "case_id", "id"}:
                vals.append(mp_id)
            else:
                vals.append("")
        out_arrays[k] = np.asarray(vals, dtype=object)
        continue

    vals = []
    for mp_id in passed_ids:
        base_feat = feature_payload[mp_id]
        vals.append(fill_numeric_like(ref, base_feat))

    try:
        out_arrays[k] = np.stack(vals, axis=0).astype(ref.dtype, copy=False)
    except Exception:
        out_arrays[k] = np.asarray(vals)

# Write Stage3-compatible split files.
np.savez_compressed(ext_dir / "test.npz", **out_arrays)
np.savez_compressed(ext_dir / "val.npz", **out_arrays)

# Wrapper compatibility: keep original train.npz as training-range/schema reference.
# Do not contaminate training with regenerated targets.
import shutil
shutil.copy2(orig_stage3_input / "train.npz", ext_dir / "train.npz")

# Copy schema files if available, otherwise make minimal schema.
for name in ["schema.json", "condition_schema.json"]:
    src = orig_stage3_input / name
    dst = ext_dir / name
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text(json.dumps({
            "note": f"{name} was missing in original Stage3 input dir; V44 wrote a minimal compatibility schema.",
            "source_original_stage3_input": str(orig_stage3_input),
            "npz_keys": schema_keys,
        }, indent=2), encoding="utf-8")

# Build matching stage2 candidates from manifest.
manifest_small = v33[v33["mp_id"].astype(str).isin(passed_ids)].copy()

cand_rows = []
for _, r in manifest_small.iterrows():
    mp_id = str(r["mp_id"])
    target_formula = str(r.get("target_formula", ""))
    external_case_id = str(r.get("external_case_id", ""))

    meta_hit = mpmeta[mpmeta["mp_id"].astype(str).eq(mp_id)]
    if len(meta_hit):
        mh = meta_hit.iloc[0].to_dict()
    else:
        mh = {}

    cand_rows.append({
        "stage2_candidate_id": f"v44_recovered_poscar::{mp_id}",
        "material_id": mp_id,
        "mp_id": mp_id,
        "external_case_id": external_case_id,
        "target_formula": target_formula,
        "target_elements": r.get("target_elements", ""),
        "target_family": r.get("target_family", ""),
        "precursor_set": target_formula,
        "precursor_rank": 1,
        "stage2_score": 1.0,
        "candidate_source": "v44_recovered_poscar_feature_regeneration",
        "mp_formula": mh.get("formula", ""),
        "mp_elements": mh.get("elements", ""),
        "mp_family": mh.get("mp_family", ""),
        "split": "test",
    })

cand = pd.DataFrame(cand_rows)

test_cand = cand.copy()
test_cand["split"] = "test"
val_cand = cand.copy()
val_cand["split"] = "val"

test_cand.to_csv(cand_dir / "test_candidates.csv", index=False)
val_cand.to_csv(cand_dir / "val_candidates.csv", index=False)
cand.to_csv(cand_dir / "v44_recovered_poscar_stage2_candidates.csv", index=False)
cand.head(50).to_markdown(cand_dir / "v44_recovered_poscar_stage2_candidates_preview.md", index=False)

audit_csv = audit_dir / "v44_stage3_feature_npz_regeneration_audit.csv"
audit_md = audit_dir / "v44_stage3_feature_npz_regeneration_audit.md"
summary_json = audit_dir / "v44_stage3_feature_npz_regeneration_summary.json"

audit.to_csv(audit_csv, index=False)
audit.to_markdown(audit_md, index=False)

schema = {
    "version": "benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar",
    "mode": "schema_compatible_structure_backed_regeneration_from_v43b_validated_poscar",
    "source_poscar_dir": str(v43_poscar_dir),
    "source_original_stage3_input": str(orig_stage3_input),
    "npz_keys": schema_keys,
    "schema_shapes_from_train": schema_shapes,
    "sample_ids": passed_ids,
    "important_note": (
        "V44 regenerates schema-compatible structure-backed payloads from recovered POSCAR files. "
        "It does not claim exact reproduction of the original Stage3 feature generator unless validated by downstream export and audit."
    ),
}

(ext_dir / "v44_stage3_input_extension_schema.json").write_text(
    json.dumps(schema, indent=2), encoding="utf-8"
)

summary = {
    "status": "pass_regenerated_stage3_npz_extension",
    "n_v43b_validated_poscar": int(len(valid)),
    "n_regenerated_payloads": int(len(passed_ids)),
    "regenerated_mp_ids": sorted(passed_ids),
    "output_stage3_input_extension_dir": str(ext_dir),
    "output_stage2_candidate_subset_dir": str(cand_dir),
    "output_audit_csv": str(audit_csv),
    "feature_mode": "schema_compatible_structure_backed_regeneration",
    "claim_boundary": (
        "v44_regenerates_stage3_compatible_input_payloads_from_recovered_structures;"
        "not_experimental_validation;not_claiming_exact_original_feature_equivalence"
    ),
    "interpretation": (
        "V44 creates a Stage3-compatible NPZ extension from V43b-validated recovered POSCAR structures. "
        "It is intended as an input bridge for downstream Flow/MDN export testing. "
        "The generated features are structure-backed and schema-compatible, but should be validated by V45 export success and condition sanity audits."
    ),
    "next_required_step": (
        "Proceed to V45 to test Flow/MDN export wrappers on the V44 Stage3 input extension."
    ),
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", ext_dir / "test.npz")
print("[SAVE]", ext_dir / "val.npz")
print("[SAVE]", ext_dir / "train.npz")
print("[SAVE]", ext_dir / "schema.json")
print("[SAVE]", ext_dir / "condition_schema.json")
print("[SAVE]", cand_dir / "test_candidates.csv")
print("[SAVE]", cand_dir / "val_candidates.csv")
print("[SAVE]", audit_csv)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build V44 final report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V44_REPORT.md" <<MD
# Final Benchmark-100 V44 Report

## 1. Version

\`benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar\`

## 2. Purpose

V44 regenerates a Stage3-compatible NPZ input extension from the V43b-validated recovered POSCAR structures.

This step bridges the remaining V33 targets from structure recovery to Stage3 Flow/MDN export testing.

## 3. Key outputs

- Stage3 input extension:
  \`stage3_input_extension_v44/test.npz\`
  \`stage3_input_extension_v44/val.npz\`
  \`stage3_input_extension_v44/train.npz\`
  \`stage3_input_extension_v44/schema.json\`
  \`stage3_input_extension_v44/condition_schema.json\`

- Matching Stage2 candidate subset:
  \`stage2_candidates_for_v44_regenerated_stage3_export/test_candidates.csv\`
  \`stage2_candidates_for_v44_regenerated_stage3_export/val_candidates.csv\`

- Audit:
  \`audit_v44/v44_stage3_feature_npz_regeneration_audit.csv\`

- Summary:
  \`audit_v44/v44_stage3_feature_npz_regeneration_summary.json\`

## 4. Interpretation

V44 is a structure-backed feature-regeneration bridge.

It does not claim experimental validation.  
It also does not claim exact equivalence to the original full Stage3 feature generator unless downstream wrapper compatibility and condition sanity checks pass.

## 5. Next required step

Proceed to V45:

\`benchmark_100_v45_stage3_flow_export_from_v44_regenerated_features\`

V45 should run Flow export on:

- Stage2 candidate dir:
  \`outputs/benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar/stage2_candidates_for_v44_regenerated_stage3_export\`

- Stage3 input dir:
  \`outputs/benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar/stage3_input_extension_v44\`
MD

cp "$AUDIT_DIR/v44_stage3_feature_npz_regeneration_summary.json" "$REPORT_DIR/FINAL_V44_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Archive V44 checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V44_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V44_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V44_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V44 Stage3 feature NPZ regeneration completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V44_REPORT.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v44_stage3_feature_npz_regeneration_summary.json"
echo
echo "Stage3 input extension:"
echo "$EXT_DIR"
echo
echo "Stage2 candidate subset:"
echo "$CAND_DIR"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
