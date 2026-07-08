#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V41_VERSION="benchmark_100_v41_stage3_feature_regeneration_for_remaining_v33_targets"
OUT_ROOT="$PROJECT_ROOT/outputs/$V41_VERSION"

V40D_SUMMARY="$PROJECT_ROOT/outputs/benchmark_100_v40d_merge_partial_flow_patch_and_realign/audit_v40d/v40d_realignment_summary.json"
V33_MANIFEST="$PROJECT_ROOT/outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv"
MP_META="$PROJECT_ROOT/outputs/benchmark_100_v32_mp_metadata_aware_stage3_alignment/mp_metadata_table_v32/v32_mp_metadata_table.csv"

RAW_MP_ROOT="$PROJECT_ROOT/data/raw/mp_full_archive_export"
POSCAR_DIR="$RAW_MP_ROOT/poscar"
SUMMARY_JSON_DIR="$RAW_MP_ROOT/summary_json"
PROVENANCE_JSON_DIR="$RAW_MP_ROOT/provenance_json"

AUDIT_DIR="$OUT_ROOT/stage3_feature_regeneration_readiness_v41"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V41"

mkdir -p "$AUDIT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V41 Stage3 Feature Regeneration Readiness"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

echo
echo "[STEP 1] Check required upstream files"

for f in "$V33_MANIFEST" "$MP_META"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "[STEP 2] Build remaining-target regeneration manifest"

python - <<PY
import json
import pandas as pd
from pathlib import Path

project_root = Path("$PROJECT_ROOT")
out_root = Path("$OUT_ROOT")
audit_dir = Path("$AUDIT_DIR")
audit_dir.mkdir(parents=True, exist_ok=True)

v33_manifest_path = Path("$V33_MANIFEST")
mp_meta_path = Path("$MP_META")

poscar_dir = Path("$POSCAR_DIR")
summary_json_dir = Path("$SUMMARY_JSON_DIR")
provenance_json_dir = Path("$PROVENANCE_JSON_DIR")

covered_by_v40d = {
    "mp-1190568",  # BaSO4
    "mp-12372",   # CaSO4
}

v33 = pd.read_csv(v33_manifest_path)
mpmeta = pd.read_csv(mp_meta_path)

# 剩余未被 V40d partial Flow patch 覆盖的 V33 targets
remaining = v33[~v33["mp_id"].astype(str).isin(covered_by_v40d)].copy()

rows = []

for _, r in remaining.iterrows():
    mp_id = str(r["mp_id"])
    target_formula = str(r.get("target_formula", ""))
    external_case_id = str(r.get("external_case_id", ""))
    target_family = str(r.get("target_family", ""))

    poscar_candidates = [
        poscar_dir / f"{mp_id}.vasp",
        poscar_dir / f"{mp_id}.poscar",
        poscar_dir / mp_id,
        poscar_dir / f"POSCAR_{mp_id}",
    ]

    summary_candidates = [
        summary_json_dir / f"{mp_id}.json",
        summary_json_dir / f"{mp_id}.summary.json",
    ]

    provenance_candidates = [
        provenance_json_dir / f"{mp_id}.json",
        provenance_json_dir / f"{mp_id}.provenance.json",
    ]

    poscar_hit = next((p for p in poscar_candidates if p.exists()), None)
    summary_hit = next((p for p in summary_candidates if p.exists()), None)
    provenance_hit = next((p for p in provenance_candidates if p.exists()), None)

    meta_hit = mpmeta[mpmeta["mp_id"].astype(str) == mp_id]

    if len(meta_hit):
        mh = meta_hit.iloc[0].to_dict()
    else:
        mh = {}

    has_structure_source = poscar_hit is not None
    has_json_metadata = summary_hit is not None or provenance_hit is not None
    can_regenerate_formula_features = len(meta_hit) > 0
    can_regenerate_structure_features = has_structure_source

    if can_regenerate_structure_features:
        recommended_action = "regenerate_stage3_structural_and_formula_features"
        readiness_status = "ready_for_feature_regeneration"
    elif can_regenerate_formula_features:
        recommended_action = "regenerate_formula_only_stage3_features_or_fetch_structure"
        readiness_status = "partial_ready_formula_metadata_only"
    else:
        recommended_action = "missing_mp_metadata_and_structure_source"
        readiness_status = "blocked_missing_sources"

    rows.append({
        "external_case_id": external_case_id,
        "mp_id": mp_id,
        "target_formula": target_formula,
        "target_family": target_family,
        "mp_formula": mh.get("formula", ""),
        "mp_elements": mh.get("elements", ""),
        "mp_family": mh.get("mp_family", ""),
        "has_poscar_source": bool(poscar_hit),
        "poscar_path": str(poscar_hit) if poscar_hit else "",
        "has_summary_json": bool(summary_hit),
        "summary_json_path": str(summary_hit) if summary_hit else "",
        "has_provenance_json": bool(provenance_hit),
        "provenance_json_path": str(provenance_hit) if provenance_hit else "",
        "has_mp_metadata": bool(len(meta_hit)),
        "can_regenerate_formula_features": bool(can_regenerate_formula_features),
        "can_regenerate_structure_features": bool(can_regenerate_structure_features),
        "readiness_status": readiness_status,
        "recommended_action": recommended_action,
    })

df = pd.DataFrame(rows)

out_csv = audit_dir / "v41_remaining_targets_feature_regeneration_readiness.csv"
out_md = audit_dir / "v41_remaining_targets_feature_regeneration_readiness.md"
summary_json = audit_dir / "v41_feature_regeneration_readiness_summary.json"

df.to_csv(out_csv, index=False)
df.to_markdown(out_md, index=False)

summary = {
    "status": "pass_readiness_audit",
    "n_remaining_targets": int(len(df)),
    "n_ready_for_structure_feature_regeneration": int((df["readiness_status"] == "ready_for_feature_regeneration").sum()),
    "n_partial_ready_formula_metadata_only": int((df["readiness_status"] == "partial_ready_formula_metadata_only").sum()),
    "n_blocked_missing_sources": int((df["readiness_status"] == "blocked_missing_sources").sum()),
    "remaining_mp_ids": sorted(df["mp_id"].astype(str).tolist()),
    "ready_mp_ids": sorted(df.loc[df["readiness_status"] == "ready_for_feature_regeneration", "mp_id"].astype(str).tolist()),
    "partial_ready_mp_ids": sorted(df.loc[df["readiness_status"] == "partial_ready_formula_metadata_only", "mp_id"].astype(str).tolist()),
    "blocked_mp_ids": sorted(df.loc[df["readiness_status"] == "blocked_missing_sources", "mp_id"].astype(str).tolist()),
    "output_csv": str(out_csv),
    "interpretation": (
        "V41 audits whether the seven still-blocked V33 targets have enough local MP archive sources "
        "to regenerate Stage3 input features. Structure-backed targets can proceed to descriptor/NPZ regeneration; "
        "formula-only targets need either structure fetching or formula-only fallback features."
    ),
    "next_required_step": (
        "Use this readiness table to build a Stage3 input feature extension for the remaining targets. "
        "Do not report MDN/Flow outputs for targets without regenerated Stage3 input features."
    ),
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", out_csv)
print("[SAVE]", out_md)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build V41 report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V41_READINESS_REPORT.md" <<MD
# Final Benchmark-100 V41 Readiness Report

## 1. Version

\`benchmark_100_v41_stage3_feature_regeneration_for_remaining_v33_targets\`

## 2. Purpose

V41 checks whether the seven V33 targets still blocked after V40d have enough local sources to regenerate Stage3 input features.

V40d partially resolved:

- \`mp-1190568\` / BaSO4
- \`mp-12372\` / CaSO4

The remaining targets are:

- \`mp-1079918\` / CaCO3
- \`mp-1198150\` / Bi2Se3
- \`mp-1330\` / AlN
- \`mp-1986\` / ZnO
- \`mp-2251\` / Li3N
- \`mp-22851\` / NaCl
- \`mp-2652\` / Y2O3

## 3. Key outputs

- Readiness CSV:
  \`stage3_feature_regeneration_readiness_v41/v41_remaining_targets_feature_regeneration_readiness.csv\`

- Readiness Markdown:
  \`stage3_feature_regeneration_readiness_v41/v41_remaining_targets_feature_regeneration_readiness.md\`

- Summary JSON:
  \`stage3_feature_regeneration_readiness_v41/v41_feature_regeneration_readiness_summary.json\`

## 4. Interpretation

V41 is not a generation layer.

It is a source-readiness layer that decides whether Stage3 input features can be regenerated from local MP archive files and metadata.

Only targets with real regenerated Stage3 input features should be passed into Flow or MDN export.

## 5. Next required step

After this audit:

1. if POSCAR sources exist, regenerate structural/formula features and build a new Stage3 NPZ extension;
2. if only formula metadata exists, either fetch/build missing structures or create a clearly labeled formula-only fallback feature set;
3. rerun Flow/MDN export only after compatible \`train.npz\`, \`val.npz\`, \`test.npz\`, \`schema.json\`, and \`condition_schema.json\` exist.
MD

cp "$AUDIT_DIR/v41_feature_regeneration_readiness_summary.json" "$REPORT_DIR/FINAL_V41_READINESS_SUMMARY.json"

echo
echo "[STEP 4] Archive V41 readiness checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V41_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V41_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V41_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V41 Stage3 feature regeneration readiness completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V41_READINESS_REPORT.md"
echo
echo "Readiness table:"
echo "$AUDIT_DIR/v41_remaining_targets_feature_regeneration_readiness.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v41_feature_regeneration_readiness_summary.json"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
