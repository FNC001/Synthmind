#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V42_VERSION="benchmark_100_v42_recover_structure_sources_from_mp_json_for_remaining_v33_targets"
OUT_ROOT="$PROJECT_ROOT/outputs/$V42_VERSION"

V41_READINESS="$PROJECT_ROOT/outputs/benchmark_100_v41_stage3_feature_regeneration_for_remaining_v33_targets/stage3_feature_regeneration_readiness_v41/v41_remaining_targets_feature_regeneration_readiness.csv"

AUDIT_DIR="$OUT_ROOT/mp_json_structure_source_audit_v42"
RECOVER_DIR="$OUT_ROOT/recovered_structure_sources_v42"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V42"

mkdir -p "$AUDIT_DIR" "$RECOVER_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V42 Recover Structure Sources from MP JSON"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

echo
echo "[STEP 1] Check V41 readiness input"

if [[ ! -f "$V41_READINESS" ]]; then
  echo "[ERROR] missing V41 readiness table: $V41_READINESS"
  exit 1
fi

echo "[OK] $V41_READINESS"

echo
echo "[STEP 2] Inspect summary_json / provenance_json for structure payloads"

python - <<PY
import json
import pandas as pd
from pathlib import Path

v41_path = Path("$V41_READINESS")
audit_dir = Path("$AUDIT_DIR")
recover_dir = Path("$RECOVER_DIR")

audit_dir.mkdir(parents=True, exist_ok=True)
recover_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(v41_path)

def read_json_safe(path):
    p = Path(str(path))
    if not p.exists():
        return None, "missing"
    try:
        return json.loads(p.read_text(encoding="utf-8")), ""
    except Exception as e:
        return None, repr(e)

def flatten_keys(obj, prefix="", max_depth=4):
    keys = []
    if max_depth <= 0:
        return keys
    if isinstance(obj, dict):
        for k, v in obj.items():
            name = f"{prefix}.{k}" if prefix else str(k)
            keys.append(name)
            keys.extend(flatten_keys(v, name, max_depth - 1))
    elif isinstance(obj, list) and obj:
        keys.extend(flatten_keys(obj[0], prefix + "[]", max_depth - 1))
    return keys

def find_structure_like(obj):
    keys = flatten_keys(obj, max_depth=5) if obj is not None else []
    lower = [k.lower() for k in keys]
    structure_hits = []
    patterns = [
        "structure",
        "lattice",
        "sites",
        "species",
        "coords",
        "cart_coords",
        "frac_coords",
        "abc",
        "angles",
    ]
    for k, lk in zip(keys, lower):
        if any(p in lk for p in patterns):
            structure_hits.append(k)
    return keys, structure_hits

rows = []
recovered_json_rows = []

for _, r in df.iterrows():
    mp_id = str(r["mp_id"])
    summary_path = str(r.get("summary_json_path", ""))
    provenance_path = str(r.get("provenance_json_path", ""))

    summary_obj, summary_err = read_json_safe(summary_path)
    prov_obj, prov_err = read_json_safe(provenance_path)

    summary_keys, summary_struct_hits = find_structure_like(summary_obj)
    prov_keys, prov_struct_hits = find_structure_like(prov_obj)

    has_summary_structure_like = len(summary_struct_hits) > 0
    has_provenance_structure_like = len(prov_struct_hits) > 0

    # 保存结构相关 key preview，方便人工检查
    structure_key_preview = {
        "mp_id": mp_id,
        "summary_json_path": summary_path,
        "provenance_json_path": provenance_path,
        "summary_structure_like_keys": summary_struct_hits[:200],
        "provenance_structure_like_keys": prov_struct_hits[:200],
        "summary_top_level_keys": list(summary_obj.keys()) if isinstance(summary_obj, dict) else [],
        "provenance_top_level_keys": list(prov_obj.keys()) if isinstance(prov_obj, dict) else [],
    }

    preview_path = recover_dir / f"{mp_id}_json_structure_key_preview.json"
    preview_path.write_text(json.dumps(structure_key_preview, indent=2), encoding="utf-8")

    if has_summary_structure_like or has_provenance_structure_like:
        status = "possible_structure_payload_in_json"
        action = "inspect_json_payload_and_reconstruct_poscar_or_structure_features"
    else:
        status = "no_structure_payload_detected"
        action = "need_external_structure_fetch_or_formula_only_fallback"

    rows.append({
        "external_case_id": r.get("external_case_id", ""),
        "mp_id": mp_id,
        "target_formula": r.get("target_formula", ""),
        "summary_json_path": summary_path,
        "provenance_json_path": provenance_path,
        "summary_json_read_error": summary_err,
        "provenance_json_read_error": prov_err,
        "has_summary_structure_like_keys": has_summary_structure_like,
        "n_summary_structure_like_keys": len(summary_struct_hits),
        "has_provenance_structure_like_keys": has_provenance_structure_like,
        "n_provenance_structure_like_keys": len(prov_struct_hits),
        "json_key_preview_path": str(preview_path),
        "v42_status": status,
        "recommended_action": action,
    })

audit = pd.DataFrame(rows)

audit_csv = audit_dir / "v42_mp_json_structure_source_audit.csv"
audit_md = audit_dir / "v42_mp_json_structure_source_audit.md"
summary_json = audit_dir / "v42_mp_json_structure_source_audit_summary.json"

audit.to_csv(audit_csv, index=False)
audit.to_markdown(audit_md, index=False)

summary = {
    "status": "pass_json_structure_source_audit",
    "n_targets": int(len(audit)),
    "n_possible_structure_payload_in_json": int((audit["v42_status"] == "possible_structure_payload_in_json").sum()),
    "n_no_structure_payload_detected": int((audit["v42_status"] == "no_structure_payload_detected").sum()),
    "possible_structure_mp_ids": sorted(audit.loc[audit["v42_status"] == "possible_structure_payload_in_json", "mp_id"].astype(str).tolist()),
    "no_structure_payload_mp_ids": sorted(audit.loc[audit["v42_status"] == "no_structure_payload_detected", "mp_id"].astype(str).tolist()),
    "output_csv": str(audit_csv),
    "interpretation": (
        "V42 checks whether the local MP summary/provenance JSON files contain structure-like payloads. "
        "Targets with structure-like JSON payloads may be recoverable without online fetching. "
        "Targets without structure-like payloads need external structure fetching or a clearly labeled formula-only fallback."
    ),
    "next_required_step": (
        "If structure-like payloads exist, build V43 to reconstruct POSCAR/structure features. "
        "If not, build V43 as a formula-only fallback layer with explicit non-structural labeling, or fetch missing structures."
    ),
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", audit_csv)
print("[SAVE]", audit_md)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build V42 report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V42_REPORT.md" <<MD
# Final Benchmark-100 V42 Report

## 1. Version

\`benchmark_100_v42_recover_structure_sources_from_mp_json_for_remaining_v33_targets\`

## 2. Purpose

V42 checks whether the seven remaining V33 targets can recover structure sources from local MP \`summary_json\` or \`provenance_json\` files.

This follows V41, which showed that the seven remaining targets have formula metadata but no detected POSCAR files.

## 3. Key outputs

- JSON structure-source audit:
  \`mp_json_structure_source_audit_v42/v42_mp_json_structure_source_audit.csv\`

- Markdown preview:
  \`mp_json_structure_source_audit_v42/v42_mp_json_structure_source_audit.md\`

- Summary:
  \`mp_json_structure_source_audit_v42/v42_mp_json_structure_source_audit_summary.json\`

- Per-target JSON key previews:
  \`recovered_structure_sources_v42/*_json_structure_key_preview.json\`

## 4. Interpretation

V42 is still not a Stage3 generation layer.

It only checks whether local MP JSON files contain enough structure-like information to support later POSCAR reconstruction or Stage3 feature regeneration.

## 5. Next required step

If V42 finds structure-like payloads, proceed to V43 to reconstruct POSCAR or structure-derived features.

If V42 does not find structure-like payloads, the remaining choices are:

1. fetch/build missing structures for the seven MP IDs;
2. create a clearly labeled formula-only Stage3 fallback feature set;
3. keep the targets blocked from real Flow/MDN export until structure-backed features are available.
MD

cp "$AUDIT_DIR/v42_mp_json_structure_source_audit_summary.json" "$REPORT_DIR/FINAL_V42_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Archive V42 checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V42_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V42_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V42_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V42 MP JSON structure source audit completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V42_REPORT.md"
echo
echo "Audit:"
echo "$AUDIT_DIR/v42_mp_json_structure_source_audit.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v42_mp_json_structure_source_audit_summary.json"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
