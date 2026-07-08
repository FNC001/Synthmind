#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V38_VERSION="benchmark_100_v38_stage3_input_feature_extension_for_v33_targets"
OUT_ROOT="$PROJECT_ROOT/outputs/$V38_VERSION"
AUDIT_DIR="$OUT_ROOT/stage3_feature_source_audit_v38"
FINAL_DIR="$OUT_ROOT/FINAL_REPORT_V38"

V33_MANIFEST="$PROJECT_ROOT/outputs/benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping/stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv"
V37_PREFLIGHT="$PROJECT_ROOT/outputs/benchmark_100_v37_real_stage3_generation_from_v36_targetaware_stage2/stage3_input_preflight_v37/v37_stage3_input_payload_compatibility.csv"

mkdir -p "$AUDIT_DIR" "$FINAL_DIR"

echo "============================================================"
echo "Benchmark-100 V38 Stage3 Feature Source Audit"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "AUDIT_DIR    = $AUDIT_DIR"
echo "============================================================"

echo
echo "[STEP 1] Check required inputs"

for p in "$V33_MANIFEST" "$V37_PREFLIGHT"; do
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] Missing required file: $p"
    exit 1
  fi
  echo "[OK] $p"
done

echo
echo "[STEP 2] Audit possible Stage3 feature sources"

python - <<PY
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path("$PROJECT_ROOT")
audit_dir = Path("$AUDIT_DIR")
audit_dir.mkdir(parents=True, exist_ok=True)

v33_manifest = Path("$V33_MANIFEST")
v37_preflight = Path("$V37_PREFLIGHT")

mp_re = re.compile(r"(mp-\d+)")

def extract_mp_id(x):
    s = str(x) if x is not None else ""
    m = mp_re.search(s)
    return m.group(1) if m else s.strip()

manifest = pd.read_csv(v33_manifest)
targets = sorted(set(manifest["mp_id"].astype(str).map(extract_mp_id)))

candidate_sources = []

# 1. Existing Stage3 condition dataset NPZ files
for p in [
    project_root / "data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/train.npz",
    project_root / "data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/val.npz",
    project_root / "data/interim/generative/stage3_condition_dataset/hybrid_mixed_v1/test.npz",
]:
    if p.exists():
        candidate_sources.append(("stage3_condition_npz", p))

# 2. Stage3 task-view CSVs
for p in (project_root / "data/interim/features/stage3_task_views").glob("*.csv"):
    candidate_sources.append(("stage3_task_view_csv", p))

# 3. Structdesc Stage3 CSVs
for root in [
    project_root / "data/interim/features/structdesc_features_stage3_v2",
    project_root / "data/interim/features/structdesc_features",
]:
    if root.exists():
        for p in root.glob("*stage3*.csv"):
            candidate_sources.append(("stage3_structdesc_csv", p))

# 4. Inference Stage3 hybrid CSVs
infer_root = project_root / "data/interim/infer"
if infer_root.exists():
    for p in infer_root.glob("benchmark_*/stage3_hybrid/*.csv"):
        candidate_sources.append(("infer_stage3_hybrid_csv", p))

# 5. Graph cache PKLs
for root in [
    project_root / "data/interim/graph_cache/cgcnn_stage3",
    project_root / "data/interim/graph_cache/chgnet_stage3",
]:
    if root.exists():
        for p in root.rglob("*.pkl"):
            candidate_sources.append(("stage3_graph_cache_pkl", p))

rows = []

def scan_npz(source_class, path):
    try:
        arr = np.load(path, allow_pickle=True)
        sample_ids = arr["sample_id"] if "sample_id" in arr else []
        found = set()
        for sid in sample_ids:
            mid = extract_mp_id(sid)
            if mid in targets:
                found.add(mid)
        return found, "", list(arr.files)
    except Exception as e:
        return set(), str(e), []

def scan_csv(source_class, path):
    try:
        df = pd.read_csv(path, nrows=None)
        cols = list(df.columns)
        id_cols = [c for c in cols if c.lower() in {
            "mp_id", "material_id", "case_id", "sample_id", "id", "target_id"
        } or "material" in c.lower() or "sample" in c.lower() or c.lower().endswith("_id")]
        found = set()
        for c in id_cols:
            vals = df[c].astype(str).map(extract_mp_id)
            found.update([v for v in vals if v in targets])
        return found, "", cols
    except Exception as e:
        return set(), str(e), []

def scan_pkl(source_class, path):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        found = set()
        keys_seen = []

        if isinstance(obj, dict):
            keys_seen = list(obj.keys())[:20]
            for k in obj.keys():
                mid = extract_mp_id(k)
                if mid in targets:
                    found.add(mid)
        elif isinstance(obj, pd.DataFrame):
            cols = list(obj.columns)
            keys_seen = cols[:20]
            for c in cols:
                if "id" in c.lower() or "material" in c.lower() or "sample" in c.lower():
                    vals = obj[c].astype(str).map(extract_mp_id)
                    found.update([v for v in vals if v in targets])
        elif isinstance(obj, list):
            keys_seen = [type(x).__name__ for x in obj[:5]]
            for x in obj:
                if isinstance(x, dict):
                    for k, v in x.items():
                        if "id" in str(k).lower() or "material" in str(k).lower() or "sample" in str(k).lower():
                            mid = extract_mp_id(v)
                            if mid in targets:
                                found.add(mid)
        return found, "", keys_seen
    except Exception as e:
        return set(), str(e), []

for source_class, path in candidate_sources:
    if path.suffix == ".npz":
        found, err, meta = scan_npz(source_class, path)
    elif path.suffix == ".csv":
        found, err, meta = scan_csv(source_class, path)
    elif path.suffix == ".pkl":
        found, err, meta = scan_pkl(source_class, path)
    else:
        found, err, meta = set(), "unsupported_suffix", []

    rows.append({
        "source_class": source_class,
        "path": str(path),
        "exists": path.exists(),
        "n_found_targets": len(found),
        "found_mp_ids": ";".join(sorted(found)),
        "read_error": err,
        "source_meta_preview": json.dumps(meta[:40] if isinstance(meta, list) else meta, ensure_ascii=False),
    })

source_df = pd.DataFrame(rows)

target_rows = []
for mp_id in targets:
    matched = source_df[source_df["found_mp_ids"].astype(str).str.contains(mp_id, regex=False, na=False)]
    target_rows.append({
        "mp_id": mp_id,
        "has_any_stage3_feature_source": len(matched) > 0,
        "n_matching_sources": int(len(matched)),
        "matching_source_classes": ";".join(sorted(set(matched["source_class"].astype(str)))) if len(matched) else "",
        "matching_sources": ";".join(matched["path"].astype(str).head(10).tolist()) if len(matched) else "",
    })

target_df = pd.DataFrame(target_rows)

summary = {
    "status": "pass_with_feature_sources" if int(target_df["has_any_stage3_feature_source"].sum()) == len(target_df) else "blocked_missing_stage3_feature_sources",
    "n_v33_targets": int(len(target_df)),
    "n_targets_with_any_feature_source": int(target_df["has_any_stage3_feature_source"].sum()),
    "n_targets_missing_any_feature_source": int((~target_df["has_any_stage3_feature_source"]).sum()),
    "targets_with_feature_source": sorted(target_df.loc[target_df["has_any_stage3_feature_source"], "mp_id"].tolist()),
    "targets_missing_feature_source": sorted(target_df.loc[~target_df["has_any_stage3_feature_source"], "mp_id"].tolist()),
    "n_scanned_sources": int(len(source_df)),
    "n_sources_with_hits": int((source_df["n_found_targets"] > 0).sum()),
    "interpretation": (
        "V38 checks whether the nine V33 formula-exact MP targets have any reusable Stage3-compatible "
        "feature source in existing NPZ/CSV/PKL artifacts. If targets are missing from all feature sources, "
        "the correct next step is to regenerate Stage3 features rather than fabricate real MDN/Flow candidates."
    ),
    "next_required_action": (
        "If all targets have feature sources, build a Stage3 NPZ extension. "
        "If not, regenerate Stage3 input features for the missing MP IDs from structures/graph descriptors."
    ),
}

source_csv = audit_dir / "v38_stage3_feature_source_inventory.csv"
source_md = audit_dir / "v38_stage3_feature_source_inventory_hits.md"
target_csv = audit_dir / "v38_target_feature_source_status.csv"
target_md = audit_dir / "v38_target_feature_source_status.md"
summary_json = audit_dir / "v38_stage3_feature_source_audit_summary.json"
summary_md = audit_dir / "v38_stage3_feature_source_audit_summary.md"

source_df.to_csv(source_csv, index=False)
source_df[source_df["n_found_targets"] > 0].to_markdown(source_md, index=False)
target_df.to_csv(target_csv, index=False)
target_df.to_markdown(target_md, index=False)
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

with open(summary_md, "w", encoding="utf-8") as f:
    f.write("| metric | value |\\n|:--|:--|\\n")
    for k, v in summary.items():
        f.write(f"| {k} | {v} |\\n")

print(json.dumps(summary, indent=2))
print("[SAVE]", source_csv)
print("[SAVE]", source_md)
print("[SAVE]", target_csv)
print("[SAVE]", target_md)
print("[SAVE]", summary_json)
print("[SAVE]", summary_md)
PY

echo
echo "[STEP 3] Build V38 final report"

cat > "$FINAL_DIR/FINAL_BENCHMARK_100_V38_FEATURE_SOURCE_AUDIT_REPORT.md" <<MD
# Final Benchmark-100 V38 Feature Source Audit Report

## 1. Version

\`benchmark_100_v38_stage3_input_feature_extension_for_v33_targets\`

## 2. Purpose

V38 audits whether the nine V33 formula-exact MP targets have reusable Stage3-compatible feature sources.

This follows the chain:

- V34 confirmed that these MP IDs are absent from the real Stage3 candidate library.
- V36 created Stage2-candidate-compatible target rows.
- V37 showed that existing MDN/Flow wrappers cannot consume most V36 rows because Stage3 NPZ feature payloads are missing.
- V38 checks whether those feature payloads can be recovered from existing Stage3 NPZ/CSV/PKL artifacts.

## 3. Key outputs

- Source inventory: \`stage3_feature_source_audit_v38/v38_stage3_feature_source_inventory.csv\`
- Source hits: \`stage3_feature_source_audit_v38/v38_stage3_feature_source_inventory_hits.md\`
- Target status: \`stage3_feature_source_audit_v38/v38_target_feature_source_status.csv\`
- Target status preview: \`stage3_feature_source_audit_v38/v38_target_feature_source_status.md\`
- Summary: \`stage3_feature_source_audit_v38/v38_stage3_feature_source_audit_summary.json\`

## 4. Interpretation

If all nine targets have feature sources, the next step is to build a Stage3 NPZ extension.

If some targets are still missing, the correct next step is to regenerate Stage3 input features for those MP IDs from their structures and descriptor pipeline.

V38 should not fabricate Stage3 features or report placeholder-generated candidates as real MDN/Flow outputs.

MD

cp "$AUDIT_DIR/v38_stage3_feature_source_audit_summary.json" "$FINAL_DIR/FINAL_V38_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Show V38 summary"
cat "$AUDIT_DIR/v38_stage3_feature_source_audit_summary.json"

echo
echo "============================================================"
echo "[DONE] V38 Stage3 feature source audit completed."
echo "Report:"
echo "$FINAL_DIR/FINAL_BENCHMARK_100_V38_FEATURE_SOURCE_AUDIT_REPORT.md"
echo
echo "Check:"
echo "cat $AUDIT_DIR/v38_target_feature_source_status.md"
echo "cat $AUDIT_DIR/v38_stage3_feature_source_inventory_hits.md"
echo "cat $AUDIT_DIR/v38_stage3_feature_source_audit_summary.json"
echo "============================================================"
