#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V45_VERSION="benchmark_100_v45_real_stage3_export_from_v44_regenerated_features"
OUT_ROOT="$PROJECT_ROOT/outputs/$V45_VERSION"

STAGE2_DIR="$PROJECT_ROOT/outputs/benchmark_100_v36_v33_target_aware_stage2_candidate_construction/target_aware_stage2_candidates_v36"
STAGE3_INPUT_DIR="$PROJECT_ROOT/outputs/benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar/stage3_input_extension_v44"

FLOW_EXPORT_SCRIPT="$PROJECT_ROOT/scripts/04_train/joint/arset_mixflow/2_export_stage3_precursor_conditioned_candidates_mixture_flow.py"
FLOW_STAGE3_SCRIPT="$PROJECT_ROOT/scripts/04_train/stage3/mixed/train_condition_mixture_flow_mixed.py"
FLOW_CKPT="$PROJECT_ROOT/runs/stage3/condition_mixture_flow_hybrid_mixed_v1_yset_conditioned_v2/best_stage3_condition_mixture_flow_mixed.pt"
BASELINE_CKPT="$PROJECT_ROOT/runs/stage3/stage3_baseline_commonized_v1/best_model.pkl"

MDN_EXPORT_SCRIPT="$PROJECT_ROOT/scripts/04_train/joint/arset_mdn/2_export_stage3_precursor_conditioned_candidates_mdn.py"
MDN_STAGE3_SCRIPT="$PROJECT_ROOT/scripts/04_train/stage3/train_condition_residual_mdn_mixed.py"
MDN_CKPT="$PROJECT_ROOT/runs/stage3/condition_residual_mdn_hybrid_mixed_v1_yset_conditioned/best_stage3_residual_mdn_mixed.pt"

FLOW_OUT="$OUT_ROOT/flow_export_v45"
MDN_OUT="$OUT_ROOT/mdn_export_v45"
MERGED_DIR="$OUT_ROOT/merged_stage3_candidates_v45"
AUDIT_DIR="$OUT_ROOT/audit_v45"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V45"

mkdir -p "$FLOW_OUT" "$MDN_OUT" "$MERGED_DIR" "$AUDIT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V45 Real Stage3 Export from V44 Regenerated Features"
echo "PROJECT_ROOT     = $PROJECT_ROOT"
echo "OUT_ROOT         = $OUT_ROOT"
echo "STAGE2_DIR       = $STAGE2_DIR"
echo "STAGE3_INPUT_DIR = $STAGE3_INPUT_DIR"
echo "============================================================"

echo
echo "[STEP 1] Check required inputs"

for p in \
  "$STAGE2_DIR/test_candidates.csv" \
  "$STAGE2_DIR/val_candidates.csv" \
  "$STAGE3_INPUT_DIR/train.npz" \
  "$STAGE3_INPUT_DIR/val.npz" \
  "$STAGE3_INPUT_DIR/test.npz" \
  "$STAGE3_INPUT_DIR/schema.json" \
  "$STAGE3_INPUT_DIR/condition_schema.json" \
  "$FLOW_EXPORT_SCRIPT" \
  "$FLOW_STAGE3_SCRIPT" \
  "$FLOW_CKPT" \
  "$BASELINE_CKPT"
do
  if [[ ! -e "$p" ]]; then
    echo "[ERROR] missing required input: $p"
    exit 1
  fi
  echo "[OK] $p"
done

echo
echo "[STEP 2] Run Flow export from V44 regenerated Stage3 input"

set +e
python "$FLOW_EXPORT_SCRIPT" \
  --stage2_candidates_dir "$STAGE2_DIR" \
  --stage3_input_dir "$STAGE3_INPUT_DIR" \
  --stage3_script "$FLOW_STAGE3_SCRIPT" \
  --stage3_ckpt "$FLOW_CKPT" \
  --baseline_ckpt "$BASELINE_CKPT" \
  --output_dir "$FLOW_OUT" \
  --splits "val,test" \
  --model_class_name "MixtureResidualConditionFlowMixed" \
  --n_gen_samples 8 \
  --max_stage2_candidates 16 \
  --device cpu \
  --clip_to_train_range \
  > "$AUDIT_DIR/v45_flow_export_stdout.log" \
  2> "$AUDIT_DIR/v45_flow_export_stderr.log"

FLOW_STATUS=$?
set -e

echo "[FLOW_STATUS] $FLOW_STATUS"

echo
echo "[STEP 3] Optional MDN export attempt"

MDN_STATUS=999

if [[ -f "$MDN_EXPORT_SCRIPT" && -f "$MDN_STAGE3_SCRIPT" && -f "$MDN_CKPT" ]]; then
  set +e
  python "$MDN_EXPORT_SCRIPT" \
    --stage2_candidates_dir "$STAGE2_DIR" \
    --stage3_input_dir "$STAGE3_INPUT_DIR" \
    --stage3_script "$MDN_STAGE3_SCRIPT" \
    --stage3_ckpt "$MDN_CKPT" \
    --output_dir "$MDN_OUT" \
    --splits "val,test" \
    --n_gen_samples 5 \
    --max_stage2_candidates 16 \
    --temperature_tol 150 \
    --time_tol 24 \
    --device cpu \
    > "$AUDIT_DIR/v45_mdn_export_stdout.log" \
    2> "$AUDIT_DIR/v45_mdn_export_stderr.log"

  MDN_STATUS=$?
  set -e
else
  echo "[SKIP] MDN inputs are incomplete; Flow remains the main V45 path." | tee "$AUDIT_DIR/v45_mdn_export_stdout.log"
  echo "" > "$AUDIT_DIR/v45_mdn_export_stderr.log"
fi

echo "[MDN_STATUS] $MDN_STATUS"

echo
echo "[STEP 4] Collect candidate outputs"

python - <<PY
import json
from pathlib import Path

import pandas as pd

out_root = Path("$OUT_ROOT")
flow_out = Path("$FLOW_OUT")
mdn_out = Path("$MDN_OUT")
merged_dir = Path("$MERGED_DIR")
audit_dir = Path("$AUDIT_DIR")

merged_dir.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)

candidate_files = []
for root in [flow_out, mdn_out]:
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in [".csv", ".jsonl"]:
                candidate_files.append(p)

inventory_rows = []
frames = []

def read_candidate_file(p):
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".jsonl":
        return pd.read_json(p, lines=True)
    raise ValueError(p)

for p in candidate_files:
    row = {
        "path": str(p),
        "suffix": p.suffix,
        "read_status": "pending",
        "n_rows": 0,
        "columns": "",
        "error": "",
    }
    try:
        df = read_candidate_file(p)
        row["read_status"] = "pass"
        row["n_rows"] = int(len(df))
        row["columns"] = ";".join(map(str, df.columns))
        df["v45_source_file"] = str(p)
        frames.append(df)
    except Exception as e:
        row["read_status"] = "failed"
        row["error"] = repr(e)
    inventory_rows.append(row)

inventory = pd.DataFrame(inventory_rows)
inventory_csv = audit_dir / "v45_exported_candidate_file_inventory.csv"
inventory_md = audit_dir / "v45_exported_candidate_file_inventory.md"
inventory.to_csv(inventory_csv, index=False)
inventory.to_markdown(inventory_md, index=False)

if frames:
    merged = pd.concat(frames, ignore_index=True, sort=False)
else:
    merged = pd.DataFrame()

merged_csv = merged_dir / "v45_real_stage3_candidates_from_v44_features_raw.csv"
merged.to_csv(merged_csv, index=False)

summary = {
    "status": "pass_with_candidates" if len(merged) > 0 else "blocked_no_candidates",
    "flow_exit_status": int("$FLOW_STATUS"),
    "mdn_exit_status": int("$MDN_STATUS"),
    "n_candidate_files_detected": int(len(candidate_files)),
    "n_merged_candidate_rows": int(len(merged)),
    "merged_output_csv": str(merged_csv) if len(merged) > 0 else "",
    "candidate_inventory_csv": str(inventory_csv),
    "candidate_columns": list(merged.columns) if len(merged) > 0 else [],
    "interpretation": (
        "V45 runs Stage3 Flow/MDN export using V44 regenerated Stage3-compatible input features. "
        "Flow is the primary supported path; MDN remains optional because earlier attempts showed checkpoint/interface mismatch."
    ),
    "important_caution": (
        "V44 features are structure/formula regenerated and zero-fill unavailable precursor/graph-specific columns. "
        "Therefore V45 candidates should be labeled as real Stage3 Flow outputs from regenerated V44 features, not as fully original-distribution V30 candidates."
    ),
    "logs": {
        "flow_stdout": str(audit_dir / "v45_flow_export_stdout.log"),
        "flow_stderr": str(audit_dir / "v45_flow_export_stderr.log"),
        "mdn_stdout": str(audit_dir / "v45_mdn_export_stdout.log"),
        "mdn_stderr": str(audit_dir / "v45_mdn_export_stderr.log"),
    },
    "next_required_step": (
        "If candidate rows exist, proceed to V45b: normalize Flow candidates into the V32/V40c Stage3 candidate schema, "
        "attach MP metadata, merge with the Stage3 library, and rerun metadata-aware alignment for the seven V44 targets."
    )
}

summary_json = audit_dir / "v45_stage3_export_summary.json"
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", inventory_csv)
print("[SAVE]", inventory_md)
print("[SAVE]", merged_csv)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 5] Build V45 final report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V45_REPORT.md" <<MD
# Final Benchmark-100 V45 Report

## 1. Version

\`benchmark_100_v45_real_stage3_export_from_v44_regenerated_features\`

## 2. Purpose

V45 reruns Stage3 Flow/MDN export using the V44 regenerated Stage3-compatible NPZ inputs for the seven remaining V33 targets.

## 3. Inputs

- Stage2 candidate directory:
  \`outputs/benchmark_100_v36_v33_target_aware_stage2_candidate_construction/target_aware_stage2_candidates_v36\`

- Stage3 input directory:
  \`outputs/benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar/stage3_input_extension_v44\`

## 4. Outputs

- Flow export:
  \`flow_export_v45\`

- MDN export:
  \`mdn_export_v45\`

- Raw merged candidates:
  \`merged_stage3_candidates_v45/v45_real_stage3_candidates_from_v44_features_raw.csv\`

- Export audit:
  \`audit_v45/v45_stage3_export_summary.json\`

## 5. Interpretation

V45 is a controlled Stage3 export layer.

The main expected successful path is Flow export, because V40b already showed that the Flow interface can produce candidate rows when compatible Stage3 NPZ payloads exist.

MDN is retained as an optional diagnostic path because previous attempts showed checkpoint/interface mismatch.

## 6. Caution

V45 candidates should be labeled conservatively as:

\`real_stage3_flow_from_v44_regenerated_features\`

They should not be described as fully equivalent to original V30 real Stage3 candidates unless later feature-parity checks confirm that all descriptor families were reconstructed.

## 7. Next required step

If V45 produces candidate rows, proceed to V45b:

\`benchmark_100_v45b_normalize_v45_stage3_candidates_and_realign\`

V45b should:

1. normalize raw Flow outputs into the V32/V40c Stage3 candidate schema;
2. attach MP metadata;
3. merge with the existing Stage3 library;
4. rerun metadata-aware alignment for the seven V44 targets;
5. produce a final comparison against V32/V40d.
MD

cp "$AUDIT_DIR/v45_stage3_export_summary.json" "$REPORT_DIR/FINAL_V45_METRIC_SUMMARY.json"

echo
echo "[STEP 6] Archive V45 checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V45_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V45_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V45_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V45 Stage3 export from V44 regenerated features completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V45_REPORT.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v45_stage3_export_summary.json"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
