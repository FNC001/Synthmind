#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/Users/wyc/SynPred}"

V43_VERSION="benchmark_100_v43_reconstruct_poscar_from_mp_json_for_remaining_v33_targets"
OUT_ROOT="$PROJECT_ROOT/outputs/$V43_VERSION"

V42_AUDIT="$PROJECT_ROOT/outputs/benchmark_100_v42_recover_structure_sources_from_mp_json_for_remaining_v33_targets/mp_json_structure_source_audit_v42/v42_mp_json_structure_source_audit.csv"

RECOVER_DIR="$OUT_ROOT/recovered_poscar_v43"
AUDIT_DIR="$OUT_ROOT/audit_v43"
REPORT_DIR="$OUT_ROOT/FINAL_REPORT_V43"

mkdir -p "$RECOVER_DIR" "$AUDIT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Benchmark-100 V43 Reconstruct POSCAR from MP JSON"
echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "OUT_ROOT     = $OUT_ROOT"
echo "============================================================"

echo
echo "[STEP 1] Check V42 audit input"

if [[ ! -f "$V42_AUDIT" ]]; then
  echo "[ERROR] missing V42 audit table: $V42_AUDIT"
  exit 1
fi

echo "[OK] $V42_AUDIT"

echo
echo "[STEP 2] Extract structure payloads and write POSCAR files"

python - <<PY
import json
import math
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

v42_audit = Path("$V42_AUDIT")
recover_dir = Path("$RECOVER_DIR")
audit_dir = Path("$AUDIT_DIR")

recover_dir.mkdir(parents=True, exist_ok=True)
audit_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(v42_audit)

def read_json(path):
    p = Path(str(path))
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def looks_like_matrix(x):
    if not isinstance(x, list) or len(x) != 3:
        return False
    for row in x:
        if not isinstance(row, list) or len(row) != 3:
            return False
        if not all(is_number(v) for v in row):
            return False
    return True

def extract_lattice_matrix(lattice):
    if not isinstance(lattice, dict):
        return None

    # pymatgen-style
    if looks_like_matrix(lattice.get("matrix")):
        return [[float(v) for v in row] for row in lattice["matrix"]]

    # sometimes nested as lattice -> matrix
    if "lattice" in lattice and isinstance(lattice["lattice"], dict):
        m = extract_lattice_matrix(lattice["lattice"])
        if m is not None:
            return m

    return None

def extract_species(site):
    # pymatgen-style: species: [{"element": "Na", "occu": 1}]
    sp = site.get("species")
    if isinstance(sp, list) and len(sp) > 0:
        first = sp[0]
        if isinstance(first, dict):
            for key in ["element", "label", "species", "symbol"]:
                if key in first and first[key]:
                    return str(first[key])
        elif isinstance(first, str):
            return first

    # older style
    for key in ["label", "species_string", "element", "specie"]:
        if key in site and site[key]:
            val = site[key]
            if isinstance(val, dict):
                for k2 in ["element", "symbol", "label"]:
                    if k2 in val:
                        return str(val[k2])
            return str(val)

    return None

def extract_frac_coords(site):
    for key in ["abc", "frac_coords", "fractional_coords"]:
        if key in site and isinstance(site[key], list) and len(site[key]) == 3:
            vals = site[key]
            if all(is_number(v) for v in vals):
                return [float(v) for v in vals]
    return None

def looks_like_structure(obj):
    if not isinstance(obj, dict):
        return False
    lattice = obj.get("lattice")
    sites = obj.get("sites")
    if not isinstance(lattice, dict) or not isinstance(sites, list) or len(sites) == 0:
        return False
    matrix = extract_lattice_matrix(lattice)
    if matrix is None:
        return False
    good_sites = 0
    for site in sites:
        if isinstance(site, dict) and extract_species(site) and extract_frac_coords(site):
            good_sites += 1
    return good_sites > 0

def find_structure_candidates(obj, path="root", out=None):
    if out is None:
        out = []
    if looks_like_structure(obj):
        out.append((path, obj))
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_structure_candidates(v, f"{path}.{k}", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:20]):
            find_structure_candidates(v, f"{path}[{i}]", out)
    return out

def choose_structure(summary_obj, prov_obj):
    candidates = []

    # 优先 summary_json 中的 structure/final_structure
    for source_name, obj in [("summary_json", summary_obj), ("provenance_json", prov_obj)]:
        if obj is None:
            continue
        found = find_structure_candidates(obj)
        for path, struct in found:
            priority = 100
            low = path.lower()
            if low.endswith(".structure") or ".structure" in low:
                priority -= 30
            if "final" in low:
                priority -= 20
            if "initial" in low:
                priority -= 5
            if source_name == "summary_json":
                priority -= 10
            candidates.append((priority, source_name, path, struct))

    if not candidates:
        return None, "", "", 0

    candidates.sort(key=lambda x: x[0])
    priority, source_name, path, struct = candidates[0]
    return struct, source_name, path, len(candidates)

def normalize_element_symbol(s):
    s = str(s).strip()
    # remove oxidation-like decorations if present
    m = re.match(r"([A-Z][a-z]?)", s)
    return m.group(1) if m else s

def write_poscar(struct, title, out_path):
    lattice = extract_lattice_matrix(struct.get("lattice", {}))
    sites = struct.get("sites", [])

    parsed = []
    for site in sites:
        if not isinstance(site, dict):
            continue
        elem = extract_species(site)
        frac = extract_frac_coords(site)
        if elem and frac:
            parsed.append((normalize_element_symbol(elem), frac))

    if lattice is None or not parsed:
        raise ValueError("missing lattice matrix or parsed sites")

    # group by element preserving first occurrence
    elems = []
    grouped = defaultdict(list)
    for elem, frac in parsed:
        if elem not in grouped:
            elems.append(elem)
        grouped[elem].append(frac)

    lines = []
    lines.append(str(title))
    lines.append("1.0")
    for row in lattice:
        lines.append("  " + "  ".join(f"{float(v): .16f}" for v in row))
    lines.append("  " + "  ".join(elems))
    lines.append("  " + "  ".join(str(len(grouped[e])) for e in elems))
    lines.append("Direct")
    for elem in elems:
        for frac in grouped[elem]:
            lines.append("  " + "  ".join(f"{float(v): .16f}" for v in frac))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "n_sites": len(parsed),
        "elements": ";".join(elems),
        "element_counts": ";".join(f"{e}:{len(grouped[e])}" for e in elems),
    }

rows = []

for _, r in df.iterrows():
    mp_id = str(r["mp_id"])
    formula = str(r.get("target_formula", ""))

    summary_path = str(r.get("summary_json_path", ""))
    prov_path = str(r.get("provenance_json_path", ""))

    summary_obj = read_json(summary_path)
    prov_obj = read_json(prov_path)

    struct, source_name, struct_path, n_candidates = choose_structure(summary_obj, prov_obj)

    structure_json_path = recover_dir / f"{mp_id}_recovered_structure.json"
    poscar_path = recover_dir / f"{mp_id}.vasp"

    status = "blocked_no_valid_structure_payload"
    err = ""
    info = {}

    if struct is not None:
        try:
            structure_json_path.write_text(json.dumps(struct, indent=2), encoding="utf-8")
            info = write_poscar(struct, f"{mp_id} {formula} recovered_from_{source_name}", poscar_path)
            status = "pass_recovered_poscar"
        except Exception as e:
            err = repr(e)
            status = "failed_poscar_write"

    rows.append({
        "external_case_id": r.get("external_case_id", ""),
        "mp_id": mp_id,
        "target_formula": formula,
        "structure_source": source_name,
        "structure_json_path_in_source": struct_path,
        "n_structure_candidates_found": int(n_candidates),
        "recovered_structure_json": str(structure_json_path) if structure_json_path.exists() else "",
        "recovered_poscar": str(poscar_path) if poscar_path.exists() else "",
        "n_sites": info.get("n_sites", ""),
        "elements": info.get("elements", ""),
        "element_counts": info.get("element_counts", ""),
        "v43_status": status,
        "error": err,
    })

audit = pd.DataFrame(rows)

audit_csv = audit_dir / "v43_recovered_poscar_audit.csv"
audit_md = audit_dir / "v43_recovered_poscar_audit.md"
summary_json = audit_dir / "v43_recovered_poscar_summary.json"

audit.to_csv(audit_csv, index=False)
audit.to_markdown(audit_md, index=False)

summary = {
    "status": "pass" if (audit["v43_status"] == "pass_recovered_poscar").all() else "pass_with_blocked_or_failed_targets",
    "n_targets": int(len(audit)),
    "n_recovered_poscar": int((audit["v43_status"] == "pass_recovered_poscar").sum()),
    "n_failed_or_blocked": int((audit["v43_status"] != "pass_recovered_poscar").sum()),
    "recovered_mp_ids": sorted(audit.loc[audit["v43_status"] == "pass_recovered_poscar", "mp_id"].astype(str).tolist()),
    "failed_or_blocked_mp_ids": sorted(audit.loc[audit["v43_status"] != "pass_recovered_poscar", "mp_id"].astype(str).tolist()),
    "output_poscar_dir": str(recover_dir),
    "output_audit_csv": str(audit_csv),
    "interpretation": (
        "V43 reconstructs POSCAR files from local MP summary/provenance JSON structure payloads. "
        "Recovered POSCAR files can be used by the next Stage3 feature-regeneration layer."
    ),
    "next_required_step": (
        "Proceed to V44: regenerate Stage3-compatible formula/structure features and NPZ inputs "
        "for the recovered MP targets, then rerun Flow/MDN export."
    ),
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("[SAVE]", audit_csv)
print("[SAVE]", audit_md)
print("[SAVE]", summary_json)
PY

echo
echo "[STEP 3] Build V43 final report"

cat > "$REPORT_DIR/FINAL_BENCHMARK_100_V43_REPORT.md" <<MD
# Final Benchmark-100 V43 Report

## 1. Version

\`benchmark_100_v43_reconstruct_poscar_from_mp_json_for_remaining_v33_targets\`

## 2. Purpose

V43 reconstructs structure files from the local MP JSON payloads identified by V42.

V42 showed that all seven remaining V33 targets contain structure-like keys in local \`summary_json\` or \`provenance_json\` files. V43 converts those payloads into reusable POSCAR-style structure files.

## 3. Key outputs

- Recovered POSCAR directory:
  \`recovered_poscar_v43/\`

- POSCAR recovery audit:
  \`audit_v43/v43_recovered_poscar_audit.csv\`

- Markdown audit:
  \`audit_v43/v43_recovered_poscar_audit.md\`

- Summary:
  \`audit_v43/v43_recovered_poscar_summary.json\`

## 4. Interpretation

V43 is a structure-source recovery layer.

It does not generate Stage3 condition candidates yet. It only prepares structure-backed inputs for the next feature-regeneration step.

## 5. Next required step

Proceed to V44:

\`benchmark_100_v44_stage3_feature_npz_regeneration_from_v43_poscar\`

V44 should regenerate Stage3-compatible feature arrays and NPZ files for recovered targets, including:

- \`train.npz\`
- \`val.npz\`
- \`test.npz\`
- \`schema.json\`
- \`condition_schema.json\`

Only after those files exist should Flow/MDN export be rerun.
MD

cp "$AUDIT_DIR/v43_recovered_poscar_summary.json" "$REPORT_DIR/FINAL_V43_METRIC_SUMMARY.json"

echo
echo "[STEP 4] Archive V43 checkpoint"

cd "$PROJECT_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
BASE="checkpoint_${V43_VERSION}_${TS}"

tar -czf "outputs/${BASE}.tar.gz" \
  "outputs/$V43_VERSION" \
  "scripts/07_infer/structure_to_synthesis_route/benchmark/$V43_VERSION"

cd outputs
shasum -a 256 "${BASE}.tar.gz" > "${BASE}.tar.gz.sha256"

echo
echo "============================================================"
echo "[DONE] V43 POSCAR reconstruction completed."
echo "Report:"
echo "$REPORT_DIR/FINAL_BENCHMARK_100_V43_REPORT.md"
echo
echo "Audit:"
echo "$AUDIT_DIR/v43_recovered_poscar_audit.md"
echo
echo "Summary:"
echo "$AUDIT_DIR/v43_recovered_poscar_summary.json"
echo
echo "Archive:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz"
echo "Checksum:"
echo "$PROJECT_ROOT/outputs/${BASE}.tar.gz.sha256"
echo "============================================================"
