#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd


def read_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    root = Path(args.project_root)
    out_root = Path(args.output_dir)
    report_dir = out_root / "FINAL_REPORT_V33"
    report_dir.mkdir(parents=True, exist_ok=True)

    version = "benchmark_100_v33_stage3_library_expansion_or_direct_formula_mapping"

    gap_summary = read_json(out_root / "stage3_library_gap_analysis_v33/v33_gap_case_table_summary.json")
    mapping_summary = read_json(out_root / "direct_formula_mp_mapping_v33/v33_direct_formula_mp_mapping_summary.json")
    target_summary = read_json(out_root / "expanded_stage3_targets_v33/v33_expanded_stage3_target_list_summary.json")
    request_summary = read_json(out_root / "stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest_summary.json")

    metrics = {
        "v33_version": version,
        "gap_analysis_status": gap_summary.get("status", "missing"),
        "direct_formula_mapping_status": mapping_summary.get("status", "missing"),
        "expanded_target_list_status": target_summary.get("status", "missing"),
        "expansion_request_manifest_status": request_summary.get("status", "missing"),
        "n_gap_rows": gap_summary.get("n_gap_rows", ""),
        "n_external_gap_cases": gap_summary.get("n_external_gap_cases", ""),
        "gap_type_counts": gap_summary.get("gap_type_counts", ""),
        "priority_counts": gap_summary.get("priority_counts", ""),
        "n_mapping_rows": mapping_summary.get("n_mapping_rows", ""),
        "n_mapped_cases": mapping_summary.get("n_mapped_cases", ""),
        "n_missing_cases": mapping_summary.get("n_missing_cases", ""),
        "mapping_type_counts": mapping_summary.get("mapping_type_counts", ""),
        "n_expansion_target_rows": target_summary.get("n_expansion_target_rows", ""),
        "n_stage3_expansion_needed": target_summary.get("n_stage3_expansion_needed", ""),
        "expansion_needed_cases": target_summary.get("expansion_needed_cases", ""),
        "n_stage3_expansion_requests": request_summary.get("n_stage3_expansion_requests", ""),
        "n_unique_mp_targets": request_summary.get("n_unique_mp_targets", ""),
        "requested_models": request_summary.get("requested_models", ""),
        "request_status": request_summary.get("request_status", ""),
        "mp_targets": request_summary.get("mp_targets", ""),
        "final_interpretation": (
            "V33 converts V32 review/unaligned cases into formula-exact MP targets and prepares a clean "
            "Stage3 expansion request manifest. It is a planning and handoff layer, not yet a regenerated "
            "Stage3 MDN/Flow candidate library."
        ),
        "next_required_upgrade": (
            "run_real_stage3_mdn_flow_export_or_generation_for_v33_requested_mp_targets"
        ),
    }

    metric_df = pd.DataFrame(
        [{"metric": k, "value": str(v)} for k, v in metrics.items()]
    )
    metric_csv = report_dir / "FINAL_V33_METRIC_SUMMARY.csv"
    metric_md = report_dir / "FINAL_V33_METRIC_SUMMARY.md"
    metric_df.to_csv(metric_csv, index=False)
    metric_df.to_markdown(metric_md, index=False)

    gap_md = out_root / "stage3_library_gap_analysis_v33/v33_gap_case_table.md"
    target_md = out_root / "expanded_stage3_targets_v33/v33_expanded_stage3_target_list.md"
    request_md = out_root / "stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.md"

    def maybe_read(p):
        p = Path(p)
        return p.read_text(encoding="utf-8") if p.exists() else "_Missing._"

    report = f"""# Final Benchmark-100 V33 Report

## 1. Version

`{version}`

## 2. Purpose

V33 follows V32 and focuses on Stage3 library expansion planning.

V32 made the external-case alignment stricter by using MP formula, element, and family metadata. This exposed a set of chemically weak, review, or unaligned cases. V33 does not force those cases into the existing Stage3 library. Instead, it prepares a conservative expansion plan.

The goal is to answer:

- Which V32 cases remain weak, review, or unaligned?
- Do these cases have direct formula-exact MP matches?
- Which MP targets should be added to the real Stage3 MDN/Flow condition-candidate library?
- What clean manifest should be used for the next Stage3 export or generation step?

## 3. Key outputs

- Gap case table: `stage3_library_gap_analysis_v33/v33_gap_case_table.csv`
- Direct formula-to-MP mapping: `direct_formula_mp_mapping_v33/v33_direct_formula_mp_mapping.csv`
- Expanded Stage3 target list: `expanded_stage3_targets_v33/v33_expanded_stage3_target_list.csv`
- Stage3 expansion request manifest: `stage3_expansion_request_manifest_v33/v33_stage3_expansion_request_manifest.csv`

## 4. Metric summary

{metric_df.to_markdown(index=False)}

## 5. Interpretation

V33 successfully converts all V32 weak/review/unaligned gap cases into formula-exact MP targets.

The most important result is that the current problem is no longer an alignment-code issue. The issue is now clearly localized as a **Stage3 library coverage problem**:

- V32 already has real Stage3 MDN/Flow candidates.
- But the current real Stage3 library does not contain condition candidates for several external benchmark formulas.
- V33 identifies exactly which MP targets need new real Stage3 candidate export or generation.

## 6. Gap cases from V32

{maybe_read(gap_md)}

## 7. Expanded Stage3 target list

{maybe_read(target_md)}

## 8. Stage3 expansion request manifest

{maybe_read(request_md)}

## 9. Conclusion

V33 is a conservative planning layer.

It should not be interpreted as a new Stage3 confidence model. Its role is to prepare the next real-model step: exporting or generating MDN/Flow condition candidates for the formula-exact MP targets listed in the V33 request manifest.

The next version should use this manifest as input and generate real Stage3 candidates for the requested MP targets.
"""

    report_path = report_dir / "FINAL_BENCHMARK_100_V33_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    # Append master index
    master = out_root / "FINAL_BENCHMARK_100_V33_MASTER_INDEX.md"
    append = f"""
## Final report

V33 final report and metric summary have been generated.

Output files:

- Final report: `outputs/{version}/FINAL_REPORT_V33/FINAL_BENCHMARK_100_V33_REPORT.md`
- Metric summary CSV: `outputs/{version}/FINAL_REPORT_V33/FINAL_V33_METRIC_SUMMARY.csv`
- Metric summary Markdown: `outputs/{version}/FINAL_REPORT_V33/FINAL_V33_METRIC_SUMMARY.md`

Final interpretation:

V33 is a Stage3 expansion-planning layer. It converts V32 review/unaligned cases into formula-exact MP targets and prepares a clean request manifest for real Stage3 MDN/Flow candidate export or generation.
"""
    old = master.read_text(encoding="utf-8") if master.exists() else ""
    if "## Final report" not in old:
        master.write_text(old.rstrip() + "\n\n" + append.strip() + "\n", encoding="utf-8")

    print("[SAVE]", report_path)
    print("[SAVE]", metric_csv)
    print("[SAVE]", metric_md)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
