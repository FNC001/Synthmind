from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from synpred.research.attribution import attribute_candidates, summarize_attribution
from synpred.research.candidate_pool import stable_file_hash
from synpred.research.metrics.registry import MetricRegistry
from synpred.research.splits import choose_group_key, fingerprint_rows


LEGACY_BASELINES = {
    "rsp_legacy_stage2_v5_all": {
        "precursor_exact@1": 0.3947,
        "precursor_exact@10": 0.6335,
        "precursor_exact@200": 0.7702,
        "precursor_exact@500": 0.8024,
    },
    "cdg_legacy_stage3_v3_missing_aware": {
        "condition_ref_match@1": 0.3649,
        "condition_ref_match@10": 0.6927,
    },
    "cdg_legacy_stage3_v3_strict_comparable": {
        "condition_ref_match@1": 0.1796,
        "condition_ref_match@10": 0.3319,
    },
    "grv_legacy_stage35_v3_missing_aware": {
        "route_ref_match@1": 0.2072,
        "route_ref_match@10": 0.3455,
    },
    "grv_candidate_v12": {
        "missing_aware_route_ref_match@1": 0.2473,
        "strict_comparable_route_ref_match@1": 0.1834,
        "operational_usable_route@1": 0.3231,
    },
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(root.glob(pattern))
    return sorted({p for p in out if p.is_file()})


def audit_repo(project_root: Path) -> dict[str, Any]:
    scripts = discover_files(project_root / "scripts", ("**/*.py", "**/*.sh"))
    artifacts = discover_files(project_root, ("outputs/evaluation.local_bak_20260612/**/*.json", "outputs/evaluation.local_bak_20260612/**/*.csv", "runs.local_bak_20260612/**/*"))
    selectors = discover_files(project_root, ("scripts/08_auto_improve/model_selector.py", "outputs/autorun/**/model_selection_decision.json"))
    report_lines = [
        "# Repository Audit v1",
        "",
        "## Confirmed Facts",
        f"- Project root: `{project_root}`",
        f"- Git status: unavailable here (`.git` not found) unless this directory is nested inside a copied workspace.",
        f"- Script files discovered: {len(scripts)}",
        f"- Legacy artifact files discovered under `outputs/evaluation.local_bak_20260612` and `runs.local_bak_20260612`: {len(artifacts)}",
        "- Public functional module names are now RSP/CDG/GRV; legacy Stage2/Stage3/Stage35 names are compatibility aliases.",
        "",
        "## Main Entry Points",
        "- RSP legacy training: `scripts/04_train/stage2/*`",
        "- CDG legacy training: `scripts/04_train/stage3/*`",
        "- GRV legacy training: `scripts/04_train/stage35/*` and `scripts/10_autorun/run_autodl_gpu_training_queue.py`",
        "- Legacy route candidate/evaluator logic: `scripts/06_eval/build_stage35_route_candidates_v4.py`",
        "- Selector compatibility logic: `scripts/08_auto_improve/model_selector.py`",
        "",
        "## Risks",
        "- This local checkout is not a git repository, so commit and dirty-tree provenance cannot be recorded locally.",
        "- Historical v3/v12 comparison cannot be claimed as pure ranking gain until a shared frozen candidate pool is verified.",
        "- Legacy missing-aware route metrics can hide missing fields; new protocol separates complete_case, available_field, and operational_validity.",
        "",
        "## Blockers Before New Training",
        "- Freeze and hash a shared candidate pool for v3/v12/new GRV.",
        "- Reproduce legacy metrics from frozen artifacts using the exact legacy evaluator.",
        "- Confirm chemistry-disjoint split group key availability.",
    ]
    write_text(project_root / "reports/repo_audit_v1.md", "\n".join(report_lines))

    inv_path = project_root / "reports/artifact_inventory_v1.csv"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    with inv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256", "category"])
        writer.writeheader()
        for p in artifacts[:5000]:
            category = "checkpoint" if p.suffix in {".joblib", ".pt", ".pth"} else ("metrics" if p.suffix == ".json" else "table")
            try:
                digest = stable_file_hash(p)
                size = p.stat().st_size
            except Exception:
                digest, size = "", 0
            writer.writerow({"path": str(p.relative_to(project_root)), "size_bytes": size, "sha256": digest, "category": category})
    return {"scripts": len(scripts), "artifacts": len(artifacts), "selectors": [str(p) for p in selectors]}


def audit_metrics(project_root: Path) -> None:
    registry = MetricRegistry.load(project_root / "research/specs/metric_registry_v1.yaml")
    errors = registry.validate()
    lines = [
        "# Legacy Metric Definition Audit v1",
        "",
        "## Confirmed Definitions From Code",
        "- Legacy relaxed route hit in `build_stage35_route_candidates_v4.py`: precursor exact AND relaxed condition hit.",
        "- Legacy usable relaxed route hit: precursor Jaccard >= 0.5 AND relaxed condition hit; this is operational, not reference-match accuracy.",
        "- Strict-comparable condition relaxed hit uses temperature error <= 200 C, time error <= 48 h, and exact known atmosphere match.",
        "- Strict-comparable condition strict hit uses temperature error <= 100 C, time error <= 24 h, and exact known atmosphere match.",
        "",
        "## Registry Validation",
        f"- Registered metrics: {len(registry.ids())}",
        f"- Validation errors: {len(errors)}",
    ]
    lines.extend([f"- {e}" for e in errors] or ["- none"])
    write_text(project_root / "reports/legacy_metric_definition_audit_v1.md", "\n".join(lines))


def audit_leakage_and_split(project_root: Path) -> dict[str, Any]:
    candidates = discover_files(project_root / "data", ("**/train.csv", "**/val.csv", "**/test.csv", "**/*train*.jsonl", "**/*val*.jsonl", "**/*test*.jsonl"))
    chosen_file = next((p for p in candidates if p.suffix == ".csv"), None)
    group_key = None
    manifest: dict[str, Any] = {
        "split_version": "chemistry_disjoint_v1",
        "status": "not_created",
        "reason": "No reliable group split was written by this audit-only pass.",
        "group_key": None,
        "source_file": str(chosen_file) if chosen_file else None,
    }
    if chosen_file:
        df = pd.read_csv(chosen_file, nrows=2000)
        group_key = choose_group_key(list(df.columns))
        manifest["group_key"] = group_key
        manifest["source_columns_sample"] = list(df.columns)[:80]
        manifest["source_fingerprint_sample"] = fingerprint_rows(df, list(df.columns)[:20])
        if group_key:
            manifest["status"] = "field_available_manifest_only"
            manifest["reason"] = "A candidate chemistry grouping key exists, but no split is created until full distribution and leakage checks are run."
    write_json(project_root / "research/specs/split_manifest_v1.json", manifest)
    lines = [
        "# Data Leakage Audit v1",
        "",
        f"- Candidate split/data files scanned: {len(candidates)}",
        f"- Chemistry-disjoint manifest status: {manifest['status']}",
        f"- Candidate group key: {group_key or 'not found'}",
        "",
        "## Leakage Risks",
        "- Historical method-stratified/IID split remains useful for continuity but is not chemistry-disjoint.",
        "- No new chemistry-disjoint split was written because this pass does not yet perform full group overlap and method divergence balancing.",
    ]
    write_text(project_root / "reports/data_leakage_audit_v1.md", "\n".join(lines))
    write_text(
        project_root / "reports/chemistry_disjoint_split_report_v1.md",
        "\n".join([
            "# Chemistry-Disjoint Split Report v1",
            "",
            f"Status: `{manifest['status']}`",
            f"Selected group key candidate: `{group_key or 'none'}`",
            "",
            "No legacy split was overwritten. A full split builder should use the manifest and write to `data/splits/chemistry_disjoint_v1.*` only after distribution checks pass.",
        ]),
    )
    return manifest


def collect_legacy_reproduction(project_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for model_id, metrics in LEGACY_BASELINES.items():
        for metric, documented in metrics.items():
            rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "documented_value": documented,
                    "actual_value": "",
                    "delta": "",
                    "status": "documented_not_rerun",
                    "reason": "P0/P1 audit pass records baselines but does not touch test for new model selection.",
                }
            )
    out = project_root / "reports/legacy_baseline_reproduction_v1.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    write_text(
        project_root / "reports/legacy_baseline_reproduction_v1.md",
        "\n".join([
            "# Legacy Baseline Reproduction v1",
            "",
            "Status: documented, not rerun in this pass.",
            "",
            "Training is paused until exact frozen artifact/evaluator/candidate-pool reproduction is established. The CSV records documented values and marks actual rerun fields blank.",
        ]),
    )
    write_json(
        project_root / "artifacts/reproduction_manifest_v1.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": None,
            "git_status": "unavailable_not_a_git_repository",
            "python": sys.version,
            "platform": platform.platform(),
            "selector_update_status": "unchanged",
            "test_access_status": "not_used_for_new_selection",
        },
    )


def candidate_pool_comparability(project_root: Path) -> dict[str, Any]:
    paths = {
        "v3_final_val": project_root / "outputs/evaluation.local_bak_20260612/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv",
        "v4_val": project_root / "outputs/evaluation.local_bak_20260612/stage35_route_candidates_v4_20260612/val_route_candidates.csv",
        "v3_final_test_presence": project_root / "outputs/evaluation.local_bak_20260612/stage35_route_candidates_v3_final_20260612/test_route_candidates.csv",
        "v4_test_presence": project_root / "outputs/evaluation.local_bak_20260612/stage35_route_candidates_v4_20260612/test_route_candidates.csv",
    }
    rows = []
    for name, path in paths.items():
        if path.exists():
            df = pd.read_csv(path, nrows=5000)
            rows.append({"pool": name, "path": str(path.relative_to(project_root)), "exists": True, "columns": list(df.columns), "sha256": stable_file_hash(path), "sample_id_preview_n": int(df["sample_id"].nunique()) if "sample_id" in df.columns else None})
        else:
            rows.append({"pool": name, "path": str(path.relative_to(project_root)), "exists": False, "columns": [], "sha256": "", "sample_id_preview_n": None})
    comparable = all(r["exists"] for r in rows) and rows[0]["sha256"] == rows[1]["sha256"]
    lines = [
        "# Candidate Pool Comparability v1",
        "",
        f"Direct same-file/hash comparability: `{comparable}`",
        "",
        "Historical v3/v4/v12-like runs should not be described as pure ranking improvements unless the shared candidate table is verified or re-scored.",
        "",
        "| pool | exists | preview sample ids | sha256 prefix |",
        "|---|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['pool']} | {r['exists']} | {r['sample_id_preview_n']} | {str(r['sha256'])[:12]} |")
    write_text(project_root / "reports/candidate_pool_comparability_v1.md", "\n".join(lines))
    return {"same_hash": comparable, "pools": rows}


def run_attribution(project_root: Path) -> dict[str, Any]:
    src = project_root / "outputs/evaluation.local_bak_20260612/stage35_route_candidates_v3_final_20260612/val_route_candidates.csv"
    if not src.exists():
        write_text(project_root / "reports/error_attribution_v1.md", "# Error Attribution v1\n\nBlocked: frozen route candidate table not found.")
        pd.DataFrame(columns=["attribution", "count", "fraction"]).to_csv(project_root / "reports/error_attribution_summary_v1.csv", index=False)
        return {"status": "blocked", "reason": str(src)}
    usecols = None
    df = pd.read_csv(src, nrows=250000, usecols=usecols)
    rank_col = "route_rank_calibrated_v3_final" if "route_rank_calibrated_v3_final" in df.columns else ("route_rank_raw" if "route_rank_raw" in df.columns else "final_rank")
    if rank_col not in df.columns:
        df[rank_col] = df.groupby("sample_id").cumcount() + 1
    attrs = attribute_candidates(df.rename(columns={rank_col: "final_rank"}))
    summary = summarize_attribution(attrs)
    attrs.to_csv(project_root / "reports/error_attribution_v1.csv", index=False)
    try:
        attrs.to_parquet(project_root / "reports/error_attribution_v1.parquet", index=False)
    except Exception:
        (project_root / "reports/error_attribution_v1.parquet").write_text("parquet unavailable; see error_attribution_v1.csv\n", encoding="utf-8")
    summary.to_csv(project_root / "reports/error_attribution_summary_v1.csv", index=False)
    table_lines = ["| attribution | count | fraction |", "|---|---:|---:|"]
    for _, row in summary.iterrows():
        table_lines.append(f"| {row['attribution']} | {int(row['count'])} | {float(row['fraction']):.4f} |")
    write_text(
        project_root / "reports/error_attribution_v1.md",
        "\n".join([
            "# Error Attribution v1",
            "",
            f"Source: `{src.relative_to(project_root)}`",
            f"Rank column used: `{rank_col}`",
            "",
            *table_lines,
        ]),
    )
    return {"status": "ok", "source": str(src), "rank_col": rank_col, "summary": summary.to_dict(orient="records")}


def write_same_pool_baselines(project_root: Path) -> None:
    rows = [
        {"model_id": "frequency_baseline", "run_id": "not_run_p0", "task": "e2e_route", "split": "iid", "protocol": "complete_case", "metric_id": "e2e_route.iid.complete_case.route_ref_match@1", "value": "", "sample_count": "", "candidate_budget": "candidate_budget_v1", "canonicalization_version": "canonicalization_v1", "seed": "", "checkpoint": "", "candidate_pool_hash": ""},
        {"model_id": "legacy_grv_v3", "run_id": "documented", "task": "e2e_route", "split": "iid", "protocol": "legacy_missing_aware", "metric_id": "legacy.route_ref_match@1", "value": 0.2072, "sample_count": "", "candidate_budget": "legacy", "canonicalization_version": "legacy", "seed": "", "checkpoint": "runs.local_bak_20260612/stage35/route_reranker_v3_final_20260612", "candidate_pool_hash": ""},
        {"model_id": "candidate_grv_v12", "run_id": "documented", "task": "e2e_route", "split": "iid", "protocol": "legacy_missing_aware", "metric_id": "legacy.route_ref_match@1", "value": 0.2473, "sample_count": "", "candidate_budget": "legacy", "canonicalization_version": "legacy", "seed": "", "checkpoint": "", "candidate_pool_hash": ""},
    ]
    pd.DataFrame(rows).to_csv(project_root / "reports/same_pool_baseline_results_v1.csv", index=False)


def select_branch(project_root: Path, attribution_result: dict[str, Any]) -> str:
    branch = "GRV"
    reason = "Prior documented v12 mainly improves route ranking metrics, while recent RSP/CDG auxiliary sweeps are plateauing; full branch decision remains provisional until same-pool attribution is complete."
    if attribution_result.get("status") == "ok":
        rows = attribution_result.get("summary", [])
        fixable = {r["attribution"]: r["fraction"] for r in rows if r["attribution"] in {"skeleton_miss", "condition_miss", "ranking_miss", "normalization_or_reference_issue"}}
        if fixable:
            best = max(fixable, key=fixable.get)
            branch = {"skeleton_miss": "RSP", "condition_miss": "CDG", "ranking_miss": "GRV", "normalization_or_reference_issue": "DATA_EVALUATOR"}[best]
            reason = f"largest measured fixable attribution category is {best} ({fixable[best]:.3f})."
    write_text(
        project_root / "reports/selected_research_branch_v1.md",
        "\n".join([
            "# Selected Research Branch v1",
            "",
            f"Selected branch: `{branch}`",
            "",
            f"Reason: {reason}",
            "",
            "Default inference selector remains unchanged.",
        ]),
    )
    return branch


def final_summary(project_root: Path, branch: str) -> None:
    write_text(
        project_root / "reports/synpred_next_research_summary_v1.md",
        "\n".join([
            "# SynPred Next Research Summary v1",
            "",
            "## Direct Answers",
            "1. Current largest error source: provisional, see `reports/error_attribution_summary_v1.csv`; full conclusion requires frozen shared candidate pool.",
            "2. v12 uplift source: not yet proven to be pure ranking gain because candidate-pool comparability is not verified.",
            f"3. Priority module: `{branch}` provisional from current attribution/context.",
            "4. Ranking loss when reference route is in pool: see ranking_miss fraction and first_hit_rank fields in attribution output.",
            "5. Chemistry-disjoint performance drop: not measured yet; split manifest is audit-only.",
            "6. Protocol confusion: legacy missing-aware route metrics and usable metrics mix reference-match with operational validity.",
            "7. Reproducible historical metrics: documented in `legacy_baseline_reproduction_v1.csv`; exact rerun is blocked pending frozen artifact checks.",
            "8. True end-to-end lift this round: no new model lift claimed.",
            "9. Fixed-budget lift: not claimed until same-pool baselines run.",
            "10. Default selector: unchanged.",
        ]),
    )


def write_manifest(project_root: Path) -> None:
    files = [
        "reports/repo_audit_v1.md",
        "research/specs/task_spec_v1.yaml",
        "research/specs/naming_map_v1.yaml",
        "research/specs/metric_registry_v1.yaml",
        "research/specs/canonicalization_v1.yaml",
        "research/specs/split_manifest_v1.json",
        "research/specs/candidate_budget_v1.yaml",
        "reports/legacy_baseline_reproduction_v1.md",
        "reports/candidate_pool_comparability_v1.md",
        "reports/error_attribution_v1.parquet",
        "reports/error_attribution_summary_v1.csv",
        "reports/error_attribution_v1.md",
        "reports/same_pool_baseline_results_v1.csv",
        "reports/selected_research_branch_v1.md",
        "reports/synpred_next_research_summary_v1.md",
    ]
    rows = []
    for rel in files:
        p = project_root / rel
        rows.append({"path": rel, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0, "sha256": stable_file_hash(p) if p.exists() else ""})
    write_json(project_root / "artifacts/artifact_manifest_v1.json", {"created_at": datetime.now(timezone.utc).isoformat(), "files": rows})


def run(project_root: Path, phases: set[str]) -> None:
    if phases & {"audit", "freeze", "all"}:
        audit_repo(project_root)
        audit_metrics(project_root)
        audit_leakage_and_split(project_root)
    if phases & {"reproduce", "all"}:
        collect_legacy_reproduction(project_root)
        candidate_pool_comparability(project_root)
    attr = {"status": "not_run"}
    if phases & {"attribute", "all"}:
        attr = run_attribution(project_root)
        write_same_pool_baselines(project_root)
    branch = "GRV"
    if phases & {"baseline", "develop", "all"}:
        branch = select_branch(project_root, attr)
    final_summary(project_root, branch)
    write_manifest(project_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the next RSP/CDG/GRV research plan without modifying production selectors.")
    parser.add_argument("--config", default="research/configs/next_research_v1.yaml")
    parser.add_argument("--phase", default="audit,freeze,reproduce,attribute,baseline")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-budget", type=int, default=200)
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--allow-test-eval", action="store_true", default=False)
    parser.add_argument("--allow-selector-update", action="store_true", default=False)
    args = parser.parse_args()

    if args.allow_selector_update:
        raise SystemExit("--allow-selector-update is disabled for this research plan.")
    if args.dry_run:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return 0
    phases = {x.strip() for x in args.phase.split(",") if x.strip()}
    run(Path.cwd(), phases)
    print("Wrote RSP/CDG/GRV research reports and manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
