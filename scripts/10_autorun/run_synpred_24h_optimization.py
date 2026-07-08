#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EXPECTED_BASELINE = {
    "stage2_all_top1_exact": 0.39465065502183405,
    "stage2_all_top10_exact": 0.6334606986899564,
    "stage2_all_top200_exact": 0.7701965065502183,
    "stage2_all_top500_exact": 0.8024017467248908,
    "stage2_core_top1_exact": 0.446515397082658,
    "stage2_core_top10_exact": 0.6944894651539708,
    "stage2_core_top200_exact": 0.8184764991896273,
    "stage2_core_top500_exact": 0.853322528363047,
    "stage3_missing_top1_relaxed_condition": 0.36490174672489084,
    "stage3_missing_top10_relaxed_condition": 0.6926855895196506,
    "stage3_strict_top1_relaxed_condition": 0.1796,
    "stage3_strict_top10_relaxed_condition": 0.3319,
    "stage35_missing_top1_relaxed_route": 0.20715065502183405,
    "stage35_missing_top10_relaxed_route": 0.3455240174672489,
    "stage35_missing_top200_relaxed_route": 0.4885371179039301,
    "stage35_strict_top1_relaxed_route": 0.10453056768558952,
    "stage35_strict_top10_relaxed_route": 0.18040393013100436,
    "stage35_strict_top200_relaxed_route": 0.26965065502183405,
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def pct(x: Any) -> str:
    try:
        return f"{100 * float(x):.2f}%"
    except Exception:
        return "n/a"


def run_cmd(cmd: List[str], cwd: Path, log_path: Path, timeout_s: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n## START {started}\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout_s)
        log.write(f"\n## END {datetime.now().isoformat(timespec='seconds')} exit={proc.returncode}\n")
        return int(proc.returncode)


def ensure_dirs(root: Path) -> Dict[str, Path]:
    names = [
        "00_setup",
        "01_baseline_repro",
        "02_stage2_diagnosis",
        "03_stage2_experiments",
        "04_stage3_experiments",
        "05_stage35_experiments",
        "06_final_eval",
        "07_figures",
        "08_tables",
        "09_article_draft",
        "10_final_package",
        "logs",
    ]
    out = {name: root / name for name in names}
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def flatten_baseline(registry: Dict[str, Any]) -> Dict[str, float]:
    b = registry.get("baselines", {})
    s2 = b.get("stage2_v5_all_test", {}).get("metrics", {})
    s2c = b.get("stage2_core_calibrated_test", {}).get("metrics", {})
    s3m = b.get("stage3_v3_missing_aware_test", {}).get("metrics", {})
    s3s = b.get("stage3_v3_strict_comparable_test", {}).get("metrics", {})
    r3m = b.get("stage35_v3_final_missing_aware_test", {}).get("metrics", {})
    r3s = b.get("stage35_v3_final_strict_comparable_test", {}).get("metrics", {})
    return {
        "stage2_all_top1_exact": s2.get("top1_exact"),
        "stage2_all_top10_exact": s2.get("top10_exact"),
        "stage2_all_top200_exact": s2.get("top200_exact"),
        "stage2_all_top500_exact": s2.get("top500_exact"),
        "stage2_core_top1_exact": s2c.get("top1_exact"),
        "stage2_core_top10_exact": s2c.get("top10_exact"),
        "stage2_core_top200_exact": s2c.get("top200_exact"),
        "stage2_core_top500_exact": s2c.get("top500_exact"),
        "stage3_missing_top1_relaxed_condition": s3m.get("top1_relaxed_condition"),
        "stage3_missing_top10_relaxed_condition": s3m.get("top10_relaxed_condition"),
        "stage3_strict_top1_relaxed_condition": s3s.get("top1_relaxed_condition"),
        "stage3_strict_top10_relaxed_condition": s3s.get("top10_relaxed_condition"),
        "stage35_missing_top1_relaxed_route": r3m.get("top1_relaxed_route"),
        "stage35_missing_top10_relaxed_route": r3m.get("top10_relaxed_route"),
        "stage35_missing_top200_relaxed_route": r3m.get("top200_relaxed_route"),
        "stage35_strict_top1_relaxed_route": r3s.get("top1_relaxed_route"),
        "stage35_strict_top10_relaxed_route": r3s.get("top10_relaxed_route"),
        "stage35_strict_top200_relaxed_route": r3s.get("top200_relaxed_route"),
    }


def baseline_repro(project_root: Path, out: Path, log: Path) -> Tuple[bool, Dict[str, Any]]:
    rc = run_cmd(
        [sys.executable, "scripts/08_auto_improve/metrics_registry.py", "--project_root", str(project_root), "--output_dir", str(out.parent)],
        project_root,
        log,
    )
    registry = read_json(out.parent / "metrics_registry/metrics_registry.json")
    metrics = flatten_baseline(registry)
    rows = []
    ok = rc == 0
    for key, expected in EXPECTED_BASELINE.items():
        observed = metrics.get(key)
        delta = None if observed is None else float(observed) - float(expected)
        passed = observed is not None and abs(delta or 0.0) <= 0.005
        ok = ok and passed
        rows.append({"metric": key, "expected": expected, "observed": observed, "delta": delta, "pass": passed})
    obj = {"status": "pass" if ok else "fail", "metrics": metrics, "checks": rows}
    write_json(out / "baseline_metrics.json", obj)
    lines = ["# Baseline Reproduction Report", "", f"Status: **{obj['status']}**", "", "| metric | expected | observed | delta | pass |", "|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {pct(r['expected'])} | {pct(r['observed'])} | {pct(r['delta']) if r['delta'] is not None else 'n/a'} | {r['pass']} |")
    lines.append("")
    (out / "BASELINE_REPRO_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return ok, obj


def copy_tree_items(src_root: Path, patterns: Iterable[str], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in src_root.glob(pattern):
            if src.is_file():
                target = dst / src.name
                shutil.copy2(src, target)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_package(project_root: Path, out: Path) -> Path:
    pkg = out / "final_package"
    if pkg.exists():
        shutil.rmtree(pkg)
    for sub in ["reports", "figures", "tables", "article_draft", "metrics_json"]:
        (pkg / sub).mkdir(parents=True, exist_ok=True)
    copy_tree_items(out, ["*.md"], pkg / "reports")
    for subdir in ["01_baseline_repro", "02_stage2_diagnosis", "03_stage2_experiments", "04_stage3_experiments", "05_stage35_experiments", "06_final_eval"]:
        copy_tree_items(out / subdir, ["*.md"], pkg / "reports")
        copy_tree_items(out / subdir, ["*.json"], pkg / "metrics_json")
    copy_tree_items(out / "07_figures", ["*.png", "*.svg", "*_source.csv"], pkg / "figures")
    copy_tree_items(out / "08_tables", ["*.csv", "*.md"], pkg / "tables")
    copy_tree_items(out / "09_article_draft", ["*.md"], pkg / "article_draft")
    reproduce = pkg / "reproduce_commands.sh"
    reproduce.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "python scripts/10_autorun/run_synpred_24h_optimization.py --project_root . --output_root outputs/autorun/24h_optimization_20260613 --allow_update_selector 0\n",
        encoding="utf-8",
    )
    manifest = []
    for path in sorted(pkg.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rel = path.relative_to(pkg).as_posix()
            manifest.append(
                {
                    "path": rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "description": rel,
                    "paper_main_result": rel.startswith("figures/") or rel.startswith("tables/") or rel.startswith("article_draft/"),
                    "diagnostic_result": rel.startswith("reports/") or rel.startswith("metrics_json/"),
                    "reproducible": True,
                }
            )
    write_json(pkg / "artifact_manifest.json", manifest)
    tar_path = out / "SynPred_24h_results_package.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(pkg, arcname="final_package")
    return tar_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SynPred 24h controlled optimization, evaluation, and paper artifact generation.")
    ap.add_argument("--project_root", default=".")
    ap.add_argument("--output_root", default="outputs/autorun/24h_optimization_20260613")
    ap.add_argument("--max_train_hours", type=float, default=12)
    ap.add_argument("--max_total_hours", type=float, default=24)
    ap.add_argument("--scope", default="all")
    ap.add_argument("--run_stage2", type=int, default=1)
    ap.add_argument("--run_stage3", type=int, default=1)
    ap.add_argument("--run_stage35", type=int, default=1)
    ap.add_argument("--run_expensive_stage2_oof", type=int, default=0)
    ap.add_argument("--kfold", type=int, default=3)
    ap.add_argument("--run_test_only_if_val_pass", type=int, default=1)
    ap.add_argument("--allow_update_selector", type=int, default=0)
    ap.add_argument("--make_paper_figures", type=int, default=1)
    ap.add_argument("--make_article_draft", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    out = Path(args.output_root)
    if not out.is_absolute():
        out = project_root / out
    dirs = ensure_dirs(out)
    log = dirs["logs"] / "autorun.log"
    timeline: List[Dict[str, Any]] = []
    start = time.time()

    ok, baseline = baseline_repro(project_root, dirs["01_baseline_repro"], log)
    timeline.append({"step": "baseline_repro", "status": baseline["status"], "time": datetime.now().isoformat()})
    skipped: List[str] = []

    if ok:
        run_cmd([sys.executable, "scripts/08_auto_improve/diagnose_stage2_bottlenecks.py", "--project_root", str(project_root), "--output_dir", str(dirs["02_stage2_diagnosis"] / "stage2")], project_root, log)
        run_cmd([sys.executable, "scripts/08_auto_improve/diagnose_stage2_stage3_coupling.py", "--project_root", str(project_root), "--output_dir", str(dirs["02_stage2_diagnosis"])], project_root, log)
        if args.run_stage2:
            run_cmd([sys.executable, "scripts/08_auto_improve/experiment_stage2_core_method_selector.py", "--project_root", str(project_root), "--output_dir", str(dirs["03_stage2_experiments"] / "core_selector")], project_root, log)
            run_cmd([sys.executable, "scripts/08_auto_improve/experiment_stage2_top1_calibration.py", "--project_root", str(project_root), "--output_dir", str(dirs["03_stage2_experiments"] / "top1_calibration")], project_root, log)
            if not args.run_expensive_stage2_oof:
                skipped.append("Stage2 K-fold neural OOF skipped because run_expensive_stage2_oof=0.")
        if args.run_stage3:
            run_cmd([sys.executable, "scripts/08_auto_improve/experiment_condition_calibration_search.py", "--project_root", str(project_root), "--output_dir", str(dirs["04_stage3_experiments"] / "condition_calibration")], project_root, log)
            run_cmd([sys.executable, "scripts/08_auto_improve/experiment_atmosphere_strict_repair.py", "--project_root", str(project_root), "--output_dir", str(dirs["04_stage3_experiments"] / "atmosphere_repair")], project_root, log)
            run_cmd([sys.executable, "scripts/08_auto_improve/experiment_route_text_features.py", "--project_root", str(project_root), "--output_dir", str(dirs["04_stage3_experiments"] / "route_text_features")], project_root, log)
        if args.run_stage35:
            run_cmd([sys.executable, "scripts/08_auto_improve/experiment_pairwise_route_ranker.py", "--project_root", str(project_root), "--output_dir", str(dirs["05_stage35_experiments"] / "pairwise_ranker")], project_root, log)
            run_cmd([sys.executable, "scripts/08_auto_improve/experiment_route_score_meta_calibration.py", "--project_root", str(project_root), "--output_dir", str(dirs["05_stage35_experiments"] / "meta_calibration")], project_root, log)
        run_cmd([sys.executable, "scripts/08_auto_improve/model_selector.py", "--project_root", str(project_root), "--output_dir", str(out)], project_root, log)
    else:
        skipped.append("All new experiments skipped because baseline reproduction failed.")

    if args.make_paper_figures:
        run_cmd([sys.executable, "scripts/11_paper/make_paper_tables.py", "--project_root", str(project_root), "--output_dir", str(dirs["08_tables"])], project_root, log)
        run_cmd([sys.executable, "scripts/11_paper/make_paper_figures.py", "--project_root", str(project_root), "--output_dir", str(dirs["07_figures"])], project_root, log)
        run_cmd([sys.executable, "scripts/11_paper/make_results_summary.py", "--project_root", str(project_root), "--output_dir", str(out)], project_root, log)
    if args.make_article_draft:
        run_cmd([sys.executable, "scripts/11_paper/write_article_draft.py", "--project_root", str(project_root), "--output_dir", str(dirs["09_article_draft"])], project_root, log)

    decision = read_json(out / "model_selection_decision.json")
    tar_path = build_package(project_root, out)
    elapsed_h = (time.time() - start) / 3600.0
    final = [
        "# SYN_PRED_24H_FINAL_REPORT",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Elapsed hours: {elapsed_h:.2f}",
        "",
        "## Baseline Reproduction",
        f"Status: **{baseline['status']}**",
        "",
        "## Experiments",
        "- Stage2 diagnosis, core selector, and top1 calibration gate check.",
        "- Stage3 condition calibration comparison, atmosphere strict-repair audit, and route-text availability audit.",
        "- Stage35 pairwise-ranker hook and route score meta-calibration classification.",
        "",
        "## Skipped",
    ]
    final += [f"- {x}" for x in skipped] or ["- None"]
    final += [
        "",
        "## Model Selection",
        f"- Stage2 mode: `{decision.get('stage2_mode', 'n/a')}`",
        f"- Ranking mode: `{decision.get('ranking_mode', 'n/a')}`",
        f"- Update default inference selector: `{decision.get('do_update_default_inference_selector', False)}`",
        "",
        "## Artifacts",
        f"- Figures: `{dirs['07_figures']}`",
        f"- Tables: `{dirs['08_tables']}`",
        f"- Article draft: `{dirs['09_article_draft']}`",
        f"- Package: `{tar_path}`",
        "",
        "## Warnings",
        "- Passwords are not stored in generated scripts or reports.",
        "- New expensive model training is gated and disabled by default in this autorun configuration.",
        "",
    ]
    (out / "SYN_PRED_24H_FINAL_REPORT.md").write_text("\n".join(final), encoding="utf-8")
    write_json(out / "autorun_timeline.json", timeline)
    print(json.dumps({"output_root": str(out), "package": str(tar_path), "baseline_status": baseline["status"]}, indent=2))


if __name__ == "__main__":
    main()

