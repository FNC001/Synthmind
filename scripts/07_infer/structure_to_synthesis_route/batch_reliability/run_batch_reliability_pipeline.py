#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import yaml
except Exception:
    yaml = None


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Please install pyyaml.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_step(name: str, cmd: list[str], log_dir: Path, controller_dry_run: bool, child_dry_run: bool) -> dict:
    mkdir(log_dir)
    log_file = log_dir / f"{name}.log"

    print()
    print("=" * 60)
    print(f"[STEP] {name}")
    print("[CMD]", " ".join(map(str, cmd)))
    print("[CONTROLLER_EXECUTES_CHILD]", not controller_dry_run)
    print("[CHILD_DRY_RUN]", child_dry_run)
    print("=" * 60)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[TIME] {now()}\n")
        f.write("[CMD] " + " ".join(map(str, cmd)) + "\n")
        f.write(f"[CONTROLLER_EXECUTES_CHILD] {not controller_dry_run}\n")
        f.write(f"[CHILD_DRY_RUN] {child_dry_run}\n\n")
        f.flush()

        if controller_dry_run:
            rc = 0
        else:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            rc = proc.returncode
            f.write(f"\n[RETURN_CODE] {rc}\n")

    status = "planned" if child_dry_run else ("finished" if rc == 0 else "failed")
    return {
        "step": name,
        "status": status,
        "return_code": rc,
        "log_file": str(log_file),
        "command": " ".join(map(str, cmd)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/batch_reliability.yaml")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--auto_recover", action="store_true",
                    help="After audit steps, automatically run recovery + re-audit for failed cases.")
    ap.add_argument("--start_from", default="audit_batch")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    cfg = load_yaml(config_path)

    project_root = Path(cfg["project_root"])
    batch_name = cfg["batch_name"]
    out_root = Path(cfg["reliability_out_root"]) / batch_name
    log_dir = out_root / "logs"
    mkdir(out_root)
    mkdir(log_dir)

    dry_run = bool(cfg.get("dry_run", True))
    if args.execute:
        dry_run = False
    if args.dry_run:
        dry_run = True

    modules = cfg.get("modules", {})

    print("=" * 60)
    print("Batch Reliability Pipeline")
    print("config      =", config_path)
    print("project_root=", project_root)
    print("batch_name  =", batch_name)
    print("out_root    =", out_root)
    print("dry_run     =", dry_run)
    print("start_from  =", args.start_from)
    print("=" * 60)

    steps = [
        ("audit_batch", "01_audit_layer1_outputs.py"),
        ("stage3_feature_source_audit", "02_stage3_feature_source_audit.py"),
        ("stage3_input_extension", "03_build_stage3_input_extension.py"),
        ("recover_missing_structures", "04_recover_missing_structures.py"),
        ("validate_recovered_poscars", "05_validate_recovered_poscars.py"),
        ("regenerate_stage3_npz", "06_regenerate_stage3_npz.py"),
        ("export_stage3_conditions", "07_reexport_stage3_conditions.py"),
        ("normalize_recovered_candidates", "08_normalize_recovered_candidates.py"),
        ("clipping_diagnostic", "09_condition_clipping_diagnostic.py"),
        ("global_clip_export", "10_global_clip_reexport.py"),
        ("merge_recovered_results", "11_merge_recovered_results.py"),
        ("gap_closure_report", "12_build_batch_closure_report.py"),
        ("auto_recover_and_reaudit", "13_auto_recover_and_reaudit.py"),
    ]

    names = [x[0] for x in steps]
    if args.start_from not in names:
        raise ValueError(f"Unknown start_from={args.start_from}. Available: {names}")

    start_idx = names.index(args.start_from)

    results = []

    for name, script_name in steps[start_idx:]:
        if not modules.get(name, True):
            print(f"[SKIP disabled] {name}")
            continue

        # auto_recover_and_reaudit only runs when --auto_recover is passed.
        if name == "auto_recover_and_reaudit" and not args.auto_recover:
            print(f"[SKIP] {name} (use --auto_recover to enable)")
            continue

        script = script_dir / "scripts" / script_name
        if not script.exists():
            print(f"[WARN] missing script: {script}")
            results.append({
                "step": name,
                "status": "missing_script",
                "return_code": 127,
                "log_file": "",
                "command": str(script),
            })
            continue

        cmd = [
            "python",
            str(script),
            "--config",
            str(config_path),
        ]
        if dry_run:
            cmd.append("--dry_run")
        else:
            cmd.append("--execute")

        # We still execute the child script so that audit/manifest/report files are generated.
        # The child script receives --dry_run/--execute and decides whether to run heavy actions.
        res = run_step(name, cmd, log_dir, controller_dry_run=False, child_dry_run=dry_run)
        results.append(res)

        if res["return_code"] != 0:
            print(f"[STOP] step failed: {name}")
            break

    summary_json = out_root / "batch_reliability_run_summary.json"
    summary_md = out_root / "batch_reliability_run_summary.md"

    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Batch Reliability Run Summary", ""]
    md.append(f"- Batch: `{batch_name}`")
    md.append(f"- Config: `{config_path}`")
    md.append("")
    md.append("| step | status | return_code | log_file |")
    md.append("|---|---|---:|---|")
    for r in results:
        md.append(f"| {r['step']} | {r['status']} | {r['return_code']} | `{r['log_file']}` |")
    summary_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    print("[SAVE]", summary_json)
    print("[SAVE]", summary_md)


if __name__ == "__main__":
    main()
