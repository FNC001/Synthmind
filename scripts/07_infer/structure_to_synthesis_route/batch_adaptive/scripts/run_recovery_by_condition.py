#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], log_file: Path, dry_run: bool = True) -> int:
    mkdir(log_file.parent)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"[TIME] {now()}\n")
        f.write("[CMD] " + " ".join(map(str, cmd)) + "\n")
        f.write(f"[DRY_RUN] {dry_run}\n\n")
        f.flush()

        if dry_run:
            return 0

        proc = subprocess.run(
            list(map(str, cmd)),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(f"\n[RETURN_CODE] {proc.returncode}\n")
        return proc.returncode


def write_json(path: Path, obj: dict) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def handle_pipeline_failed(project_root: Path, batch_name: str, row: dict, out_dir: Path) -> dict:
    case_id = row["case_id"]
    case_dir = out_dir / "pipeline_failed" / case_id
    mkdir(case_dir)

    pipeline_log = str(row.get("pipeline_log", ""))

    result = {
        "case_id": case_id,
        "recovery_module": "inspect_pipeline_log",
        "status": "manual_debug_required",
        "reason": "Layer-1 pipeline failed. This usually indicates path/script/checkpoint/input errors.",
        "pipeline_log": pipeline_log,
        "suggested_next_steps": [
            "Open the pipeline log.",
            "Check POSCAR validity.",
            "Check missing scripts/checkpoints.",
            "After fixing the upstream issue, rerun adaptive batch for this case with --force.",
        ],
    }

    write_json(case_dir / "recovery_result.json", result)

    md = []
    md.append(f"# Pipeline Failed: {case_id}")
    md.append("")
    md.append(f"- Pipeline log: `{pipeline_log}`")
    md.append("")
    md.append("## Suggested next steps")
    md.append("")
    for x in result["suggested_next_steps"]:
        md.append(f"- {x}")
    md.append("")
    (case_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    return result


def handle_stage2_recovery(project_root: Path, batch_name: str, row: dict, out_dir: Path, dry_run: bool) -> dict:
    """
    Stage2 没有候选时的处理。
    第一版不直接改 pipeline 参数，而是生成可执行 TODO 和任务目录。
    """
    case_id = row["case_id"]
    case_dir = out_dir / "stage2_recovery" / case_id
    mkdir(case_dir)

    # 这里先生成策略说明。后面如果你确定 pipeline 中 sample_stage2_gflownet 的参数位置，
    # 可以把 echo 替换成真实 run_pipeline.py --start_from sample_stage2_gflownet 的调用。
    commands = [
        f'echo "[Stage2 recovery] {case_id}"',
        f'echo "Suggested: increase n_samples, top_k; enable fallback, retrieval, ExtraTrees baseline."',
        f'echo "Then rerun from sample_stage2_gflownet or summarize_stage2 for infer_name={case_id}."',
    ]

    sh = case_dir / "run_stage2_recovery.sh"
    sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n", encoding="utf-8")
    sh.chmod(0o755)

    rc = run_cmd(["bash", str(sh)], case_dir / "stage2_recovery.log", dry_run=dry_run)

    result = {
        "case_id": case_id,
        "recovery_module": "stage2_recovery",
        "status": "planned" if dry_run else ("finished" if rc == 0 else "failed"),
        "return_code": rc,
        "recovery_script": str(sh),
        "recovery_log": str(case_dir / "stage2_recovery.log"),
        "reason": "No Stage2 precursor candidates.",
        "recommended_parameters": {
            "n_samples": "increase, e.g. 200 -> 500/1000",
            "top_k": "increase, e.g. 50 -> 100/200",
            "fallback": "enable composition fallback",
            "retrieval": "enable retrieval_npz candidates",
            "baseline": "enable ExtraTrees baseline",
        },
    }
    write_json(case_dir / "recovery_result.json", result)
    return result


def handle_stage3_gap_recovery(project_root: Path, batch_name: str, row: dict, out_dir: Path, dry_run: bool) -> dict:
    """
    Stage2 有候选但 Stage3 没条件。
    """
    case_id = row["case_id"]
    case_dir = out_dir / "stage3_gap_recovery" / case_id
    mkdir(case_dir)

    commands = [
        f'echo "[Stage3 gap recovery] {case_id}"',
        f'echo "Step 1: audit Stage3 feature source."',
        f'echo "Step 2: if features exist, rebuild conditioned table."',
        f'echo "Step 3: if features missing but POSCAR exists, regenerate structdesc/embedding/stage3 hybrid."',
        f'echo "Step 4: rerun Stage3 condition export."',
        f'echo "Step 5: rerun route finalizer."',
    ]

    sh = case_dir / "run_stage3_gap_recovery.sh"
    sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n", encoding="utf-8")
    sh.chmod(0o755)

    rc = run_cmd(["bash", str(sh)], case_dir / "stage3_gap_recovery.log", dry_run=dry_run)

    result = {
        "case_id": case_id,
        "recovery_module": "stage3_gap_recovery",
        "status": "planned" if dry_run else ("finished" if rc == 0 else "failed"),
        "return_code": rc,
        "recovery_script": str(sh),
        "recovery_log": str(case_dir / "stage3_gap_recovery.log"),
        "reason": "Stage2 candidates exist, but Stage3 conditions are missing.",
        "planned_steps": [
            "stage3_feature_source_audit",
            "stage3_input_feature_extension_or_regeneration",
            "stage3_condition_export",
            "route_finalizer",
        ],
    }
    write_json(case_dir / "recovery_result.json", result)
    return result


def handle_condition_reexport(project_root: Path, batch_name: str, row: dict, out_dir: Path, dry_run: bool) -> dict:
    """
    Stage3 有条件但 condition support 太低。
    轻量处理：建议从 run_stage3_flow 重新导出，必要时做 clipping diagnostic。
    """
    case_id = row["case_id"]
    case_dir = out_dir / "condition_reexport" / case_id
    mkdir(case_dir)

    pipeline_dir = project_root / "scripts/07_infer/structure_to_synthesis_route/pipeline"
    config = pipeline_dir / "configs/full_route_stage3.yaml"

    # 如果 pipeline 支持 start_from run_stage3_flow，则可以真实调用。
    # dry_run=False 时会执行；默认 dry_run=True 只记录命令。
    cmd = [
        "python",
        str(pipeline_dir / "run_pipeline.py"),
        "--config",
        str(config),
        "--infer_name",
        case_id,
        "--start_from",
        "run_stage3_flow",
    ]

    rc = run_cmd(cmd, case_dir / "condition_reexport.log", dry_run=dry_run)

    result = {
        "case_id": case_id,
        "recovery_module": "condition_reexport",
        "status": "planned" if dry_run else ("finished" if rc == 0 else "failed"),
        "return_code": rc,
        "command": " ".join(cmd),
        "recovery_log": str(case_dir / "condition_reexport.log"),
        "reason": "Stage3 conditions exist but condition support is low.",
        "suggested_follow_up": [
            "If support remains low, run clipping diagnostic.",
            "Increase n_gen_samples if supported by stage3 flow script/config.",
            "Then rerun summarize_routes/finalizer.",
        ],
    }
    write_json(case_dir / "recovery_result.json", result)
    return result


def handle_route_finalizer_recovery(project_root: Path, batch_name: str, row: dict, out_dir: Path, dry_run: bool) -> dict:
    """
    Stage2/Stage3 有输出，但 final route 缺失。
    轻量处理：从 summarize_routes 或后处理阶段重跑。
    """
    case_id = row["case_id"]
    case_dir = out_dir / "route_finalizer_recovery" / case_id
    mkdir(case_dir)

    pipeline_dir = project_root / "scripts/07_infer/structure_to_synthesis_route/pipeline"
    config = pipeline_dir / "configs/full_route_stage3.yaml"

    # 从 summarize_routes 开始，尽量避免重跑前面的模型。
    cmd = [
        "python",
        str(pipeline_dir / "run_pipeline.py"),
        "--config",
        str(config),
        "--infer_name",
        case_id,
        "--start_from",
        "summarize_routes",
    ]

    rc = run_cmd(cmd, case_dir / "route_finalizer_recovery.log", dry_run=dry_run)

    result = {
        "case_id": case_id,
        "recovery_module": "route_finalizer_recovery",
        "status": "planned" if dry_run else ("finished" if rc == 0 else "failed"),
        "return_code": rc,
        "command": " ".join(cmd),
        "recovery_log": str(case_dir / "route_finalizer_recovery.log"),
        "reason": "Stage2 and Stage3 outputs exist, but final route table is missing.",
    }
    write_json(case_dir / "recovery_result.json", result)
    return result


def handle_manual_or_rule_recovery(project_root: Path, batch_name: str, row: dict, out_dir: Path) -> dict:
    case_id = row["case_id"]
    case_dir = out_dir / "manual_or_rule_recovery" / case_id
    mkdir(case_dir)

    result = {
        "case_id": case_id,
        "recovery_module": "manual_or_rule_recovery",
        "status": "manual_review_required",
        "reason": "Major warning or suspicious precursor/condition rule.",
        "top1_precursor_set": row.get("top1_precursor_set", ""),
        "audit_level": row.get("audit_level", ""),
        "condition_support_score": row.get("condition_support_score", ""),
        "top1_final_score": row.get("top1_final_score", ""),
    }

    write_json(case_dir / "recovery_result.json", result)

    md = []
    md.append(f"# Manual or Rule Recovery: {case_id}")
    md.append("")
    for k, v in result.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    (case_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default="/Users/wyc/SynPred")
    ap.add_argument("--batch_root", default="/Users/wyc/SynPred/outputs/batch_adaptive")
    ap.add_argument("--batch_name", default="batch_001")
    ap.add_argument("--plan_csv", default="")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    # 默认安全：如果没有 --execute，就 dry run
    dry_run = True
    if args.execute:
        dry_run = False
    if args.dry_run:
        dry_run = True

    project_root = Path(args.project_root)
    batch_dir = Path(args.batch_root) / args.batch_name

    if args.plan_csv:
        plan_csv = Path(args.plan_csv)
    else:
        plan_csv = batch_dir / "recovery_plan" / "recovery_plan.csv"

    if not plan_csv.exists():
        raise FileNotFoundError(f"Missing recovery plan: {plan_csv}")

    plan = pd.read_csv(plan_csv)
    out_dir = batch_dir / "recovery_runs"
    mkdir(out_dir)

    results = []

    for _, row in plan.iterrows():
        rec = row.to_dict()
        module = str(rec.get("recovery_module", ""))

        if module == "inspect_pipeline_log":
            result = handle_pipeline_failed(project_root, args.batch_name, rec, out_dir)

        elif module == "stage2_recovery":
            result = handle_stage2_recovery(project_root, args.batch_name, rec, out_dir, dry_run=dry_run)

        elif module == "stage3_gap_recovery":
            result = handle_stage3_gap_recovery(project_root, args.batch_name, rec, out_dir, dry_run=dry_run)

        elif module == "condition_reexport":
            result = handle_condition_reexport(project_root, args.batch_name, rec, out_dir, dry_run=dry_run)

        elif module == "route_finalizer_recovery":
            result = handle_route_finalizer_recovery(project_root, args.batch_name, rec, out_dir, dry_run=dry_run)

        elif module == "manual_or_rule_recovery":
            result = handle_manual_or_rule_recovery(project_root, args.batch_name, rec, out_dir)

        else:
            result = {
                "case_id": rec.get("case_id", ""),
                "recovery_module": module,
                "status": "unknown_module",
                "reason": "No handler implemented for this recovery module.",
            }

        results.append(result)

    result_df = pd.DataFrame(results)
    result_csv = out_dir / "recovery_run_results.csv"
    result_json = out_dir / "recovery_run_results.json"
    result_md = out_dir / "recovery_run_results.md"

    result_df.to_csv(result_csv, index=False)
    result_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Recovery Run Results")
    lines.append("")
    lines.append(f"- Batch name: `{args.batch_name}`")
    lines.append(f"- Plan csv: `{plan_csv}`")
    lines.append(f"- Dry run: `{dry_run}`")
    lines.append(f"- Number of tasks: `{len(results)}`")
    lines.append("")

    if len(result_df) > 0:
        cols = ["case_id", "recovery_module", "status", "return_code", "reason", "recovery_log"]
        cols = [c for c in cols if c in result_df.columns]
        lines.append(result_df[cols].to_markdown(index=False))
    else:
        lines.append("No recovery tasks.")

    result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[SAVE]", result_csv)
    print("[SAVE]", result_json)
    print("[SAVE]", result_md)
    print()
    if len(result_df) > 0:
        print(result_df.to_string(index=False))
    else:
        print("No recovery tasks.")


if __name__ == "__main__":
    main()
