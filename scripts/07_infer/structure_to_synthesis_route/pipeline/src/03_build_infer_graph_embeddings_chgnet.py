#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_infer_graph_embeddings_chgnet.py

功能
----
把 inference 场景下的新结构图嵌入导出流程整合为一个脚本：

infer.jsonl
  -> graph_cache_input/stage2_*.jsonl
  -> CHGNet cache (inference_mode)
  -> CHGNet embeddings
  -> infer_graph_embed.csv

依赖
----
- scripts/07_infer/03d_build_chgnet_cache_stage2_inference_mode.py
- scripts/03_graph/export_chgnet_stage2_embeddings.py

推荐用法
--------
python 03_build_infer_graph_embeddings_chgnet.py \
  --infer_jsonl /Users/wyc/SynPred/infer_runs/demo1/infer_split/infer.jsonl \
  --work_dir /Users/wyc/SynPred/infer_runs/demo1

输出
----
work_dir/
  ├── graph_cache_input/
  ├── chgnet_cache/
  ├── chgnet_embed/
  ├── graph_embed/infer_graph_embed.csv
  └── infer_graph_embed_summary.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_command(cmd: str) -> None:
    print(f"[RUN] {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {cmd}")


def build_graph_cache_input(infer_jsonl: Path, graph_cache_input_dir: Path) -> Dict[str, Any]:
    rows = read_jsonl(infer_jsonl)
    if not rows:
        raise ValueError(f"No rows found in {infer_jsonl}")

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        rr = dict(row)
        rr.setdefault("formula", rr.get("formula_guess"))
        rr.setdefault("split", "infer")
        rr.setdefault("source_dataset", "infer")
        rr.setdefault("synthesis_type", "infer")
        rr.setdefault("split_group", rr.get("sample_id"))
        normalized_rows.append(rr)

    ensure_dir(graph_cache_input_dir)
    split_files = [
        "stage2_train.jsonl",
        "stage2_val.jsonl",
        "stage2_test.jsonl",
        "stage2_gold_train_holdout.jsonl",
    ]
    for name in split_files:
        write_jsonl(graph_cache_input_dir / name, normalized_rows)

    return {
        "n_rows": len(normalized_rows),
        "graph_cache_input_dir": str(graph_cache_input_dir),
        "split_files": [str(graph_cache_input_dir / x) for x in split_files],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference-only CHGNet graph embedding pipeline: infer.jsonl -> cache -> embeddings -> infer_graph_embed.csv"
    )
    parser.add_argument("--infer_jsonl", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True, help="Working directory under infer_runs/demoX")
    parser.add_argument("--project_root", type=str, default="/Users/wyc/SynPred")
    parser.add_argument(
        "--precursor_vocab_json",
        type=str,
        default="/Users/wyc/SynPred/data/interim/graph_cache/chgnet_stage2/gold_only/precursor_vocab.json",
    )
    parser.add_argument("--train_mode", type=str, default="gold_only")
    parser.add_argument("--max_sites", type=int, default=256)
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep intermediate files if they already exist; default is overwrite by rerun.")
    args = parser.parse_args()

    infer_jsonl = Path(args.infer_jsonl).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    precursor_vocab_json = Path(args.precursor_vocab_json).expanduser().resolve()

    if not infer_jsonl.exists():
        raise FileNotFoundError(f"Missing infer_jsonl: {infer_jsonl}")
    if not precursor_vocab_json.exists():
        raise FileNotFoundError(f"Missing precursor_vocab_json: {precursor_vocab_json}")

    graph_cache_input_dir = work_dir / "graph_cache_input"
    chgnet_cache_dir = work_dir / "chgnet_cache"
    chgnet_embed_dir = work_dir / "chgnet_embed"
    graph_embed_dir = work_dir / "graph_embed"
    infer_graph_embed_csv = graph_embed_dir / "infer_graph_embed.csv"

    ensure_dir(work_dir)
    ensure_dir(graph_embed_dir)

    summary: Dict[str, Any] = {
        "config": {
            "infer_jsonl": str(infer_jsonl),
            "work_dir": str(work_dir),
            "project_root": str(project_root),
            "precursor_vocab_json": str(precursor_vocab_json),
            "train_mode": args.train_mode,
            "max_sites": int(args.max_sites),
            "keep_intermediate": bool(args.keep_intermediate),
        },
        "steps": {},
    }

    # Step 1: build graph_cache_input
    summary["steps"]["graph_cache_input"] = build_graph_cache_input(
        infer_jsonl=infer_jsonl,
        graph_cache_input_dir=graph_cache_input_dir,
    )

    # Step 2: build chgnet cache in inference mode
    chgnet_cache_script = project_root / "scripts/07_infer/structure_to_synthesis_route/pipeline/src/03d_build_chgnet_cache_stage2_inference_mode.py"
    if not chgnet_cache_script.exists():
        raise FileNotFoundError(f"Missing script: {chgnet_cache_script}")

    cmd_cache = " ".join([
        "python",
        shlex.quote(str(chgnet_cache_script)),
        "--input_dir", shlex.quote(str(graph_cache_input_dir)),
        "--output_dir", shlex.quote(str(chgnet_cache_dir)),
        "--train_mode", shlex.quote(str(args.train_mode)),
        "--precursor_vocab_json", shlex.quote(str(precursor_vocab_json)),
        "--max_sites", str(int(args.max_sites)),
        "--inference_mode",
    ])
    run_command(cmd_cache)
    summary["steps"]["chgnet_cache"] = {
        "script": str(chgnet_cache_script),
        "output_dir": str(chgnet_cache_dir),
    }

    # Step 3: export chgnet embeddings
    export_script = project_root / "scripts/03_graph/export_chgnet_stage2_embeddings.py"
    if not export_script.exists():
        raise FileNotFoundError(f"Missing script: {export_script}")

    cmd_export = " ".join([
        "python",
        shlex.quote(str(export_script)),
        "--cache_dir", shlex.quote(str(chgnet_cache_dir)),
        "--output_dir", shlex.quote(str(chgnet_embed_dir)),
    ])
    run_command(cmd_export)
    summary["steps"]["chgnet_export"] = {
        "script": str(export_script),
        "output_dir": str(chgnet_embed_dir),
    }

    # Step 4: copy stage2_test_graph_embed.csv -> infer_graph_embed.csv
    stage2_test_csv = chgnet_embed_dir / "stage2_test_graph_embed.csv"
    if not stage2_test_csv.exists():
        raise FileNotFoundError(f"Expected output missing: {stage2_test_csv}")

    shutil.copyfile(stage2_test_csv, infer_graph_embed_csv)
    summary["steps"]["final_copy"] = {
        "source": str(stage2_test_csv),
        "target": str(infer_graph_embed_csv),
    }

    summary_path = work_dir / "infer_graph_embed_summary.json"
    write_json(summary_path, summary)

    print(f"[DONE] infer_graph_embed_csv -> {infer_graph_embed_csv}")
    print(f"[DONE] summary -> {summary_path}")


if __name__ == "__main__":
    main()
