#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


def preflight(r):
    cfg = r.cfg
    paths = cfg["paths"]
    stage2 = cfg["stage2"]

    r.log("===== STEP 0: preflight =====")
    r.require_dir(paths["poscar_dir"])
    r.require_file(stage2["gflownet_ckpt"])
    r.require_dir(stage2["template_dir"])
    r.require_file(Path(stage2["template_dir"]) / "feature_cols.json")
    r.require_file(Path(stage2["template_dir"]) / "feature_mean.npy")
    r.require_file(Path(stage2["template_dir"]) / "feature_std.npy")
    r.require_file(Path(stage2["template_dir"]) / "action_to_id.json")
    r.require_file(Path(stage2["template_dir"]) / "action_vocab.json")
    r.require_file(Path(stage2["template_dir"]) / "precursor_names.json")
    r.require_file(stage2["precursor_vocab_json"])

    if cfg["steps"].get("run_stage3_flow", False):
        r.require_file(cfg["stage3"]["schema_json"])
        r.require_file(cfg["stage3"]["flow_ckpt"])
        r.require_file(cfg["stage3"]["flow_script"])


def make_infer_split(r):
    r.log("===== STEP 1: make infer split from POSCARs =====")
    out = r.work_dir / "split"
    r.run([
        "python",
        r.project_root / "scripts/07_infer/structure_to_synthesis_route/pipeline/src/01_make_infer_split_from_poscars.py",
        "--poscar_dir", r.cfg["paths"]["poscar_dir"],
        "--output_dir", out,
    ])
    p = out / "infer.jsonl"
    r.require_file(p)
    r.outputs["infer_jsonl"] = str(p)


def build_structdesc(r):
    r.log("===== STEP 2: build structural descriptors =====")
    out = r.work_dir / "infer_structdesc.csv"
    r.run([
        "python",
        r.project_root / "scripts/07_infer/structure_to_synthesis_route/pipeline/src/02_build_infer_structdesc_direct.py",
        "--infer_jsonl", r.outputs["infer_jsonl"],
        "--output_csv", out,
    ])
    r.require_file(out)
    r.outputs["infer_structdesc_csv"] = str(out)


def build_chgnet_embedding(r):
    r.log("===== STEP 3: build CHGNet graph embeddings =====")
    out_dir = r.work_dir / "chgnet"
    r.run([
        "python",
        r.project_root / "scripts/07_infer/structure_to_synthesis_route/pipeline/src/03_build_infer_graph_embeddings_chgnet.py",
        "--infer_jsonl", r.outputs["infer_jsonl"],
        "--work_dir", out_dir,
        "--project_root", r.project_root,
        "--precursor_vocab_json", r.cfg["stage2"]["precursor_vocab_json"],
        "--train_mode", "gold_only",
        "--max_sites", r.cfg["graph"]["max_sites"],
    ])
    p = out_dir / "graph_embed/infer_graph_embed.csv"
    r.require_file(p)
    r.outputs["chgnet_embed_csv"] = str(p)


def finalize_graph_embedding(r):
    r.log("===== STEP 4: finalize graph embeddings =====")
    cgcnn_cache = r.work_dir / "cgcnn_cache_infer"
    graph = r.cfg["graph"]
    final = r.work_dir / "graph_embed/infer_graph_embed.csv"

    if cgcnn_cache.is_dir() and Path(graph["cgcnn_checkpoint"]).exists() and Path(graph["cgcnn_model_py"]).exists():
        r.run([
            "python",
            r.project_root / "scripts/07_infer/structure_to_synthesis_route/pipeline/src/04_finalize_infer_graph_embeddings.py",
            "--project_root", r.project_root,
            "--work_dir", r.work_dir,
            "--checkpoint", graph["cgcnn_checkpoint"],
            "--device", r.device,
            "--model_py", graph["cgcnn_model_py"],
            "--model_class", graph["cgcnn_model_class"],
        ])
        if final.exists():
            r.outputs["final_graph_embed_csv"] = str(final)
            return

    r.log("[WARN] CGCNN cache/model unavailable; use CHGNet-only embedding.")
    r.record_degradation("finalize_graph_embedding", "CGCNN unavailable, using CHGNet-only")
    if "chgnet_embed_csv" not in r.outputs:
        raise FileNotFoundError("Cannot fall back to CHGNet embedding: chgnet_embed_csv not available")
    r.outputs["final_graph_embed_csv"] = r.outputs["chgnet_embed_csv"]
