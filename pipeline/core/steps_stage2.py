#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
import pandas as pd


def build_stage2_features(r):
    r.log("===== STEP 5: build Stage2 hybrid feature table =====")
    out_dir = r.work_dir / "stage2_hybrid_csv"
    r.run([
        "python",
        r.project_root / "pipeline/core/05_build_hybrid_features_infer.py",
        "--task", "stage2",
        "--output_dir", out_dir,
        "--infer_descriptor_csv", r.outputs["infer_structdesc_csv"],
        "--infer_embedding_csv", r.outputs["final_graph_embed_csv"],
        "--embedding_prefix", "graph_emb",
        "--replicate_to_train_val",
    ])
    p = out_dir / "stage2_train_hybrid.csv"
    r.require_file(p)
    r.outputs["stage2_hybrid_csv"] = str(p)


def build_stage2_npz(r):
    r.log("===== STEP 6: build Stage2 GFlowNet inference NPZ =====")
    out_dir = r.work_dir / "stage2_hybrid"
    r.run([
        "python",
        r.project_root / "pipeline/core/07_build_stage2_gflownet_infer_npz.py",
        "--infer_hybrid_csv", r.outputs["stage2_hybrid_csv"],
        "--template_dir", r.cfg["stage2"]["template_dir"],
        "--output_dir", out_dir,
        "--split_name", "test",
    ])
    r.require_file(out_dir / "test.npz")
    r.require_file(out_dir / "test_meta.csv")
    r.outputs["stage2_npz_dir"] = str(out_dir)


def sample_stage2_gflownet(r):
    """
    Stage2 precursor candidate sampling.

    Supported modes:
      - composition_biased:
          use 19_sample_stage2_gflownet_composition_constrained.py
          This adds target-element-aware bias during decoding, before post-filtering.
      - standard:
          use the same sampler without the composition bias flags.
    """
    cfg = r.cfg
    project_root = r.project_root
    work_dir = r.work_dir
    device = r.device

    stage2_cfg = cfg.get("stage2", {}) or {}
    sampler_mode = stage2_cfg.get("sampler_mode", "standard")

    input_dir = Path(r.outputs.get("stage2_npz_dir", work_dir / "stage2_hybrid"))

    ckpt_path = Path(
        _expand_path(r, 
            stage2_cfg.get(
                "gflownet_ckpt",
                "{project_root}/runs/stage2/gflownet_joint_rerank_hybrid_gold_only_v1/best_model.pt",
            )
        )
    )

    batch_size = int(stage2_cfg.get("batch_size", 128))
    n_samples = int(stage2_cfg.get("n_samples", 100))
    temperature = float(stage2_cfg.get("temperature", 1.0))
    top_k = int(stage2_cfg.get("top_k", 20))
    seed = stage2_cfg.get("seed", None)

    if sampler_mode == "composition_biased":
        print("===== STEP 7: sample Stage2 GFlowNet precursor candidates with composition-biased decoding =====")

        script = project_root / "pipeline/core/19_sample_stage2_gflownet_composition_constrained.py"
        output_dir = work_dir / "stage2_gflownet_candidates_composition_decoding"

        target_hit_bonus = float(stage2_cfg.get("target_hit_bonus", 6.0))
        extra_element_penalty = float(stage2_cfg.get("extra_element_penalty", 1.0))
        no_overlap_penalty = float(stage2_cfg.get("no_overlap_penalty", 6.0))
        stop_bias = float(stage2_cfg.get("stop_bias", -2.0))
        ignore_elements = stage2_cfg.get("ignore_elements", ["H", "O"])

        if isinstance(ignore_elements, str):
            ignore_elements_arg = ignore_elements
        else:
            ignore_elements_arg = ",".join(str(x) for x in ignore_elements)

        _run_cmd([
            "python",
            str(script),
            "--input_dir", str(input_dir),
            "--output_dir", str(output_dir),
            "--ckpt_path", str(ckpt_path),
            "--split", "test",
            "--batch_size", str(batch_size),
            "--n_samples", str(n_samples),
            "--temperature", str(temperature),
            "--top_k", str(top_k),
            "--use_greedy_as_first",
            "--composition_constrained",
            "--target_hit_bonus", str(target_hit_bonus),
            "--extra_element_penalty", str(extra_element_penalty),
            "--no_overlap_penalty", str(no_overlap_penalty),
            "--stop_bias", str(stop_bias),
            "--ignore_elements", ignore_elements_arg,
            "--device", str(device),
        ] + (["--seed", str(seed)] if seed is not None else []))

        sample_csv = output_dir / "test_samples.csv"

        if not sample_csv.exists():
            raise FileNotFoundError(
                f"composition-biased Stage2 sampling did not create expected file: {sample_csv}"
            )

        r.outputs["stage2_sample_csv"] = str(sample_csv)
        r.outputs["stage2_sample_dir"] = str(output_dir)
        r.outputs["stage2_sampler_mode"] = "composition_biased"

        print(f"[OK] {sample_csv}")

    elif sampler_mode == "standard":
        print("===== STEP 7: sample Stage2 GFlowNet precursor candidates =====")

        script = project_root / "pipeline/core/19_sample_stage2_gflownet_composition_constrained.py"
        output_dir = work_dir / "stage2_gflownet_candidates"

        _run_cmd([
            "python",
            str(script),
            "--project_root", str(project_root),
            "--input_dir", str(input_dir),
            "--output_dir", str(output_dir),
            "--ckpt_path", str(ckpt_path),
            "--split", "test",
            "--batch_size", str(batch_size),
            "--n_samples", str(n_samples),
            "--temperature", str(temperature),
            "--top_k", str(top_k),
            "--use_greedy_as_first",
            "--device", str(device),
        ] + (["--seed", str(seed)] if seed is not None else []))

        sample_csv = output_dir / "test_samples.csv"

        if not sample_csv.exists():
            raise FileNotFoundError(
                f"standard Stage2 sampling did not create expected file: {sample_csv}"
            )

        r.outputs["stage2_sample_csv"] = str(sample_csv)
        r.outputs["stage2_sample_dir"] = str(output_dir)
        r.outputs["stage2_sampler_mode"] = "standard"

        print(f"[OK] {sample_csv}")

    else:
        raise ValueError(
            f"Unknown stage2.sampler_mode = {sampler_mode}. "
            "Allowed values: standard, composition_biased."
        )



def _run_cmd(cmd):
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def _cfg_get(r, key, default=None):
    cfg = getattr(r, "cfg", None)
    if cfg is None:
        return default

    if isinstance(key, str):
        parts = key.split(".")
    else:
        parts = list(key)

    cur = cfg
    for part in parts:
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _project_root(r) -> Path:
    return Path(str(_cfg_get(r, "project_root", getattr(r, "project_root", ""))))


def _route_scripts_dir(r) -> Path:
    return _project_root(r)


def _work_dir(r) -> Path:
    return Path(str(getattr(r, "work_dir")))


def _out_dir(r) -> Path:
    return Path(str(getattr(r, "out_dir")))


def _device(r) -> str:
    return str(getattr(r, "device", _cfg_get(r, "device", "cpu")))


def _set_output(r, key, value):
    if not hasattr(r, "outputs"):
        r.outputs = {}
    r.outputs[key] = str(value)


def _get_output(r, key, default=None):
    return getattr(r, "outputs", {}).get(key, default)


def _require_file(path: Path, name: str = None):
    if not path.exists():
        label = name or str(path)
        raise FileNotFoundError(f"Missing required file for {label}: {path}")
    print(f"[OK] {path}")


def constrain_stage2_by_composition(r):
    """
    STEP 7b:
    Post-rerank/filter Stage2 sampled precursor candidates by target composition.
    """
    work_dir = _work_dir(r)
    route_scripts = _route_scripts_dir(r)

    input_csv = Path(_get_output(
        r,
        "stage2_sample_csv",
        work_dir / "stage2_gflownet_candidates" / "test_samples.csv",
    ))

    output_csv = work_dir / "stage2_gflownet_candidates" / "test_samples_composition_constrained.csv"
    summary_json = work_dir / "stage2_gflownet_candidates" / "test_samples_composition_constrained_summary.json"

    script = route_scripts / "pipeline/core/18_constrain_stage2_sample_candidates_by_composition.py"
    _require_file(script, "composition constraint script")
    _require_file(input_csv, "stage2_sample_csv")

    min_coverage = _cfg_get(r, "stage2.composition_constraint_min_coverage", 0.0)
    coverage_weight = _cfg_get(r, "stage2.composition_constraint_coverage_weight", 20.0)
    extra_penalty_weight = _cfg_get(r, "stage2.composition_constraint_extra_penalty_weight", 5.0)
    rank_weight = _cfg_get(r, "stage2.composition_constraint_rank_weight", 0.01)
    top_n_per_sample = _cfg_get(r, "stage2.composition_constraint_top_n_per_sample", 100)

    cmd = [
        "python", script,
        "--input_csv", input_csv,
        "--output_csv", output_csv,
        "--summary_json", summary_json,
        "--min_coverage", str(min_coverage),
        "--coverage_weight", str(coverage_weight),
        "--extra_penalty_weight", str(extra_penalty_weight),
        "--rank_weight", str(rank_weight),
        "--top_n_per_sample", str(top_n_per_sample),
        "--dedup",
    ]

    if bool(_cfg_get(r, "stage2.composition_constraint_drop_zero_overlap", False)):
        cmd.append("--drop_zero_overlap")

    _run_cmd(cmd)

    _require_file(output_csv, "stage2_constrained_sample_csv")
    _set_output(r, "stage2_constrained_sample_csv", output_csv)
    _set_output(r, "stage2_constrained_sample_summary_json", summary_json)


def add_composition_fallback(r):
    """
    STEP 9:
    Add composition-complete fallback precursor sets.
    """
    work_dir = _work_dir(r)
    route_scripts = _route_scripts_dir(r)

    input_csv = Path(_get_output(
        r,
        "stage2_unique_csv",
        work_dir / "stage2_summary" / "unique_sets_ranked.csv",
    ))

    output_csv = work_dir / "stage2_summary" / "unique_sets_ranked_with_fallback.csv"
    summary_json = work_dir / "stage2_summary" / "composition_fallback_summary.json"

    script = route_scripts / "pipeline/core/11_add_composition_fallback_precursors.py"
    _require_file(script, "composition fallback script")
    _require_file(input_csv, "stage2_unique_csv")

    top_n_fallback = _cfg_get(r, "stage2.fallback_top_n", 20)

    _run_cmd([
        "python", script,
        "--input_csv", input_csv,
        "--output_csv", output_csv,
        "--summary_json", summary_json,
        "--top_n_fallback", str(top_n_fallback),
        "--rank_col", "rank",
        "--precursor_col", "precursor_set",
    ])

    _require_file(output_csv, "stage2_fallback_csv")
    _set_output(r, "stage2_fallback_csv", output_csv)
    _set_output(r, "stage2_fallback_summary_json", summary_json)


def retrieve_stage2_candidates(r):
    """
    STEP 10:
    Retrieve historical Stage2 precursor candidates from existing NPZ labels.
    """
    project_root = _project_root(r)
    work_dir = _work_dir(r)
    route_scripts = _route_scripts_dir(r)

    input_csv = Path(_get_output(
        r,
        "stage2_unique_csv",
        work_dir / "stage2_summary" / "unique_sets_ranked.csv",
    ))

    template_dir = Path(str(_cfg_get(
        r,
        "stage2.template_dir",
        project_root / "data/interim/generative/stage2_gflownet_dataset/hybrid/gold_only",
    )))

    output_csv = work_dir / "stage2_summary" / "retrieval_npz_candidates.csv"
    summary_json = work_dir / "stage2_summary" / "retrieval_npz_candidates_summary.json"

    script = route_scripts / "pipeline/core/17_retrieve_stage2_precursor_candidates_from_npz.py"
    _require_file(script, "retrieval script")
    _require_file(input_csv, "stage2_unique_csv")
    _require_file(template_dir / "precursor_names.json", "precursor_names.json")

    top_k = _cfg_get(r, "stage2.retrieval_top_k", 50)
    min_similarity = _cfg_get(r, "stage2.retrieval_min_similarity", 0.0)
    label_threshold = _cfg_get(r, "stage2.retrieval_label_threshold", 0.5)

    _run_cmd([
        "python", script,
        "--target_csv", input_csv,
        "--dataset_dir", template_dir,
        "--precursor_names_json", template_dir / "precursor_names.json",
        "--output_csv", output_csv,
        "--summary_json", summary_json,
        "--top_k", str(top_k),
        "--min_similarity", str(min_similarity),
        "--label_threshold", str(label_threshold),
    ])

    _require_file(output_csv, "stage2_retrieval_csv")
    _set_output(r, "stage2_retrieval_csv", output_csv)
    _set_output(r, "stage2_retrieval_summary_json", summary_json)


def predict_stage2_baseline(r):
    """
    STEP 11:
    Predict Stage2 candidates with ExtraTrees baseline if available.
    If model/script is missing, write an empty CSV so downstream merge still works.
    """
    project_root = _project_root(r)
    work_dir = _work_dir(r)
    route_scripts = _route_scripts_dir(r)

    output_csv = work_dir / "stage2_summary" / "extratrees_baseline_candidates.csv"
    summary_json = work_dir / "stage2_summary" / "extratrees_baseline_candidates_summary.json"

    model_path = Path(str(_cfg_get(
        r,
        "stage2.baseline_model",
        project_root / "runs/stage2/extratrees_multilabel_hybrid_gold_only_v1/stage2_extratrees_multilabel.joblib",
    )))

    script_candidates = [
        route_scripts / "22_predict_stage2_extratrees_baseline_candidates.py",
        route_scripts / "20_predict_stage2_extratrees_baseline_infer.py",
    ]
    script = next((p for p in script_candidates if p.exists()), None)

    if script is not None and model_path.exists():
        input_dir = Path(_get_output(
            r,
            "stage2_npz_dir",
            work_dir / "stage2_hybrid",
        ))

        top_k_labels = _cfg_get(r, "stage2.baseline_top_k_labels", 12)
        top_k_sets = _cfg_get(r, "stage2.baseline_top_k_sets", 30)
        min_prob = _cfg_get(r, "stage2.baseline_min_prob", 0.02)
        max_set_size = _cfg_get(r, "stage2.baseline_max_set_size", 4)

        _run_cmd([
            "python", script,
            "--input_dir", input_dir,
            "--model_path", model_path,
            "--split", "test",
            "--output_csv", output_csv,
            "--summary_json", summary_json,
            "--top_k_labels", str(top_k_labels),
            "--top_k_sets", str(top_k_sets),
            "--min_prob", str(min_prob),
            "--max_set_size", str(max_set_size),
        ])
    else:
        print("[WARN] baseline model or script missing; write empty baseline csv.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "rank", "set_key", "precursor_set", "n_precursors", "count", "frequency",
            "sample_id", "formula", "formula_x", "formula_y", "doi", "split_group",
            "decode_method", "decode_methods_seen",
            "sample_rank_min", "sample_rank_mean", "sample_rank_max",
            "is_baseline_candidate", "baseline_score",
        ]
        pd.DataFrame(columns=cols).to_csv(output_csv, index=False)
        summary_json.write_text(
            '{"status": "skipped", "reason": "missing baseline model or script"}\n',
            encoding="utf-8",
        )

    _require_file(output_csv, "stage2_baseline_csv")
    _set_output(r, "stage2_baseline_csv", output_csv)
    _set_output(r, "stage2_baseline_summary_json", summary_json)


def merge_stage2_sources(r):
    """
    STEP 12:
    Merge GFlowNet/fallback/retrieval/baseline Stage2 candidate pools.
    """
    work_dir = _work_dir(r)
    route_scripts = _route_scripts_dir(r)

    fallback_csv = Path(_get_output(
        r,
        "stage2_fallback_csv",
        work_dir / "stage2_summary" / "unique_sets_ranked_with_fallback.csv",
    ))
    retrieval_csv = Path(_get_output(
        r,
        "stage2_retrieval_csv",
        work_dir / "stage2_summary" / "retrieval_npz_candidates.csv",
    ))
    baseline_csv = Path(_get_output(
        r,
        "stage2_baseline_csv",
        work_dir / "stage2_summary" / "extratrees_baseline_candidates.csv",
    ))

    output_csv = work_dir / "stage2_summary" / "unique_sets_ranked_with_fallback_retrieval_baseline.csv"
    summary_json = work_dir / "stage2_summary" / "merge_fallback_retrieval_baseline_summary.json"

    script = route_scripts / "pipeline/core/16_merge_stage2_candidate_sources.py"
    _require_file(script, "merge stage2 sources script")
    _require_file(fallback_csv, "stage2_fallback_csv")
    _require_file(retrieval_csv, "stage2_retrieval_csv")
    _require_file(baseline_csv, "stage2_baseline_csv")

    _run_cmd([
        "python", script,
        "--input_csvs", fallback_csv, retrieval_csv, baseline_csv,
        "--output_csv", output_csv,
        "--summary_json", summary_json,
        "--precursor_col", "precursor_set",
    ])

    _require_file(output_csv, "stage2_merged_csv")
    _set_output(r, "stage2_merged_csv", output_csv)
    _set_output(r, "stage2_merged_summary_json", summary_json)


def rerank_stage2_by_elements(r):
    """
    STEP 13:
    Element-aware rerank of merged Stage2 precursor candidates.
    """
    work_dir = _work_dir(r)
    route_scripts = _route_scripts_dir(r)

    input_csv = Path(_get_output(
        r,
        "stage2_merged_csv",
        work_dir / "stage2_summary" / "unique_sets_ranked_with_fallback_retrieval_baseline.csv",
    ))

    output_csv = work_dir / "stage2_summary" / "unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv"
    summary_json = work_dir / "stage2_summary" / "element_rerank_with_fallback_retrieval_baseline_summary.json"

    script = route_scripts / "pipeline/core/10_rerank_stage2_candidates_by_elements.py"
    _require_file(script, "element-aware Stage2 rerank script")
    _require_file(input_csv, "stage2_merged_csv")

    top_n = _cfg_get(r, "stage2.element_top_n", 30)
    coverage_weight = _cfg_get(r, "stage2.element_coverage_weight", 20.0)
    extra_penalty_weight = _cfg_get(r, "stage2.element_extra_penalty_weight", 5.0)
    rank_weight = _cfg_get(r, "stage2.element_rank_weight", 0.01)

    _run_cmd([
        "python", script,
        "--input_csv", input_csv,
        "--output_csv", output_csv,
        "--summary_json", summary_json,
        "--precursor_col", "precursor_set",
        "--rank_col", "rank",
        "--top_n", str(top_n),
        "--coverage_weight", str(coverage_weight),
        "--extra_penalty_weight", str(extra_penalty_weight),
        "--rank_weight", str(rank_weight),
    ])

    _require_file(output_csv, "stage2_final_csv")
    _set_output(r, "stage2_final_csv", output_csv)
    _set_output(r, "stage2_element_rerank_summary_json", summary_json)


def fix_stage2_global_rank(r):
    """
    STEP 13b:
    Fix global rank after multi-source Stage2 merge/rerank.
    """
    work_dir = _work_dir(r)
    csv = Path(_get_output(
        r,
        "stage2_final_csv",
        work_dir / "stage2_summary" / "unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv",
    ))

    _require_file(csv, "stage2_final_csv")

    df = pd.read_csv(csv)
    if df.empty:
        print(f"[WARN] empty csv, rank unchanged: {csv}")
        return

    if "rank" in df.columns:
        df["source_rank"] = df["rank"]

    sort_cols = []
    ascending = []

    if "element_rerank_score" in df.columns:
        sort_cols.append("element_rerank_score")
        ascending.append(False)
    if "element_coverage" in df.columns:
        sort_cols.append("element_coverage")
        ascending.append(False)
    if "source_rank" in df.columns:
        sort_cols.append("source_rank")
        ascending.append(True)

    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)
    df.to_csv(csv, index=False)

    print(f"[OK] fixed rank: {csv}")
    preview_cols = [
        c for c in [
            "rank", "source_rank", "precursor_set", "candidate_source",
            "element_coverage", "element_rerank_score",
        ]
        if c in df.columns
    ]
    if preview_cols:
        print(df[preview_cols].head(30).to_string(index=False))

    _set_output(r, "stage2_final_csv", csv)


def export_precursor_only_report(r):
    """
    Final precursor-only report.
    """
    work_dir = _work_dir(r)
    out_dir = _out_dir(r)

    input_csv = Path(_get_output(
        r,
        "stage2_final_csv",
        work_dir / "stage2_summary" / "unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv",
    ))

    _require_file(input_csv, "stage2_final_csv")

    out = out_dir / "precursor_only"
    out.mkdir(parents=True, exist_ok=True)

    output_csv = out / "precursor_only_recommendations.csv"
    output_md = out / "precursor_only_recommendations.md"

    df = pd.read_csv(input_csv)
    df.to_csv(output_csv, index=False)

    show_cols = [
        c for c in [
            "rank", "precursor_set", "element_coverage", "element_hit",
            "element_missing", "missing_count", "extra_element_penalty",
            "element_rerank_score", "candidate_source",
        ]
        if c in df.columns
    ]

    lines = []
    lines.append("# Precursor-only Recommendations")
    lines.append("")
    lines.append("These recommendations are generated from the Stage2 precursor candidate pipeline.")
    lines.append("")
    if show_cols:
        lines.append(df[show_cols].head(30).to_markdown(index=False))
    else:
        lines.append(df.head(30).to_markdown(index=False))
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[SAVE] {output_csv}")
    print(f"[SAVE] {output_md}")

    _set_output(r, "precursor_only_csv", output_csv)
    _set_output(r, "precursor_only_md", output_md)

def summarize_stage2(r):
    """
    STEP 8:
    Summarize Stage2 sampled precursor candidates into unique precursor sets.
    Prefer composition-constrained samples if available.
    """
    work_dir = _work_dir(r)
    project_root = _project_root(r)

    input_csv = Path(_get_output(
        r,
        "stage2_constrained_sample_csv",
        work_dir / "stage2_gflownet_candidates" / "test_samples_composition_constrained.csv",
    ))

    # If strong composition-biased sampler is enabled, its constrained output may be in another directory.
    if not input_csv.exists():
        strong_csv = work_dir / "stage2_gflownet_candidates_composition_decoding_strong" / "test_samples_composition_constrained.csv"
        if strong_csv.exists():
            input_csv = strong_csv

    output_dir = work_dir / "stage2_summary"
    output_csv = output_dir / "unique_sets_ranked.csv"

    script = project_root / "pipeline" / "core" / "08_summarize_stage2_gflownet_candidates.py"

    _require_file(script, "summarize Stage2 candidates script")
    _require_file(input_csv, "stage2 candidate sample csv")

    top_n = _cfg_get(r, "stage2.summary_top_n", 100)

    _run_cmd([
        "python", script,
        "--input_csv", input_csv,
        "--output_dir", output_dir,
        "--top_n", str(top_n),
    ])

    _require_file(output_csv, "stage2_unique_csv")

    _set_output(r, "stage2_unique_csv", output_csv)
    _set_output(r, "stage2_summary_dir", output_dir)


def _expand_path(r, value):
    """
    Expand a path value from config/output.

    Supports:
    - absolute paths
    - relative paths under project_root
    - placeholders like {project_root}, {work_dir}, {out_dir}, {infer_name}
    - simple nested placeholders like {stage2.gflownet_run_dir}
    """
    if value is None:
        return None

    cfg = getattr(r, "cfg", None)
    if cfg is None:
        cfg = getattr(r, "config", {})
    if cfg is None:
        cfg = {}

    def cfg_get(dotted_key, default=""):
        cur = cfg
        for part in str(dotted_key).split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    project_root = str(
        cfg.get("project_root")
        or getattr(r, "project_root", "")
    )
    infer_name = str(cfg.get("infer_name") or "")
    work_dir = str(getattr(r, "work_dir", "") or cfg.get("work_dir", ""))
    out_dir = str(getattr(r, "out_dir", "") or cfg.get("out_dir", ""))

    s = str(value)

    placeholders = {
        "project_root": project_root,
        "infer_name": infer_name,
        "work_dir": work_dir,
        "out_dir": out_dir,
    }

    # Replace basic placeholders.
    for k, v in placeholders.items():
        s = s.replace("{" + k + "}", str(v))

    # Replace nested placeholders such as {stage2.gflownet_run_dir}
    import re
    for m in re.findall(r"\{([^{}]+)\}", s):
        if "." in m:
            s = s.replace("{" + m + "}", str(cfg_get(m, "")))

    path = Path(s).expanduser()

    # Relative paths are resolved under project_root.
    if not path.is_absolute():
        path = Path(project_root) / path


    return path
