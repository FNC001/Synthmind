# Pipeline Flow

This document summarizes the overall logic of pipeline_v3.

============================================================
1. Input
============================================================

The input is a folder of POSCAR files:

    data/infer/<infer_name>/poscars

Each POSCAR represents a target structure for synthesis-route inference.

============================================================
2. Structure feature construction
============================================================

The pipeline first converts POSCAR files into model-readable structure features.

Main steps:

    POSCAR
    -> infer.jsonl
    -> direct structural descriptors
    -> CHGNet graph embeddings
    -> hybrid feature table

The hybrid feature table is used as the input representation for Stage2 precursor prediction.

============================================================
3. Stage2 precursor generation
============================================================

Stage2 predicts candidate precursor sets.

Main candidate sources:

    GFlowNet generation
    composition fallback
    retrieval from historical precursor labels
    ExtraTrees baseline if available

The raw GFlowNet samples can be noisy, so the pipeline applies several correction layers.

Main correction layers:

    composition constraint
    fallback completion
    retrieval augmentation
    multi-source merge
    element-aware reranking

The final Stage2 output is:

    unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv

This file is the main precursor candidate table.

============================================================
4. Stage3 condition generation
============================================================

Stage3 predicts synthesis conditions for each selected precursor set.

The condition model estimates:

    temperature
    time
    condition confidence or score

Stage3 is needed for complete route recommendation, but it can be skipped in precursor-only mode.

============================================================
5. Route construction
============================================================

After Stage3, the pipeline combines:

    target structure
    precursor set
    temperature
    time
    condition score
    element coverage

into readable synthesis-route candidates.

The readable route files are useful for inspection but are not the final recommendation.

============================================================
6. Stage35 route reranking
============================================================

Stage35 performs final route-level reranking.

It uses information from:

    precursor quality
    element coverage
    missing elements
    extra element penalty
    Stage3 condition score
    learned route-ranker probability
    precursor rank

The preferred full-route output is:

    final_top_routes.md
    final_top_routes.csv

============================================================
7. Two recommended modes
============================================================

Full route mode:

    python run_pipeline.py --config configs/full_route_stage3.yaml

Final output:

    final_top_routes.md

Precursor-only mode:

    python run_pipeline.py --config configs/precursor_only.yaml

Final output:

    precursor_only_recommendations.md

============================================================
8. Practical interpretation
============================================================

The stable result should be understood as a multi-stage recommendation:

    raw generation
    -> chemical filtering
    -> fallback and retrieval augmentation
    -> element-aware reranking
    -> optional condition generation
    -> final route reranking

The raw Stage2 output alone should not be treated as the final synthesis recommendation.
