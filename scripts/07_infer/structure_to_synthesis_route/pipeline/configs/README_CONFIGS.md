# Config Guide

This folder contains YAML configuration files for pipeline_v3.

============================================================
1. full_route_stage3.yaml
============================================================

Use this config for complete synthesis-route prediction.

It enables:

    Stage2 precursor prediction
    Stage3 condition generation
    Stage35 route reranking
    final top-route export

Run:

    python run_pipeline.py --config configs/full_route_stage3.yaml

Main final outputs:

    final_top_routes.md
    final_top_routes.csv

============================================================
2. precursor_only.yaml
============================================================

Use this config when only precursor-set prediction is needed.

It enables:

    Stage2 precursor prediction
    composition constraint
    fallback completion
    retrieval augmentation
    element-aware reranking
    precursor-only export

It disables:

    Stage3 condition generation
    route construction
    Stage35 route reranking

Run:

    python run_pipeline.py --config configs/precursor_only.yaml

Main final outputs:

    precursor_only_recommendations.md
    precursor_only_recommendations.csv

============================================================
3. Choosing a config
============================================================

Use full_route_stage3.yaml for:

    precursor set + temperature + time

Use precursor_only.yaml for:

    precursor set only

============================================================
4. Resume from a later step
============================================================

For example:

    python run_pipeline.py --config configs/full_route_stage3.yaml --start_from stage35_v21_rerank

This is useful when earlier intermediate files already exist.
