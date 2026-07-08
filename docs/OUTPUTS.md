# Output Guide

The full-route pipeline writes outputs under:

```text
outputs/inference/<infer_name>/
```

The main route directory is:

```text
outputs/inference/<infer_name>/routes_flow_fallback_retrieval_baseline_element_reranked/
```

Stable user-facing files:

```text
final_recommended_routes.csv
final_recommended_routes.md
final_recommended_routes_summary.json
pipeline_v3_manifest.json
```

Useful intermediate route tables:

```text
synthesis_routes_readable.csv
synthesis_routes_display_filtered.csv
final_top_routes.csv
final_top_routes_with_confidence.csv
final_top_routes_v3_joint_reranked.csv
final_top_routes_v3_learned_reranked.csv
final_top_routes_v43_template_chemonly_reranked.csv
```

Precursor-only mode writes:

```text
outputs/inference/<infer_name>_precursor_only/precursor_only/
  precursor_only_recommendations.csv
  precursor_only_recommendations.md
```

Important Stage2 intermediate files live in:

```text
data/interim/infer/<infer_name>/stage2_summary/
  unique_sets_ranked.csv
  unique_sets_ranked_with_fallback.csv
  retrieval_npz_candidates.csv
  unique_sets_ranked_with_fallback_retrieval_baseline.csv
  unique_sets_ranked_with_fallback_retrieval_baseline_element_reranked.csv
```
