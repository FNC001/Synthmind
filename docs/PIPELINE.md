# Pipeline Flow

Input POSCAR files are read from:

```text
data/infer/<infer_name>/poscars/
```

The main flow is:

```text
POSCAR
-> infer split
-> structural descriptors
-> CHGNet graph embeddings
-> Stage2 hybrid features
-> GFlowNet precursor sampling
-> composition constraint
-> fallback/retrieval/baseline candidate merge
-> element-aware precursor rerank
-> Stage3 conditioned feature table
-> LightGBM condition prediction
-> readable synthesis routes
-> display filtering
-> QC/confidence postprocess
-> v3/v43 route ranking
-> final_recommended_routes.csv
```

Full-route mode:

```bash
scripts/run_full_route.sh demo_poscar_test
```

Precursor-only mode:

```bash
scripts/run_precursor_only.sh demo_poscar_test
```

The raw Stage2 output is useful for diagnostics, but the final synthesis recommendation should come from `final_recommended_routes.csv`.
