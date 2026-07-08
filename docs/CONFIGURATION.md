# Config Guide

The public configs live in `configs/`.

## Full Route

Use `configs/full_route.yaml` for complete synthesis-route prediction:

```bash
python pipeline/run_pipeline.py \
  --config configs/full_route.yaml \
  --project_root "$(pwd)" \
  --infer_name demo_poscar_test
```

This enables structure features, Stage2 precursor candidates, Stage3 LightGBM condition prediction, route assembly, QC/confidence post-processing, and final recommendation export.

## Precursor Only

Use `configs/precursor_only.yaml` when only precursor sets are needed:

```bash
python pipeline/run_pipeline.py \
  --config configs/precursor_only.yaml \
  --project_root "$(pwd)" \
  --infer_name demo_poscar_test
```

This stops after the Stage2 precursor candidate stack and writes the precursor-only recommendation table.

## Resume

Use `--start_from <step_name>` when earlier intermediate files already exist:

```bash
python pipeline/run_pipeline.py \
  --config configs/full_route.yaml \
  --project_root "$(pwd)" \
  --infer_name demo_poscar_test \
  --start_from run_stage3_lgbm
```
