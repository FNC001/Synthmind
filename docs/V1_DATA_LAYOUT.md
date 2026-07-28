# Synthmind V1.0 external data layout

V1.0 keeps executable source in Git and reads large or restricted artifacts
from a separate root selected with `SYNTHMIND_DATA_ROOT` or `--data-root`.

## Required top-level directories

| Directory | Purpose |
|---|---|
| `00_OVERVIEW_AND_MANIFEST` | Lineage, model selection and integrity metadata |
| `02_RAW_DATA` | Authorized raw synthesis and structure assets |
| `03_CLEANED_AND_MERGED_DATA` | Portable standardized and cleaned records |
| `04_SPLITS` | Base, route and family-grouped splits |
| `05_FEATURES_AND_EMBEDDINGS` | Descriptors, graph caches and embeddings |
| `06_TRAIN_READY_DATA` | Final Stage2 and Stage3 training arrays |
| `07_BEST_MODELS` | Imported dependencies and selected checkpoints |
| `08_GENERATED_OUTPUTS` | Frozen candidates and generative samples |
| `09_ACCURACY_EVALUATION` | Metric references and independent evaluator |
| `11_TESTS_AND_AUDITS` | Release and structure integrity tools |

The formal code lives in the Git checkout. An old `01_SOURCE_CODE` directory
inside an artifact archive is not used by V1.0.

## Environment variables

```bash
export SYNTHMIND_DATA_ROOT=/path/to/authorized/artifacts
export WORK_ROOT=/path/to/new/output
export PYTHON=/path/to/python
export DEVICE=cuda
```

All supported workflow steps read artifacts from `SYNTHMIND_DATA_ROOT` and
write new data, checkpoints and reports to `WORK_ROOT`.

## Profiles and minimum artifact levels

- `validate-full`: prepared data, selected weights, frozen candidates/samples,
  evaluation tools and audits.
- `validate-fast`: prepared data and selected weights; skips metric replay.
- `train`: final prepared Stage2/Stage3 datasets plus the imported Stage2 base
  candidate dependency.
- `rebuild-train`: relaxed/gold Stage2 inputs, Stage3 chem-checked source,
  schema and imported Stage2 base dependency.

The raw alignment and feature-building source is fully included, but historical
raw inputs may require source-specific options. The supported automatic
training profiles begin at the documented prepared checkpoints.
