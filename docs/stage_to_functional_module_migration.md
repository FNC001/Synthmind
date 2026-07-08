# Stage to Functional Module Migration

SynPred now uses functional module names in new research-facing code and reports:

| Legacy name | New ID | Public class | Role |
|---|---|---|---|
| Stage2 | RSP | `RouteSkeletonProposer` | proposes method/precursor skeletons |
| Stage3 | CDG | `ConditionDistributionGenerator` | generates condition distributions |
| Stage35 | GRV | `GlobalRouteVerifier` | verifies and ranks full route candidates |

Legacy names remain valid compatibility aliases in historical scripts, paths, and checkpoints. New public configs, metric registries, reports, and model cards should use RSP/CDG/GRV. Bulk destructive renaming is intentionally avoided.

Versioning rules:

- `model_id`, `run_id`, `split`, `protocol`, `metric_id`, `candidate_budget`, and `canonicalization_version` are separate fields.
- Metric IDs must not include model versions, run IDs, or dates.
- Top200/Top500 coverage may be reported as appendix candidate coverage, not as primary route-specification accuracy.
- `usable` is operational validity, not reference-match accuracy.
