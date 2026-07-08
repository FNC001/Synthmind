# Evaluation Protocol v1

This protocol separates reference-match accuracy from operational validity.

## Protocols

`complete_case` evaluates joint route matching only on samples where all required route fields are available and comparable. Every complete-case metric must report sample count `N`.

`available_field` evaluates each field only on samples where that field has a label. These results must not be compressed into a joint route accuracy by ignoring missing fields.

`operational_validity` measures parseability, element coverage, chemistry-rule pass status, field completeness, and confidence thresholds. It is not reference-match accuracy.

## Main K Values

Main text reports `K = 1, 5, 10`. RSP may additionally report `recall@50`. Top200 and Top500 are appendix candidate-coverage curves only.

## Legacy Compatibility

Legacy missing-aware metrics are retained for reproduction only. They are not directly comparable with `complete_case` route-reference metrics.

The current legacy strict-comparable relaxed condition implementation uses:

- temperature absolute error <= 200 C;
- time absolute error <= 48 h;
- exact lower-case atmosphere match when the reference atmosphere is known.

The current legacy strict-comparable strict condition implementation uses:

- temperature absolute error <= 100 C;
- time absolute error <= 24 h;
- exact lower-case atmosphere match when the reference atmosphere is known.

Any change to these tolerances creates a new protocol version.

