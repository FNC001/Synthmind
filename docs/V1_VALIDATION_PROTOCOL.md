# Synthmind V1.0 publication gate

The V1.0 tag may be pushed only when all of the following pass:

1. No credential or private-key pattern is present in tracked files.
2. Every tracked Python source compiles.
3. Every tracked shell script passes `bash -n`.
4. The V1.0 test suite executes 86/86 tests successfully.
5. `synthmind --version`, `check-data`, workflow `--help`, and dry-run work
   from an installed checkout.
6. `validate-fast` runs against the authorized full artifact root.
7. `validate-full` reproduces the frozen Stage2 candidate hash, Stage3
   ensemble and three validation metrics.
8. The authorized artifact root passes its internal 192,790-file manifest
   both before and after validation.
9. One-epoch Stage2 and Stage3 GPU training smoke tests save loadable model
   artifacts from the complete prepared datasets.
10. A real structure completes GPU inference and passes all 23 output,
    model-hash and sample-count checks.

Large data and model files are not part of the public Git tag and are checked
through the external artifact root.
