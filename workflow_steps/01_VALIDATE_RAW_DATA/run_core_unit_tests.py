#!/usr/bin/env python3
"""Run the complete synthmind chemistry and research unit-test suite."""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path


EXPECTED_CHEMISTRY = 68
EXPECTED_RESEARCH = 9


def discover(loader: unittest.TestLoader, start: Path) -> unittest.TestSuite:
    return loader.discover(start_dir=str(start), pattern="test_*.py")


def count_cases(suite: unittest.TestSuite) -> int:
    return suite.countTestCases()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, default=Path.cwd())
    args = parser.parse_args()
    core = args.core.resolve()
    os.chdir(core)
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))

    # Separate loaders mirror two independent discovery roots.  A single
    # loader retains its first top-level directory and would skip/fail the
    # sibling research tree because these historical test folders are not
    # Python packages.
    chemistry = discover(unittest.TestLoader(), Path("tests/chemistry"))
    research = discover(unittest.TestLoader(), Path("tests/research"))
    chemistry_count = count_cases(chemistry)
    research_count = count_cases(research)
    expected_total = EXPECTED_CHEMISTRY + EXPECTED_RESEARCH
    if chemistry_count != EXPECTED_CHEMISTRY or research_count != EXPECTED_RESEARCH:
        raise SystemExit(
            f"Unexpected discovery counts: chemistry={chemistry_count}, research={research_count}, "
            f"expected={EXPECTED_CHEMISTRY}+{EXPECTED_RESEARCH}"
        )
    suite = unittest.TestSuite([chemistry, research])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun != expected_total:
        raise SystemExit(f"Unexpected executed test count: {result.testsRun} != {expected_total}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
