"""Release-level regression tests for the public Synthmind V1.0 interface."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import synthmind
from synthmind.release_layout import check_data_root, required_paths


REPO_ROOT = Path(__file__).resolve().parents[1]


class V1ReleaseTests(unittest.TestCase):
    def test_public_version_is_v1(self) -> None:
        self.assertEqual(synthmind.__version__, "1.0.0")
        self.assertEqual((REPO_ROOT / "VERSION").read_text().strip(), "1.0.0")

    def test_external_data_layout_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in required_paths("validate-full"):
                path = root / relative
                if Path(relative).suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                else:
                    path.mkdir(parents=True, exist_ok=True)

            report = check_data_root(root, profile="validate-full")
            self.assertTrue(report["passed"])
            self.assertEqual(report["present"], report["required"])

            metric = root / (
                "09_ACCURACY_EVALUATION/03_THREE_METRICS/"
                "final_three_metrics.json"
            )
            metric.unlink()
            report = check_data_root(root, profile="validate-full")
            self.assertFalse(report["passed"])
            self.assertEqual(report["present"], report["required"] - 1)

    def test_workflow_dry_run_covers_ordered_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            work_root = root / "work"
            data_root.mkdir()
            process = subprocess.run(
                [
                    "bash",
                    str(REPO_ROOT / "scripts" / "run_v1_workflow.sh"),
                    "--data-root",
                    str(data_root),
                    "--work-root",
                    str(work_root),
                    "--python",
                    sys.executable,
                    "--profile",
                    "validate-full",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            output = process.stdout
            expected = [
                "01_VALIDATE_RAW_DATA",
                "02_ALIGN_SYNTHESIS_STRUCTURES",
                "03_CLEAN_AND_STRATIFY",
                "04_BUILD_FEATURES",
                "05_BUILD_STAGE2_DATASET",
                "06_TRAIN_STAGE2",
                "07_BUILD_STAGE3_DATASET",
                "08_TRAIN_STAGE3",
                "09_EVALUATE_THREE_METRICS",
                "10_INFER_END_TO_END",
            ]
            positions = [output.index(step) for step in expected]
            self.assertEqual(positions, sorted(positions))
            self.assertIn("completed", output)


if __name__ == "__main__":
    unittest.main()
