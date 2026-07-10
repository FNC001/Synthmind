import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import patch

from pipeline.core.runner import PipelineRunner
from pipeline.core import steps_stage2
from pipeline.core.steps_reliability import apply_v43_template_ranker_if_enabled
from pipeline.run_pipeline import select_final_recommended_routes


class Stage2RegressionTest(unittest.TestCase):
    def test_public_config_keys_take_precedence_and_legacy_keys_still_work(self):
        public = SimpleNamespace(
            cfg={
                "composition_constraint": {"min_coverage": 0.75},
                "stage2": {"composition_constraint_min_coverage": 0.25},
            }
        )
        legacy = SimpleNamespace(
            cfg={"stage2": {"composition_constraint_min_coverage": 0.25}}
        )

        self.assertEqual(
            steps_stage2._cfg_get_compat(
                public,
                "composition_constraint.min_coverage",
                "stage2.composition_constraint_min_coverage",
                0.0,
            ),
            0.75,
        )
        self.assertEqual(
            steps_stage2._cfg_get_compat(
                legacy,
                "composition_constraint.min_coverage",
                "stage2.composition_constraint_min_coverage",
                0.0,
            ),
            0.25,
        )

    def test_stage2_commands_use_the_current_python_executable(self):
        with patch.object(steps_stage2.subprocess, "run") as run:
            steps_stage2._run_cmd(["python", "child.py", "--flag"])

        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:], ["child.py", "--flag"])
        self.assertTrue(run.call_args.kwargs["check"])


class GraphEmbeddingRegressionTest(unittest.TestCase):
    def test_nested_graph_commands_use_the_current_python_executable(self):
        module_path = (
            Path(__file__).parents[1]
            / "pipeline"
            / "core"
            / "03_build_infer_graph_embeddings_chgnet.py"
        )
        spec = importlib.util.spec_from_file_location("graph_embedding_pipeline", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            infer_jsonl = root / "infer.jsonl"
            vocab_json = root / "precursor_vocab.json"
            work_dir = root / "work"
            infer_jsonl.write_text(
                json.dumps({"sample_id": "s1", "formula": "SrTiO3"}) + "\n",
                encoding="utf-8",
            )
            vocab_json.write_text("{}\n", encoding="utf-8")
            commands = []

            def fake_run(command):
                commands.append(command)
                if command[1].endswith("export_chgnet_stage2_embeddings.py"):
                    output = work_dir / "chgnet_embed" / "stage2_test_graph_embed.csv"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text("sample_id,value\ns1,1\n", encoding="utf-8")

            argv = [
                str(module_path),
                "--infer_jsonl",
                str(infer_jsonl),
                "--work_dir",
                str(work_dir),
                "--project_root",
                str(Path(__file__).parents[1]),
                "--precursor_vocab_json",
                str(vocab_json),
            ]
            with patch.object(module, "run_command", side_effect=fake_run), patch.object(
                sys, "argv", argv
            ):
                module.main()

            self.assertEqual(len(commands), 2)
            self.assertTrue(all(command[0] == sys.executable for command in commands))


class RestoreRegressionTest(unittest.TestCase):
    def test_lgbm_output_is_restored_as_the_downstream_stage3_input(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_dir = root / "work"
            out_dir = root / "out"
            lgbm_csv = (
                out_dir
                / "stage3_condition_predictions_lgbm"
                / "test_candidates_flat.csv"
            )
            lgbm_csv.parent.mkdir(parents=True)
            lgbm_csv.write_text("sample_id,value\ns1,1\n", encoding="utf-8")

            runner = PipelineRunner(
                {
                    "project_root": str(root),
                    "paths": {
                        "work_dir": str(work_dir),
                        "out_dir": str(out_dir),
                    },
                }
            )
            runner.restore_existing_outputs()

            self.assertEqual(runner.outputs["lgbm_flat_csv"], str(lgbm_csv))
            self.assertEqual(runner.outputs["flow_flat_csv"], str(lgbm_csv))


class FinalSelectionRegressionTest(unittest.TestCase):
    def test_safe_strict_output_has_priority_over_template_only_output(self):
        with TemporaryDirectory() as tmp:
            route_out_dir = Path(tmp)
            safe_csv = route_out_dir / "safe.csv"
            template_csv = route_out_dir / "template.csv"
            safe_csv.write_text("kind\nsafe\n", encoding="utf-8")
            template_csv.write_text("kind\ntemplate\n", encoding="utf-8")

            runner = SimpleNamespace(
                out_dir=route_out_dir,
                outputs={
                    "route_out_dir": str(route_out_dir),
                    "final_top_routes_v43_safe_strict_reranked_csv": str(safe_csv),
                    "final_top_routes_v43_template_chemonly_reranked_csv": str(
                        template_csv
                    ),
                },
            )

            select_final_recommended_routes(runner)

            self.assertEqual(
                runner.outputs["final_recommended_routes_source"],
                "stage35_v43_safe_strict",
            )
            self.assertEqual(
                (route_out_dir / "final_recommended_routes.csv").read_text(
                    encoding="utf-8"
                ),
                safe_csv.read_text(encoding="utf-8"),
            )


class V43ConfigRegressionTest(unittest.TestCase):
    def test_public_v43_paths_and_output_names_are_honored(self):
        with TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            script_dir = project_root / "custom_scripts"
            route_out_dir = project_root / "routes"
            script_dir.mkdir()
            route_out_dir.mkdir()

            script = script_dir / "04_apply_v43_template_pairwise_ranker_chemonly.py"
            model = project_root / "models" / "ranker.joblib"
            feature_cols = project_root / "models" / "features.json"
            input_csv = route_out_dir / "custom_features.csv"
            model.parent.mkdir()
            for path in [script, model, feature_cols]:
                path.write_text("placeholder\n", encoding="utf-8")
            input_csv.write_text("sample_id,value\ns1,1\n", encoding="utf-8")

            commands = []
            runner = SimpleNamespace(
                outputs={"route_out_dir": str(route_out_dir)},
                step_enabled=lambda name: True,
                run=lambda command: commands.append(command),
            )
            cfg = {
                "project_root": str(project_root),
                "stage35_v43": {
                    "enabled": True,
                    "script_dir": "custom_scripts",
                    "model_path": "models/ranker.joblib",
                    "feature_cols_json": "models/features.json",
                    "feature_csv_name": "custom_features.csv",
                    "output_csv_name": "custom_output.csv",
                    "output_md_name": "custom_output.md",
                    "summary_json_name": "custom_summary.json",
                    "top_n": 17,
                },
            }

            apply_v43_template_ranker_if_enabled(runner, cfg)

            self.assertEqual(len(commands), 1)
            command = [str(value) for value in commands[0]]
            self.assertEqual(command[1], str(script))
            self.assertEqual(command[command.index("--model_path") + 1], str(model))
            self.assertEqual(
                command[command.index("--feature_cols_json") + 1], str(feature_cols)
            )
            self.assertEqual(
                command[command.index("--output_csv") + 1],
                str(route_out_dir / "custom_output.csv"),
            )
            self.assertEqual(command[command.index("--top_n") + 1], "17")


if __name__ == "__main__":
    unittest.main()
