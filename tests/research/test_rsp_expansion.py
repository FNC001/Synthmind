from __future__ import annotations

import unittest
from collections import Counter

import pandas as pd

from synpred.research.run_rsp_expansion import (
    ExpansionConfig,
    PrecursorLibrary,
    canonical_set_key,
    evaluate_gate,
    metric_dict,
    normalize_candidates,
    rank_variant,
)
from synpred.research.run_rsp_prune_recovery import PruneConfig, generate_prune_candidates


class RspExpansionTest(unittest.TestCase):
    def test_canonical_set_key_is_order_invariant(self) -> None:
        a = canonical_set_key([" TiO2 ", "SrCO3"])
        b = canonical_set_key(["SrCO3", "TiO2"])
        self.assertEqual(a, b)

    def test_rank_variant_keeps_fixed_budget_and_source(self) -> None:
        base = normalize_candidates(
            pd.DataFrame(
                [
                    {
                        "id": "s1",
                        "sample_index": 0,
                        "rank": 1,
                        "formula": "SrTiO3",
                        "reaction_method": "solid_state",
                        "true_precursors": '["SrCO3","TiO2"]',
                        "candidate_set": '["SrCO3","TiO2"]',
                        "candidate_source": "v4_base",
                        "total_score_v5": 10,
                    },
                    {
                        "id": "s1",
                        "sample_index": 0,
                        "rank": 2,
                        "formula": "SrTiO3",
                        "reaction_method": "solid_state",
                        "true_precursors": '["SrCO3","TiO2"]',
                        "candidate_set": '["SrO","TiO2"]',
                        "candidate_source": "v4_base",
                        "total_score_v5": 9,
                    },
                ]
            )
        )
        expansion = pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "sample_index": 0,
                    "id": "s1",
                    "formula": "SrTiO3",
                    "reaction_method": "solid_state",
                    "true_precursors": '["SrCO3","TiO2"]',
                    "candidate_set": '["Sr(NO3)2","TiO2"]',
                    "candidate_source": "rare",
                    "source_group": "rare",
                    "rank": 10**9,
                    "total_score_v5": 1.0,
                    "target_elements": {"Sr", "Ti"},
                    "candidate_elements": {"Sr", "Ti"},
                    "true_key": canonical_set_key(["SrCO3", "TiO2"]),
                    "candidate_key": canonical_set_key(["Sr(NO3)2", "TiO2"]),
                    "label_exact": 0,
                    "jaccard_label": 1 / 3,
                    "is_rare_reference": False,
                    "is_oov_reference": False,
                }
            ]
        )
        ranked = rank_variant(base, expansion, preserve_base_top=1, budgets=[2])
        self.assertLessEqual(len(ranked), 2)
        self.assertEqual(ranked.iloc[0]["source_group"], "base")
        self.assertIn("rare", set(ranked["source_group"]))

    def test_metrics_and_gate(self) -> None:
        base = normalize_candidates(
            pd.DataFrame(
                [
                    {
                        "id": "s1",
                        "sample_index": 0,
                        "rank": 1,
                        "formula": "SrTiO3",
                        "reaction_method": "solid_state",
                        "true_precursors": '["SrCO3","TiO2"]',
                        "candidate_set": '["SrO","TiO2"]',
                        "candidate_source": "base",
                    },
                    {
                        "id": "s1",
                        "sample_index": 0,
                        "rank": 2,
                        "formula": "SrTiO3",
                        "reaction_method": "solid_state",
                        "true_precursors": '["SrCO3","TiO2"]',
                        "candidate_set": '["SrCO3","TiO2"]',
                        "candidate_source": "base",
                    },
                ]
            )
        )
        ranked = rank_variant(base, pd.DataFrame(), preserve_base_top=2, budgets=[1, 2])
        metrics = metric_dict(ranked, [1, 2])
        self.assertEqual(metrics["exact@1"], 0.0)
        self.assertEqual(metrics["exact@1"], metrics["skeleton_oracle@1"])

        result = {
            "variants": {
                "rsp_v5_baseline": {"exact@1": 0.0, "skeleton_oracle@50": 0.0},
                "base_plus_family_rare_chemistry": {"exact@1": 0.0, "skeleton_oracle@50": 0.1},
            }
        }
        gate = evaluate_gate(result, [50], max_exact1_drop=0.005)
        self.assertTrue(gate["passed"])

    def test_prune_candidates_remove_one_label_without_budget_leak(self) -> None:
        base = normalize_candidates(
            pd.DataFrame(
                [
                    {
                        "id": "s1",
                        "sample_index": 0,
                        "rank": 1,
                        "formula": "SrTiO3",
                        "reaction_method": "solid_state",
                        "true_precursors": '["SrCO3","TiO2"]',
                        "candidate_set": '["SrCO3","TiO2","NH4F"]',
                        "candidate_source": "base",
                        "total_score_v5": 3.0,
                    }
                ]
            )
        )
        base["true_key"] = base["true_precursors"].map(lambda x: canonical_set_key([v.strip('"') for v in x.strip("[]").split(",")]))
        base["candidate_key"] = base["candidate_set"].map(lambda x: canonical_set_key([v.strip('"') for v in x.strip("[]").split(",")]))
        cfg = PruneConfig(
            output_dir="unused",  # type: ignore[arg-type]
            train_candidates="unused",  # type: ignore[arg-type]
            split_candidates="unused",  # type: ignore[arg-type]
            budgets=(50, 200, 500),
            seed_top_grid=(1,),
            preserve_base_top_grid=(1,),
            max_candidate_size=8,
            prune_score_offset=-0.02,
            max_exact1_drop=0.005,
            bootstrap_iterations=10,
            ci=0.95,
            seed=42,
            raw_config_path="unused",
        )
        pruned = generate_prune_candidates(base, seed_top=1, cfg=cfg)
        self.assertEqual(len(pruned), 3)
        self.assertIn("prune", set(pruned["source_group"]))
        self.assertTrue((pruned["candidate_source"] == "skeleton_prune").all())


if __name__ == "__main__":
    unittest.main()
