import pandas as pd
import unittest

from synpred.research.attribution import attribute_candidates, summarize_attribution


class AttributionTest(unittest.TestCase):
    def test_attribution_success_and_conservation(self):
        df = pd.DataFrame(
            [
                {"sample_id": "s1", "final_rank": 1, "precursor_exact_if_eval": 1, "relaxed_condition_hit_if_eval": 1, "relaxed_route_hit_if_eval": 1},
                {"sample_id": "s2", "final_rank": 1, "precursor_exact_if_eval": 0, "relaxed_condition_hit_if_eval": 1, "relaxed_route_hit_if_eval": 0},
                {"sample_id": "s3", "final_rank": 1, "precursor_exact_if_eval": 1, "relaxed_condition_hit_if_eval": 0, "relaxed_route_hit_if_eval": 0},
                {"sample_id": "s4", "final_rank": 1, "precursor_exact_if_eval": 1, "relaxed_condition_hit_if_eval": 1, "relaxed_route_hit_if_eval": 0},
                {"sample_id": "s4", "final_rank": 2, "precursor_exact_if_eval": 1, "relaxed_condition_hit_if_eval": 1, "relaxed_route_hit_if_eval": 1},
            ]
        )
        out = attribute_candidates(df)
        self.assertEqual(len(out), 4)
        self.assertEqual(set(out["attribution"]), {"success", "skeleton_miss", "condition_miss", "ranking_miss"})
        summary = summarize_attribution(out)
        self.assertEqual(int(summary["count"].sum()), 4)
