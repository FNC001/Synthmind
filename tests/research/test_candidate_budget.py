import pandas as pd
import unittest

from synthmind.research.candidate_pool import BudgetConfig, dedupe_and_budget


class CandidateBudgetTest(unittest.TestCase):
    def test_dedupe_happens_before_budget(self):
        df = pd.DataFrame(
            [
                {"sample_id": "s1", "candidate_id": "a", "predicted_method": "solid_state", "canonical_precursor_set": "A|B", "temperature": 700, "time": 12, "atmosphere": "air", "solvent": "", "final_rank": 1},
                {"sample_id": "s1", "candidate_id": "b", "predicted_method": "solid_state", "canonical_precursor_set": "A|B", "temperature": 700, "time": 12, "atmosphere": "air", "solvent": "", "final_rank": 2},
                {"sample_id": "s1", "candidate_id": "c", "predicted_method": "solid_state", "canonical_precursor_set": "A|C", "temperature": 700, "time": 12, "atmosphere": "air", "solvent": "", "final_rank": 3},
            ]
        )
        out = dedupe_and_budget(df, BudgetConfig("test", route_budget=2, skeleton_budget=2))
        self.assertEqual(list(out["candidate_id"]), ["a", "c"])
