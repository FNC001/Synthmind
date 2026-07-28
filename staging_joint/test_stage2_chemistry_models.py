from __future__ import annotations

import unittest

import torch

from training.family.train_stage2_autoregressive_set import (
    AutoregressiveSetGenerator,
    beam_decode_batch,
    order_invariant_teacher_loss,
)
from training.family.train_stage2_listwise_ranker import (
    ListwiseSetRanker,
    precursor_formula_features,
)


class ChemistryModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.names = ["LiCl", "NaCl", "NaBr", "Li2O", "Na2S"]
        self.chemistry = torch.from_numpy(precursor_formula_features(self.names))

    def test_relative_ranker_forward_is_finite(self) -> None:
        model = ListwiseSetRanker(
            x_dim=157,
            n_labels=len(self.names),
            hidden=32,
            blocks=1,
            dropout=0.0,
            max_set_len=3,
            label_chemistry=self.chemistry,
            query_chemistry_dim=147,
            use_relative_chemistry_features=True,
            candidate_transformer_heads=4,
            joint_transformer_layers=1,
            joint_transformer_heads=4,
        )
        x = torch.randn(2, 157)
        labels = torch.tensor([[[0, 1, 5], [2, 5, 5]], [[3, 4, 5], [0, 5, 5]]])
        scores = model(x, labels, torch.zeros(2, 2, 2))
        self.assertEqual(tuple(scores.shape), (2, 2))
        self.assertTrue(bool(torch.isfinite(scores).all()))

    def test_relative_ranker_requires_both_formula_sides(self) -> None:
        with self.assertRaisesRegex(ValueError, "relative chemistry"):
            ListwiseSetRanker(10, 5, 32, 1, 0.0, 3, use_relative_chemistry_features=True)

    def test_autoregressive_loss_backpropagates_and_decodes_unique_sets(self) -> None:
        model = AutoregressiveSetGenerator(157, 5, 3, 32, 1, 0.0, self.chemistry)
        x = torch.randn(3, 157)
        y = torch.tensor(
            [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.float32
        )
        lengths = y.sum(dim=1).long()
        loss, _, _ = order_invariant_teacher_loss(model, x, y, lengths, 0.0, 0.25)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        rows, scores = beam_decode_batch(model, x, beam_width=12, branch_factor=6)
        self.assertEqual(len(rows), len(x))
        for candidates, candidate_scores in zip(rows, scores):
            self.assertTrue(candidates)
            self.assertEqual(len(candidates), len(set(candidates)))
            self.assertEqual(len(candidates), len(candidate_scores))
            self.assertTrue(all(a >= b for a, b in zip(candidate_scores, candidate_scores[1:])))


if __name__ == "__main__":
    unittest.main()
