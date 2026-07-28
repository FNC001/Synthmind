from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from training.family.audit_stage2_group_macro import grouped_metrics
from training.family.audit_stage2_formula_group_headroom import group_report
from training.family.evaluate_stage2_family_template_oof import stable_group_fold
from training.family.build_stage2_final_refit_pack import refit_features
from training.family.build_stage2_protected_candidate_union import protected_union_row
from training.family.evaluate_stage2_canonical_chemistry import canonical_pattern_score, label_score
from training.family.evaluate_stage2_oof_score_calibrator import row_score_features
from training.family.evaluate_stage2_matscibert_set_generator import generate_candidate_sets
from training.family.evaluate_stage2_score_ensemble import (
    candidate_fingerprint,
    candidate_fingerprints,
    rowwise_zscore,
)
from training.family.evaluate_stage2_score_ensemble_grid import parse_mix_grid, trial_key

from training.family.evaluate_stage2_multi_expert_routing import (
    anion_group_signature,
    formula_element_count_bucket,
    parse_count_bins,
)
from training.family.evaluate_stage2_candidate_fusion import fuse_row, fuse_row_topk
from training.family.evaluate_stage2_unseen_label_rescore import unseen_rescore_row
from training.family.evaluate_stage2_oof_chemistry_rescore import (
    chemistry_features_for_candidate,
)
from training.family.evaluate_stage2_family_substitution import (
    all_group_mapping_variants,
    periodic_group_features,
)
from training.family.train_stage2_oof_expert_gate import overlap_features
from training.family.train_stage2_train_oof_multi_expert_router import (
    merge_expert_row,
    parse_expert_pair,
    select_experts,
)
from training.family.train_stage2_oof_candidate_stacker import (
    MatSciFeatureBuilder,
    append_matsci_features,
    build_row_candidates_and_features,
    formula_group_folds,
    merge_preserved_base_prefix,
    query_balance_weights,
    query_route_features,
    multilabel_candidate_features,
    source_rank_features,
)
from training.family.train_stage2_within_family_variant_ranker import family_slot_rerank
from training.family.train_stage2_oof_safe_slot_selector import (
    multi_slot_rows,
    safe_slot_merge,
    slot_rows,
)
from training.family.train_stage2_structured_energy_ranker import (
    _global_score_rerank_prepared,
    _global_safe_swap_prepared,
    build_training_pools,
    apply_fixed_trial,
    CandidateRegistry,
    PoolData,
    family_slot_rerank as structured_family_slot_rerank,
    label_structured_features,
    merge_candidate_sources as merge_structured_candidate_sources,
)
from training.family.train_stage2_autoregressive_set import (
    AutoregressiveSetGenerator,
    beam_decode_batch,
    order_invariant_teacher_loss,
    parse_family_filter,
)
from training.family.train_stage2_listwise_ranker import (
    balanced_row_weights,
    CandidateDataset,
    ListwiseSetRanker,
    precursor_formula_features,
    restrict_dataset_positive_rank,
)
from training.family.train_stage2_matscibert_cross_encoder import (
    build_multi_positive_supervision,
    candidate_rank_feature_maps,
    candidate_rank_features,
    evaluate_specialist_grid,
)
from training.family.train_stage2_lightgbm_ranker import (
    flatten_evaluation_pairs,
    flatten_training_pool,
)
from training.family.train_stage2_validation_meta_lambdarank import (
    parse_expert_source as parse_meta_expert_source,
    protected_expert_union,
    row_subset as meta_row_subset,
    score_sorted_rows,
)
from training.family.train_stage2_validation_miss_gate import (
    agreement_features,
    gated_merge,
)


class ChemistryModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.names = ["LiCl", "NaCl", "NaBr", "Li2O", "Na2S"]
        self.chemistry = torch.from_numpy(precursor_formula_features(self.names))

    def test_lightgbm_pool_flattening_preserves_query_groups(self) -> None:
        pairs = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
        mask = np.asarray([[True, True, False], [True, True, True]])
        query_indices = np.asarray([1, 0], dtype=np.int64)
        queries = np.asarray([[10.0], [20.0]], dtype=np.float32)
        features, labels, groups = flatten_training_pool(
            pairs, mask, query_indices, queries, include_query_features=True
        )
        self.assertEqual(features.shape, (5, 3))
        self.assertEqual(groups.tolist(), [2, 3])
        self.assertEqual(labels.tolist(), [1.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(features[:, -1].tolist(), [20.0, 20.0, 10.0, 10.0, 10.0])

    def test_lightgbm_evaluation_flattening_aligns_queries(self) -> None:
        pair = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        query_indices = np.asarray([1, 0, 1], dtype=np.int64)
        queries = np.asarray([[10.0], [20.0]], dtype=np.float32)
        features = flatten_evaluation_pairs(
            pair, query_indices, queries, include_query_features=True
        )
        self.assertEqual(features.tolist(), [[1.0, 20.0], [2.0, 10.0], [3.0, 20.0]])

    def test_validation_meta_union_preserves_base_and_adds_experts(self) -> None:
        rows = protected_expert_union(
            [[(0,), (1,), (2,)]],
            [[[(3,), (1,)]], [[(4,), (3,)]]],
            base_limit=2,
            expert_limit=2,
        )
        self.assertEqual(rows, [[(0,), (1,), (3,), (4,)]])
        name, path = parse_meta_expert_source("groupbal=/tmp/candidates.jsonl")
        self.assertEqual(name, "groupbal")
        self.assertTrue(path.endswith("/tmp/candidates.jsonl"))

    def test_validation_meta_row_subset_keeps_group_contiguity(self) -> None:
        matrix = np.arange(12, dtype=np.float32).reshape(6, 2)
        labels = np.asarray([1, 0, 0, 1, 0, 0], dtype=np.float32)
        features, selected_labels, groups = meta_row_subset(
            matrix, labels, [(0, 2), (2, 3), (3, 6)], [2, 0]
        )
        self.assertEqual(groups.tolist(), [3, 2])
        self.assertEqual(selected_labels.tolist(), [1.0, 0.0, 0.0, 1.0, 0.0])
        self.assertEqual(features[:, 0].tolist(), [6.0, 8.0, 10.0, 0.0, 2.0])

    def test_validation_meta_score_sorting_is_span_aligned(self) -> None:
        rows = [[(0,), (1,), (2,)], [(3,), (4,)]]
        sorted_rows = score_sorted_rows(
            rows,
            np.asarray([0.1, 0.7, 0.2, -1.0, 2.0], dtype=np.float32),
            [(0, 3), (3, 5)],
        )
        self.assertEqual(sorted_rows, [[(1,), (2,), (0,)], [(4,), (3,)]])

    def test_validation_miss_gate_features_capture_expert_disagreement(self) -> None:
        agreeing = agreement_features(
            [(0,), (1,)], [[(0,), (1,)], [(0,), (1,)]], limit=2
        )
        disagreeing = agreement_features(
            [(0,), (1,)], [[(2,), (3,)], [(4,), (5,)]], limit=2
        )
        self.assertEqual(agreeing.shape, disagreeing.shape)
        self.assertGreater(float(agreeing[6]), float(disagreeing[6]))
        self.assertGreater(float(agreeing[8]), float(disagreeing[8]))

    def test_validation_miss_gate_merge_protects_prefix_and_is_unique(self) -> None:
        base = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]
        ranked = [(20,), (21,), (0,), (22,), (23,), (24,), (25,), (26,), (27,), (28,)]
        merged = gated_merge(base, ranked, protected_prefix=3)
        self.assertEqual(merged[:3], base[:3])
        self.assertEqual(len(merged), len(set(merged)))
        self.assertEqual(merged[3], (20,))

    def test_protected_candidate_union_keeps_prefix_and_ranks_consensus(self) -> None:
        row = protected_union_row(
            [(0,), (1,), (2,), (3,)],
            [[(4,), (5,), (2,)], [(4,), (6,), (5,)]],
            protected_prefix=2,
            expert_limit=3,
            rrf_constant=10.0,
            output_limit=7,
        )
        self.assertEqual(row[:2], [(0,), (1,)])
        self.assertEqual(row[2], (4,))
        self.assertEqual(len(row), len(set(row)))
        self.assertIn((2,), row)

    def test_train_oof_multi_expert_router_parses_paired_sources(self) -> None:
        self.assertEqual(
            parse_expert_pair("base=train.jsonl::val.jsonl"),
            ("base", "train.jsonl", "val.jsonl"),
        )
        with self.assertRaisesRegex(ValueError, "expert_pair"):
            parse_expert_pair("base=train.jsonl")

    def test_train_oof_multi_expert_router_applies_risk_threshold(self) -> None:
        gain = np.asarray([[0.4, 0.2], [0.3, 0.35], [0.1, 0.2]], dtype=np.float32)
        loss = np.asarray([[0.1, 0.01], [0.2, 0.05], [0.1, 0.15]], dtype=np.float32)
        selected = select_experts(gain, loss, risk_weight=2.0, threshold=0.15)
        self.assertEqual(selected.tolist(), [1, 2, 0])

    def test_train_oof_multi_expert_router_preserves_base_prefix(self) -> None:
        base = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]
        expert = [(20,), (21,), (22,), (23,), (24,), (25,), (26,), (27,), (28,), (29,)]
        merged = merge_expert_row(base, expert, base_keep=7)
        self.assertEqual(merged[:7], base[:7])
        self.assertEqual(merged[7:10], expert[:3])
        self.assertEqual(len(merged), len(set(merged)))
        self.assertIn((7,), merged[10:])
        self.assertEqual(merge_expert_row(base, expert, base_keep=10), expert)

    def test_matscibert_specialist_preserves_inactive_rows(self) -> None:
        inactive = [(index,) for index in range(11)]
        active = [(index, 11) for index in range(11)]
        targets = [(0,), (10, 11)]
        raw_scores = np.asarray(
            list(range(11)) + list(range(11)), dtype=np.float32
        )
        trial, ranked, _ = evaluate_specialist_grid(
            targets,
            [inactive, active],
            raw_scores,
            [(0, 11), (11, 22)],
            ["same"] * 12,
            np.asarray([1], dtype=np.int64),
            alphas=(0.0,),
            protected_prefixes=(0,),
            minimum_gains=(0.0,),
        )
        self.assertEqual(ranked[0], inactive)
        self.assertIn(targets[1], ranked[1][:10])
        self.assertEqual(float(trial["exact_hit@10"]), 1.0)

    def test_matscibert_multi_positive_supervision_uses_train_routes_only(self) -> None:
        registry = CandidateRegistry()
        candidate_ids = [registry.id_for((index,)) for index in range(3)]
        pool = PoolData(
            query_indices=np.asarray([0, 1, 2], dtype=np.int64),
            candidate_ids=np.asarray(
                [
                    [candidate_ids[0], candidate_ids[1], candidate_ids[2]],
                    [candidate_ids[1], candidate_ids[0], candidate_ids[2]],
                    [candidate_ids[1], candidate_ids[0], candidate_ids[2]],
                ],
                dtype=np.int64,
            ),
            mask=np.ones((3, 3), dtype=bool),
            pair_features=np.zeros((3, 3, 1), dtype=np.float32),
            target_in_source_pool=0,
            rows_with_same_family_negative=3,
            rows_with_cross_family_negative=0,
        )
        meta = pd.DataFrame(
            {
                "family_signature_primary": ["G01", "G01", "G01"],
                "target_anion_elements": ['["O"]', '["S"]', '["O"]'],
                "synthesis_type": ["solid", "solid", "solid"],
                "source_dataset": ["a", "a", "a"],
            }
        )
        mask, audit = build_multi_positive_supervision(
            [(0,), (1,), (1,)],
            meta,
            pool,
            registry,
            route_kind="family",
            minimum_count=2,
            maximum_candidates=3,
        )
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(mask[0, 1]))
        self.assertFalse(bool(mask[0, 2]))
        self.assertEqual(int(audit["extra_positive_pairs"]), 1)

    def test_oof_calibrator_score_features_are_finite_and_rank_aware(self) -> None:
        features = row_score_features(np.asarray([3.0, 2.0, 1.0], dtype=np.float32))
        self.assertEqual(features.shape, (3, 6))
        self.assertTrue(bool(np.isfinite(features).all()))
        self.assertGreater(float(features[0, 4]), float(features[-1, 4]))

    def test_multi_slot_rows_only_replaces_low_scoring_unprotected_entries(self) -> None:
        base = [[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]]
        candidates = [[(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]]
        scores = [np.asarray([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 6.5], dtype=np.float32)]
        rows, switched = multi_slot_rows(base, candidates, scores, margin=1.0, protected=5)
        self.assertTrue(bool(switched[0]))
        self.assertEqual(rows[0][:5], base[0][:5])
        self.assertIn((10,), rows[0][:10])
        self.assertNotIn((9,), rows[0][:10])
        self.assertEqual(len(rows[0]), len(set(rows[0])))

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

    def test_chemistry_only_encoder_does_not_use_random_label_ids(self) -> None:
        model = ListwiseSetRanker(
            x_dim=10,
            n_labels=len(self.names),
            hidden=32,
            blocks=1,
            dropout=0.0,
            max_set_len=3,
            label_chemistry=self.chemistry,
            chemistry_only_label_encoder=True,
        ).eval()
        x = torch.randn(2, 10)
        labels = torch.tensor([[[0, 1, 5], [2, 5, 5]], [[3, 4, 5], [0, 5, 5]]])
        numeric = torch.zeros(2, 2, 2)
        before = model(x, labels, numeric)
        with torch.no_grad():
            model.label_embedding.weight.normal_(mean=100.0, std=20.0)
        after = model(x, labels, numeric)
        self.assertTrue(torch.allclose(before, after))

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

    def test_formula_complexity_gate_is_deterministic_and_observable(self) -> None:
        bins = parse_count_bins("4,2,3,3")
        self.assertEqual(bins, [2, 3, 4])
        self.assertEqual(formula_element_count_bucket("Li2O", bins), "E<=2")
        self.assertEqual(formula_element_count_bucket("LiFePO4", bins), "E<=4")
        self.assertEqual(formula_element_count_bucket("Na2LiFePO4Cl", bins), "E>4")
        with self.assertRaisesRegex(ValueError, "positive integers"):
            parse_count_bins("0,2")
        self.assertEqual(anion_group_signature('["O", "S"]'), "G16")
        self.assertEqual(anion_group_signature(["Cl", "Br"]), "G17")
        self.assertEqual(anion_group_signature(""), "A_NONE")

    def test_hard_case_expert_uses_only_honest_positive_rank_band(self) -> None:
        y = np.eye(3, dtype=np.float32)
        candidates = [
            [(0,), (1,), (2,)],
            [(0,), (2,), (1,)],
            [(0,), (2,), (1,)],
        ]
        dataset = CandidateDataset(
            np.zeros((3, 2), dtype=np.float32),
            y,
            candidates,
            [[0.0, -1.0, -2.0]] * 3,
            n_candidates=3,
            pool_limit=3,
            max_set_len=1,
            pad_id=3,
            training=True,
            seed=7,
        )
        restrict_dataset_positive_rank(dataset, minimum=2, maximum=2)
        self.assertEqual(dataset.row_indices, [2])

    def test_group_balanced_sampling_equalizes_formula_group_mass(self) -> None:
        weights = balanced_row_weights(
            [0, 1, 2, 3],
            ["TiO2", "TiO2", "TiO2", "CeO2"],
            ["G04", "G04", "G04", "LN"],
            group_power=1.0,
            family_power=0.0,
        )
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]), places=7)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=7)

    def test_final_refit_scaler_uses_trainval_and_preserves_family_bits(self) -> None:
        trainval = np.asarray([
            [1.0, 10.0, 1.0, 0.0],
            [3.0, 30.0, 0.0, 1.0],
        ], dtype=np.float32)
        test = np.asarray([[5.0, 50.0, 1.0, 1.0]], dtype=np.float32)
        train_scaled, test_scaled, mean, std = refit_features(
            trainval, test, family_feature_count=2
        )
        self.assertTrue(np.allclose(train_scaled[:, :2].mean(axis=0), 0.0))
        self.assertTrue(np.array_equal(train_scaled[:, 2:], trainval[:, 2:]))
        self.assertTrue(np.array_equal(test_scaled[:, 2:], test[:, 2:]))
        self.assertTrue(np.allclose(mean, [2.0, 20.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(std, [1.0, 10.0, 1.0, 1.0]))

    def test_base_prefix_merge_preserves_trusted_candidates_without_duplicates(self) -> None:
        base = [(0,), (1,), (2,), (3,)]
        ranked = [(2,), (4,), (1,), (5,)]
        merged = merge_preserved_base_prefix(base, ranked, prefix=2)
        self.assertEqual(merged, [(0,), (1,), (2,), (4,), (5,), (3,)])
        self.assertEqual(len(merged), len(set(merged)))

    def test_family_slot_rerank_preserves_family_sequence(self) -> None:
        label_families = ["A", "A", "B", "B", "C"]
        base = [(0,), (2,), (1,), (3,), (4,)]
        scores = np.asarray([1.0, 1.0, 10.0, 9.0, 0.0], dtype=np.float32)
        reranked = family_slot_rerank(
            base,
            scores,
            label_families,
            slate_size=4,
            protected_prefix=0,
        )
        before = [label_families[candidate[0]] for candidate in base[:4]]
        after = [label_families[candidate[0]] for candidate in reranked[:4]]
        self.assertEqual(after, before)
        self.assertEqual(reranked[:4], [(1,), (3,), (0,), (2,)])

    def test_safe_slot_merge_changes_only_the_requested_slot(self) -> None:
        base = [(index,) for index in range(12)]
        merged = safe_slot_merge(base, (99,), [(99,), (7,), (13,)], protected=9)
        self.assertEqual(merged[:9], base[:9])
        self.assertEqual(merged[9], (99,))
        self.assertNotIn((9,), merged[:10])
        self.assertEqual(len(merged), len(set(merged)))

    def test_safe_slot_margin_falls_back_to_the_base_tenth_candidate(self) -> None:
        base = [[(index,) for index in range(10)]]
        candidates = [[(9,), (99,)]]
        scores = [np.asarray([0.60, 0.70], dtype=np.float32)]
        switched_rows, switched = slot_rows(base, candidates, scores, margin=0.0, protected=9)
        protected_rows, protected = slot_rows(base, candidates, scores, margin=0.2, protected=9)
        self.assertEqual(switched_rows[0][9], (99,))
        self.assertTrue(bool(switched[0]))
        self.assertEqual(protected_rows[0][9], (9,))
        self.assertFalse(bool(protected[0]))

    def test_structured_energy_features_support_unseen_formula_labels(self) -> None:
        features, audit = label_structured_features(["LiCl", "Na2S", "not-a-formula"])
        self.assertEqual(features.shape[0], 3)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(audit["labels"], 3)
        self.assertGreaterEqual(audit["composition_parsed"], 2)
        self.assertGreater(float(features[0].sum()), 0.0)
        self.assertGreater(float(features[1].sum()), 0.0)

    def test_structured_energy_rerank_changes_variants_not_family_slots(self) -> None:
        label_families = ["G01", "G01", "G07", "G07", "G13"]
        base = [(0,), (2,), (1,), (3,), (4,)]
        scores = np.asarray([0.0, 0.0, 9.0, 8.0, 0.0], dtype=np.float32)
        ranked = structured_family_slot_rerank(
            base, scores, label_families, protected_prefix=0, slate_size=4
        )
        self.assertEqual(ranked[:4], [(1,), (3,), (0,), (2,)])
        self.assertEqual(
            [label_families[row[0]] for row in ranked[:4]],
            [label_families[row[0]] for row in base[:4]],
        )
        protected = structured_family_slot_rerank(
            base,
            scores,
            label_families,
            protected_prefix=0,
            slate_size=4,
            minimum_gain=10.0,
        )
        self.assertEqual(protected[:4], base[:4])

    def test_structured_global_rerank_can_reallocate_family_slots(self) -> None:
        base = [(0,), (1,), (2,), (3,), (4,)]
        scores = np.asarray([0.0, 0.1, 0.2, 5.0, 4.0], dtype=np.float32)
        ranked = _global_score_rerank_prepared(
            base,
            scores,
            protected_prefix=1,
            slate_size=3,
            candidate_window=5,
            append_remaining=True,
        )
        self.assertEqual(ranked[:3], [(0,), (3,), (4,)])
        self.assertEqual(set(ranked), set(base))

    def test_structured_global_safe_swap_respects_margin(self) -> None:
        base = [(0,), (1,), (2,), (3,), (4,)]
        scores = np.asarray([5.0, 4.0, 1.0, 1.4, 1.2], dtype=np.float32)
        swapped = _global_safe_swap_prepared(
            base,
            scores,
            protected_prefix=1,
            slate_size=3,
            candidate_window=5,
            minimum_gain=0.25,
            append_remaining=False,
        )
        protected = _global_safe_swap_prepared(
            base,
            scores,
            protected_prefix=1,
            slate_size=3,
            candidate_window=5,
            minimum_gain=0.5,
            append_remaining=False,
        )
        self.assertEqual(swapped, [(0,), (1,), (3,)])
        self.assertEqual(protected, base[:3])

    def test_score_ensemble_normalizes_each_candidate_row(self) -> None:
        normalized = rowwise_zscore(
            np.asarray([1.0, 2.0, 3.0, 10.0, 14.0], dtype=np.float32),
            [(0, 3), (3, 5)],
        )
        self.assertAlmostEqual(float(normalized[:3].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[3:].mean()), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[:3].std()), 1.0, places=6)
        self.assertAlmostEqual(float(normalized[3:].std()), 1.0, places=6)

    def test_score_ensemble_grid_parses_and_prioritizes_exact_top10(self) -> None:
        self.assertEqual(parse_mix_grid("0,0.5,0.5,1"), (0.0, 0.5, 1.0))
        better_top10 = {
            "exact_hit@10": 0.8,
            "exact_hit@5": 0.5,
            "exact_hit@1": 0.2,
            "lost_hits_vs_base": 3,
        }
        better_top5 = {
            "exact_hit@10": 0.79,
            "exact_hit@5": 0.7,
            "exact_hit@1": 0.4,
            "lost_hits_vs_base": 0,
        }
        self.assertGreater(
            trial_key(better_top10, 0.5), trial_key(better_top5, 0.5)
        )
        with self.assertRaises(ValueError):
            parse_mix_grid("-0.1,0.5")

    def test_score_candidate_fingerprints_are_order_sensitive(self) -> None:
        first = candidate_fingerprints([[(1, 2), (3,)]])
        second = candidate_fingerprints([[(3,), (1, 2)]])
        self.assertEqual(first[0], candidate_fingerprint((1, 2)))
        self.assertFalse(np.array_equal(first, second))

    def test_formula_group_headroom_counts_recoverable_route_frequency(self) -> None:
        frame = pd.DataFrame(
            {
                "group": ["g", "g", "g"],
                "formula": ["LiCl", "LiCl", "LiCl"],
                "family": ["G01", "G01", "G01"],
            }
        )
        targets = [(0,), (1,), (1,)]
        rows = [[(0,), (2,)], [(0,), (2,)], [(0,), (2,)]]
        report = group_report(frame, targets, rows, ["LiCl", "NaCl", "KCl"], 1)
        self.assertEqual(report[0]["current_hits"], 1)
        self.assertEqual(report[0]["oracle_hits"], 2)
        self.assertEqual(report[0]["potential_extra_hits"], 1)

    def test_cross_encoder_rank_features_encode_boundary_and_missing_candidates(self) -> None:
        maps = candidate_rank_feature_maps([[(1,), (2,), (3,)]], limit=100)
        first = candidate_rank_features(maps, 0, (1,))
        third = candidate_rank_features(maps, 0, (3,))
        missing = candidate_rank_features(maps, 0, (9,))
        self.assertGreater(float(first[0]), float(third[0]))
        self.assertEqual(float(first[2]), 1.0)
        np.testing.assert_array_equal(missing, np.zeros(6, dtype=np.float32))

    def test_family_template_oof_keeps_identical_formula_groups_together(self) -> None:
        first = stable_group_fold("formula-group-a", folds=5, seed=7)
        self.assertEqual(first, stable_group_fold("formula-group-a", folds=5, seed=7))
        self.assertTrue(0 <= first < 5)

    def test_fixed_slate_trial_replays_without_parameter_search(self) -> None:
        targets = [(3,)]
        base = [[(0,), (1,), (2,), (3,)]]
        trial, rows = apply_fixed_trial(
            targets,
            base,
            np.asarray([3.0, 2.0, 1.0, 4.0], dtype=np.float32),
            [(0, 4)],
            ["G01", "G02", "G03", "G04"],
            {
                "strategy": "global_score",
                "alpha": 0.0,
                "protected_prefix": 1,
                "candidate_window": 4,
            },
        )
        self.assertEqual(rows[0][0], (0,))
        self.assertIn((3,), rows[0][:3])
        self.assertEqual(trial["selection_mode"], "fixed_from_validation")

    def test_structured_pool_can_mix_same_and_cross_family_negatives(self) -> None:
        registry = CandidateRegistry()
        pool = build_training_pools(
            targets=[(0,), (1,)],
            meta=pd.DataFrame({"formula": ["LiCl", "NaCl"]}),
            source_rows=[[[(0,), (2,)], [(1,), (2,)]]],
            label_families=["G01", "G01", "G16"],
            pool_limit=3,
            source_union_limit=5,
            registry=registry,
            feature_builder=lambda row, candidate: np.asarray(
                [float(row), float(len(candidate))], dtype=np.float32
            ),
            seed=7,
            cross_family_negatives=1,
        )
        self.assertEqual(len(pool.query_indices), 2)
        self.assertEqual(pool.rows_with_cross_family_negative, 2)
        decoded = [
            [registry.keys[int(candidate_id)] for candidate_id in row]
            for row in pool.candidate_ids
        ]
        self.assertTrue(all((2,) in row for row in decoded))

    def test_structured_candidate_union_prefers_best_rank_then_votes(self) -> None:
        sources = [
            [[(0,), (1,), (2,)]],
            [[(1,), (0,), (3,)]],
        ]
        merged = merge_structured_candidate_sources(sources, row_index=0, limit=4)
        self.assertEqual(merged[:2], [(0,), (1,)])
        self.assertEqual(set(merged), {(0,), (1,), (2,), (3,)})

    def test_canonical_chemistry_prior_rewards_carbohydrates_and_clean_salts(self) -> None:
        self.assertGreater(
            canonical_pattern_score("C12H22O11", {"C"}),
            canonical_pattern_score("CO(NH2)2", {"C"}),
        )
        frequency = np.asarray([1.0, 100.0], dtype=np.float32)
        clean = label_score(
            0, "Ce(NO3)4", {"Ce", "N", "O"}, {"Ce"}, {"Ce"}, {"O"}, frequency
        )
        extra_metal = label_score(
            1, "CeSnO3", {"Ce", "Sn", "O"}, {"Ce", "Sn"}, {"Ce"}, {"O"}, frequency
        )
        self.assertGreater(clean, extra_metal)

    def test_group_macro_metric_prevents_large_formula_domination(self) -> None:
        metrics = grouped_metrics(
            [(0,), (0,), (0,), (1,)],
            [[(0,)], [(0,)], [(0,)], [(2,)]],
            ["TiO2", "TiO2", "TiO2", "CeO2"],
            ["G04", "G04", "G04", "LN"],
        )
        self.assertEqual(metrics["row_exact_hit@1"], 0.75)
        self.assertEqual(metrics["group_macro_exact_hit@1"], 0.5)

    def test_autoregressive_family_filter_is_stable(self) -> None:
        self.assertEqual(parse_family_filter("G11,G07,G11"), ["G07", "G11"])

    def test_rrf_topk_matches_full_fusion_prefix(self) -> None:
        rows = [
            [(0,), (1,), (2,), (3,)],
            [(2,), (0,), (4,), (1,)],
            [(4,), (3,), (0,), (5,)],
        ]
        weights = [1.0, 0.5, 2.0]
        for constant in (1.0, 10.0, 100.0):
            for k in (0, 1, 3, 20):
                self.assertEqual(
                    fuse_row_topk(rows, weights, constant, k),
                    fuse_row(rows, weights, constant)[:k],
                )

    def test_unseen_label_rescore_uses_training_observable_frequency(self) -> None:
        candidates = [(0,), (1,), (2,), (0, 2)]
        train_seen = np.asarray([True, True, False])
        self.assertEqual(
            unseen_rescore_row(candidates, train_seen, unseen_bonus=2.0),
            [(2,), (0, 2), (0,), (1,)],
        )
        self.assertEqual(
            unseen_rescore_row(candidates, train_seen, unseen_bonus=0.0),
            candidates,
        )

    def test_oof_chemistry_features_reward_family_coverage(self) -> None:
        names = ["LiCl", "NaBr", "Fe2O3"]
        elements = [{"Li", "Cl"}, {"Na", "Br"}, {"Fe", "O"}]
        groups = [{1, 17}, {1, 17}, {8, 16}]
        metals = [{"Li"}, {"Na"}, {"Fe"}]
        features = chemistry_features_for_candidate(
            (0,), {"Li"}, {"O"}, elements, groups, metals,
            np.asarray([True, True, True]), expected_length=1,
        )
        self.assertEqual(names[0], "LiCl")
        self.assertEqual(float(features[0]), 1.0)
        self.assertEqual(float(features[1]), 1.0)
        self.assertEqual(float(features[2]), 0.0)
        self.assertEqual(float(features[5]), 0.0)

    def test_periodic_group_retrieval_collapses_same_family_elements(self) -> None:
        values = periodic_group_features(["Li2O", "Na2S", "Fe2O3"])
        self.assertTrue(np.allclose(values[0], values[1]))
        li_na_similarity = float(np.dot(values[0], values[1]) / (
            np.linalg.norm(values[0]) * np.linalg.norm(values[1])
        ))
        li_fe_similarity = float(np.dot(values[0], values[2]) / (
            np.linalg.norm(values[0]) * np.linalg.norm(values[2])
        ))
        self.assertGreater(li_na_similarity, li_fe_similarity)

    def test_all_group_substitution_maps_cations_and_anions(self) -> None:
        self.assertIn(
            {"Li": "Na", "O": "S"},
            all_group_mapping_variants("Li2O", "Na2S"),
        )
        self.assertIn(
            {"Li": "Na", "Cl": "Br"},
            all_group_mapping_variants("LiCl", "NaBr"),
        )

    def test_expert_gate_agreement_features_are_prediction_only(self) -> None:
        experts = [
            [[(0,), (1,), (2,)]],
            [[(0,), (2,), (3,)]],
        ]
        features = overlap_features(experts)
        self.assertEqual(features.shape, (1, 14))
        self.assertEqual(float(features[0, 0]), 1.0)
        self.assertGreater(float(features[0, 4]), 0.0)

    def test_candidate_stacker_uses_source_ranks_and_chemistry(self) -> None:
        self.assertEqual(source_rank_features(None), [0.0, 0.0, 0.0])
        rows, features = build_row_candidates_and_features(
            [[(0,), (1,)], [(1,), (2,)]],
            {"Li"}, {"O"},
            [{"Li", "Cl"}, {"Na", "Br"}, {"Fe", "O"}],
            [{1, 17}, {1, 17}, {8, 16}],
            [{"Li"}, {"Na"}, {"Fe"}],
            np.asarray([True, True, True]),
            expected_length=1,
            union_limit=10,
        )
        self.assertEqual(rows, [(1,), (0,), (2,)])
        self.assertEqual(features.shape, (3, 57))
        self.assertGreater(float(features[0, 0]), 0.0)
        route = query_route_features({"Li", "Na"}, {"O", "S"})
        self.assertEqual(float(route[0]), 1.0)
        self.assertEqual(float(route[18 + 15]), 1.0)

    def test_candidate_stacker_formula_folds_are_group_disjoint(self) -> None:
        groups = np.asarray(["A", "A", "A", "B", "B", "C", "D", "E"], dtype=object)
        folds = formula_group_folds(groups, n_splits=3, seed=17)
        covered = []
        for train_rows, query_rows in folds:
            self.assertFalse(set(groups[train_rows]) & set(groups[query_rows]))
            covered.extend(query_rows.tolist())
        self.assertEqual(sorted(covered), list(range(len(groups))))

    def test_candidate_stacker_group_weights_equalize_query_mass(self) -> None:
        weights = query_balance_weights(
            [0, 1, 2, 3], ["A", "A", "A", "B"], power=1.0
        )
        self.assertAlmostEqual(float(weights[:3].sum()), float(weights[3]), places=6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_matscibert_features_are_train_fitted_and_candidate_aligned(self) -> None:
        rng = np.random.default_rng(11)
        label_views = rng.normal(size=(5, 12)).astype(np.float32)
        train_query_views = rng.normal(size=(8, 12)).astype(np.float32)
        train_y = np.zeros((8, 5), dtype=np.float32)
        train_y[np.arange(8), np.arange(8) % 5] = 1.0
        builder = MatSciFeatureBuilder(
            label_views, train_query_views, train_y,
            components=4, ridge_alpha=1.0, seed=11,
        )
        direct, projected = builder.transform_queries(train_query_views[:1])
        base = np.ones((2, 3), dtype=np.float32)
        combined = append_matsci_features(
            base, [(0,), (1, 2)], builder, direct[0], projected[0]
        )
        self.assertEqual(builder.feature_dim, 29)
        self.assertEqual(combined.shape, (2, 32))
        self.assertTrue(np.isfinite(combined).all())

    def test_matscibert_multilabel_features_reward_high_ranked_sets(self) -> None:
        features = multilabel_candidate_features(
            [(0, 1), (2, 3)], np.asarray([5.0, 4.0, -2.0, -3.0], dtype=np.float32)
        )
        self.assertEqual(features.shape, (2, 20))
        self.assertGreater(float(features[0, 3]), float(features[1, 3]))
        self.assertEqual(float(features[0, 15]), 1.0)

    def test_matscibert_set_generator_is_unique_and_bottleneck_ranked(self) -> None:
        rows, scores = generate_candidate_sets(
            np.asarray([5.0, 4.0, 1.0, -1.0], dtype=np.float32),
            top_labels=4, max_set_length=2, limit=10,
        )
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(rows[0], (0,))
        self.assertTrue(all(a >= b for a, b in zip(scores, scores[1:])))


if __name__ == "__main__":
    unittest.main()
