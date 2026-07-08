import unittest

from synthmind.research.schema import (
    CandidatePool,
    ConditionDistributionGenerator,
    GlobalRouteVerifier,
    RouteCandidate,
    RouteSkeletonProposer,
)


class SchemaTest(unittest.TestCase):
    def test_public_module_aliases_and_route_candidate_schema(self):
        self.assertEqual(RouteSkeletonProposer.legacy_alias, "stage2")
        self.assertEqual(ConditionDistributionGenerator.legacy_alias, "stage3")
        self.assertEqual(GlobalRouteVerifier.legacy_alias, "stage35")
        candidate = RouteCandidate(
            sample_id="s1",
            candidate_id="c1",
            method="solid_state",
            canonical_precursor_set=("BaCO3", "TiO2"),
        )
        pool = CandidatePool(pool_id="p1", split="val", candidates=(candidate,))
        self.assertEqual(pool.candidates[0].canonicalization_version, "canonicalization_v1")
