from pathlib import Path
import unittest

from synthmind.research.metrics.registry import MetricRegistry


class MetricRegistryTest(unittest.TestCase):
    def test_metric_registry_loads_and_has_stable_ids(self):
        registry = MetricRegistry.load(Path("research/specs/metric_registry_v1.yaml"))
        ids = registry.ids()
        self.assertIn("e2e_route.iid.complete_case.route_ref_match@1", ids)
        self.assertIn("rsp.iid.complete_case.precursor_recall@50", ids)
        self.assertEqual(registry.validate(), [])

    def test_operational_metric_is_not_reference_accuracy(self):
        registry = MetricRegistry.load(Path("research/specs/metric_registry_v1.yaml"))
        metric = registry.get("e2e_route.iid.operational.valid_route_rate")
        self.assertEqual(metric.protocol, "operational_validity")
        self.assertNotIn("accuracy", metric.metric_id)
