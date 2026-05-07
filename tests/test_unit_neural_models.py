import os
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from integration_hub.global_workspace import GlobalWorkspace, Representation
from integration_hub.triple_network import TripleNetworkModel
from neural_models.plasticity import hebbian_update, homeostatic_scaling, stdp_update
from neural_models.predictive import PredictiveProcessingHierarchy


class NeuralModelsUnitTests(unittest.TestCase):
    def test_hebbian_update_uses_pre_post_product(self):
        delta = hebbian_update(pre=0.5, post=0.8, learning_rate=0.1)
        self.assertAlmostEqual(delta, 0.04)

    def test_stdp_update_sign_matches_timing_order(self):
        ltp = stdp_update(pre_time=10.0, post_time=12.0)
        ltd = stdp_update(pre_time=12.0, post_time=10.0)
        zero = stdp_update(pre_time=5.0, post_time=5.0)

        self.assertGreater(ltp, 0.0)
        self.assertLess(ltd, 0.0)
        self.assertEqual(zero, 0.0)

    def test_homeostatic_scaling_moves_toward_target_rate(self):
        gain_up = homeostatic_scaling(current_rate=2.0, target_rate=5.0, scaling_rate=0.01)
        gain_down = homeostatic_scaling(current_rate=5.0, target_rate=2.0, scaling_rate=0.01)

        self.assertGreater(gain_up, 1.0)
        self.assertLess(gain_down, 1.0)

    def test_predictive_processing_returns_expected_shapes(self):
        np.random.seed(7)
        hierarchy = PredictiveProcessingHierarchy(n_levels=3, n_units=8)
        output = hierarchy.process(np.array([0.1, 0.2, 0.3], dtype=float))

        self.assertIn("predictions", output)
        self.assertIn("errors", output)
        self.assertIn("total_error", output)
        self.assertEqual(len(output["predictions"]), 3)
        self.assertEqual(len(output["errors"]), 3)
        self.assertEqual(output["predictions"][0].shape, (8,))
        self.assertEqual(output["errors"][0].shape, (8,))
        self.assertGreaterEqual(output["total_error"], 0.0)

    def test_global_workspace_broadcast_respects_attention_weight(self):
        workspace = GlobalWorkspace(ignition_threshold=0.3)
        workspace.register_service("low_attention", attention_weight=0.2)
        workspace.register_service("high_attention", attention_weight=1.0)

        event = workspace.compete_for_access(
            [
                Representation(source="low_attention", content="x", salience=1.0),
                Representation(source="high_attention", content="y", salience=0.4),
            ]
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.representation.source, "high_attention")
        self.assertIn("low_attention", event.recipients)
        self.assertIn("high_attention", event.recipients)

    def test_triple_network_switches_between_modes(self):
        model = TripleNetworkModel(switch_threshold=0.5)

        low_salience = model.switch_networks(0.2)
        high_salience = model.switch_networks(0.9)

        self.assertEqual(low_salience["mode"], "mind_wandering")
        self.assertEqual(high_salience["mode"], "task_focused")
        self.assertGreaterEqual(high_salience["cen"], 0.5)


if __name__ == "__main__":
    unittest.main()
