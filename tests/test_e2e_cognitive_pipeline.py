import os
import sys
import unittest

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from cognitive_network import CognitiveNeuralSystem
from tasks import iowa_gambling_task, morris_maze_task, n_back_task, stroop_task


class CognitivePipelineE2ETests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        np.random.seed(42)
        self.system = CognitiveNeuralSystem()
        await self.system.initialize()

    async def asyncTearDown(self):
        await self.system.shutdown()

    async def test_full_pipeline_processes_a_cognitive_task(self):
        result = await self.system.process_cognitive_task(
            {"context": "e2e-validation", "complexity": "high", "signal": [1, 2, 3]}
        )

        for key in (
            "action",
            "homeostasis",
            "memory",
            "emotion",
            "state_machine",
            "network_mode",
            "broadcast",
        ):
            self.assertIn(key, result)

        self.assertIn(result["network_mode"], ("mind_wandering", "task_focused"))
        self.assertIsInstance(result["state_machine"], dict)
        self.assertEqual(len(self.system.services), 16)
        self.assertGreaterEqual(len(self.system.get_service_firing_rates()), 16)

    async def test_benchmark_task_suite_runs_end_to_end(self):
        n_back = await n_back_task(self.system, n=2, sequence_length=8)
        stroop = await stroop_task(self.system, n_trials=8)
        iowa = await iowa_gambling_task(self.system, n_trials=8)
        morris = await morris_maze_task(self.system, n_trials=5)

        self.assertEqual(n_back["task"], "n_back")
        self.assertGreaterEqual(n_back["hit_rate"], 0.0)
        self.assertGreaterEqual(n_back["false_alarm_rate"], 0.0)

        self.assertEqual(stroop["task"], "stroop")
        self.assertIn("stroop_effect", stroop)

        self.assertEqual(iowa["task"], "iowa_gambling")
        self.assertEqual(set(iowa["deck_distribution"].keys()), {"A", "B", "C", "D"})

        self.assertEqual(morris["task"], "morris_maze")
        self.assertGreater(morris["mean_escape_latency"], 0.0)


if __name__ == "__main__":
    unittest.main()
