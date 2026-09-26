import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).parents[3] / "verl/trainer/ppo/validation_metrics.py"
SPEC = importlib.util.spec_from_file_location("validation_metrics", MODULE_PATH)
validation_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation_metrics)
summarize_episode_successes = validation_metrics.summarize_episode_successes
unique_episode_outcomes = validation_metrics.unique_episode_outcomes


class TestValidationEpisodeMetrics(unittest.TestCase):
    def test_turn_expansion_does_not_reweight_episodes(self):
        sources, successes = unique_episode_outcomes(
            data_sources=["nq", "nq", "nq", "nq", "bamboogle"],
            traj_uids=["long", "long", "long", "short", "tiny"],
            episode_successes=[1, 1, 1, 0, 1],
        )

        np.testing.assert_array_equal(sources, ["nq", "nq", "bamboogle"])
        np.testing.assert_array_equal(successes, [1, 0, 1])
        metrics = summarize_episode_successes(sources, successes)

        self.assertAlmostEqual(metrics["success_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["nq_success_rate"], 1 / 2)
        self.assertEqual(metrics["nq_evaluated_cases"], 2)
        self.assertAlmostEqual(metrics["bamboogle_success_rate"], 1.0)
        self.assertEqual(metrics["bamboogle_evaluated_cases"], 1)

    def test_task_rates_are_weighted_by_episode_not_batch(self):
        metrics = summarize_episode_successes(
            data_sources=["popqa", "popqa", "popqa", "popqa", "popqa"],
            episode_successes=[1, 1, 1, 0, 0],
        )

        self.assertAlmostEqual(metrics["popqa_success_rate"], 3 / 5)
        self.assertEqual(metrics["popqa_evaluated_cases"], 5)

    def test_conflicting_duplicate_rows_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent validation rows"):
            unique_episode_outcomes(
                data_sources=["nq", "nq"],
                traj_uids=["same", "same"],
                episode_successes=[0, 1],
            )

    def test_length_mismatch_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            unique_episode_outcomes(["nq"], ["a", "b"], [1])


if __name__ == "__main__":
    unittest.main()
