#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinetics_eval import merge_covariance_overlay
from kinetics_tune import self_consistency


class CovarianceOverlayTest(unittest.TestCase):
    def test_changes_only_requested_covariance(self):
        base = {
            "covariances": {"contact_process": [1.0, 2.0], "gyroscope": [3.0]},
            "contact_model": {"linear_stiffness": [10.0]},
        }
        merged = merge_covariance_overlay(
            base, {"covariances": {"contact_process": [4.0, 5.0]}}
        )

        self.assertEqual(merged["covariances"]["contact_process"], [4.0, 5.0])
        self.assertEqual(merged["covariances"]["gyroscope"], [3.0])
        self.assertEqual(merged["contact_model"], base["contact_model"])
        self.assertEqual(base["covariances"]["contact_process"], [1.0, 2.0])

    def test_scales_are_relative_to_each_robots_own_value(self):
        for reference, expected in ((9.0e-4, 9.0e-1), (1.0e-2, 1.0e1)):
            merged = merge_covariance_overlay(
                {"covariances": {"contact_wrench": [1.0, reference]}},
                {"covariance_scales": {"contact_wrench": [1.0, 1000.0]}},
            )
            self.assertEqual(merged["covariances"]["contact_wrench"][0], 1.0)
            self.assertAlmostEqual(merged["covariances"]["contact_wrench"][1], expected)

    def test_absolute_and_scaled_fields_coexist(self):
        merged = merge_covariance_overlay(
            {"covariances": {"contact_process": [1.0, 2.0], "contact_wrench": [4.0]}},
            {"covariances": {"contact_process": [7.0, 8.0]}, "covariance_scales": {"contact_wrench": [0.5]}},
        )
        self.assertEqual(merged["covariances"]["contact_process"], [7.0, 8.0])
        self.assertEqual(merged["covariances"]["contact_wrench"], [2.0])

    def test_rejects_a_field_set_both_ways(self):
        with self.assertRaises(ValueError):
            merge_covariance_overlay(
                {"covariances": {"contact_wrench": [1.0]}},
                {"covariances": {"contact_wrench": [2.0]}, "covariance_scales": {"contact_wrench": [3.0]}},
            )

    def test_rejects_an_empty_overlay(self):
        with self.assertRaises(ValueError):
            merge_covariance_overlay({"covariances": {"contact_wrench": [1.0]}}, {})



class SelfConsistencyTest(unittest.TestCase):
    """The diagnostic must read 0 for an exactly consistent estimate and scale with the gap."""

    @staticmethod
    def _write(directory, name, velocity_gain):
        """A synthetic dataset whose reported velocity is velocity_gain x the true dp/dt."""
        project = Path(directory) / name
        project.mkdir(parents=True)
        step = 0.005
        times = np.arange(0, 40, step)
        position = np.column_stack([0.3 * np.sin(2 * np.pi * 1.0 * times + axis) for axis in range(3)])
        velocity = np.column_stack([0.3 * 2 * np.pi * np.cos(2 * np.pi * 1.0 * times + axis)
                                    for axis in range(3)]) * velocity_gain
        pose = np.column_stack([times, position, np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1))])
        np.savetxt(project / "kinetics.txt", pose, header="timestamp tx ty tz qx qy qz qw")
        np.savetxt(project / "kinetics_velocity.txt",
                   np.column_stack([times, velocity, np.zeros((len(times), 3))]),
                   header="timestamp vx vy vz wx wy wz")
        return project

    def test_consistent_estimate_scores_near_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "HRP5_MultiContact_1", velocity_gain=1.0)
            value = self_consistency(Path(directory), ["HRP5_MultiContact_1"])["HRP5_MultiContact_1|consistency"]
            # Only the half-sample skew of the shared backward difference remains.
            self.assertLess(value, 0.02)

    def test_score_tracks_the_size_of_the_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "HRP5_MultiContact_1", velocity_gain=1.3)
            value = self_consistency(Path(directory), ["HRP5_MultiContact_1"])["HRP5_MultiContact_1|consistency"]
            self.assertAlmostEqual(value, 0.3, delta=0.03)



if __name__ == "__main__":
    unittest.main()
