"""Unit tests for point-cloud metrics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vggt_al.metrics import (  # noqa: E402
    chamfer_distance,
    evaluate_point_cloud,
    f_score,
    normal_consistency,
    precision_recall,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.cloud = self.rng.normal(size=(300, 3)).astype(np.float32)

    def test_identical_clouds(self):
        self.assertAlmostEqual(chamfer_distance(self.cloud, self.cloud), 0.0, places=6)
        self.assertAlmostEqual(f_score(self.cloud, self.cloud, 0.05), 1.0, places=6)
        p, r = precision_recall(self.cloud, self.cloud, 0.05)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)

    def test_shifted_clouds(self):
        shifted = self.cloud.copy()
        shifted[:50] += np.array([0.3, 0, 0], dtype=np.float32)
        self.assertGreater(chamfer_distance(self.cloud, shifted), 0.0)
        score = f_score(self.cloud, shifted, 0.05)
        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 1.0)

    def test_normal_consistency_range(self):
        nc = normal_consistency(self.cloud, self.cloud + 1e-4)
        self.assertGreaterEqual(nc, 0.0)
        self.assertLessEqual(nc, 1.0)

    def test_evaluate_wrapper(self):
        result = evaluate_point_cloud(self.cloud, self.cloud, with_normals=True)
        self.assertIn("CD", result)
        self.assertIn("F-Score", result)
        self.assertIn("NC", result)
        self.assertAlmostEqual(result["F-Score"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
