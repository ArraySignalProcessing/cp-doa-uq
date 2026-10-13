from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path

# Resolve project imports when discovery runs from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_core import boundary_uv
from common import load_paper_config


class BoundaryDesignTests(unittest.TestCase):
    def test_only_controlled_pair_moves(self):
        experiment = load_paper_config()["boundary"]
        first = boundary_uv(0.20, experiment["geometry"])
        second = boundary_uv(0.05, experiment["geometry"])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertAlmostEqual(np.linalg.norm(first[0] - first[1]), 0.20)
        self.assertEqual(len(experiment["separations"]) * experiment["repeats"], 1800)


if __name__ == "__main__":
    unittest.main()
