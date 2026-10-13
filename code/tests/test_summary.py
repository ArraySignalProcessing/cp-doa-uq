from __future__ import annotations

import unittest

import sys
from pathlib import Path

# Resolve project imports when discovery runs from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_paper import calibration_error, h2_interpretation


class SummaryTests(unittest.TestCase):
    def test_calibration_contract(self):
        self.assertAlmostEqual(calibration_error([0.5, 0.8, 0.9, 0.95], [0.5, 0.8, 0.9, 0.95]), 0.0)
        self.assertEqual(h2_interpretation(-0.01, -0.02, -0.001), "direct_support")


if __name__ == "__main__":
    unittest.main()
