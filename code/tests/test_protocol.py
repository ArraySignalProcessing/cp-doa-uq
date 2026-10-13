from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path

# Resolve project imports when discovery runs from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol import draw_trial_latents, fixed_scene


class ProtocolTests(unittest.TestCase):
    def test_equal_power_latents_and_fixed_geometry_are_reproducible(self):
        first = draw_trial_latents((4, 4, 8), 3, [1.0, 1.0, 1.0], master_seed=7, repeat_index=2, scene_index=0)
        second = draw_trial_latents((4, 4, 8), 3, [1.0, 1.0, 1.0], master_seed=7, repeat_index=2, scene_index=0)
        np.testing.assert_array_equal(first["signals"], second["signals"])
        scene = fixed_scene((12, 12, 64), [
            {"azimuth": -8.0, "elevation": 30.0},
            {"azimuth": 0.0, "elevation": 35.0},
            {"azimuth": 8.0, "elevation": 40.0},
        ], maximum_spatial_factor_condition=10.0, master_seed=7)
        self.assertEqual(scene["true_uv"].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
