"""Bootstrap alignment, failure denominator and interrupted-prefix recovery tests."""
from __future__ import annotations
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import sys
from pathlib import Path

# Resolve project imports when discovery runs from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.conditional_bootstrap import extend_prefix, prefix_covariance
from common import factors_to_uv


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        uv = np.array([[.2, .1], [-.3, .25]])
        self.factors = {'ax': np.exp(1j * np.pi * np.arange(3)[:, None] * uv[:, 0]),
                        'ay': np.exp(1j * np.pi * np.arange(3)[:, None] * uv[:, 1]),
                        'signals': np.ones((8, 2), complex)}
        self.mask = np.ones((3, 3, 8), bool)
        from baselines.cp_core import reconstruct
        self.observed = reconstruct(self.factors) + .02j
        self.kwargs = dict(factors=self.factors, observed=self.observed, mask=self.mask,
                           optimizer={'restarts': 1}, fit_converged=True, budget=4,
                           seed=73, condition_index=2, repeat_index=5)

    def test_interrupted_prefix_resumes_without_repeating_committed_refits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.h5'
            noises = []
            calls = [0]
            def optimizer(observed, *args):
                calls[0] += 1
                if calls[0] == 3:
                    raise RuntimeError('simulated interruption')
                noises.append(observed.copy())
                # Reverse labels to check matching to the outer fit.
                return {'status': 'converged',
                        'factors': {key: value[:, ::-1] for key, value in self.factors.items()}}
            with h5py.File(path, 'w') as handle:
                group = handle.create_group('prefix')
                with patch('baselines.conditional_bootstrap.optimize_masked_cp', side_effect=optimizer):
                    with self.assertRaises(RuntimeError):
                        extend_prefix(group, **self.kwargs)
                    self.assertEqual(group.attrs['completed'], 2)
                    extend_prefix(group, **self.kwargs)
                self.assertEqual(calls[0], 5)  # 2 committed + interrupted attempt + 2 missing
                self.assertEqual(group.attrs['completed'], 4)
                expected = factors_to_uv(self.factors)
                np.testing.assert_allclose(group['samples'][...], np.broadcast_to(expected, (4, 2, 2)))
                self.assertEqual(int(group['valid'][...].sum()), 4)
            # A clean run must draw the same four noisy observations as the resumed run.
            clean = []
            def collect(observed, *args):
                clean.append(observed.copy())
                return {'status': 'converged', 'factors': self.factors}
            with h5py.File(Path(directory) / 'clean.h5', 'w') as handle:
                with patch('baselines.conditional_bootstrap.optimize_masked_cp', side_effect=collect):
                    extend_prefix(handle.create_group('prefix'), **self.kwargs)
            for resumed, uninterrupted in zip(noises, clean):
                np.testing.assert_array_equal(resumed, uninterrupted)

    def test_failed_refits_are_counted_in_requested_prefix(self):
        rng = np.random.default_rng(4)
        samples = rng.normal(size=(10, 2, 2))
        valid = np.array([True] * 6 + [False] * 4)
        covariance, status = prefix_covariance(samples, valid, 10, .7)
        self.assertIsNone(covariance)
        self.assertEqual(status, 'bootstrap_insufficient_valid')

if __name__ == '__main__':
    unittest.main()
