"""Numerical checks for gauge-aware DOA uncertainty propagation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import sys
from pathlib import Path

# Resolve project imports when discovery runs from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.cp_core import rebalance_columns, reconstruct
from experiment_core import evaluate_outer_instance
from baselines.uq_core import (
    complex_model_jacobian,
    doa_analytic_jacobian,
    doa_numerical_jacobian,
    gauge_generator_matrix,
    gauge_parameterization_sensitivity,
    gauge_reduced_information_diagnostics,
    gauge_restricted_doa_covariance,
    gauge_slice_basis,
    fitted_full_rate_scaled_doa_covariance,
    naive_pseudoinverse_doa_covariance,
    structural_pseudoinverse_doa_covariance,
    UQContractError,
    uq_contract_status,
)
from common import factors_to_uv


def _steering(length: int, frequencies: np.ndarray) -> np.ndarray:
    return np.exp(1j * np.pi * np.arange(length)[:, None] * frequencies[None, :])


class ColumnRebalancingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261005)
        self.factors = {
            "ax": rng.standard_normal((5, 2)) + 1j * rng.standard_normal((5, 2)),
            "ay": rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2)),
            "signals": rng.standard_normal((7, 2)) + 1j * rng.standard_normal((7, 2)),
        }
        ax_scale = np.asarray([1e-8, 1e8])
        ay_scale = np.asarray([1e3, 1e-3])
        self.factors["ax"] *= ax_scale
        self.factors["ay"] *= ay_scale
        self.factors["signals"] /= ax_scale * ay_scale

    def test_rebalancing_preserves_tensor_doa_and_equalizes_norms(self):
        tensor = reconstruct(self.factors)
        doa = factors_to_uv(self.factors)
        balanced = rebalance_columns(self.factors)
        norms = np.stack(
            [np.linalg.norm(balanced[name], axis=0) for name in ("ax", "ay", "signals")]
        )
        np.testing.assert_allclose(
            norms,
            np.broadcast_to(norms[:1], norms.shape),
            atol=0.0,
            rtol=1e-12,
        )
        np.testing.assert_allclose(reconstruct(balanced), tensor, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(factors_to_uv(balanced), doa, atol=1e-12, rtol=1e-12)

    def test_rebalancing_rejects_zero_columns(self):
        invalid = {name: value.copy() for name, value in self.factors.items()}
        invalid["ax"][:, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "zero column"):
            rebalance_columns(invalid)

    def test_rebalancing_rejects_nonfinite_columns(self):
        invalid = {name: value.copy() for name, value in self.factors.items()}
        invalid["ay"][0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rebalance_columns(invalid)


class DoaJacobianTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20260922)
        frequencies_x = np.array([-0.31, 0.24])
        frequencies_y = np.array([0.18, -0.27])
        self.factors = {
            "ax": _steering(5, frequencies_x) * np.array([1.2 - 0.4j, 0.8 + 0.3j]),
            "ay": _steering(4, frequencies_y) * np.array([0.7 + 0.2j, 1.1 - 0.5j]),
            "signals": rng.standard_normal((7, 2)) + 1j * rng.standard_normal((7, 2)),
        }

    def _observations(self):
        clean = reconstruct(self.factors)
        rng = np.random.default_rng(7)
        observed = clean + 1e-2 * (
            rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
        )
        return clean, observed, np.ones(clean.shape, dtype=bool)

    def _gauge_transform(self):
        transformed = {name: value.copy() for name, value in self.factors.items()}
        ax_scale = np.array([1.7 * np.exp(0.8j), 0.6 * np.exp(-1.1j)])
        ay_scale = np.array([0.9 * np.exp(-0.4j), 1.4 * np.exp(0.7j)])
        transformed["ax"] *= ax_scale
        transformed["ay"] *= ay_scale
        transformed["signals"] /= ax_scale * ay_scale
        return transformed

    def test_analytic_jacobian_matches_central_difference(self):
        analytic = doa_analytic_jacobian(self.factors)
        numerical = doa_numerical_jacobian(self.factors, relative_step=1e-6)
        relative_error = np.linalg.norm(analytic - numerical) / np.linalg.norm(numerical)
        self.assertLess(relative_error, 1e-8)

    def test_direction_mapping_is_invariant_to_complex_column_scales(self):
        transformed = self._gauge_transform()
        np.testing.assert_allclose(
            factors_to_uv(transformed), factors_to_uv(self.factors), atol=1e-12, rtol=1e-12
        )

    def test_propagated_covariance_is_symmetric_positive_semidefinite(self):
        clean, observed, mask = self._observations()
        covariance, _ = gauge_restricted_doa_covariance(
            self.factors, observed, mask
        )
        for block in covariance:
            np.testing.assert_allclose(block, block.T, atol=1e-12, rtol=1e-12)
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(block))), -1e-12)

    def test_propagated_covariance_is_invariant_to_cp_gauge(self):
        _, observed, mask = self._observations()
        reference, _ = gauge_restricted_doa_covariance(self.factors, observed, mask)
        transformed = self._gauge_transform()
        changed, _ = gauge_restricted_doa_covariance(transformed, observed, mask)
        np.testing.assert_allclose(changed, reference, atol=1e-12, rtol=1e-10)

    def test_fitted_full_matches_actual_mask_covariance_at_unit_rate(self):
        _, observed, mask = self._observations()
        actual, actual_diagnostics = gauge_restricted_doa_covariance(
            self.factors, observed, mask
        )
        comparison, comparison_diagnostics = fitted_full_rate_scaled_doa_covariance(
            self.factors, observed, mask
        )
        np.testing.assert_allclose(comparison, actual, atol=1e-12, rtol=1e-12)
        self.assertEqual(comparison_diagnostics["observation_rate"], 1.0)
        self.assertEqual(comparison_diagnostics["covariance_scale"], 1.0)
        self.assertEqual(
            comparison_diagnostics["real_noise_variance"],
            actual_diagnostics["real_noise_variance"],
        )

    def test_fitted_full_uses_actual_mask_residual_variance(self):
        _, observed, _ = self._observations()
        mask = np.indices(observed.shape).sum(axis=0) % 2 == 0
        _, diagnostics = fitted_full_rate_scaled_doa_covariance(
            self.factors, observed, mask
        )
        effective_parameters = (
            2
            * sum(value.size for value in self.factors.values())
            - 4 * self.factors["ax"].shape[1]
        )
        residual = np.where(mask, reconstruct(self.factors) - observed, 0.0)
        residual_degrees = 2 * np.count_nonzero(mask) - effective_parameters
        expected_variance = float(np.vdot(residual, residual).real) / residual_degrees
        self.assertAlmostEqual(diagnostics["real_noise_variance"], expected_variance)
        self.assertEqual(diagnostics["residual_observations"], float(np.count_nonzero(mask)))
        self.assertEqual(diagnostics["information_observations"], float(mask.size))
        self.assertAlmostEqual(diagnostics["covariance_scale"], 1.0 / np.mean(mask))

    def test_fitted_full_covariance_is_invariant_to_cp_gauge(self):
        _, observed, _ = self._observations()
        mask = np.indices(observed.shape).sum(axis=0) % 2 == 0
        reference, _ = fitted_full_rate_scaled_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            rcond=1e-12,
            inversion_mode="structural_svd",
        )
        changed, _ = fitted_full_rate_scaled_doa_covariance(
            self._gauge_transform(),
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            rcond=1e-12,
            inversion_mode="structural_svd",
        )
        np.testing.assert_allclose(changed, reference, atol=1e-12, rtol=1e-9)

    def test_gauge_generators_are_model_and_doa_null_directions(self):
        _, _, mask = self._observations()
        generators = gauge_generator_matrix(self.factors)
        model_jacobian = complex_model_jacobian(self.factors, mask)
        doa_jacobian = doa_analytic_jacobian(self.factors)
        self.assertEqual(np.linalg.matrix_rank(generators), 4 * self.factors["ax"].shape[1])
        np.testing.assert_allclose(model_jacobian @ generators, 0.0, atol=1e-12)
        np.testing.assert_allclose(doa_jacobian @ generators, 0.0, atol=1e-12)

    def test_information_diagnostic_is_regular_for_separated_targets(self):
        _, _, mask = self._observations()
        diagnostics, status, reason = gauge_reduced_information_diagnostics(
            self.factors, mask
        )
        self.assertEqual(status, "valid", reason)
        self.assertEqual(
            diagnostics["reduced_jacobian_rank"], diagnostics["tangent_dimension"]
        )

    def test_information_diagnostic_detects_exact_target_overlap(self):
        _, _, mask = self._observations()
        overlapped = {name: value.copy() for name, value in self.factors.items()}
        overlapped["ax"][:, 1] = overlapped["ax"][:, 0]
        overlapped["ay"][:, 1] = overlapped["ay"][:, 0]
        diagnostics, status, _ = gauge_reduced_information_diagnostics(
            overlapped, mask
        )
        self.assertEqual(status, "nonregular_information")
        self.assertLess(
            diagnostics["reduced_jacobian_rank"], diagnostics["tangent_dimension"]
        )

    def test_information_diagnostic_is_gauge_parameterization_invariant(self):
        _, _, mask = self._observations()
        reference, reference_status, _ = gauge_reduced_information_diagnostics(
            self.factors, mask
        )
        changed, changed_status, _ = gauge_reduced_information_diagnostics(
            self._gauge_transform(), mask
        )
        self.assertEqual(changed_status, reference_status)
        self.assertEqual(
            changed["reduced_jacobian_rank"], reference["reduced_jacobian_rank"]
        )

    def test_raw_gauge_covariance_is_parameterization_invariant(self):
        _, observed, mask = self._observations()
        reference, reference_diagnostics = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            rcond=1e-12,
            inversion_mode="structural_svd",
        )
        changed, changed_diagnostics = gauge_restricted_doa_covariance(
            self._gauge_transform(),
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            rcond=1e-12,
            inversion_mode="structural_svd",
        )
        self.assertEqual(reference_diagnostics["uq_valid"], 1.0)
        self.assertEqual(changed_diagnostics["uq_valid"], 1.0)
        np.testing.assert_allclose(changed, reference, atol=1e-12, rtol=1e-9)

    def test_transverse_slices_give_the_same_doa_covariance(self):
        _, observed, mask = self._observations()
        constraint_covariance, _ = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="constraints",
            rcond=1e-12,
        )
        orthogonal_covariance, _ = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            rcond=1e-12,
            inversion_mode="structural_svd",
        )
        np.testing.assert_allclose(
            constraint_covariance, orthogonal_covariance, atol=1e-12, rtol=1e-9
        )

    def test_structural_qr_gauge_basis_matches_null_space_basis(self):
        _, observed, mask = self._observations()
        null_covariance, _ = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            inversion_mode="structural_svd",
        )
        qr_covariance, diagnostics = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal_structural_qr",
            inversion_mode="structural_svd",
        )
        self.assertEqual(diagnostics["uq_valid"], 1.0)
        np.testing.assert_allclose(qr_covariance, null_covariance, atol=1e-12, rtol=1e-9)

    def test_naive_pseudoinverse_agrees_for_estimable_doa(self):
        _, observed, mask = self._observations()
        gauge_aware, _ = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            rcond=1e-12,
        )
        naive, diagnostics = naive_pseudoinverse_doa_covariance(
            self.factors, observed, mask, canonicalize=False, rcond=1e-12
        )
        self.assertEqual(diagnostics["uq_valid"], 1.0)
        np.testing.assert_allclose(naive, gauge_aware, atol=1e-12, rtol=1e-8)

    def test_structural_rank_pseudoinverse_agrees_with_gauge_reduction(self):
        _, observed, mask = self._observations()
        gauge_aware, _ = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            inversion_mode="structural_svd",
        )
        structural_mp, diagnostics = structural_pseudoinverse_doa_covariance(
            self.factors, observed, mask, canonicalize=False
        )
        self.assertEqual(diagnostics["uq_valid"], 1.0)
        np.testing.assert_allclose(structural_mp, gauge_aware, atol=1e-12, rtol=1e-8)

    def test_gauge_slice_is_transverse(self):
        _, generators, constraints = gauge_slice_basis(self._gauge_transform(), "constraints")
        self.assertEqual(np.linalg.matrix_rank(constraints @ generators), generators.shape[1])

    def test_parameterization_stress_is_numerically_invariant(self):
        _, observed, mask = self._observations()
        covariance, _ = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            rcond=1e-12,
        )
        diagnostics = gauge_parameterization_sensitivity(
            self.factors,
            observed,
            mask,
            covariance,
            method="gauge_slice",
            samples=4,
            seed=1234,
            rcond=1e-12,
            inversion_mode="structural_svd",
        )
        self.assertLess(diagnostics["parameterization_max_relative_change"], 1e-8)
        self.assertEqual(diagnostics["parameterization_rank_changes"], 0.0)
        self.assertEqual(diagnostics["parameterization_invalid_count"], 0.0)

    def test_degenerate_doa_mapping_has_explicit_contract_status(self):
        degenerate = {name: value.copy() for name, value in self.factors.items()}
        degenerate["ax"][:, 0] = np.array([1.0, 1.0, 1.0, -1.0, 1.0])
        with self.assertRaises(UQContractError) as caught:
            doa_analytic_jacobian(degenerate)
        self.assertEqual(caught.exception.status, "degenerate_doa_map")

    def test_nonpositive_residual_degrees_have_explicit_contract_status(self):
        clean = reconstruct(self.factors)
        sparse_mask = np.zeros(clean.shape, dtype=bool)
        sparse_mask.ravel()[:2] = True
        with self.assertRaises(UQContractError) as caught:
            gauge_restricted_doa_covariance(self.factors, clean, sparse_mask)
        self.assertEqual(caught.exception.status, "nonpositive_residual_df")

    def test_contract_rejects_nonconverged_fit(self):
        _, observed, mask = self._observations()
        _, diagnostics = gauge_restricted_doa_covariance(
            self.factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode="orthogonal",
            inversion_mode="structural_svd",
        )
        status, reason = uq_contract_status(diagnostics, fit_status="not_converged")
        self.assertEqual(status, "fit_not_converged")
        self.assertTrue(reason)

    def test_shared_fit_withholds_covariance_after_fit_failure(self):
        _, observed, mask = self._observations()
        optimizer_result = {
            "factors": self.factors,
            "status": "not_converged",
            "iterations": 1,
            "failure_reason": "iteration limit",
        }
        with patch("experiment_core.optimize_masked_cp", return_value=optimizer_result):
            result = evaluate_outer_instance(
                clean_tensor=observed, standard_noise=np.zeros_like(observed),
                mask_uniform=np.where(mask, 0.0, 0.99), true_uv=factors_to_uv(self.factors),
                condition={"observation_rate": 0.9, "snr_db": 10}, initialization_seed=7,
                rank=2, optimizer_config={"restarts": 1}, method_config={},
                confidence_levels=[0.5, 0.8, 0.9, 0.95], correct_branch_max_uv_error=0.1,
            )
        self.assertEqual(result["uq"], {})
        self.assertEqual(result["terminal_status"], "fit_not_converged")
        self.assertTrue(result["fit"]["failure_reason"])



if __name__ == "__main__":
    unittest.main()
