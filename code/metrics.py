"""Shared target matching and ellipse diagnostics for paper experiments."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from baselines.uq_core import UQContractError


def ellipse_diagnostics(
    errors: np.ndarray,
    covariance: np.ndarray,
    confidence_levels: list[float] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute NIS, coverage and area for matched two-dimensional targets."""
    errors = np.asarray(errors, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    levels = np.asarray(confidence_levels, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 2:
        raise UQContractError("numerical_failure", "DOA errors must have shape (rank, 2)")
    if covariance.shape != (errors.shape[0], 2, 2):
        raise UQContractError("numerical_failure", "DOA covariance has an invalid shape")
    if levels.ndim != 1 or np.any(~np.isfinite(levels)) or np.any((levels <= 0.0) | (levels >= 1.0)):
        raise UQContractError("numerical_failure", "Confidence levels must lie in (0, 1)")
    if not np.isfinite(covariance).all():
        raise UQContractError("numerical_failure", "DOA covariance contains non-finite values")
    thresholds = chi2.ppf(levels, df=2)
    nis = np.empty(errors.shape[0], dtype=float)
    coverage = np.empty((errors.shape[0], levels.size), dtype=bool)
    area = np.empty((errors.shape[0], levels.size), dtype=float)
    for target in range(errors.shape[0]):
        block = 0.5 * (covariance[target] + covariance[target].T)
        eigenvalues = np.linalg.eigvalsh(block)
        if np.min(eigenvalues) <= 0.0:
            raise UQContractError(
                "nonpositive_doa_covariance", "DOA covariance is not positive definite"
            )
        distance = float(errors[target] @ np.linalg.solve(block, errors[target]))
        determinant = float(np.linalg.det(block))
        nis[target] = distance
        coverage[target] = distance <= thresholds
        area[target] = np.pi * thresholds * np.sqrt(determinant)
    return {
        "confidence_levels": levels,
        "nis": nis,
        "ellipse_coverage": coverage,
        "ellipse_area": area,
    }
