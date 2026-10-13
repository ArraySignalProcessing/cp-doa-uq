"""Local covariance for complex CP factors and gauge-invariant 2-D DOA."""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space, qr

from common import canonicalize_factors, factors_to_uv
from .cp_core import FACTOR_NAMES, pack_factors, reconstruct, unpack_factors


DEFAULT_RCOND = 1e-10
DEFAULT_INVARIANCE_TOLERANCE = 1e-8


class UQContractError(ValueError):
    """Expected UQ-domain failure with a stable reporting category."""

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _validate_factors(factors: dict[str, np.ndarray]) -> None:
    arrays = {name: np.asarray(factors[name]) for name in FACTOR_NAMES}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise UQContractError("invalid_factor", "CP factors contain non-finite values")
    if any(value.ndim != 2 for value in arrays.values()):
        raise UQContractError("invalid_factor", "CP factors must be matrices")
    ranks = {value.shape[1] for value in arrays.values()}
    if len(ranks) != 1 or not ranks or next(iter(ranks)) <= 0:
        raise UQContractError("invalid_factor", "CP factors have inconsistent ranks")
    for name, value in arrays.items():
        column_norms = np.linalg.norm(value, axis=0)
        scale = max(float(np.linalg.norm(value)), 1.0)
        if np.any(column_norms <= np.finfo(float).eps * scale):
            raise UQContractError("invalid_factor", f"{name} contains a zero column")


def complex_model_jacobian(
    factors: dict[str, np.ndarray], mask: np.ndarray
) -> np.ndarray:
    ax, ay, signals = factors["ax"], factors["ay"], factors["signals"]
    mx, my, snapshots = mask.shape
    rank = ax.shape[1]
    coordinates = np.argwhere(mask)
    complex_size = rank * (mx + my + snapshots)
    design = np.zeros((coordinates.shape[0], complex_size), dtype=np.complex128)
    ay_offset = mx * rank
    signal_offset = (mx + my) * rank
    for row, (m, n, snapshot) in enumerate(coordinates):
        components = np.arange(rank)
        design[row, m * rank + components] = ay[n] * signals[snapshot]
        design[row, ay_offset + n * rank + components] = ax[m] * signals[snapshot]
        design[row, signal_offset + snapshot * rank + components] = ax[m] * ay[n]
    return np.block([[design.real, -design.imag], [design.imag, design.real]])


def gauge_generator_matrix(factors: dict[str, np.ndarray]) -> np.ndarray:
    """Return infinitesimal real CP gauge directions.

    Each component has two independent complex scales.  The four columns are
    log-amplitude and phase changes in ``ax`` followed by log-amplitude and
    phase changes in ``ay``; the inverse product is absorbed by ``signals``.
    """
    arrays = {name: np.asarray(factors[name]) for name in FACTOR_NAMES}
    rank = arrays["ax"].shape[1]
    parameter_count = pack_factors(arrays).size
    generators = np.zeros((parameter_count, 4 * rank), dtype=float)
    for component in range(rank):
        for column, (first_name, multiplier) in enumerate(
            (("ax", 1.0), ("ax", 1.0j), ("ay", 1.0), ("ay", 1.0j))
        ):
            direction = {name: np.zeros_like(value) for name, value in arrays.items()}
            direction[first_name][:, component] = multiplier * arrays[first_name][:, component]
            direction["signals"][:, component] = -multiplier * arrays["signals"][:, component]
            generators[:, 4 * component + column] = pack_factors(direction)
    return generators


def gauge_constraint_jacobian(factors: dict[str, np.ndarray]) -> np.ndarray:
    """Jacobian of a norm-and-phase gauge slice through the supplied factors.

    Unlike the original canonical-only implementation, the phase constraint
    is oriented at the current column and therefore remains transverse after
    arbitrary nonzero complex CP rescaling.
    """
    ax, ay, signals = factors["ax"], factors["ay"], factors["signals"]
    rank = ax.shape[1]
    complex_size = rank * (ax.shape[0] + ay.shape[0] + signals.shape[0])
    constraints = np.zeros((4 * rank, 2 * complex_size))
    ay_offset = ax.shape[0] * rank
    for component in range(rank):
        ax_indices = np.arange(ax.shape[0]) * rank + component
        ay_indices = ay_offset + np.arange(ay.shape[0]) * rank + component
        constraints[4 * component, ax_indices] = 2.0 * ax[:, component].real
        constraints[4 * component, complex_size + ax_indices] = 2.0 * ax[:, component].imag
        ax_anchor = int(np.argmax(np.abs(ax[:, component])))
        ax_value = ax[ax_anchor, component]
        if abs(ax_value) <= 1e-14:
            raise UQContractError("invalid_factor", "Cannot fix the phase of a zero ax column")
        ax_phase = np.conj(ax_value) / abs(ax_value)
        ax_anchor_index = ax_anchor * rank + component
        constraints[4 * component + 1, ax_anchor_index] = ax_phase.imag
        constraints[4 * component + 1, complex_size + ax_anchor_index] = ax_phase.real
        constraints[4 * component + 2, ay_indices] = 2.0 * ay[:, component].real
        constraints[4 * component + 2, complex_size + ay_indices] = 2.0 * ay[:, component].imag
        ay_anchor = int(np.argmax(np.abs(ay[:, component])))
        ay_value = ay[ay_anchor, component]
        if abs(ay_value) <= 1e-14:
            raise UQContractError("invalid_factor", "Cannot fix the phase of a zero ay column")
        ay_phase = np.conj(ay_value) / abs(ay_value)
        ay_anchor_index = ay_offset + ay_anchor * rank + component
        constraints[4 * component + 3, ay_anchor_index] = ay_phase.imag
        constraints[4 * component + 3, complex_size + ay_anchor_index] = ay_phase.real
    return constraints


def gauge_slice_basis(
    factors: dict[str, np.ndarray], mode: str = "constraints"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a gauge-complement basis plus generators and slice constraints."""
    generators = gauge_generator_matrix(factors)
    constraints = gauge_constraint_jacobian(factors)
    if mode == "constraints":
        basis = null_space(constraints)
    elif mode == "orthogonal":
        basis = null_space(generators.T)
    elif mode == "orthogonal_structural_qr":
        expected_gauge_rank = 4 * factors["ax"].shape[1]
        orthogonal, _, _ = qr(generators, mode="full", pivoting=True)
        basis = orthogonal[:, expected_gauge_rank:]
    else:
        raise ValueError(f"Unknown gauge slice mode: {mode}")
    return basis, generators, constraints


def doa_numerical_jacobian(
    factors: dict[str, np.ndarray], relative_step: float = 1e-6
) -> np.ndarray:
    """Finite-difference reference used to validate the analytic Jacobian."""
    vector = pack_factors(factors)
    rank = factors["ax"].shape[1]
    output_size = 2 * rank
    jacobian = np.empty((output_size, vector.size))
    shape = (factors["ax"].shape[0], factors["ay"].shape[0], factors["signals"].shape[0])
    for index in range(vector.size):
        step = relative_step * max(1.0, abs(vector[index]))
        direction = np.zeros_like(vector)
        direction[index] = step
        plus = factors_to_uv(unpack_factors(vector + direction, shape, rank)).reshape(-1)
        minus = factors_to_uv(unpack_factors(vector - direction, shape, rank)).reshape(-1)
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian


def doa_analytic_jacobian(factors: dict[str, np.ndarray]) -> np.ndarray:
    """Jacobian of adjacent-factor correlation phases in packed real coordinates.

    For q = sum(conj(a[m]) * a[m + 1]) and f = angle(q) / pi,
    df = Im(dq / q) / pi.  Rows are ordered as (u_0, v_0, ...), and
    columns follow ``pack_factors``: all real parts followed by all imaginary
    parts of (ax, ay, signals).
    """
    ax, ay, signals = (np.asarray(factors[name]) for name in FACTOR_NAMES)
    rank = ax.shape[1]
    complex_size = rank * (ax.shape[0] + ay.shape[0] + signals.shape[0])
    jacobian = np.zeros((2 * rank, 2 * complex_size), dtype=float)
    mode_offsets = (0, ax.shape[0] * rank)

    for output_offset, (factor, mode_offset) in enumerate(zip((ax, ay), mode_offsets)):
        for component in range(rank):
            column = factor[:, component]
            correlation_terms = np.conj(column[:-1]) * column[1:]
            correlation = np.sum(correlation_terms)
            scale = max(float(np.sum(np.abs(correlation_terms))), 1.0)
            if abs(correlation) <= 1e-12 * scale:
                raise UQContractError(
                    "degenerate_doa_map",
                    "DOA phase derivative is undefined for a near-zero adjacent correlation"
                )
            output_row = 2 * component + output_offset
            for sensor in range(column.size):
                derivative_real = 0.0j
                derivative_imag = 0.0j
                if sensor < column.size - 1:
                    derivative_real += column[sensor + 1]
                    derivative_imag += -1j * column[sensor + 1]
                if sensor > 0:
                    derivative_real += np.conj(column[sensor - 1])
                    derivative_imag += 1j * np.conj(column[sensor - 1])
                packed_index = mode_offset + sensor * rank + component
                jacobian[output_row, packed_index] = np.imag(
                    derivative_real / correlation
                ) / np.pi
                jacobian[output_row, complex_size + packed_index] = np.imag(
                    derivative_imag / correlation
                ) / np.pi
    return jacobian


def _noise_variance(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    mask: np.ndarray,
    effective_parameters: int,
) -> tuple[float, int]:
    residual = np.where(mask, reconstruct(factors) - observed, 0.0)
    real_observations = 2 * int(np.count_nonzero(mask))
    residual_degrees = real_observations - effective_parameters
    if residual_degrees <= 0:
        raise UQContractError(
            "nonpositive_residual_df", "Nonpositive residual degrees of freedom"
        )
    variance = float(np.vdot(residual, residual).real) / residual_degrees
    return variance, residual_degrees


def _covariance_blocks(
    factor_covariance: np.ndarray, factors: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    doa_jacobian = doa_analytic_jacobian(factors)
    doa_covariance = doa_jacobian @ factor_covariance @ doa_jacobian.T
    doa_covariance = 0.5 * (doa_covariance + doa_covariance.T)
    rank = factors["ax"].shape[1]
    blocks = np.stack(
        [
            doa_covariance[2 * target : 2 * target + 2, 2 * target : 2 * target + 2]
            for target in range(rank)
        ]
    )
    return blocks, doa_jacobian


def _matrix_rank(values: np.ndarray, rcond: float) -> tuple[int, float, float]:
    singular_values = np.linalg.svd(values, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return 0, 0.0, np.inf
    cutoff = rcond * singular_values[0]
    rank = int(np.count_nonzero(singular_values > cutoff))
    smallest = float(singular_values[rank - 1]) if rank else 0.0
    condition = float(singular_values[0] / smallest) if smallest else np.inf
    return rank, cutoff, condition


def _gauge_restricted_doa_covariance(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    residual_mask: np.ndarray,
    information_mask: np.ndarray,
    *,
    covariance_scale: float,
    canonicalize: bool = True,
    slice_mode: str = "constraints",
    rcond: float = DEFAULT_RCOND,
    inversion_mode: str = "pinv",
    invariance_tolerance: float = DEFAULT_INVARIANCE_TOLERANCE,
) -> tuple[np.ndarray, dict[str, float]]:
    _validate_factors(factors)
    tensor_shape = reconstruct(factors).shape
    for name, mask in (
        ("Residual", residual_mask),
        ("Information", information_mask),
    ):
        if mask.shape != tensor_shape:
            raise UQContractError(
                "numerical_failure", f"{name} mask and CP reconstruction shapes differ"
            )
        if np.asarray(mask).dtype != np.bool_:
            raise UQContractError("numerical_failure", f"{name} mask must be boolean")
    if not np.isfinite(covariance_scale) or covariance_scale <= 0.0:
        raise UQContractError("numerical_failure", "Covariance scale must be positive and finite")
    expected_effective_parameters = pack_factors(factors).size - 4 * factors["ax"].shape[1]
    if 2 * int(np.count_nonzero(residual_mask)) <= expected_effective_parameters:
        raise UQContractError(
            "nonpositive_residual_df", "Nonpositive residual degrees of freedom"
        )
    factors = canonicalize_factors(factors) if canonicalize else {
        name: np.asarray(factors[name], dtype=np.complex128).copy() for name in FACTOR_NAMES
    }
    model_jacobian = complex_model_jacobian(factors, information_mask)
    tangent_basis, generators, constraints = gauge_slice_basis(factors, slice_mode)
    reduced_jacobian = model_jacobian @ tangent_basis
    information = reduced_jacobian.T @ reduced_jacobian
    if inversion_mode == "pinv":
        information_inverse = np.linalg.pinv(information, rcond=rcond, hermitian=True)
        jacobian_rcond = np.sqrt(rcond)
    elif inversion_mode == "structural_svd":
        try:
            _, singular_values, right_vectors = np.linalg.svd(
                reduced_jacobian, full_matrices=False
            )
        except np.linalg.LinAlgError:
            from scipy.linalg import svd

            _, singular_values, right_vectors = svd(
                reduced_jacobian, full_matrices=False, lapack_driver="gesvd"
            )
        structural_cutoff = (
            np.finfo(float).eps
            * max(reduced_jacobian.shape)
            * singular_values[0]
        )
        if np.any(singular_values <= structural_cutoff):
            inverse_squared = np.where(
                singular_values > structural_cutoff, singular_values ** -2, 0.0
            )
        else:
            inverse_squared = singular_values ** -2
        information_inverse = (right_vectors.T * inverse_squared) @ right_vectors
        jacobian_rcond = structural_cutoff / singular_values[0]
    else:
        raise ValueError(f"Unknown inversion mode: {inversion_mode}")
    gauge_rank = int(np.linalg.matrix_rank(generators))
    effective_parameters = tangent_basis.shape[1]
    real_noise_variance, residual_degrees = _noise_variance(
        factors, observed, residual_mask, effective_parameters
    )
    factor_covariance = (
        tangent_basis
        @ information_inverse
        @ tangent_basis.T
        * real_noise_variance
        * covariance_scale
    )
    blocks, doa_jacobian = _covariance_blocks(factor_covariance, factors)
    jacobian_rank, _, jacobian_condition = _matrix_rank(model_jacobian, jacobian_rcond)
    reduced_rank, cutoff, reduced_condition = _matrix_rank(
        reduced_jacobian, jacobian_rcond
    )
    gauge_residual = np.linalg.norm(model_jacobian @ generators) / max(
        np.linalg.norm(model_jacobian) * np.linalg.norm(generators), 1e-30
    )
    doa_gauge_residual = np.linalg.norm(doa_jacobian @ generators) / max(
        np.linalg.norm(doa_jacobian) * np.linalg.norm(generators), 1e-30
    )
    transversality = np.linalg.svd(constraints @ generators, compute_uv=False)
    minimum_eigenvalue = float(min(np.min(np.linalg.eigvalsh(block)) for block in blocks))
    reduced_singular_values = np.linalg.svd(reduced_jacobian, compute_uv=False)
    diagnostics = {
        "gauge_rank": float(gauge_rank),
        "expected_gauge_rank": float(4 * factors["ax"].shape[1]),
        "tangent_dimension": float(tangent_basis.shape[1]),
        "jacobian_rank": float(jacobian_rank),
        "reduced_jacobian_rank": float(reduced_rank),
        "jacobian_condition": jacobian_condition,
        "information_condition": float(reduced_condition**2),
        "svd_cutoff": cutoff,
        "gauge_null_residual": float(gauge_residual),
        "doa_gauge_residual": float(doa_gauge_residual),
        "slice_transversality": float(np.min(transversality)),
        "residual_observations": float(np.count_nonzero(residual_mask)),
        "information_observations": float(np.count_nonzero(information_mask)),
        "residual_degrees": float(residual_degrees),
        "real_noise_variance": real_noise_variance,
        "observation_rate": float(np.mean(residual_mask)),
        "covariance_scale": float(covariance_scale),
        "minimum_doa_eigenvalue": minimum_eigenvalue,
        "smallest_identifiable_singular_value": float(reduced_singular_values[-1]),
        "uq_valid": float(
            gauge_rank == 4 * factors["ax"].shape[1]
            and reduced_rank == effective_parameters
            and np.isfinite(blocks).all()
            and minimum_eigenvalue > 0.0
            and gauge_residual <= invariance_tolerance
            and doa_gauge_residual <= invariance_tolerance
        ),
    }
    return blocks, diagnostics


def gauge_restricted_doa_covariance(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    mask: np.ndarray,
    *,
    canonicalize: bool = True,
    slice_mode: str = "constraints",
    rcond: float = DEFAULT_RCOND,
    inversion_mode: str = "pinv",
    invariance_tolerance: float = DEFAULT_INVARIANCE_TOLERANCE,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate DOA covariance from the actual observation pattern."""
    return _gauge_restricted_doa_covariance(
        factors,
        observed,
        mask,
        mask,
        covariance_scale=1.0,
        canonicalize=canonicalize,
        slice_mode=slice_mode,
        rcond=rcond,
        inversion_mode=inversion_mode,
        invariance_tolerance=invariance_tolerance,
    )


def fitted_full_rate_scaled_doa_covariance(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    observed_mask: np.ndarray,
    *,
    canonicalize: bool = True,
    slice_mode: str = "constraints",
    rcond: float = DEFAULT_RCOND,
    inversion_mode: str = "pinv",
    invariance_tolerance: float = DEFAULT_INVARIANCE_TOLERANCE,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate full-design information at the fitted point and scale by 1/rho.

    The residual variance is estimated only on ``observed_mask``.  No truth
    factors or unobserved noisy entries enter this comparison covariance.
    """
    observation_rate = float(np.mean(observed_mask))
    if not np.isfinite(observation_rate) or observation_rate <= 0.0:
        raise UQContractError("numerical_failure", "Observation rate must be positive")
    information_mask = np.ones(np.asarray(observed_mask).shape, dtype=bool)
    return _gauge_restricted_doa_covariance(
        factors,
        observed,
        observed_mask,
        information_mask,
        covariance_scale=1.0 / observation_rate,
        canonicalize=canonicalize,
        slice_mode=slice_mode,
        rcond=rcond,
        inversion_mode=inversion_mode,
        invariance_tolerance=invariance_tolerance,
    )


def gauge_reduced_information_diagnostics(
    factors: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    canonicalize: bool = False,
    slice_mode: str = "orthogonal",
) -> tuple[dict[str, float], str, str]:
    """Diagnose local regularity without observations or a noise estimate."""
    _validate_factors(factors)
    tensor_shape = reconstruct(factors).shape
    if np.asarray(mask).shape != tensor_shape:
        raise UQContractError("numerical_failure", "Mask and CP reconstruction shapes differ")
    if np.asarray(mask).dtype != np.bool_:
        raise UQContractError("numerical_failure", "Observation mask must be boolean")
    factors = canonicalize_factors(factors) if canonicalize else {
        name: np.asarray(factors[name], dtype=np.complex128).copy()
        for name in FACTOR_NAMES
    }
    model_jacobian = complex_model_jacobian(factors, mask)
    tangent_basis, generators, _ = gauge_slice_basis(factors, slice_mode)
    reduced_jacobian = model_jacobian @ tangent_basis
    singular_values = np.linalg.svd(reduced_jacobian, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    cutoff = np.finfo(float).eps * max(reduced_jacobian.shape) * largest
    reduced_rank = int(np.count_nonzero(singular_values > cutoff))
    tangent_dimension = int(tangent_basis.shape[1])
    smallest = (
        float(singular_values[-1])
        if singular_values.size == tangent_dimension
        else 0.0
    )
    condition = (
        float(largest / smallest)
        if reduced_rank == tangent_dimension and smallest > 0.0
        else np.inf
    )
    gauge_rank = int(np.linalg.matrix_rank(generators))
    expected_gauge_rank = 4 * factors["ax"].shape[1]
    gauge_residual = float(
        np.linalg.norm(model_jacobian @ generators)
        / max(np.linalg.norm(model_jacobian) * np.linalg.norm(generators), 1e-30)
    )
    diagnostics = {
        "gauge_rank": float(gauge_rank),
        "expected_gauge_rank": float(expected_gauge_rank),
        "tangent_dimension": float(tangent_dimension),
        "reduced_jacobian_rank": float(reduced_rank),
        "jacobian_condition": condition,
        "information_condition": float(condition**2),
        "svd_cutoff": float(cutoff),
        "gauge_null_residual": gauge_residual,
        "smallest_identifiable_singular_value": smallest,
    }
    if gauge_rank != expected_gauge_rank or gauge_residual > DEFAULT_INVARIANCE_TOLERANCE:
        return diagnostics, "gauge_rank_deficient", "Gauge structure is not regular"
    if reduced_rank != tangent_dimension:
        return diagnostics, "nonregular_information", "Gauge-reduced Jacobian is rank deficient"
    return diagnostics, "valid", ""


def uq_contract_status(
    diagnostics: dict[str, float], *, fit_status: str = "converged"
) -> tuple[str, str]:
    """Map diagnostics to one mutually exclusive conference-paper UQ status."""
    if fit_status != "converged":
        return "fit_not_converged", "Point estimator did not converge"
    if not np.isfinite(float(diagnostics.get("gauge_rank", np.nan))):
        return "numerical_failure", "Gauge-rank diagnostic is non-finite"
    if int(round(float(diagnostics["gauge_rank"]))) != int(
        round(float(diagnostics.get("expected_gauge_rank", -1)))
    ):
        return "gauge_rank_deficient", "Gauge generators do not have the expected structural rank"
    if float(diagnostics.get("gauge_null_residual", np.inf)) > DEFAULT_INVARIANCE_TOLERANCE:
        return "gauge_rank_deficient", "Gauge directions are not numerical model-null directions"
    if float(diagnostics.get("doa_gauge_residual", np.inf)) > DEFAULT_INVARIANCE_TOLERANCE:
        return "degenerate_doa_map", "DOA functional is not numerically gauge invariant"
    if int(round(float(diagnostics.get("reduced_jacobian_rank", -1)))) != int(
        round(float(diagnostics.get("tangent_dimension", -2)))
    ):
        return "nonregular_information", "Gauge-reduced information is rank deficient"
    if not np.isfinite(float(diagnostics.get("minimum_doa_eigenvalue", np.nan))):
        return "numerical_failure", "DOA covariance contains non-finite values"
    if float(diagnostics["minimum_doa_eigenvalue"]) <= 0.0:
        return "nonpositive_doa_covariance", "DOA covariance is not positive definite"
    if not bool(diagnostics.get("uq_valid", 0.0)):
        return "gauge_rank_deficient", "Gauge generators do not have the expected structural rank"
    return "valid", ""


def naive_pseudoinverse_doa_covariance(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    mask: np.ndarray,
    *,
    canonicalize: bool = False,
    rcond: float = DEFAULT_RCOND,
) -> tuple[np.ndarray, dict[str, float]]:
    """Propagate the full singular-information Moore--Penrose covariance."""
    factors = canonicalize_factors(factors) if canonicalize else {
        name: np.asarray(factors[name], dtype=np.complex128).copy() for name in FACTOR_NAMES
    }
    model_jacobian = complex_model_jacobian(factors, mask)
    generators = gauge_generator_matrix(factors)
    gauge_rank = int(np.linalg.matrix_rank(generators))
    effective_parameters = model_jacobian.shape[1] - gauge_rank
    real_noise_variance, residual_degrees = _noise_variance(
        factors, observed, mask, effective_parameters
    )
    information = model_jacobian.T @ model_jacobian
    factor_covariance = (
        np.linalg.pinv(information, rcond=rcond, hermitian=True) * real_noise_variance
    )
    blocks, doa_jacobian = _covariance_blocks(factor_covariance, factors)
    jacobian_rank, cutoff, jacobian_condition = _matrix_rank(model_jacobian, np.sqrt(rcond))
    gauge_residual = np.linalg.norm(model_jacobian @ generators) / max(
        np.linalg.norm(model_jacobian) * np.linalg.norm(generators), 1e-30
    )
    doa_gauge_residual = np.linalg.norm(doa_jacobian @ generators) / max(
        np.linalg.norm(doa_jacobian) * np.linalg.norm(generators), 1e-30
    )
    minimum_eigenvalue = float(min(np.min(np.linalg.eigvalsh(block)) for block in blocks))
    diagnostics = {
        "gauge_rank": float(gauge_rank),
        "tangent_dimension": float(effective_parameters),
        "jacobian_rank": float(jacobian_rank),
        "reduced_jacobian_rank": float(jacobian_rank),
        "jacobian_condition": jacobian_condition,
        "information_condition": float(jacobian_condition**2),
        "svd_cutoff": cutoff,
        "gauge_null_residual": float(gauge_residual),
        "doa_gauge_residual": float(doa_gauge_residual),
        "slice_transversality": np.nan,
        "residual_degrees": float(residual_degrees),
        "real_noise_variance": real_noise_variance,
        "minimum_doa_eigenvalue": minimum_eigenvalue,
        "smallest_identifiable_singular_value": float(
            np.linalg.svd(model_jacobian, compute_uv=False)[effective_parameters - 1]
        ),
        "uq_valid": float(
            gauge_rank == 4 * factors["ax"].shape[1]
            and jacobian_rank == effective_parameters
            and np.isfinite(blocks).all()
            and minimum_eigenvalue > 0.0
        ),
    }
    return blocks, diagnostics


def structural_pseudoinverse_doa_covariance(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    mask: np.ndarray,
    *,
    canonicalize: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Propagate a full-Jacobian pseudoinverse at the known structural rank.

    This is an E2 ablation, not the Proposed definition.  It retains exactly
    ``n - 4K`` singular directions and therefore separates structural-rank MP
    equivalence from a fixed relative-threshold pseudoinverse.
    """
    factors = canonicalize_factors(factors) if canonicalize else {
        name: np.asarray(factors[name], dtype=np.complex128).copy() for name in FACTOR_NAMES
    }
    model_jacobian = complex_model_jacobian(factors, mask)
    generators = gauge_generator_matrix(factors)
    gauge_rank = int(np.linalg.matrix_rank(generators))
    expected_rank = model_jacobian.shape[1] - 4 * factors["ax"].shape[1]
    _, singular_values, right_vectors = np.linalg.svd(model_jacobian, full_matrices=False)
    retained = singular_values[:expected_rank]
    structural_cutoff = (
        np.finfo(float).eps * max(model_jacobian.shape) * singular_values[0]
    )
    regular = bool(retained.size == expected_rank and np.all(retained > structural_cutoff))
    inverse_squared = np.where(retained > structural_cutoff, retained ** -2, 0.0)
    identifiable_vectors = right_vectors[:expected_rank].T
    factor_covariance = (identifiable_vectors * inverse_squared) @ identifiable_vectors.T
    real_noise_variance, residual_degrees = _noise_variance(
        factors, observed, mask, expected_rank
    )
    factor_covariance *= real_noise_variance
    blocks, doa_jacobian = _covariance_blocks(factor_covariance, factors)
    minimum_eigenvalue = float(min(np.min(np.linalg.eigvalsh(block)) for block in blocks))
    condition = float(retained[0] / retained[-1]) if retained[-1] > 0.0 else np.inf
    diagnostics = {
        "gauge_rank": float(gauge_rank),
        "tangent_dimension": float(expected_rank),
        "jacobian_rank": float(expected_rank if regular else np.count_nonzero(retained > structural_cutoff)),
        "reduced_jacobian_rank": float(expected_rank if regular else np.count_nonzero(retained > structural_cutoff)),
        "jacobian_condition": condition,
        "information_condition": float(condition**2),
        "svd_cutoff": float(structural_cutoff),
        "gauge_null_residual": float(
            np.linalg.norm(model_jacobian @ generators)
            / max(np.linalg.norm(model_jacobian) * np.linalg.norm(generators), 1e-30)
        ),
        "doa_gauge_residual": float(
            np.linalg.norm(doa_jacobian @ generators)
            / max(np.linalg.norm(doa_jacobian) * np.linalg.norm(generators), 1e-30)
        ),
        "slice_transversality": np.nan,
        "residual_degrees": float(residual_degrees),
        "real_noise_variance": real_noise_variance,
        "minimum_doa_eigenvalue": minimum_eigenvalue,
        "smallest_identifiable_singular_value": float(retained[-1]),
        "uq_valid": float(
            gauge_rank == 4 * factors["ax"].shape[1]
            and regular
            and np.isfinite(blocks).all()
            and minimum_eigenvalue > 0.0
        ),
    }
    return blocks, diagnostics


def gauge_parameterization_sensitivity(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    mask: np.ndarray,
    reference_covariance: np.ndarray,
    *,
    method: str,
    samples: int,
    seed: int,
    minimum_scale: float = 0.25,
    maximum_scale: float = 4.0,
    slice_mode: str = "orthogonal",
    rcond: float = DEFAULT_RCOND,
    inversion_mode: str = "pinv",
    include_canonical: bool = True,
) -> dict[str, float]:
    """Stress a DOA covariance under equivalent raw CP parameterizations."""
    if samples < 0:
        return {}
    rng = np.random.default_rng(seed)
    rank = factors["ax"].shape[1]
    relative_changes = []
    area_ratios = []
    rank_changes = 0
    invalid = 0
    reference_areas = np.sqrt(
        np.maximum(np.linalg.det(reference_covariance), np.finfo(float).tiny)
    )
    transformed_factors = []
    if include_canonical:
        transformed_factors.append(canonicalize_factors(factors))
    for _ in range(samples):
        magnitudes_ax = np.exp(
            rng.uniform(np.log(minimum_scale), np.log(maximum_scale), rank)
        )
        magnitudes_ay = np.exp(
            rng.uniform(np.log(minimum_scale), np.log(maximum_scale), rank)
        )
        ax_scale = magnitudes_ax * np.exp(1j * rng.uniform(-np.pi, np.pi, rank))
        ay_scale = magnitudes_ay * np.exp(1j * rng.uniform(-np.pi, np.pi, rank))
        transformed = {name: np.asarray(value).copy() for name, value in factors.items()}
        transformed["ax"] *= ax_scale
        transformed["ay"] *= ay_scale
        transformed["signals"] /= ax_scale * ay_scale
        transformed_factors.append(transformed)
    if not transformed_factors:
        return {}
    for transformed in transformed_factors:
        if method == "gauge_slice":
            covariance, diagnostics = gauge_restricted_doa_covariance(
                transformed,
                observed,
                mask,
                canonicalize=False,
                slice_mode=slice_mode,
                rcond=rcond,
                inversion_mode=inversion_mode,
            )
        elif method == "naive_mp":
            covariance, diagnostics = naive_pseudoinverse_doa_covariance(
                transformed, observed, mask, canonicalize=False, rcond=rcond
            )
        else:
            raise ValueError(f"Unknown covariance method: {method}")
        invalid += int(not diagnostics["uq_valid"])
        expected_rank = int(diagnostics["tangent_dimension"])
        rank_changes += int(int(diagnostics["reduced_jacobian_rank"]) != expected_rank)
        relative_changes.append(
            float(
                np.linalg.norm(covariance - reference_covariance)
                / max(np.linalg.norm(reference_covariance), np.finfo(float).tiny)
            )
        )
        areas = np.sqrt(np.maximum(np.linalg.det(covariance), np.finfo(float).tiny))
        ratios = areas / reference_areas
        area_ratios.append(float(np.max(np.maximum(ratios, 1.0 / ratios))))
    return {
        "parameterization_max_relative_change": float(np.max(relative_changes)),
        "parameterization_median_relative_change": float(np.median(relative_changes)),
        "parameterization_max_area_ratio": float(np.max(area_ratios)),
        "parameterization_rank_changes": float(rank_changes),
        "parameterization_invalid_count": float(invalid),
    }


def rcond_sensitivity(
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    mask: np.ndarray,
    reference_covariance: np.ndarray,
    *,
    method: str,
    rconds: list[float],
    slice_mode: str = "orthogonal",
    inversion_mode: str = "pinv",
) -> dict[str, float]:
    """Measure covariance and rank sensitivity to predeclared pinv thresholds."""
    if not rconds:
        return {}
    relative_changes = []
    area_ratios = []
    rank_changes = 0
    invalid = 0
    reference_areas = np.sqrt(
        np.maximum(np.linalg.det(reference_covariance), np.finfo(float).tiny)
    )
    for rcond in rconds:
        if method == "gauge_slice":
            covariance, diagnostics = gauge_restricted_doa_covariance(
                factors,
                observed,
                mask,
                canonicalize=False,
                slice_mode=slice_mode,
                rcond=float(rcond),
                inversion_mode=inversion_mode,
            )
        elif method == "naive_mp":
            covariance, diagnostics = naive_pseudoinverse_doa_covariance(
                factors, observed, mask, canonicalize=False, rcond=float(rcond)
            )
        else:
            raise ValueError(f"Unknown covariance method: {method}")
        invalid += int(not diagnostics["uq_valid"])
        expected_rank = int(diagnostics["tangent_dimension"])
        rank_changes += int(int(diagnostics["reduced_jacobian_rank"]) != expected_rank)
        relative_changes.append(
            float(
                np.linalg.norm(covariance - reference_covariance)
                / max(np.linalg.norm(reference_covariance), np.finfo(float).tiny)
            )
        )
        areas = np.sqrt(np.maximum(np.linalg.det(covariance), np.finfo(float).tiny))
        ratios = areas / reference_areas
        area_ratios.append(float(np.max(np.maximum(ratios, 1.0 / ratios))))
    return {
        "rcond_max_relative_change": float(np.max(relative_changes)),
        "rcond_max_area_ratio": float(np.max(area_ratios)),
        "rcond_rank_changes": float(rank_changes),
        "rcond_invalid_count": float(invalid),
    }
