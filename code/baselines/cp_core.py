"""Shared complex weighted CP objective and optimizer."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import minimize


FACTOR_NAMES = ("ax", "ay", "signals")


def pack_factors(factors: dict[str, np.ndarray]) -> np.ndarray:
    complex_vector = np.concatenate([np.asarray(factors[name]).ravel() for name in FACTOR_NAMES])
    return np.concatenate((complex_vector.real, complex_vector.imag))


def unpack_factors(vector: np.ndarray, shape: tuple[int, int, int], rank: int) -> dict[str, np.ndarray]:
    complex_size = rank * sum(shape)
    complex_vector = vector[:complex_size] + 1j * vector[complex_size:]
    factors: dict[str, np.ndarray] = {}
    offset = 0
    for name, rows in zip(FACTOR_NAMES, shape):
        size = rows * rank
        factors[name] = complex_vector[offset : offset + size].reshape(rows, rank)
        offset += size
    return factors


def reconstruct(factors: dict[str, np.ndarray]) -> np.ndarray:
    return np.einsum(
        "mk,nk,lk->mnl",
        factors["ax"],
        factors["ay"],
        factors["signals"],
        optimize=True,
    )


def rebalance_columns(factors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fix CP column-scale freedom without changing the reconstructed tensor."""
    balanced = {name: np.asarray(factors[name]).copy() for name in FACTOR_NAMES}
    norms = np.asarray(
        [np.linalg.norm(balanced[name], axis=0) for name in FACTOR_NAMES],
        dtype=float,
    )
    if not np.isfinite(norms).all():
        raise ValueError("Cannot rebalance CP factors with non-finite column norms")
    if np.any(norms == 0.0):
        raise ValueError("Cannot rebalance CP factors containing a zero column")
    common_log_norm = np.mean(np.log(norms), axis=0)
    for mode, name in enumerate(FACTOR_NAMES):
        scale = np.exp(common_log_norm - np.log(norms[mode]))
        balanced[name] *= scale[None, :]
    return balanced


def _weighted_als_warm_start(
    observed: np.ndarray,
    mask: np.ndarray,
    initialization: dict[str, np.ndarray],
    config: dict,
) -> tuple[dict[str, np.ndarray], int]:
    factors = rebalance_columns(initialization)
    maximum_iterations = int(config.get("warm_start_iterations", 0))
    tolerance = float(config.get("warm_start_tolerance", 1e-6))
    ridge = float(config.get("warm_start_ridge", 1e-8))
    previous_fit = np.where(mask, reconstruct(factors), 0.0)
    mode_others = (("ay", "signals"), ("ax", "signals"), ("ax", "ay"))
    for iteration in range(1, maximum_iterations + 1):
        for mode, name in enumerate(FACTOR_NAMES):
            data = np.moveaxis(observed, mode, 0)
            data_mask = np.moveaxis(mask, mode, 0)
            first_name, second_name = mode_others[mode]
            first, second = factors[first_name], factors[second_name]
            updated = factors[name].copy()
            for row in range(data.shape[0]):
                first_index, second_index = np.nonzero(data_mask[row])
                if first_index.size == 0:
                    continue
                design = first[first_index] * second[second_index]
                gram = design.conj().T @ design + ridge * np.eye(design.shape[1])
                right_hand_side = design.conj().T @ data[row][first_index, second_index]
                try:
                    updated[row] = np.linalg.solve(gram, right_hand_side)
                except np.linalg.LinAlgError:
                    updated[row] = np.linalg.lstsq(design, data[row][first_index, second_index], rcond=None)[0]
            factors[name] = updated
        factors = rebalance_columns(factors)
        current_fit = np.where(mask, reconstruct(factors), 0.0)
        relative_change = np.linalg.norm(current_fit - previous_fit) / max(
            np.linalg.norm(previous_fit), 1e-12
        )
        previous_fit = current_fit
        if relative_change < tolerance:
            return factors, iteration
    return factors, maximum_iterations


def objective_and_gradient(
    vector: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    rank: int,
    balance_weight: float = 0.0,
) -> tuple[float, np.ndarray]:
    factors = unpack_factors(vector, observed.shape, rank)
    residual = np.where(mask, reconstruct(factors) - observed, 0.0)
    observed_count = max(int(np.count_nonzero(mask)), 1)
    objective = 0.5 * float(np.vdot(residual, residual).real) / observed_count
    ax, ay, signals = factors["ax"], factors["ay"], factors["signals"]
    gradients = {
        "ax": np.einsum("mnl,nk,lk->mk", residual, np.conj(ay), np.conj(signals), optimize=True),
        "ay": np.einsum("mnl,mk,lk->nk", residual, np.conj(ax), np.conj(signals), optimize=True),
        "signals": np.einsum("mnl,mk,nk->lk", residual, np.conj(ax), np.conj(ay), optimize=True),
    }
    squared_norms = {
        name: np.sum(np.abs(factors[name]) ** 2, axis=0) for name in FACTOR_NAMES
    }
    ax_norm, ay_norm, signal_norm = (squared_norms[name] for name in FACTOR_NAMES)
    norm_differences = (
        ax_norm - ay_norm,
        ax_norm - signal_norm,
        ay_norm - signal_norm,
    )
    balance_penalty = 0.25 * balance_weight * sum(
        float(np.sum(difference**2)) for difference in norm_differences
    )
    objective += balance_penalty
    gradients["ax"] += balance_weight * (
        (ax_norm - ay_norm) + (ax_norm - signal_norm)
    )[None, :] * ax
    gradients["ay"] += balance_weight * (
        (ay_norm - ax_norm) + (ay_norm - signal_norm)
    )[None, :] * ay
    gradients["signals"] += balance_weight * (
        (signal_norm - ax_norm) + (signal_norm - ay_norm)
    )[None, :] * signals
    complex_gradient = np.concatenate([gradients[name].ravel() for name in FACTOR_NAMES]) / observed_count
    real_gradient = np.concatenate((complex_gradient.real, complex_gradient.imag))
    return objective, real_gradient


def optimize_masked_cp(
    observed: np.ndarray,
    mask: np.ndarray,
    rank: int,
    initialization,
    config: dict,
) -> dict:
    initializations = initialization if isinstance(initialization, list) else [initialization]
    balance_weight = float(config.get("balance_weight", 0.0))
    results = []
    warm_start_iterations = []
    for candidate in initializations:
        warm_start, warm_iterations = _weighted_als_warm_start(
            observed, mask, candidate, config
        )
        warm_start_iterations.append(warm_iterations)
        results.append(
            minimize(
                objective_and_gradient,
                pack_factors(warm_start),
                args=(observed, mask, rank, balance_weight),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": int(config["max_iterations"]),
                    "ftol": float(config["relative_tolerance"]),
                    "gtol": float(config["gradient_tolerance"]),
                    "maxls": 40,
                },
            )
        )
    selected_restart = int(np.argmin([candidate.fun for candidate in results]))
    result = results[selected_restart]
    factors = rebalance_columns(unpack_factors(result.x, observed.shape, rank))
    data_objective, data_gradient = objective_and_gradient(
        result.x, observed, mask, rank, balance_weight=0.0
    )
    status = "converged" if result.success else "not_converged"
    return {
        "factors": factors,
        "status": status,
        "iterations": int(result.nit),
        "warm_start_iterations": int(warm_start_iterations[selected_restart]),
        "total_iterations": int(
            sum(warm_start_iterations) + sum(candidate.nit for candidate in results)
        ),
        "selected_restart": selected_restart,
        "objective": float(data_objective),
        "regularized_objective": float(result.fun),
        "gradient_norm": float(np.linalg.norm(data_gradient)),
        "regularized_gradient_norm": float(np.linalg.norm(result.jac)),
        "failure_reason": "" if result.success else str(result.message),
    }
