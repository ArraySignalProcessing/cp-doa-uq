"""Shared fitting and diagnostic operations for the paper experiments."""
from __future__ import annotations
import time
import numpy as np
from baselines.cp_core import optimize_masked_cp, rebalance_columns
from baselines.uq_core import (UQContractError, gauge_reduced_information_diagnostics,
    gauge_restricted_doa_covariance, uq_contract_status)
from common import (angular_errors_deg, common_initialization, cp_tensor,
    factors_to_uv, match_targets, steering, uv_to_angles)
from metrics import ellipse_diagnostics
from protocol import materialize_condition, validate_terminal_status

def _fit_once(observed, mask, rank, initializations, optimizer_config):
    """Retain numerical failures as planned runs, with a finite fallback for diagnostics."""
    try:
        fit = optimize_masked_cp(observed, mask, rank, initializations, optimizer_config)
        if not all(np.isfinite(value).all() for value in fit["factors"].values()):
            raise FloatingPointError("Point fit returned non-finite factors")
        return fit
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return {"factors": initializations[0], "status": "numerical_failure",
                "failure_reason": str(error), "iterations": 0}


def _uq_result(
    covariance_function,
    factors: dict[str, np.ndarray],
    observed: np.ndarray,
    mask: np.ndarray,
    method_config: dict,
) -> dict:
    started = time.perf_counter()
    try:
        covariance, diagnostics = covariance_function(
            factors,
            observed,
            mask,
            canonicalize=False,
            slice_mode=str(method_config.get("slice_mode", "orthogonal")),
            rcond=float(method_config.get("rcond", 1.0e-10)),
            inversion_mode=str(method_config.get("inversion_mode", "structural_svd")),
        )
        status, reason = uq_contract_status(diagnostics, fit_status="converged")
        if status != "valid":
            covariance = None
    except UQContractError as error:
        covariance = None
        diagnostics = {}
        status = error.status
        reason = str(error)
    except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
        covariance = None
        diagnostics = {}
        status = "numerical_failure"
        reason = str(error)
    validate_terminal_status(status)
    return {
        "covariance": covariance,
        "diagnostics": diagnostics,
        "status": status,
        "failure_reason": reason,
        "time": time.perf_counter() - started,
    }


def evaluate_outer_instance(
    *,
    clean_tensor: np.ndarray,
    standard_noise: np.ndarray,
    mask_uniform: np.ndarray,
    true_uv: np.ndarray,
    condition: dict,
    initialization_seed: int,
    rank: int,
    optimizer_config: dict,
    method_config: dict,
    confidence_levels: list[float],
    correct_branch_max_uv_error: float,
) -> dict:
    """Fit one condition once; method scheduling is handled by the shared runner."""
    full, mask, noise_variance = materialize_condition(
        clean_tensor,
        standard_noise,
        mask_uniform,
        observation_rate=float(condition["observation_rate"]),
        snr_db=float(condition["snr_db"]),
    )
    observed = np.where(mask, full, 0.0)
    observed_rms = float(np.sqrt(np.mean(np.abs(observed[mask]) ** 2))) if np.any(mask) else 1.0
    initialization_started = time.perf_counter()
    initializations = [
        common_initialization(
            clean_tensor.shape,
            rank,
            int(initialization_seed) + restart * 1_000_003,
            observed_rms,
        )
        for restart in range(int(optimizer_config["restarts"]))
    ]
    initialization_time = time.perf_counter() - initialization_started
    fit_started = time.perf_counter()
    fit = _fit_once(observed, mask, rank, initializations, optimizer_config)
    fit_time = time.perf_counter() - fit_started
    factors = rebalance_columns(fit["factors"])
    fit["factors"] = factors
    fit_converged = str(fit.get("status", "")) == "converged"
    raw_uv = factors_to_uv(factors)
    permutation = match_targets(raw_uv, true_uv)
    estimated_uv = raw_uv[permutation]
    errors = estimated_uv - true_uv
    uv_errors = np.linalg.norm(errors, axis=1)
    angular_error = angular_errors_deg(estimated_uv, true_uv)
    correct_branch = bool(np.max(uv_errors) <= float(correct_branch_max_uv_error))

    result = {
        "fit": fit,
        "fit_converged": fit_converged,
        "initializations": initializations,
        "initialization_time": initialization_time,
        "fit_time": fit_time,
        "noise_variance": noise_variance,
        "observed_fraction": float(np.mean(mask)),
        "target_permutation": permutation,
        "true_uv": true_uv,
        "estimated_uv": estimated_uv,
        "estimated_angles_deg": uv_to_angles(estimated_uv),
        "angular_error_deg": angular_error,
        "uv_rmse": float(np.sqrt(np.mean(errors**2))),
        "angular_rmse_deg": float(np.sqrt(np.mean(angular_error**2))),
        "maximum_uv_error": float(np.max(uv_errors)),
        "correct_branch": correct_branch,
        "uq": {},
    }
    if not fit_converged:
        result["terminal_status"] = "fit_not_converged"
        result["total_time"] = initialization_time + fit_time
        return result

    result["terminal_status"] = "valid"
    result["total_time"] = initialization_time + fit_time
    return result


def boundary_uv(separation: float, geometry: dict) -> np.ndarray:
    """Move only the controlled pair while keeping its center fixed."""
    center = np.asarray(geometry["pair_center_uv"], dtype=float)
    direction = np.asarray(geometry["approach_direction"], dtype=float)
    direction /= np.linalg.norm(direction)
    offset = 0.5 * float(separation) * direction
    return np.vstack((center - offset, center + offset, geometry["fixed_target_uv"]))


def truth_factors(
    shape: tuple[int, int, int], true_uv: np.ndarray, signals: np.ndarray
) -> dict[str, np.ndarray]:
    factors = {
        "ax": np.column_stack([steering(shape[0], value) for value in true_uv[:, 0]]),
        "ay": np.column_stack([steering(shape[1], value) for value in true_uv[:, 1]]),
        "signals": np.asarray(signals).copy(),
    }
    return rebalance_columns(factors)


def evaluate_boundary_run(
    *,
    separation: float,
    geometry: dict,
    signals: np.ndarray,
    standard_noise: np.ndarray,
    mask_uniform: np.ndarray,
    initialization_seed: int,
    shape: tuple[int, int, int],
    rank: int,
    powers: list[float],
    observation_rate: float,
    snr_db: float,
    noise_reference_power: float,
    optimizer_config: dict,
    method_config: dict,
    confidence_levels: list[float],
    correct_branch_max_uv_error: float,
) -> dict:
    """Evaluate one paired separation/repeat instance."""
    true_uv = boundary_uv(separation, geometry)
    unbalanced_truth = {
        "ax": np.column_stack([steering(shape[0], value) for value in true_uv[:, 0]]),
        "ay": np.column_stack([steering(shape[1], value) for value in true_uv[:, 1]]),
        "signals": np.asarray(signals),
    }
    clean = cp_tensor(
        unbalanced_truth["ax"], unbalanced_truth["ay"], unbalanced_truth["signals"]
    )
    full, mask, noise_variance = materialize_condition(
        clean,
        standard_noise,
        mask_uniform,
        observation_rate=observation_rate,
        snr_db=snr_db,
        noise_reference_power=noise_reference_power,
    )
    observed = np.where(mask, full, 0.0)
    initialization_scale = float(np.sqrt(np.mean(np.abs(observed[mask]) ** 2))) if np.any(mask) else 1.0
    initializations = [
        common_initialization(
            shape,
            rank,
            int(initialization_seed) + restart * 1_000_003,
            initialization_scale,
        )
        for restart in range(int(optimizer_config["restarts"]))
    ]
    fit_started = time.perf_counter()
    fit = _fit_once(observed, mask, rank, initializations, optimizer_config)
    fit_time = time.perf_counter() - fit_started
    fitted = rebalance_columns(fit["factors"])
    fit["factors"] = fitted
    truth = truth_factors(shape, true_uv, signals)
    truth_diagnostics, truth_status, truth_reason = gauge_reduced_information_diagnostics(
        truth, mask, canonicalize=False, slice_mode="orthogonal"
    )
    estimated_diagnostics, estimated_regularity_status, estimated_regularity_reason = (
        gauge_reduced_information_diagnostics(
            fitted, mask, canonicalize=False, slice_mode="orthogonal"
        )
    )
    raw_uv = factors_to_uv(fitted)
    permutation = match_targets(raw_uv, true_uv)
    estimated_uv = raw_uv[permutation]
    errors = estimated_uv - true_uv
    uv_errors = np.linalg.norm(errors, axis=1)
    angular_error = angular_errors_deg(estimated_uv, true_uv)
    correct_branch = bool(np.max(uv_errors) <= correct_branch_max_uv_error)
    covariance = None
    uq_diagnostics = {}
    metrics = None
    uq_started = time.perf_counter()
    if str(fit.get("status", "")) != "converged":
        uq_status = "fit_not_converged"
        uq_reason = "Point estimator did not converge"
    else:
        try:
            covariance, uq_diagnostics = gauge_restricted_doa_covariance(
                fitted,
                observed,
                mask,
                canonicalize=False,
                slice_mode=str(method_config.get("slice_mode", "orthogonal")),
                rcond=float(method_config.get("rcond", 1.0e-10)),
                inversion_mode=str(method_config.get("inversion_mode", "structural_svd")),
            )
            uq_status, uq_reason = uq_contract_status(
                uq_diagnostics, fit_status=str(fit["status"])
            )
            if uq_status == "valid":
                covariance = np.asarray(covariance)[permutation]
                metrics = ellipse_diagnostics(errors, covariance, confidence_levels)
            else:
                covariance = None
        except UQContractError as error:
            uq_status = error.status
            uq_reason = str(error)
            covariance = None
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            uq_status = "numerical_failure"
            uq_reason = str(error)
            covariance = None
    validate_terminal_status(uq_status)
    uq_time = time.perf_counter() - uq_started
    return {
        "fit": fit,
        "fit_time": fit_time,
        "uq_time": uq_time,
        "total_time": fit_time + uq_time,
        "initializations": initializations,
        "true_uv": true_uv,
        "estimated_uv": estimated_uv,
        "target_permutation": permutation,
        "angular_error_deg": angular_error,
        "angular_rmse_deg": float(np.sqrt(np.mean(angular_error**2))),
        "maximum_uv_error": float(np.max(uv_errors)),
        "correct_branch": correct_branch,
        "noise_variance": noise_variance,
        "observed_fraction": float(np.mean(mask)),
        "truth_diagnostics": truth_diagnostics,
        "truth_regularity_status": truth_status,
        "truth_regularity_reason": truth_reason,
        "estimated_regularity_diagnostics": estimated_diagnostics,
        "estimated_regularity_status": estimated_regularity_status,
        "estimated_regularity_reason": estimated_regularity_reason,
        "uq_diagnostics": uq_diagnostics,
        "uq_status": uq_status,
        "uq_failure_reason": uq_reason,
        "terminal_status": uq_status,
        "covariance": covariance,
        "metrics": metrics,
    }


