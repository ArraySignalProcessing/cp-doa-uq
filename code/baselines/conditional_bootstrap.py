"""Fixed-mask bootstrap at a supplied fit, with per-refit prefix checkpoints."""
from __future__ import annotations
import math
import time
import numpy as np
from baselines.cp_core import optimize_masked_cp, rebalance_columns, reconstruct
from common import common_initialization, factors_to_uv, match_targets
from protocol import bootstrap_seed


def prefix_covariance(samples: np.ndarray, valid: np.ndarray, budget: int,
                      minimum_valid_fraction: float = 0.7) -> tuple[np.ndarray | None, str]:
    samples = np.asarray(samples)[:budget]
    valid = np.asarray(valid, dtype=bool)[:budget]
    if len(valid) != budget:
        raise ValueError('Bootstrap prefix is incomplete')
    if int(valid.sum()) < max(2, math.ceil(budget * minimum_valid_fraction)):
        return None, 'bootstrap_insufficient_valid'
    selected = samples[valid]
    covariance = np.stack([np.cov(selected[:, k, :], rowvar=False)
                           for k in range(selected.shape[1])])
    if not np.isfinite(covariance).all() or np.any(np.linalg.eigvalsh(covariance) <= 0):
        return None, 'nonpositive_doa_covariance'
    return covariance, 'valid'


def extend_prefix(group, *, factors: dict, observed: np.ndarray, mask: np.ndarray,
                  optimizer: dict, fit_converged: bool, budget: int, seed: int,
                  condition_index: int, repeat_index: int, stream_index: int = 0,
                  progress=None, outer_completed: int = 0, detail: str = '') -> None:
    """Append exactly the missing refits; failed refits keep their original index."""
    rank = factors['ax'].shape[1]
    if 'samples' not in group:
        group.create_dataset('samples', shape=(0, rank, 2), maxshape=(None, rank, 2), dtype=float)
        for name, dtype in [('valid', bool), ('seconds', float)]:
            group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
        import h5py
        group.create_dataset('status', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
        group.attrs['completed'] = 0
    completed = int(group.attrs['completed'])
    if completed >= budget:
        return
    # Only the committed prefix is authoritative after an interrupted append.
    for dataset in group.values():
        dataset.resize((max(budget, completed),) + dataset.shape[1:])
    setup_started = time.perf_counter()
    fitted = reconstruct(factors)
    reference = factors_to_uv(factors)
    residual = observed[mask] - fitted[mask]
    degrees = 2 * int(mask.sum()) - (2 * rank * sum(fitted.shape) - 4 * rank)
    variance = float(np.vdot(residual, residual).real / degrees) if degrees > 0 else np.nan
    blocked = ('fit_not_converged' if not fit_converged else
               'nonpositive_residual_df' if degrees <= 0 else
               'numerical_failure' if not np.isfinite(variance) or variance <= 0 else '')
    if "setup_seconds" not in group.attrs:
        group.attrs["setup_seconds"] = time.perf_counter() - setup_started
    for index in range(completed, budget):
        started = time.perf_counter()
        sample = np.full((rank, 2), np.nan)
        valid = False
        status = blocked
        if not blocked:
            noise_seed = bootstrap_seed(seed, 'bootstrap_noise', condition_index=condition_index,
                                        scene_index=stream_index, repeat_index=repeat_index,
                                        bootstrap_index=index)
            init_seed = bootstrap_seed(seed, 'bootstrap_initialization', condition_index=condition_index,
                                       scene_index=stream_index, repeat_index=repeat_index,
                                       bootstrap_index=index)
            rng = np.random.default_rng(noise_seed)
            noise = np.sqrt(variance) * (rng.standard_normal(fitted.shape) +
                                         1j * rng.standard_normal(fitted.shape))
            replicate = np.where(mask, fitted + noise, 0.0)
            rms = float(np.sqrt(np.mean(np.abs(replicate[mask]) ** 2)))
            initializations = [common_initialization(fitted.shape, rank,
                               init_seed + restart * 1_000_003, rms)
                               for restart in range(int(optimizer['restarts']))]
            try:
                fit = optimize_masked_cp(replicate, mask, rank, initializations, optimizer)
                status = 'fit_not_converged'
                if str(fit.get('status')) == 'converged':
                    uv = factors_to_uv(rebalance_columns(fit['factors']))
                    if np.isfinite(uv).all():
                        sample = uv[match_targets(uv, reference)]
                        valid, status = True, 'valid'
                    else:
                        status = 'invalid_factor'
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                status = 'numerical_failure: ' + str(error)
        seconds = time.perf_counter() - started
        group['samples'][index], group['valid'][index] = sample, valid
        group['seconds'][index], group['status'][index] = seconds, status
        # Flush data before advancing the commit boundary; I/O is outside measured UQ time.
        group.file.flush()
        group.attrs['completed'] = index + 1
        group.file.flush()
        if progress is not None:
            count = int(np.sum(group['valid'][:index + 1]))
            progress.update_nested(outer_completed=outer_completed, outer_detail=detail,
                                   inner_completed=index + 1, inner_total=budget,
                                   valid=count, failed=index + 1 - count)
