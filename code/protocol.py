"""Random-stream and status contracts for the paper experiments."""

from __future__ import annotations

from typing import Final

import numpy as np

from common import steering


STREAM_IDS: Final[dict[str, int]] = {
    "scene": 11,
    "signals": 23,
    "standard_noise": 37,
    "mask_uniform": 53,
    "initialization": 71,
    "bootstrap_noise": 89,
    "bootstrap_initialization": 107,
    "representation_phase": 127,
}

TERMINAL_STATUSES: Final[tuple[str, ...]] = (
    "valid",
    "fit_not_converged",
    "invalid_factor",
    "nonpositive_residual_df",
    "degenerate_doa_map",
    "gauge_rank_deficient",
    "nonregular_information",
    "nonpositive_doa_covariance",
    "numerical_failure",
)


def protocol_seed(
    master_seed: int,
    stream: str,
    *,
    scene_index: int = 0,
    repeat_index: int = 0,
) -> int:
    """Return a deterministic seed for one named, coordinate-indexed stream."""
    if stream not in STREAM_IDS:
        raise ValueError(f"Unknown random stream: {stream}")
    coordinates = (int(master_seed), int(scene_index), int(repeat_index))
    if any(value < 0 for value in coordinates):
        raise ValueError("Seed coordinates must be nonnegative")
    sequence = np.random.SeedSequence(
        [coordinates[0], STREAM_IDS[stream], coordinates[1], coordinates[2]]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def bootstrap_seed(
    master_seed: int,
    stream: str,
    *,
    condition_index: int,
    scene_index: int,
    repeat_index: int,
    bootstrap_index: int,
) -> int:
    """Return a deterministic seed for one conditional-bootstrap operation."""
    if stream not in ("bootstrap_noise", "bootstrap_initialization"):
        raise ValueError(f"Unknown bootstrap random stream: {stream}")
    coordinates = (
        int(master_seed),
        int(condition_index),
        int(scene_index),
        int(repeat_index),
        int(bootstrap_index),
    )
    if any(value < 0 for value in coordinates):
        raise ValueError("Bootstrap seed coordinates must be nonnegative")
    sequence = np.random.SeedSequence(
        [coordinates[0], STREAM_IDS[stream], *coordinates[1:]]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def representation_seed(
    master_seed: int,
    *,
    instance_index: int,
    exponent_index: int,
    phase_index: int,
) -> int:
    """Return a deterministic seed for one equivalent CP representation."""
    coordinates = (
        int(master_seed),
        int(instance_index),
        int(exponent_index),
        int(phase_index),
    )
    if any(value < 0 for value in coordinates):
        raise ValueError("Representation seed coordinates must be nonnegative")
    sequence = np.random.SeedSequence(
        [coordinates[0], STREAM_IDS["representation_phase"], *coordinates[1:]]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def draw_trial_latents(
    shape: tuple[int, int, int],
    rank: int,
    powers: list[float] | tuple[float, ...] | np.ndarray,
    *,
    master_seed: int,
    scene_index: int,
    repeat_index: int,
) -> dict[str, np.ndarray | int]:
    """Draw condition-independent signals, noise, mask uniforms and init seed."""
    mx, my, snapshots = (int(value) for value in shape)
    rank = int(rank)
    source_powers = np.asarray(powers, dtype=float)[:rank]
    if mx <= 0 or my <= 0 or snapshots <= 0 or rank <= 0:
        raise ValueError("Tensor dimensions and rank must be positive")
    if source_powers.shape != (rank,) or np.any(~np.isfinite(source_powers)):
        raise ValueError("One finite source power is required per component")
    if np.any(source_powers <= 0.0):
        raise ValueError("Source powers must be positive")

    signal_seed = protocol_seed(
        master_seed,
        "signals",
        scene_index=scene_index,
        repeat_index=repeat_index,
    )
    signal_rng = np.random.default_rng(signal_seed)
    signals = signal_rng.standard_normal((snapshots, rank)) + 1j * signal_rng.standard_normal(
        (snapshots, rank)
    )
    signal_norms = np.linalg.norm(signals, axis=0, keepdims=True)
    if np.any(signal_norms == 0.0):
        raise RuntimeError("Random signal draw produced a zero column")
    signals *= np.sqrt(snapshots * source_powers)[None, :] / signal_norms

    noise_seed = protocol_seed(
        master_seed,
        "standard_noise",
        scene_index=scene_index,
        repeat_index=repeat_index,
    )
    noise_rng = np.random.default_rng(noise_seed)
    standard_noise = (
        noise_rng.standard_normal((mx, my, snapshots))
        + 1j * noise_rng.standard_normal((mx, my, snapshots))
    ) / np.sqrt(2.0)

    mask_seed = protocol_seed(
        master_seed,
        "mask_uniform",
        scene_index=scene_index,
        repeat_index=repeat_index,
    )
    mask_uniform = np.random.default_rng(mask_seed).random((mx, my, snapshots))
    initialization_seed = protocol_seed(
        master_seed,
        "initialization",
        scene_index=scene_index,
        repeat_index=repeat_index,
    )
    return {
        "signals": signals,
        "standard_noise": standard_noise,
        "mask_uniform": mask_uniform,
        "signal_seed": signal_seed,
        "noise_seed": noise_seed,
        "mask_seed": mask_seed,
        "initialization_seed": initialization_seed,
    }


def sample_regular_scene(
    shape: tuple[int, int, int],
    rank: int,
    angle_bounds_deg: dict[str, list[float]],
    *,
    minimum_uv_separation: float,
    maximum_spatial_factor_condition: float,
    master_seed: int,
    scene_index: int,
    maximum_attempts: int = 20_000,
) -> dict[str, np.ndarray | int | float]:
    """Sample one fixed locally regular geometry without drawing source signals."""
    mx, my, _ = (int(value) for value in shape)
    rank = int(rank)
    elevation_bounds = np.deg2rad(angle_bounds_deg["elevation"])
    azimuth_bounds = np.deg2rad(angle_bounds_deg["azimuth"])
    scene_seed = protocol_seed(master_seed, "scene", scene_index=scene_index)
    rng = np.random.default_rng(scene_seed)
    for attempt in range(1, int(maximum_attempts) + 1):
        elevation = rng.uniform(elevation_bounds[0], elevation_bounds[1], rank)
        azimuth = rng.uniform(azimuth_bounds[0], azimuth_bounds[1], rank)
        uv = np.column_stack(
            (np.sin(elevation) * np.cos(azimuth), np.sin(elevation) * np.sin(azimuth))
        )
        distances = np.linalg.norm(uv[:, None, :] - uv[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        if rank > 1 and float(np.min(distances)) < float(minimum_uv_separation):
            continue
        ax = np.column_stack([steering(mx, value) for value in uv[:, 0]])
        ay = np.column_stack([steering(my, value) for value in uv[:, 1]])
        condition_ax = float(np.linalg.cond(ax))
        condition_ay = float(np.linalg.cond(ay))
        if max(condition_ax, condition_ay) > float(maximum_spatial_factor_condition):
            continue
        return {
            "true_uv": uv,
            "true_angles_deg": np.rad2deg(np.column_stack((elevation, azimuth))),
            "ax": ax,
            "ay": ay,
            "scene_seed": scene_seed,
            "sampling_attempts": attempt,
            "condition_ax": condition_ax,
            "condition_ay": condition_ay,
        }
    raise RuntimeError(
        f"Could not sample scene {scene_index} within {maximum_attempts} attempts"
    )


def fixed_scene(
    shape: tuple[int, int, int],
    angles_deg: list[dict[str, float]],
    *,
    maximum_spatial_factor_condition: float,
    master_seed: int,
) -> dict[str, np.ndarray | int | float]:
    """Construct the single frozen geometry used by the calibration experiment."""
    mx, my, _ = (int(value) for value in shape)
    elevation = np.deg2rad([float(value["elevation"]) for value in angles_deg])
    azimuth = np.deg2rad([float(value["azimuth"]) for value in angles_deg])
    uv = np.column_stack(
        (np.sin(elevation) * np.cos(azimuth), np.sin(elevation) * np.sin(azimuth))
    )
    ax = np.column_stack([steering(mx, value) for value in uv[:, 0]])
    ay = np.column_stack([steering(my, value) for value in uv[:, 1]])
    condition_ax = float(np.linalg.cond(ax))
    condition_ay = float(np.linalg.cond(ay))
    if max(condition_ax, condition_ay) > float(maximum_spatial_factor_condition):
        raise ValueError("Frozen geometry violates the spatial-factor condition limit")
    return {
        "true_uv": uv,
        "true_angles_deg": np.rad2deg(np.column_stack((elevation, azimuth))),
        "ax": ax,
        "ay": ay,
        "scene_seed": protocol_seed(master_seed, "scene", scene_index=0),
        "sampling_attempts": 0,
        "condition_ax": condition_ax,
        "condition_ay": condition_ay,
    }


def preset_tensor_power(
    powers: list[float] | tuple[float, ...] | np.ndarray, rank: int
) -> float:
    """Return the separation-independent tensor power used by experiment D."""
    selected = np.asarray(powers, dtype=float)[: int(rank)]
    if selected.shape != (int(rank),) or np.any(~np.isfinite(selected)):
        raise ValueError("One finite source power is required per component")
    if np.any(selected <= 0.0):
        raise ValueError("Source powers must be positive")
    return float(np.sum(selected))


def materialize_condition(
    clean_tensor: np.ndarray,
    standard_noise: np.ndarray,
    mask_uniform: np.ndarray,
    *,
    observation_rate: float,
    snr_db: float,
    noise_reference_power: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create one paired condition from shared latent random quantities.

    When ``noise_reference_power`` is omitted, experiment A uses the realized
    mean clean-tensor power.  Experiment D supplies ``preset_tensor_power`` so
    its noise scale does not change with target separation.
    """
    clean_tensor = np.asarray(clean_tensor)
    standard_noise = np.asarray(standard_noise)
    mask_uniform = np.asarray(mask_uniform)
    if clean_tensor.shape != standard_noise.shape or clean_tensor.shape != mask_uniform.shape:
        raise ValueError("Clean tensor, standard noise and mask uniforms must share a shape")
    if not np.isfinite(clean_tensor).all() or not np.isfinite(standard_noise).all():
        raise ValueError("Clean tensor and standard noise must be finite")
    if not np.isfinite(mask_uniform).all() or np.any((mask_uniform < 0.0) | (mask_uniform >= 1.0)):
        raise ValueError("Mask uniforms must lie in [0, 1)")
    observation_rate = float(observation_rate)
    if not 0.0 < observation_rate <= 1.0:
        raise ValueError("Observation rate must lie in (0, 1]")
    reference_power = (
        float(np.mean(np.abs(clean_tensor) ** 2))
        if noise_reference_power is None
        else float(noise_reference_power)
    )
    if not np.isfinite(reference_power) or reference_power <= 0.0:
        raise ValueError("Noise reference power must be positive and finite")
    noise_variance = reference_power / (10.0 ** (float(snr_db) / 10.0))
    full_noisy_tensor = clean_tensor + np.sqrt(noise_variance) * standard_noise
    observation_mask = mask_uniform < observation_rate
    return full_noisy_tensor, observation_mask, float(noise_variance)


def validate_terminal_status(status: str) -> str:
    """Validate and return one frozen mutually exclusive terminal status."""
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"Unknown terminal status: {status}")
    return status
