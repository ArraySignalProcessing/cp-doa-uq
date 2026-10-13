from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
EXPERIMENT_CONFIG_ROOT = Path(__file__).resolve().with_name("experiments")


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_paper_config() -> dict[str, Any]:
    with (EXPERIMENT_CONFIG_ROOT / "paper.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def calibration_conditions(config: dict | None = None) -> list[dict]:
    spec = (config or load_paper_config())["calibration"]
    rows = {}
    def add(rate, snr, sweep):
        rate, snr = float(rate), float(snr)
        key = (rate, snr)
        snr_tag = f"m{abs(int(snr)):02d}" if snr < 0 else f"{int(snr):02d}"
        if key in rows:
            rows[key]["sweep"] = "both"
        else:
            rows[key] = {"id": f"rho{round(rate * 100):03d}_snr{snr_tag}",
                         "observation_rate": rate, "snr_db": snr, "sweep": sweep}
    for snr in spec["snr_sweep"]["snr_db"]:
        add(spec["snr_sweep"]["observation_rate"], snr, "snr")
    for rate in spec["observation_rate_sweep"]["rates"]:
        add(rate, spec["observation_rate_sweep"]["snr_db"], "observation_rate")
    return list(rows.values())


def steering(length: int, spatial_frequency: float) -> np.ndarray:
    return np.exp(1j * np.pi * np.arange(length) * spatial_frequency)


def cp_tensor(ax: np.ndarray, ay: np.ndarray, signals: np.ndarray) -> np.ndarray:
    return np.einsum("mk,nk,lk->mnl", ax, ay, signals, optimize=True)


def common_initialization(
    shape: tuple[int, int, int], rank: int, seed: int, observed_rms: float = 1.0
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def sample(rows: int) -> np.ndarray:
        value = rng.standard_normal((rows, rank)) + 1j * rng.standard_normal((rows, rank))
        return value / np.maximum(np.linalg.norm(value, axis=0, keepdims=True), 1e-12)

    component_amplitude = observed_rms * np.sqrt(np.prod(shape) / max(rank, 1))
    common_norm = max(component_amplitude, 1e-12) ** (1.0 / 3.0)
    factors = {
        "ax": common_norm * sample(shape[0]),
        "ay": common_norm * sample(shape[1]),
        "signals": common_norm * sample(shape[2]),
    }
    return factors


def canonicalize_factors(factors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ax = np.asarray(factors["ax"], dtype=np.complex128).copy()
    ay = np.asarray(factors["ay"], dtype=np.complex128).copy()
    signals = np.asarray(factors["signals"], dtype=np.complex128).copy()
    for component in range(ax.shape[1]):
        for factor, receiver in ((ax, signals), (ay, signals)):
            norm = np.linalg.norm(factor[:, component])
            if norm <= 1e-12:
                continue
            factor[:, component] /= norm
            receiver[:, component] *= norm
            phase = np.angle(factor[0, component])
            factor[:, component] *= np.exp(-1j * phase)
            receiver[:, component] *= np.exp(1j * phase)
    return {"ax": ax, "ay": ay, "signals": signals}


def factors_to_uv(factors: dict[str, np.ndarray]) -> np.ndarray:
    factors = canonicalize_factors(factors)

    def phase_increment(matrix: np.ndarray) -> np.ndarray:
        correlation = np.sum(np.conj(matrix[:-1, :]) * matrix[1:, :], axis=0)
        return np.angle(correlation) / np.pi

    return np.column_stack((phase_increment(factors["ax"]), phase_increment(factors["ay"])))


def match_targets(estimated_uv: np.ndarray, true_uv: np.ndarray) -> np.ndarray:
    cost = np.sum((estimated_uv[:, None, :] - true_uv[None, :, :]) ** 2, axis=2)
    rank = true_uv.shape[0]
    candidates = itertools.permutations(range(rank))
    return np.asarray(min(candidates, key=lambda order: sum(cost[order[j], j] for j in range(rank))))


def uv_to_angles(uv: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(uv, axis=1)
    elevation = np.arcsin(np.clip(radius, 0.0, 1.0))
    azimuth = np.arctan2(uv[:, 1], uv[:, 0])
    return np.rad2deg(np.column_stack((elevation, azimuth)))


def angular_errors_deg(estimated_uv: np.ndarray, true_uv: np.ndarray) -> np.ndarray:
    def directions(uv: np.ndarray) -> np.ndarray:
        w = np.sqrt(np.maximum(1.0 - np.sum(uv**2, axis=1), 0.0))
        return np.column_stack((uv, w))

    estimated = directions(estimated_uv)
    truth = directions(true_uv)
    cosine = np.sum(estimated * truth, axis=1)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
