from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from onset_common import wrap_symmetric
from onset_odf import fold_phase, fold_profiles

METHOD_DIRECT = "直接フィット"
METHOD_WINDOW = "窓回帰"

DIRECT_MAX_SLOPE = 5e-4
DIRECT_STEP_RATIO = 0.25
DIRECT_MAX_STEPS = 6000
DIRECT_SCAN_BIN_MS = 0.25
DIRECT_SIGMA_MS = 1.0
DIRECT_MIN_EVENTS = 60
DIRECT_MIN_SPAN_MS = 10000.0
DIRECT_MIN_BLOCKS = 5
DIRECT_BOOTSTRAPS = 200
DIRECT_BOOTSTRAP_HALF_STEPS = 12
DIRECT_CI_PERCENTILES = (2.5, 97.5)
DIRECT_MIN_EXCESS = 0.25
DIRECT_SEED = 0
SCAN_CHUNK_ELEMENTS = 1 << 19


@dataclass(frozen=True)
class DriftFit:
    phase_ms: float
    slope: float
    slope_low: float
    slope_high: float
    reference_ms: float
    method: str = METHOD_WINDOW

    @property
    def significant(self) -> bool:
        return self.slope_low > 0.0 or self.slope_high < 0.0

    def phase_at(self, time_ms: float) -> float:
        return self.phase_ms + self.slope * time_ms


def chunk_scores(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    slopes: np.ndarray,
) -> np.ndarray:
    shifted = times_ms[None, :] * (1.0 - slopes)[:, None]
    profiles, step = fold_profiles(shifted, weights, period_ms, DIRECT_SCAN_BIN_MS)
    smoothed = ndimage.gaussian_filter1d(
        profiles, DIRECT_SIGMA_MS / step, axis=1, mode="wrap"
    )
    peaks = smoothed.max(axis=1)
    means = smoothed.mean(axis=1)
    return np.divide(peaks, means, out=np.zeros_like(peaks), where=means > 0.0)


def scan_scores(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    slopes: np.ndarray,
) -> np.ndarray:
    if slopes.size == 0:
        return np.empty(0)
    chunk = max(1, SCAN_CHUNK_ELEMENTS // max(1, times_ms.size))
    return np.concatenate(
        [
            chunk_scores(times_ms, weights, period_ms, slopes[start : start + chunk])
            for start in range(0, slopes.size, chunk)
        ]
    )


def peak_slope(slopes: np.ndarray, scores: np.ndarray) -> float | None:
    best = int(np.argmax(scores))
    if best == 0 or best == scores.size - 1:
        return None
    left, middle, right = scores[best - 1], scores[best], scores[best + 1]
    curvature = left - 2.0 * middle + right
    if curvature >= 0.0:
        return float(slopes[best])
    shift = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(slopes[best] + shift * (slopes[1] - slopes[0]))


def slope_grid(span_ms: float) -> tuple[np.ndarray, float]:
    step = DIRECT_STEP_RATIO * DIRECT_SIGMA_MS / span_ms
    half = min(DIRECT_MAX_SLOPE, step * DIRECT_MAX_STEPS / 2.0)
    count = max(2, int(half / step))
    return step * np.arange(-count, count + 1), step


def block_slices(times_ms: np.ndarray, window_ms: float) -> list[tuple[int, int]]:
    edges = np.arange(0.0, times_ms[-1] + window_ms, window_ms)
    bounds = np.searchsorted(times_ms, edges)
    return [
        (int(low), int(high))
        for low, high in zip(bounds[:-1], bounds[1:])
        if high > low
    ]


def bootstrap_slopes(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    blocks: list[tuple[int, int]],
    center: float,
    step: float,
) -> np.ndarray:
    rng = np.random.default_rng(DIRECT_SEED)
    local = center + step * np.arange(
        -DIRECT_BOOTSTRAP_HALF_STEPS, DIRECT_BOOTSTRAP_HALF_STEPS + 1
    )
    found = []
    for _ in range(DIRECT_BOOTSTRAPS):
        draw = rng.integers(0, len(blocks), len(blocks))
        chosen = np.concatenate([np.arange(*blocks[index]) for index in draw])
        scores = scan_scores(times_ms[chosen], weights[chosen], period_ms, local)
        slope = peak_slope(local, scores)
        found.append(float(local[int(np.argmax(scores))]) if slope is None else slope)
    return np.asarray(found)


def inlier_excess(
    shifted_ms: np.ndarray,
    weights: np.ndarray,
    phase_ms: float,
    period_ms: float,
    radius_ms: float,
) -> float:
    residual = wrap_symmetric(shifted_ms - phase_ms, period_ms)
    inlier = float(np.average(np.abs(residual) < radius_ms, weights=weights))
    chance = min(1.0, 2.0 * radius_ms / period_ms)
    return (inlier - chance) / (1.0 - chance)


def fit_direct(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    radius_ms: float,
    window_ms: float,
) -> DriftFit | None:
    if times_ms.size < DIRECT_MIN_EVENTS or window_ms <= 0.0:
        return None
    order = np.argsort(times_ms)
    times, weight = times_ms[order], weights[order]
    span = float(times[-1] - times[0])
    if span < DIRECT_MIN_SPAN_MS:
        return None
    blocks = block_slices(times, window_ms)
    if len(blocks) < DIRECT_MIN_BLOCKS:
        return None
    slopes, step = slope_grid(span)
    slope = peak_slope(slopes, scan_scores(times, weight, period_ms, slopes))
    if slope is None:
        return None
    shifted = times * (1.0 - slope)
    phase, _ = fold_phase(shifted, weight, period_ms)
    if inlier_excess(shifted, weight, phase, period_ms, radius_ms) < DIRECT_MIN_EXCESS:
        return None
    samples = bootstrap_slopes(times, weight, period_ms, blocks, slope, step)
    low, high = np.percentile(samples, DIRECT_CI_PERCENTILES)
    return DriftFit(
        phase,
        slope,
        float(low),
        float(high),
        float(np.average(times, weights=weight)),
        METHOD_DIRECT,
    )
