from __future__ import annotations

import numpy as np
from scipy import ndimage

from onset_common import wrap_symmetric

MIN_ONSETS = 8
STABLE_WINDOW_RATIO = 0.7
MIN_GRID_FIT = 0.1

CLUSTER_BIN_MS = 1.0
CLUSTER_SMOOTH_BINS = 5
CLUSTER_RADIUS_MS = 6.0
CLUSTER_RADIUS_PERIOD_RATIO = 0.2
SE_MEDIAN_FACTOR = 1.2533
MAD_TO_SIGMA = 1.4826
SE_CONFIDENCE_Z = 1.96


def circular_mode(offsets: np.ndarray, period: float) -> float:
    bins = max(CLUSTER_SMOOTH_BINS, int(round(period / CLUSTER_BIN_MS)))
    counts, edges = np.histogram(offsets, bins=bins, range=(-period / 2.0, period / 2.0))
    smoothed = ndimage.uniform_filter1d(
        counts.astype(float), CLUSTER_SMOOTH_BINS, mode="wrap"
    )
    best = int(np.argmax(smoothed))
    return float((edges[best] + edges[best + 1]) / 2.0)


def recentered_offsets(
    heads_ms: np.ndarray, period: float
) -> tuple[np.ndarray, float]:
    offsets = wrap_symmetric(heads_ms, period)
    center = circular_mode(offsets, period)
    return wrap_symmetric(offsets - center, period) + center, center


def cluster_radius(period: float) -> float:
    return min(CLUSTER_RADIUS_MS, CLUSTER_RADIUS_PERIOD_RATIO * period)


def chance_stability(period: float) -> float:
    return min(1.0, 2.0 * cluster_radius(period) / period)


def excess_stability(stability: float, period: float) -> float:
    chance = chance_stability(period)
    return (stability - chance) / (1.0 - chance)


def is_stable_value(stability: float | None, grid_fit: float | None) -> bool:
    return (
        stability is not None
        and grid_fit is not None
        and stability >= STABLE_WINDOW_RATIO
        and grid_fit >= MIN_GRID_FIT
    )


def screening_fit(peaks_ms: np.ndarray, period: float) -> float:
    offsets, center = recentered_offsets(peaks_ms, period)
    inside = float(np.mean(np.abs(offsets - center) < cluster_radius(period)))
    return max(0.0, excess_stability(inside, period))


def standard_error_ms(spread_ms: float, count: int) -> float:
    return float(SE_MEDIAN_FACTOR * MAD_TO_SIGMA * spread_ms / np.sqrt(max(count, 1)))


def dominant_cluster(
    offsets: np.ndarray, center: float, period: float
) -> tuple[float, float, int]:
    members = offsets[np.abs(offsets - center) < cluster_radius(period)]
    median = float(np.median(members))
    spread = float(np.median(np.abs(members - median)))
    return median, spread, int(members.size)


def concentration_score(count: int, total: int, period: float) -> float:
    chance = min(1.0, 2.0 * cluster_radius(period) / period)
    deviation = np.sqrt(max(total * chance * (1.0 - chance), 1.0))
    return float((count - total * chance) / deviation)


def inlier_ratio(residuals: np.ndarray, period: float) -> float:
    return float(np.mean(np.abs(residuals) < cluster_radius(period)))
