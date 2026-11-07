from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from onset_common import window_bounds, wrap_symmetric
from onset_config import AnalysisConfig, DEFAULT_DRIFT_WINDOW_SEC
from onset_drift import DriftFit, fit_direct
from onset_groups import GroupAnalysis
from onset_phase import (
    cluster_radius,
    concentration_score,
    dominant_cluster,
    inlier_ratio,
    recentered_offsets,
)

MIN_WINDOW_ONSETS = 15
MIN_DRIFT_WINDOWS = 5
MIN_DRIFT_WINDOW_SEC = 1.0
MIN_TREND_INLIER_RATIO = 0.8
WINDOW_SUPPORT_Z = 3.0
MIN_SUPPORT_WINDOWS = 3


@dataclass(frozen=True)
class WindowPhases:
    starts_ms: np.ndarray
    centers_ms: np.ndarray
    phases_ms: np.ndarray
    counts: np.ndarray


def window_phases(
    group: GroupAnalysis, period: float, window_sec: float
) -> WindowPhases | None:
    window_ms = window_sec * 1000.0
    starts = np.arange(0.0, group.heads_ms[-1], window_ms)
    lows = np.searchsorted(group.heads_ms, starts)
    highs = np.searchsorted(group.heads_ms, starts + window_ms)
    counts = highs - lows
    keep = counts >= MIN_WINDOW_ONSETS
    centers, phases = [], []
    for low, high in zip(lows[keep], highs[keep]):
        heads = group.heads_ms[low:high]
        offsets, center = recentered_offsets(heads, period)
        phases.append(dominant_cluster(offsets, center, period)[0])
        centers.append(float(np.median(heads)))
    if not phases:
        return None
    continuous = np.unwrap(wrap_symmetric(np.asarray(phases), period), period=period)
    return WindowPhases(starts[keep], np.asarray(centers), continuous, counts[keep])


def fit_drift(windows: WindowPhases | None, period: float) -> DriftFit | None:
    if windows is None or windows.centers_ms.size < MIN_DRIFT_WINDOWS:
        return None
    slope, intercept, low, high = stats.theilslopes(
        windows.phases_ms, windows.centers_ms
    )
    residuals = windows.phases_ms - (intercept + slope * windows.centers_ms)
    if inlier_ratio(residuals, period) < MIN_TREND_INLIER_RATIO:
        return None
    reference = float(np.average(windows.centers_ms, weights=windows.counts))
    return DriftFit(float(intercept), float(slope), float(low), float(high), reference)


def support_window_ms(config: AnalysisConfig) -> float:
    seconds = (
        config.drift_window_sec
        if config.drift_window_sec > 0.0
        else DEFAULT_DRIFT_WINDOW_SEC
    )
    return seconds * 1000.0


def supported_window_ratio(
    times_ms: np.ndarray, inside: np.ndarray, period: float, window_ms: float
) -> float | None:
    bounds = window_bounds(times_ms, window_ms)
    scores = [
        concentration_score(int(inside[low:high].sum()), int(high - low), period)
        for low, high in zip(bounds[:-1], bounds[1:])
        if high - low >= MIN_WINDOW_ONSETS
    ]
    if len(scores) < MIN_SUPPORT_WINDOWS:
        return None
    return float(np.mean(np.asarray(scores) >= WINDOW_SUPPORT_Z))


def timeline_stability(
    times_ms: np.ndarray, offsets_ms: np.ndarray, period: float, window_ms: float
) -> float | None:
    order = np.argsort(times_ms)
    residuals = wrap_symmetric(offsets_ms[order], period)
    inside = np.abs(residuals) < cluster_radius(period)
    return supported_window_ratio(times_ms[order], inside, period, window_ms)


def estimate_drift(
    heads_ms: np.ndarray,
    windows: WindowPhases | None,
    config: AnalysisConfig,
) -> DriftFit | None:
    if config.drift_window_sec <= 0.0:
        return None
    period = config.period_ms
    direct = fit_direct(
        heads_ms,
        np.ones(heads_ms.size),
        period,
        cluster_radius(period),
        config.drift_window_sec * 1000.0,
    )
    return direct if direct is not None else fit_drift(windows, period)
