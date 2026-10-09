from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig, DEFAULT_DRIFT_WINDOW_SEC
from onset_detection import Heads
from onset_drift import DriftFit, fit_direct
from onset_groups import GroupAnalysis
from onset_phase import (
    cluster_radius,
    concentration_score,
    dominant_cluster,
    excess_stability,
    inlier_ratio,
    recentered_offsets,
    standard_error_ms,
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


@dataclass(frozen=True)
class EventSupport:
    stability: float | None
    grid_fit: float


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


def window_residuals(
    windows: WindowPhases | None,
    drift: DriftFit | None,
    conclusion_ms: float,
    uses_regression: bool,
    period: float,
) -> np.ndarray | None:
    if windows is None:
        return None
    trend = (
        drift.slope * (windows.centers_ms - drift.reference_ms)
        if uses_regression and drift is not None
        else 0.0
    )
    return np.asarray(wrap_symmetric(windows.phases_ms - conclusion_ms - trend, period))


def regression_trend_ms(
    times_ms: np.ndarray, drift: DriftFit | None, uses_regression: bool
) -> np.ndarray:
    if not uses_regression or drift is None:
        return np.zeros(times_ms.shape)
    return drift.slope * (times_ms - drift.reference_ms)


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
    edges = np.arange(0.0, times_ms[-1] + window_ms, window_ms)
    bounds = np.searchsorted(times_ms, edges)
    scores = [
        concentration_score(int(inside[low:high].sum()), int(high - low), period)
        for low, high in zip(bounds[:-1], bounds[1:])
        if high - low >= MIN_WINDOW_ONSETS
    ]
    if len(scores) < MIN_SUPPORT_WINDOWS:
        return None
    return float(np.mean(np.asarray(scores) >= WINDOW_SUPPORT_Z))


def event_support(
    selected: Heads,
    conclusion_ms: float,
    drift: DriftFit | None,
    uses_regression: bool,
    period: float,
    window_ms: float,
) -> EventSupport:
    order = np.argsort(selected.peaks_ms)
    times_ms = selected.peaks_ms[order]
    trend = regression_trend_ms(times_ms, drift, uses_regression)
    residuals = wrap_symmetric(selected.heads_ms[order] - trend - conclusion_ms, period)
    inside = np.abs(residuals) < cluster_radius(period)
    return EventSupport(
        supported_window_ratio(times_ms, inside, period, window_ms),
        max(0.0, excess_stability(float(inside.mean()), period)),
    )


def window_standard_error_ms(residuals: np.ndarray | None) -> float:
    if residuals is None or residuals.size < MIN_DRIFT_WINDOWS:
        return 0.0
    spread = float(np.median(np.abs(residuals - np.median(residuals))))
    return standard_error_ms(spread, residuals.size)


def estimate_drift(
    heads_ms: np.ndarray,
    windows: WindowPhases | None,
    config: AnalysisConfig,
    weights: np.ndarray | None = None,
) -> DriftFit | None:
    if config.drift_window_sec <= 0.0:
        return None
    period = config.period_ms
    direct = fit_direct(
        heads_ms,
        np.ones(heads_ms.size) if weights is None else weights,
        period,
        cluster_radius(period),
        config.drift_window_sec * 1000.0,
    )
    return direct if direct is not None else fit_drift(windows, period)


def regression_phase(fit: DriftFit, period: float) -> float:
    return float(wrap_symmetric(fit.phase_at(fit.reference_ms), period))


def resolve_phase(
    fit: DriftFit | None, fallback_ms: float, period: float
) -> tuple[float, bool]:
    if fit is not None and fit.significant:
        return regression_phase(fit, period), True
    return fallback_ms, False
