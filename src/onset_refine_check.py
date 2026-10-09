from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_analysis import Analysis
from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_detection import Detection, Heads
from onset_kick import kick_peaks_ms, KickEnvelope, primary_keep_mask
from onset_phase import cluster_radius, MIN_ONSETS, standard_error_ms
from onset_refine import refine_heads
from onset_warp import GridWarp, IDENTITY_WARP
from onset_window_phase import regression_trend_ms

REFINE_RADIUS_MS = 15.0


@dataclass(frozen=True)
class RefineBand:
    low_hz: float
    high_hz: float
    shift_ms: float
    count: int


@dataclass(frozen=True)
class RefineCheck:
    offset_ms: float
    shift_ms: float
    spread_ms: float
    count: int
    standard_error_ms: float
    bands: tuple[RefineBand, ...]


def event_trend_ms(analysis: Analysis, times_ms: np.ndarray) -> np.ndarray:
    drift = analysis.reference.drift if analysis.reference is not None else analysis.drift
    return regression_trend_ms(times_ms, drift, analysis.final_uses_regression)


def event_residuals(
    values_ms: np.ndarray, times_ms: np.ndarray, analysis: Analysis, period: float
) -> np.ndarray:
    trend = event_trend_ms(analysis, times_ms)
    return np.asarray(wrap_symmetric(values_ms - trend - analysis.final_ms, period))


def refine_members(
    analysis: Analysis, selected: Heads, kick_ms: np.ndarray, period: float
) -> np.ndarray:
    if analysis.reference is not None:
        return analysis.reference.pool.members
    residual = event_residuals(selected.heads_ms, selected.peaks_ms, analysis, period)
    keep = primary_keep_mask(selected.heads_ms, kick_ms, analysis.primary.label)
    return np.flatnonzero(keep & (np.abs(residual) < cluster_radius(period)))


def refine_alignment(analysis: Analysis, members: np.ndarray) -> np.ndarray:
    if analysis.reference is None:
        return np.zeros(members.size)
    return analysis.reference.pool.shifts_ms[members]


def within_radius(residuals: np.ndarray) -> np.ndarray:
    return residuals[np.abs(residuals) <= REFINE_RADIUS_MS]


def refine_check(
    samples: np.ndarray,
    sample_rate: int,
    attacks: Detection,
    kick: KickEnvelope,
    config: AnalysisConfig,
    analysis: Analysis,
    warp: GridWarp = IDENTITY_WARP,
) -> RefineCheck | None:
    period = config.period_ms
    kick_ms = warp.kicks(kick_peaks_ms(kick, period))
    real = attacks.heads[config.head_level]
    selected = warp.heads(real)
    members = refine_members(analysis, selected, kick_ms, period)
    if members.size < MIN_ONSETS:
        return None
    alignment = refine_alignment(analysis, members)
    times_ms = real.peaks_ms[members]
    shift = selected.peaks_ms[members] - times_ms
    refined = refine_heads(samples, sample_rate, times_ms, config.head_level)
    if refined is None:
        return None
    warped_ms = times_ms + shift
    combined = within_radius(
        event_residuals(
            refined.heads_ms + shift - alignment, warped_ms, analysis, period
        )
    )
    if combined.size < MIN_ONSETS:
        return None
    shift = float(np.median(combined))
    spread = float(np.median(np.abs(combined - shift)))
    bands = []
    for (low, high), heads in zip(refined.bands, refined.band_heads_ms):
        residual = within_radius(
            event_residuals(heads + shift - alignment, warped_ms, analysis, period)
        )
        if residual.size >= MIN_ONSETS:
            bands.append(RefineBand(low, high, float(np.median(residual)), residual.size))
    return RefineCheck(
        float(wrap_symmetric(analysis.final_ms + shift, period)),
        shift,
        spread,
        combined.size,
        standard_error_ms(spread, combined.size),
        tuple(bands),
    )
