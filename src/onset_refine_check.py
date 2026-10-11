from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_analysis import aligned_members, Analysis
from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_detection import Detection
from onset_phase import MIN_ONSETS, standard_error_ms
from onset_refine import refine_heads
from onset_warp import GridWarp, IDENTITY_WARP

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


def within_radius(residuals: np.ndarray) -> np.ndarray:
    return residuals[np.abs(residuals) <= REFINE_RADIUS_MS]


def refine_check(
    samples: np.ndarray,
    sample_rate: int,
    detection: Detection,
    config: AnalysisConfig,
    analysis: Analysis,
    warp: GridWarp = IDENTITY_WARP,
) -> RefineCheck | None:
    period = config.period_ms
    real = detection.heads[config.head_level]
    selected = warp.heads(real)
    members, alignment = aligned_members(analysis, selected)
    if members.size < MIN_ONSETS:
        return None
    times_ms = real.peaks_ms[members]
    shift = selected.peaks_ms[members] - times_ms
    refined = refine_heads(samples, sample_rate, times_ms, config.head_level)
    if refined is None:
        return None
    warped_ms = times_ms + shift
    combined = within_radius(
        analysis.residuals_ms(refined.heads_ms + shift - alignment, warped_ms)
    )
    if combined.size < MIN_ONSETS:
        return None
    shift = float(np.median(combined))
    spread = float(np.median(np.abs(combined - shift)))
    bands = []
    for (low, high), heads in zip(refined.bands, refined.band_heads_ms):
        residual = within_radius(
            analysis.residuals_ms(heads + shift - alignment, warped_ms)
        )
        if residual.size >= MIN_ONSETS:
            bands.append(RefineBand(low, high, float(np.median(residual)), residual.size))
    return RefineCheck(
        float(wrap_symmetric(analysis.conclusion_ms + shift, period)),
        shift,
        spread,
        combined.size,
        standard_error_ms(spread, combined.size),
        tuple(bands),
    )
