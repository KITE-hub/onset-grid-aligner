from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_drift import DriftFit
from onset_phase import (
    circular_mode,
    cluster_radius,
    MIN_GRID_FIT,
    MIN_ONSETS,
    SE_CONFIDENCE_Z,
    standard_error_ms,
)
from onset_reference import group_trend_ms
from onset_stem_grid import COMB_FRACTION
from onset_window_phase import estimate_drift, resolve_phase

CONFIDENCE_HIGH = "高"
CONFIDENCE_MID = "中"
CONFIDENCE_LOW = "低"
HIGH_MIN_BANDS = 2
HIGH_MIN_CLASSES = 2
HIGH_MAX_SPAN_MS = 1.5
HIGH_MAX_SE_MS = 0.5
MID_MAX_SPAN_MS = 4.0


@dataclass(frozen=True)
class ClassPhase:
    label: int
    count: int
    inliers: int
    phase_ms: float
    spread_ms: float
    fit: float

    @property
    def qualified(self) -> bool:
        return self.count >= MIN_ONSETS and self.fit >= MIN_GRID_FIT


@dataclass(frozen=True)
class StemEstimate:
    classes: tuple[ClassPhase, ...]
    used: tuple[int, ...]
    phase_ms: float
    span_ms: float
    spread_ms: float
    event_count: int
    drift: DriftFit | None
    uses_regression: bool

    @property
    def delay_ms(self) -> float:
        return -self.phase_ms

    @property
    def se95_ms(self) -> float:
        return SE_CONFIDENCE_Z * standard_error_ms(self.spread_ms, self.event_count)


@dataclass(frozen=True)
class BandEstimate:
    low_hz: float
    high_hz: float
    estimate: StemEstimate
    times_ms: np.ndarray
    strengths: np.ndarray


@dataclass(frozen=True)
class StemResult:
    period_ms: float
    bands: tuple[BandEstimate, ...]

    @property
    def relative_phases_ms(self) -> np.ndarray:
        phases = np.array([band.estimate.phase_ms for band in self.bands])
        return wrap_symmetric(phases - phases[0], self.period_ms)

    @property
    def phase_ms(self) -> float:
        return float(self.bands[0].estimate.phase_ms + self.relative_phases_ms.mean())

    @property
    def delay_ms(self) -> float:
        return -self.phase_ms

    @property
    def band_gap_ms(self) -> float:
        return float(np.ptp(self.relative_phases_ms))

    @property
    def span_ms(self) -> float:
        return max(band.estimate.span_ms for band in self.bands)

    @property
    def se95_ms(self) -> float:
        squares = sum(band.estimate.se95_ms**2 for band in self.bands)
        return float(np.sqrt(squares)) / len(self.bands)

    @property
    def uncertainty_ms(self) -> float:
        return float(
            np.sqrt(
                self.se95_ms**2
                + (self.span_ms / 2.0) ** 2
                + (self.band_gap_ms / 2.0) ** 2
            )
        )

    @property
    def confidence(self) -> str:
        systematic = max(self.span_ms, self.band_gap_ms)
        if (
            len(self.bands) >= HIGH_MIN_BANDS
            and min(len(band.estimate.used) for band in self.bands) >= HIGH_MIN_CLASSES
            and systematic <= HIGH_MAX_SPAN_MS
            and self.se95_ms <= HIGH_MAX_SE_MS
        ):
            return CONFIDENCE_HIGH
        if systematic <= MID_MAX_SPAN_MS:
            return CONFIDENCE_MID
        return CONFIDENCE_LOW


def inlier_mask(times_ms: np.ndarray, phase_ms: float, period: float) -> np.ndarray:
    return np.abs(wrap_symmetric(times_ms - phase_ms, period)) < cluster_radius(period)


def line_chance(period: float, line_fraction: float) -> float:
    return line_fraction * min(1.0, 2.0 * cluster_radius(period) / period)


def excess_fit(inliers: int, total: int, chance: float) -> float:
    return (inliers / total - chance) / (1.0 - chance)


def class_phase(
    label: int, times_ms: np.ndarray, period: float, on_comb: np.ndarray | None
) -> ClassPhase:
    total = int(times_ms.size)
    on_line = np.ones(total, dtype=bool) if on_comb is None else on_comb
    chance = line_chance(period, 1.0 if on_comb is None else COMB_FRACTION)
    if not on_line.any():
        return ClassPhase(label, total, 0, 0.0, 0.0, excess_fit(0, total, chance))
    offsets = wrap_symmetric(times_ms, period)
    center = circular_mode(offsets[on_line], period)
    deltas = wrap_symmetric(offsets - center, period)
    members = deltas[on_line & (np.abs(deltas) < cluster_radius(period))]
    median = float(np.median(members)) if members.size else 0.0
    spread = float(np.median(np.abs(members - median))) if members.size else 0.0
    return ClassPhase(
        label,
        total,
        int(members.size),
        center + median,
        spread,
        excess_fit(int(members.size), total, chance),
    )


def pooled_events(
    times_ms: np.ndarray,
    labels: np.ndarray,
    chosen: list[ClassPhase],
    anchor_ms: float,
    period: float,
    on_comb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    weights = []
    for item in chosen:
        selected = labels == item.label
        members = times_ms[selected]
        keep = inlier_mask(members, item.phase_ms, period)
        if on_comb is not None:
            keep &= on_comb[selected]
        members = members[keep]
        shift = wrap_symmetric(item.phase_ms - anchor_ms, period)
        parts.append(members - shift)
        weights.append(np.full(members.size, 1.0 / members.size))
    return np.concatenate(parts), np.concatenate(weights)


def estimate_stem(
    times_ms: np.ndarray,
    labels: np.ndarray,
    config: AnalysisConfig,
    on_comb: np.ndarray | None = None,
) -> StemEstimate | None:
    period = config.period_ms
    classes = tuple(
        class_phase(
            int(label),
            times_ms[labels == label],
            period,
            None if on_comb is None else on_comb[labels == label],
        )
        for label in np.unique(labels)
    )
    used = tuple(index for index, item in enumerate(classes) if item.qualified)
    if not used:
        return None
    phases = np.array([classes[index].phase_ms for index in used])
    relative = wrap_symmetric(phases - phases[0], period)
    anchor_ms = float(phases[0] + relative.mean())
    aligned, weights = pooled_events(
        times_ms,
        labels,
        [classes[index] for index in used],
        anchor_ms,
        period,
        on_comb,
    )
    drift = estimate_drift(aligned, None, config, weights)
    phase_ms, uses_regression = resolve_phase(
        drift, float(wrap_symmetric(anchor_ms, period)), period
    )
    residuals = wrap_symmetric(
        aligned - group_trend_ms(aligned, drift) - anchor_ms, period
    )
    spread_ms = float(np.median(np.abs(residuals - np.median(residuals))))
    return StemEstimate(
        classes,
        used,
        phase_ms,
        float(np.ptp(relative)),
        spread_ms,
        int(aligned.size),
        drift,
        uses_regression,
    )
