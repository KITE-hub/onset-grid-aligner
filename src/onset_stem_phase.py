from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_drift import DriftFit
from onset_phase import (
    circular_mode,
    cluster_radius,
    concentration_score,
    MIN_GRID_FIT,
    MIN_ONSETS,
    SE_CONFIDENCE_Z,
    standard_error_ms,
)
from onset_reference import group_trend_ms
from onset_window_phase import estimate_drift, resolve_phase

CONFIDENCE_HIGH = "高"
CONFIDENCE_MID = "中"
CONFIDENCE_LOW = "低"
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

    @property
    def uncertainty_ms(self) -> float:
        return float(np.hypot(self.se95_ms, self.span_ms / 2.0))

    @property
    def confidence(self) -> str:
        if (
            len(self.used) >= HIGH_MIN_CLASSES
            and self.span_ms <= HIGH_MAX_SPAN_MS
            and self.se95_ms <= HIGH_MAX_SE_MS
        ):
            return CONFIDENCE_HIGH
        if self.span_ms <= MID_MAX_SPAN_MS:
            return CONFIDENCE_MID
        return CONFIDENCE_LOW


def inlier_mask(times_ms: np.ndarray, phase_ms: float, period: float) -> np.ndarray:
    return np.abs(wrap_symmetric(times_ms - phase_ms, period)) < cluster_radius(period)


def class_phase(label: int, times_ms: np.ndarray, period: float) -> ClassPhase:
    offsets = wrap_symmetric(times_ms, period)
    center = circular_mode(offsets, period)
    deltas = wrap_symmetric(offsets - center, period)
    members = deltas[np.abs(deltas) < cluster_radius(period)]
    median = float(np.median(members)) if members.size else 0.0
    spread = float(np.median(np.abs(members - median))) if members.size else 0.0
    return ClassPhase(
        label,
        int(times_ms.size),
        int(members.size),
        center + median,
        spread,
        concentration_score(int(members.size), int(times_ms.size), period),
    )


def pooled_events(
    times_ms: np.ndarray,
    labels: np.ndarray,
    chosen: list[ClassPhase],
    anchor_ms: float,
    period: float,
) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    weights = []
    for item in chosen:
        members = times_ms[labels == item.label]
        members = members[inlier_mask(members, item.phase_ms, period)]
        shift = wrap_symmetric(item.phase_ms - anchor_ms, period)
        parts.append(members - shift)
        weights.append(np.full(members.size, 1.0 / members.size))
    return np.concatenate(parts), np.concatenate(weights)


def estimate_stem(
    times_ms: np.ndarray, labels: np.ndarray, config: AnalysisConfig
) -> StemEstimate | None:
    period = config.period_ms
    classes = tuple(
        class_phase(int(label), times_ms[labels == label], period)
        for label in np.unique(labels)
    )
    used = tuple(index for index, item in enumerate(classes) if item.qualified)
    if not used:
        return None
    phases = np.array([classes[index].phase_ms for index in used])
    relative = wrap_symmetric(phases - phases[0], period)
    anchor_ms = float(phases[0] + relative.mean())
    aligned, weights = pooled_events(
        times_ms, labels, [classes[index] for index in used], anchor_ms, period
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
