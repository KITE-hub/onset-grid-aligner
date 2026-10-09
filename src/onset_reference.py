from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from onset_common import wrap_symmetric
from onset_config import (
    AnalysisConfig,
    REFERENCE_ALL,
    REFERENCE_FAST,
    REFERENCE_MIXED,
    REFERENCE_SLOW,
    SENSITIVITY_LEVELS,
)
from onset_detection import Detection, Heads
from onset_drift import DriftFit
from onset_phase import MIN_ONSETS, recentered_offsets
from onset_window_phase import (
    estimate_drift,
    event_support,
    resolve_phase,
    support_window_ms,
    WindowPhases,
)

GROUP_REGION_MS = 8.0
GROUP_BIN_MS = 0.25
GROUP_SMOOTH_SIGMA_MS = 0.6
GROUP_MIN_HEIGHT_RATIO = 0.3
GROUP_MIN_PROMINENCE_RATIO = 0.2
GROUP_MIN_SEPARATION_MS = 1.5
GROUP_ASSIGN_RADIUS_MS = 2.0
GROUP_MIN_EVENTS = 30
GROUP_MIN_SHARE = 0.15
GROUP_MIN_WINDOW_EVENTS = 8


@dataclass(frozen=True)
class WindowSlices:
    starts_ms: np.ndarray
    lows: np.ndarray
    highs: np.ndarray

    @property
    def counts(self) -> np.ndarray:
        return self.highs - self.lows


@dataclass(frozen=True)
class PhaseGroup:
    peak_ms: float
    head_ms: float
    gap_ms: float
    spread_ms: float
    members: np.ndarray

    @property
    def count(self) -> int:
        return int(self.members.size)


@dataclass(frozen=True)
class ReferencePool:
    used: tuple[int, ...]
    members: np.ndarray
    shifts_ms: np.ndarray
    weights: np.ndarray
    anchor_ms: float
    gap_ms: float
    span_ms: float
    spread_ms: float

    @property
    def count(self) -> int:
        return int(self.members.size)


@dataclass(frozen=True)
class ReferenceEstimate:
    groups: tuple[PhaseGroup, ...]
    pool: ReferencePool
    windows: WindowPhases | None
    window_counts: np.ndarray | None
    drift: DriftFit | None
    conclusion_ms: float
    uses_regression: bool
    stability: float | None
    grid_fit: float | None
    difference_ms: float
    sensitivity: tuple[tuple[float, float], ...]


def window_slices(
    heads_ms: np.ndarray, window_ms: float, window_count: int
) -> WindowSlices:
    starts = np.arange(window_count) * window_ms
    return WindowSlices(
        starts,
        np.searchsorted(heads_ms, starts),
        np.searchsorted(heads_ms, starts + window_ms),
    )


def group_modes(offsets: np.ndarray, center: float) -> np.ndarray:
    edges = np.arange(
        center - GROUP_REGION_MS, center + GROUP_REGION_MS + GROUP_BIN_MS, GROUP_BIN_MS
    )
    counts, _ = np.histogram(offsets, bins=edges)
    smoothed = ndimage.gaussian_filter1d(
        counts.astype(float), GROUP_SMOOTH_SIGMA_MS / GROUP_BIN_MS, mode="nearest"
    )
    top = float(smoothed.max())
    if top <= 0.0:
        return np.empty(0)
    peaks, _ = signal.find_peaks(
        smoothed,
        height=GROUP_MIN_HEIGHT_RATIO * top,
        prominence=GROUP_MIN_PROMINENCE_RATIO * top,
        distance=max(1, int(round(GROUP_MIN_SEPARATION_MS / GROUP_BIN_MS))),
    )
    return (edges[:-1] + edges[1:])[peaks] / 2.0


def assign_groups(offsets: np.ndarray, modes: np.ndarray) -> np.ndarray:
    distances = np.abs(offsets[:, None] - modes[None, :])
    nearest = np.argmin(distances, axis=1)
    within = distances[np.arange(offsets.size), nearest] <= GROUP_ASSIGN_RADIUS_MS
    return np.where(within, nearest, -1)


def build_phase_groups(
    offsets: np.ndarray, gaps_ms: np.ndarray, center: float
) -> tuple[PhaseGroup, ...]:
    modes = group_modes(offsets, center)
    if modes.size == 0:
        return ()
    labels = assign_groups(offsets, modes)
    groups = []
    for index in range(modes.size):
        members = np.flatnonzero(labels == index)
        if members.size == 0:
            continue
        heads = offsets[members] - gaps_ms[members]
        groups.append(
            PhaseGroup(
                float(np.median(offsets[members])),
                float(np.median(heads)),
                float(np.median(gaps_ms[members])),
                float(np.median(np.abs(heads - np.median(heads)))),
                members,
            )
        )
    return tuple(groups)


def eligible_groups(groups: tuple[PhaseGroup, ...]) -> list[int]:
    grouped = sum(group.count for group in groups)
    return [
        index
        for index, group in enumerate(groups)
        if group.count >= GROUP_MIN_EVENTS and group.count >= GROUP_MIN_SHARE * grouped
    ]


def choose_groups(groups: tuple[PhaseGroup, ...], rule: str) -> tuple[int, ...]:
    eligible = eligible_groups(groups)
    if not eligible:
        return ()
    if rule == REFERENCE_ALL:
        return tuple(eligible)
    if rule == REFERENCE_FAST:
        return (min(eligible, key=lambda index: groups[index].gap_ms),)
    if rule == REFERENCE_SLOW:
        return (max(eligible, key=lambda index: groups[index].gap_ms),)
    return (max(eligible, key=lambda index: groups[index].count),)


def pool_groups(
    groups: tuple[PhaseGroup, ...],
    used: tuple[int, ...],
    offsets: np.ndarray,
    gaps_ms: np.ndarray,
    period: float,
) -> ReferencePool:
    chosen = [groups[index] for index in used]
    heads = np.array([group.head_ms for group in chosen])
    relative = wrap_symmetric(heads - heads[0], period)
    anchor = float(heads[0] + relative.mean())
    shifts = np.zeros(offsets.size)
    weights = np.zeros(offsets.size)
    for group, head in zip(chosen, heads):
        shifts[group.members] = wrap_symmetric(head - anchor, period)
        weights[group.members] = 1.0 / group.count
    members = np.flatnonzero(weights)
    aligned = offsets[members] - gaps_ms[members] - shifts[members]
    return ReferencePool(
        used,
        members,
        shifts,
        weights,
        anchor,
        float(np.mean([group.gap_ms for group in chosen])),
        float(np.ptp(relative)),
        float(np.median(np.abs(aligned - np.median(aligned)))),
    )


def group_window_counts(
    times_ms: np.ndarray, starts_ms: np.ndarray, window_ms: float
) -> np.ndarray:
    return np.searchsorted(times_ms, starts_ms + window_ms) - np.searchsorted(
        times_ms, starts_ms
    )


def reference_windows(
    peaks_ms: np.ndarray,
    offsets: np.ndarray,
    gaps_ms: np.ndarray,
    groups: tuple[PhaseGroup, ...],
    pool: ReferencePool,
    window_ms: float,
    window_count: int,
) -> tuple[WindowPhases | None, np.ndarray | None]:
    members = pool.members
    times = peaks_ms[members]
    values = offsets[members] - gaps_ms[members] - pool.shifts_ms[members]
    slices = window_slices(times, window_ms, window_count)
    kept = slices.counts >= GROUP_MIN_WINDOW_EVENTS
    if not kept.any():
        return None, None
    bounds = list(zip(slices.lows[kept], slices.highs[kept]))
    phases = np.array([np.median(values[low:high]) for low, high in bounds])
    centers = np.array([np.median(times[low:high]) for low, high in bounds])
    starts = slices.starts_ms[kept]
    counts = np.stack(
        [
            group_window_counts(peaks_ms[other.members], starts, window_ms)
            for other in groups
        ],
        axis=1,
    )
    return WindowPhases(starts, centers, phases, slices.counts[kept]), counts


def reference_sensitivity(
    detection: Detection,
    selected: Heads,
    offsets: np.ndarray,
    pool: ReferencePool,
    period: float,
) -> tuple[tuple[float, float], ...]:
    pooled_peaks = selected.peaks_ms[pool.members]
    results = []
    for level in SENSITIVITY_LEVELS:
        heads = detection.heads[level]
        inside = np.isin(heads.peaks_ms, pooled_peaks)
        if not inside.any():
            continue
        indices = np.searchsorted(selected.peaks_ms, heads.peaks_ms[inside])
        gaps = heads.peaks_ms[inside] - heads.heads_ms[inside]
        phases = offsets[indices] - gaps - pool.shifts_ms[indices]
        results.append((level, float(wrap_symmetric(np.median(phases), period))))
    return tuple(results)


def event_window_layout(peaks_ms: np.ndarray, window_sec: float) -> tuple[float, int]:
    window_ms = window_sec * 1000.0
    return window_ms, max(1, int(np.ceil((float(peaks_ms[-1]) + 1.0) / window_ms)))


def group_trend_ms(peaks_ms: np.ndarray, drift: DriftFit | None) -> np.ndarray:
    if drift is None or not drift.significant:
        return np.zeros(peaks_ms.shape)
    return drift.slope * (peaks_ms - drift.reference_ms)


def estimate_reference(
    detection: Detection,
    selected: Heads,
    config: AnalysisConfig,
    reference_ms: float,
    mixed_drift: DriftFit | None,
) -> ReferenceEstimate | None:
    period = config.period_ms
    if config.reference_group == REFERENCE_MIXED:
        return None
    if selected.peaks_ms.size < MIN_ONSETS:
        return None
    gaps_ms = selected.peaks_ms - selected.heads_ms
    trend = group_trend_ms(selected.peaks_ms, mixed_drift)
    offsets, center = recentered_offsets(selected.peaks_ms - trend, period)
    groups = build_phase_groups(offsets, gaps_ms, center)
    used = choose_groups(groups, config.reference_group)
    if not used:
        return None
    pool = pool_groups(groups, used, offsets, gaps_ms, period)
    windows, window_counts = (None, None)
    if config.drift_window_sec > 0.0:
        window_ms, window_count = event_window_layout(
            selected.peaks_ms, config.drift_window_sec
        )
        windows, window_counts = reference_windows(
            selected.peaks_ms,
            offsets + trend,
            gaps_ms,
            groups,
            pool,
            window_ms,
            window_count,
        )
    members = pool.members
    drift = estimate_drift(
        selected.heads_ms[members] - pool.shifts_ms[members],
        windows,
        config,
        pool.weights[members],
    )
    fallback_ms = float(wrap_symmetric(pool.anchor_ms, period))
    conclusion_ms, uses_regression = resolve_phase(drift, fallback_ms, period)
    support = event_support(
        selected,
        conclusion_ms,
        drift,
        uses_regression,
        period,
        support_window_ms(config),
    )
    return ReferenceEstimate(
        groups,
        pool,
        windows,
        window_counts,
        drift,
        conclusion_ms,
        uses_regression,
        support.stability,
        support.grid_fit,
        float(wrap_symmetric(conclusion_ms - reference_ms, period)),
        reference_sensitivity(detection, selected, offsets, pool, period),
    )
