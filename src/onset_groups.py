from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import SENSITIVITY_LEVELS
from onset_detection import Detection
from onset_kick import split_by_kick
from onset_labels import LABEL_ALL, LABEL_KICK, LABEL_OTHER
from onset_phase import (
    concentration_score,
    dominant_cluster,
    MIN_ONSETS,
    recentered_offsets,
)


@dataclass(frozen=True)
class GroupAnalysis:
    label: str
    heads_ms: np.ndarray
    offsets_ms: np.ndarray
    center_ms: float
    phase_ms: float
    spread_ms: float
    cluster_count: int
    concentration: float


def analyze_group(
    label: str, heads_ms: np.ndarray, period: float
) -> GroupAnalysis | None:
    if heads_ms.size < MIN_ONSETS:
        return None
    offsets, center = recentered_offsets(heads_ms, period)
    phase, spread, count = dominant_cluster(offsets, center, period)
    return GroupAnalysis(
        label,
        heads_ms,
        offsets,
        center,
        float(wrap_symmetric(phase, period)),
        spread,
        count,
        concentration_score(count, heads_ms.size, period),
    )


def build_groups(
    heads_ms: np.ndarray, kick_ms: np.ndarray, period: float
) -> dict[str, GroupAnalysis]:
    other_ms, kick_coincident_ms = split_by_kick(heads_ms, kick_ms)
    candidates = (
        (LABEL_ALL, heads_ms),
        (LABEL_OTHER, other_ms),
        (LABEL_KICK, kick_coincident_ms),
    )
    analyzed = (analyze_group(label, values, period) for label, values in candidates)
    return {group.label: group for group in analyzed if group is not None}


def pick_primary(groups: dict[str, GroupAnalysis]) -> GroupAnalysis | None:
    if not groups:
        return None
    return max(groups.values(), key=lambda group: group.concentration)


def head_level_sensitivity(
    detection: Detection, period: float, label: str
) -> tuple[tuple[float, float | None], ...]:
    results = []
    for level in SENSITIVITY_LEVELS:
        heads_ms = detection.heads[level].heads_ms
        group = build_groups(heads_ms, detection.kick_ms, period).get(label)
        results.append((level, None if group is None else group.phase_ms))
    return tuple(results)
