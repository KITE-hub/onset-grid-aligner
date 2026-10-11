from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import SENSITIVITY_LEVELS
from onset_detection import Detection
from onset_labels import LABEL_ALL
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


def build_groups(heads_ms: np.ndarray, period: float) -> dict[str, GroupAnalysis]:
    group = analyze_group(LABEL_ALL, heads_ms, period)
    return {} if group is None else {LABEL_ALL: group}


def head_level_sensitivity(
    detection: Detection, period: float
) -> tuple[tuple[float, float | None], ...]:
    results = []
    for level in SENSITIVITY_LEVELS:
        group = analyze_group(LABEL_ALL, detection.heads[level].heads_ms, period)
        results.append((level, None if group is None else group.phase_ms))
    return tuple(results)
