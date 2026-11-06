from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_phase import dominant_cluster, MIN_ONSETS, recentered_offsets


@dataclass(frozen=True)
class GroupAnalysis:
    heads_ms: np.ndarray
    offsets_ms: np.ndarray
    center_ms: float
    phase_ms: float


def analyze_group(heads_ms: np.ndarray, period: float) -> GroupAnalysis | None:
    if heads_ms.size < MIN_ONSETS:
        return None
    offsets, center = recentered_offsets(heads_ms, period)
    phase, _, _ = dominant_cluster(offsets, center, period)
    return GroupAnalysis(
        heads_ms,
        offsets,
        center,
        float(wrap_symmetric(phase, period)),
    )
