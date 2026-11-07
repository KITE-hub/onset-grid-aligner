from __future__ import annotations

import numpy as np

from onset_grid import align_grid, CombPhaseEstimate, GridAlignment, GridLayout
from onset_kick import kick_coincident_mask
from onset_phase import MIN_ONSETS

KICK_BEAT_WEIGHT = 1.0


def beat_weights(
    times_ms: np.ndarray, weights: np.ndarray, kick_ms: np.ndarray
) -> np.ndarray:
    return weights + KICK_BEAT_WEIGHT * kick_coincident_mask(times_ms, kick_ms)


def align_beat(
    times_ms: np.ndarray,
    weights: np.ndarray,
    kick_ms: np.ndarray,
    phase_ms: float,
    layout: GridLayout,
) -> GridAlignment | None:
    if times_ms.size < MIN_ONSETS:
        return None
    return align_grid(
        times_ms,
        weights,
        phase_ms,
        layout,
        beat_weights(times_ms, weights, kick_ms),
        CombPhaseEstimate(phase_ms),
    )
