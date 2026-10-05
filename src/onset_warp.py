from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from onset_config import AnalysisConfig, MAX_BPM, MIN_BPM
from onset_detection import Detection, Heads
from onset_odf import OnsetFunction
from onset_phase import circular_mode, MIN_ONSETS
from onset_timing import MS_PER_MINUTE, TempoMap


@dataclass(frozen=True)
class GridWarp:
    tempo: TempoMap | None = None
    subdivision: int = 1
    period_ms: float = 0.0
    anchor_ms: float = 0.0

    def shifts(self, times_ms: np.ndarray) -> np.ndarray:
        if self.tempo is None:
            return np.zeros(np.shape(times_ms))
        index = self.tempo.grid_index(times_ms - self.anchor_ms, self.subdivision)
        return index * self.period_ms - self.tempo.line_ms(index, self.subdivision)

    def times(self, times_ms: np.ndarray) -> np.ndarray:
        return times_ms + self.shifts(times_ms)

    def kicks(self, kick_ms: np.ndarray) -> np.ndarray:
        return np.sort(self.times(kick_ms))

    def heads(self, heads: Heads) -> Heads:
        shift = self.shifts(heads.peaks_ms)
        return Heads(heads.heads_ms + shift, heads.peaks_ms + shift)


IDENTITY_WARP = GridWarp()


def fold_factor(bpm: float) -> float:
    factor = 1.0
    while bpm * factor > MAX_BPM:
        factor /= 2.0
    while bpm * factor < MIN_BPM:
        factor *= 2.0
    return factor


def resolve_tempo(tempo: TempoMap, total_ms: float) -> tuple[TempoMap, float]:
    folded = tempo.scaled(fold_factor(tempo.dominant_bpm(total_ms)))
    return folded, folded.dominant_bpm(total_ms)


def grid_anchor_ms(tempo: TempoMap, subdivision: int, peaks_ms: np.ndarray) -> float:
    index = tempo.grid_index(peaks_ms, subdivision)
    residual = peaks_ms - tempo.line_ms(index, subdivision)
    widest_ms = MS_PER_MINUTE / tempo.slowest_bpm / subdivision
    return circular_mode(residual, widest_ms)


def make_warp(
    tempo: TempoMap | None, config: AnalysisConfig, detection: Detection
) -> GridWarp:
    if tempo is None:
        return IDENTITY_WARP
    peaks_ms = detection.heads[config.head_level].peaks_ms
    anchor_ms = (
        grid_anchor_ms(tempo, config.subdivision, peaks_ms)
        if peaks_ms.size >= MIN_ONSETS
        else 0.0
    )
    return GridWarp(tempo, config.subdivision, config.period_ms, anchor_ms)


def warped_detection(detection: Detection, warp: GridWarp) -> Detection:
    return replace(
        detection,
        heads={level: warp.heads(item) for level, item in detection.heads.items()},
        kick_ms=warp.kicks(detection.kick_ms),
    )


def warped_functions(
    functions: tuple[OnsetFunction, ...], warp: GridWarp
) -> tuple[OnsetFunction, ...]:
    return tuple(
        replace(function, times_ms=warp.times(function.times_ms))
        for function in functions
    )
