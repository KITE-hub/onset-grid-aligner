from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from onset_config import AnalysisConfig
from onset_grid import estimate_comb_phase
from onset_odf import OnsetFunction
from onset_phase import MIN_ONSETS
from onset_timing import MS_PER_MINUTE, TempoMap


ANCHOR_BOUNDARY_BEAT_RATIO = 0.5
BPM_MATCH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GridWarp:
    tempo: TempoMap | None = None
    steps_per_beat: int = 1
    period_ms: float = 0.0
    anchor_ms: float = 0.0

    def shifts(self, times_ms: np.ndarray) -> np.ndarray:
        if self.tempo is None:
            return np.zeros(np.shape(times_ms))
        index = self.tempo.grid_index(times_ms - self.anchor_ms, self.steps_per_beat)
        return index * self.period_ms - self.tempo.line_ms(index, self.steps_per_beat)

    def times(self, times_ms: np.ndarray) -> np.ndarray:
        return times_ms + self.shifts(times_ms)

    def kicks(self, kick_ms: np.ndarray) -> np.ndarray:
        return np.sort(self.times(kick_ms))

    def function(self, function: OnsetFunction) -> OnsetFunction:
        return replace(function, times_ms=self.times(function.times_ms))


IDENTITY_WARP = GridWarp()
ANCHOR_UNAVAILABLE_MESSAGE = "優勢テンポ区間のオンセットが不足しており、グリッド位相を推定できません"


def dominant_tempo_times(
    tempo: TempoMap, bpm: float, times_ms: np.ndarray, margin_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    beats, starts, bpms = tempo.columns()
    ends = np.append(starts[1:], np.inf if tempo.end_ms is None else tempo.end_ms)
    index = np.maximum(np.searchsorted(starts, times_ms, side="right") - 1, 0)
    inside = (
        (np.abs(bpms[index] - bpm) <= BPM_MATCH_TOLERANCE * bpm)
        & (times_ms - starts[index] >= margin_ms)
        & (ends[index] - times_ms >= margin_ms)
    )
    translation = beats[index] * MS_PER_MINUTE / bpm - starts[index]
    return inside, times_ms + translation


def grid_anchor_ms(
    tempo: TempoMap, config: AnalysisConfig, function: OnsetFunction
) -> float | None:
    margin_ms = ANCHOR_BOUNDARY_BEAT_RATIO * config.layout.beat_ms
    inside, mapped = dominant_tempo_times(
        tempo, config.bpm, function.times_ms, margin_ms
    )
    if np.count_nonzero(inside) < MIN_ONSETS:
        return None
    estimate = estimate_comb_phase(
        mapped[inside], np.maximum(function.weights[inside], 0.0), config.layout
    )
    return None if estimate is None else estimate.phase_ms


def make_warp(
    tempo: TempoMap | None, config: AnalysisConfig, function: OnsetFunction
) -> GridWarp | None:
    if tempo is None:
        return IDENTITY_WARP
    anchor_ms = grid_anchor_ms(tempo, config, function)
    if anchor_ms is None:
        return None
    return GridWarp(tempo, config.layout.beat_steps, config.period_ms, anchor_ms)
