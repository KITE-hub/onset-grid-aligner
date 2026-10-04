from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

MS_PER_MINUTE = 60000.0


class TempoPoint(NamedTuple):
    beats: float
    ms: float
    bpm: float


@dataclass(frozen=True)
class TempoMap:
    points: tuple[TempoPoint, ...]
    end_ms: float | None = None

    def columns(self) -> np.ndarray:
        return np.asarray(self.points, dtype=float).T

    @property
    def slowest_bpm(self) -> float:
        return min(point.bpm for point in self.points)

    def beats_at(self, times_ms: np.ndarray) -> np.ndarray:
        beats, starts, bpms = self.columns()
        index = np.maximum(np.searchsorted(starts, times_ms, side="right") - 1, 0)
        return beats[index] + (times_ms - starts[index]) * bpms[index] / MS_PER_MINUTE

    def ms_at(self, beats: np.ndarray) -> np.ndarray:
        starts_beats, starts_ms, bpms = self.columns()
        index = np.maximum(np.searchsorted(starts_beats, beats, side="right") - 1, 0)
        return starts_ms[index] + (beats - starts_beats[index]) * MS_PER_MINUTE / bpms[index]

    def grid_index(self, times_ms: np.ndarray, subdivision: int) -> np.ndarray:
        return np.round(self.beats_at(times_ms) * subdivision)

    def line_ms(self, index: np.ndarray, subdivision: int) -> np.ndarray:
        return self.ms_at(index / subdivision)

    def dominant_bpm(self, total_ms: float) -> float:
        _, starts, bpms = self.columns()
        stop = total_ms if self.end_ms is None else min(self.end_ms, total_ms)
        durations = np.append(starts[1:], stop) - starts
        return float(bpms[int(np.argmax(durations))])

    def scaled(self, factor: float) -> TempoMap:
        return TempoMap(
            tuple(
                TempoPoint(point.beats * factor, point.ms, point.bpm * factor)
                for point in self.points
            ),
            self.end_ms,
        )
