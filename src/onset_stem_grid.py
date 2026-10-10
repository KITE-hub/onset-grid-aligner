from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STEPS_PER_BEAT = 24
GRID_LATTICES = (2, 3, 4, 6, 8)
BEAT_POINTS = 4.0
LINE_POINTS = 1.0
LATTICE_RADIUS_RATIO = 0.25
MIN_MARGIN = 0.1
TIE_TOLERANCE = 1e-9


def comb_steps() -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                step
                for lattice in GRID_LATTICES
                for step in range(0, STEPS_PER_BEAT, STEPS_PER_BEAT // lattice)
            }
        )
    )


COMB_STEPS = comb_steps()
COMB_FRACTION = len(COMB_STEPS) / STEPS_PER_BEAT


@dataclass(frozen=True)
class GridLayout:
    octave: int
    beat_steps: int
    period_steps: int
    lattice_ms: float
    points: np.ndarray

    @property
    def beat_ms(self) -> float:
        return self.lattice_ms * self.beat_steps


@dataclass(frozen=True)
class GridAlignment:
    offset_ms: float
    runner_up_ms: float
    beat_margin: float
    phase_ms: float
    shift_steps: int

    @property
    def beat_decided(self) -> bool:
        return self.beat_margin >= MIN_MARGIN


def build_layout(bpm: float, grid_bpm: float) -> GridLayout:
    octave = int(round(np.log2(grid_bpm / bpm)))
    beat_steps = max(1, int(round(STEPS_PER_BEAT * 2.0**octave)))
    period_steps = math.lcm(beat_steps, STEPS_PER_BEAT)
    steps = np.arange(period_steps)
    on_comb = np.isin(steps % STEPS_PER_BEAT, COMB_STEPS)
    points = np.where(steps % beat_steps == 0, BEAT_POINTS, LINE_POINTS)
    return GridLayout(
        octave,
        beat_steps,
        period_steps,
        60000.0 / grid_bpm / STEPS_PER_BEAT,
        points * on_comb,
    )


def circular_scores(profile: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.array(
        [np.dot(profile, np.roll(points, shift)) for shift in range(points.size)]
    )


def signed_steps(shifts: np.ndarray, period_steps: int) -> np.ndarray:
    return (shifts + period_steps // 2) % period_steps - period_steps // 2


def margin_against(scores: np.ndarray, chosen: int, rivals: np.ndarray) -> float:
    if scores[chosen] <= 0.0 or not rivals.any():
        return 0.0
    return float((scores[chosen] - scores[rivals].max()) / scores[chosen])


def align_grid(
    times_ms: np.ndarray,
    strengths: np.ndarray,
    phase_ms: float,
    layout: GridLayout,
) -> GridAlignment | None:
    index = np.round((times_ms - phase_ms) / layout.lattice_ms).astype(np.int64)
    residual = np.abs(times_ms - phase_ms - index * layout.lattice_ms)
    on_lattice = residual <= LATTICE_RADIUS_RATIO * layout.lattice_ms
    if not on_lattice.any():
        return None
    profile = np.bincount(
        index[on_lattice] % layout.period_steps,
        weights=strengths[on_lattice],
        minlength=layout.period_steps,
    )
    scores = circular_scores(profile, layout.points)
    shifts = np.arange(layout.period_steps)
    steps = signed_steps(shifts, layout.period_steps)
    tied = np.flatnonzero(scores >= scores.max() * (1.0 - TIE_TOLERANCE))
    chosen = int(tied[np.argmin(np.abs(steps[tied]))])
    rivals = (shifts - chosen) % layout.beat_steps != 0
    runner = int(np.flatnonzero(rivals)[np.argmax(scores[rivals])]) if rivals.any() else chosen
    return GridAlignment(
        phase_ms + steps[chosen] * layout.lattice_ms,
        phase_ms + steps[runner] * layout.lattice_ms,
        margin_against(scores, chosen, rivals),
        phase_ms,
        int(steps[chosen]),
    )


def comb_mask(
    times_ms: np.ndarray, alignment: GridAlignment, layout: GridLayout
) -> np.ndarray:
    index = np.round((times_ms - alignment.phase_ms) / layout.lattice_ms).astype(np.int64)
    residual = np.abs(times_ms - alignment.phase_ms - index * layout.lattice_ms)
    on_comb = np.isin((index - alignment.shift_steps) % STEPS_PER_BEAT, COMB_STEPS)
    return on_comb & (residual <= LATTICE_RADIUS_RATIO * layout.lattice_ms)
