from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from onset_common import wrap_symmetric
from onset_timing import MS_PER_MINUTE

STEPS_PER_BEAT = 24
GRID_LATTICES = (2, 3, 4, 6, 8)
BEAT_POINTS = 4.0
LINE_POINTS = 1.0
FOLD_MIN_BPM = 150.0
MIN_BEAT_MARGIN = 0.1
TIE_TOLERANCE = 1e-9
PHASE_BIN_MS = 0.1
PHASE_SMOOTH_MS = 1.0
PHASE_BRANCH_SCORE_RATIO = 0.6
PHASE_BRANCH_SUPPORT_RATIO = 0.8
SUPPORT_SIGMA_MS = 1.5
SUPPORT_SIGMA_LATTICE_RATIO = 0.15
MAX_PHASE_CANDIDATES = 24
FIT_TOLERANCE_MS = 2.0
FIT_TOLERANCE_LATTICE_RATIO = 0.15
PHASE_REFINE_RADIUS_MS = 3.0
PHASE_REFINE_LATTICE_RATIO = 0.45


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


def line_priors() -> np.ndarray:
    coarsest: dict[int, int] = {}
    for lattice in GRID_LATTICES:
        for step in range(0, STEPS_PER_BEAT, STEPS_PER_BEAT // lattice):
            coarsest.setdefault(step, lattice)
    return np.asarray([min(GRID_LATTICES) / coarsest[step] for step in COMB_STEPS])


LINE_PRIORS = line_priors()


def fold_octave(bpm: float) -> int:
    return int(math.ceil(math.log2(FOLD_MIN_BPM / bpm)))


def fold_bpm(bpm: float) -> float:
    return bpm * 2.0 ** fold_octave(bpm)


@dataclass(frozen=True)
class GridLayout:
    bpm: float

    @property
    def octave(self) -> int:
        return fold_octave(self.bpm)

    @property
    def grid_bpm(self) -> float:
        return fold_bpm(self.bpm)

    @property
    def lattice_ms(self) -> float:
        return MS_PER_MINUTE / self.grid_bpm / STEPS_PER_BEAT

    @property
    def grid_beat_ms(self) -> float:
        return self.lattice_ms * STEPS_PER_BEAT

    @property
    def beat_ms(self) -> float:
        return MS_PER_MINUTE / self.bpm

    @property
    def comb_lattice_ms(self) -> float:
        return self.beat_ms / STEPS_PER_BEAT

    @property
    def beat_steps(self) -> int:
        return max(1, int(round(STEPS_PER_BEAT * 2.0**self.octave)))

    @property
    def period_steps(self) -> int:
        return math.lcm(self.beat_steps, STEPS_PER_BEAT)

    @property
    def comb_points(self) -> np.ndarray:
        steps = np.arange(self.period_steps)
        return np.isin(steps % STEPS_PER_BEAT, COMB_STEPS).astype(float)

    @property
    def beat_score_points(self) -> np.ndarray:
        steps = np.arange(self.period_steps)
        points = np.where(steps % self.beat_steps == 0, BEAT_POINTS, LINE_POINTS)
        return points * self.comb_points


@dataclass(frozen=True)
class PhasePeak:
    phase_ms: float
    score: float


@dataclass(frozen=True)
class PhaseSupport:
    attack_count: int
    support_count: int
    support_mad_ms: float | None

    @property
    def support_ratio(self) -> float | None:
        if self.attack_count == 0:
            return None
        return self.support_count / self.attack_count


@dataclass(frozen=True)
class CombPhaseEstimate:
    phase_ms: float


@dataclass(frozen=True)
class GridAlignment:
    phase_ms: float
    shift_steps: int
    runner_up_steps: int
    candidate_steps: tuple[int, ...]
    beat_margin: float
    lattice_ms: float
    beat_ms: float
    reference_phase_ms: float

    @property
    def beat_decided(self) -> bool:
        return self.beat_margin >= MIN_BEAT_MARGIN

    def beat_offset_ms(self, phase_ms: float) -> float:
        nearby = self.reference_phase_ms + wrap_symmetric(
            phase_ms - self.reference_phase_ms, self.lattice_ms
        )
        return float(wrap_symmetric(nearby + self.shift_steps * self.lattice_ms, self.beat_ms))



def fit_tolerance_ms(layout: GridLayout) -> float:
    return min(FIT_TOLERANCE_MS, layout.comb_lattice_ms * FIT_TOLERANCE_LATTICE_RATIO)


def lattice_index(times_ms: np.ndarray, phase_ms: float, lattice_ms: float) -> np.ndarray:
    return np.round((times_ms - phase_ms) / lattice_ms).astype(np.int64)


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


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0:
        return 0.0
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    middle = np.searchsorted(cumulative, cumulative[-1] * 0.5, side="left")
    return float(sorted_values[middle])


def comb_line_times(layout: GridLayout) -> np.ndarray:
    return np.asarray(COMB_STEPS, dtype=float) * layout.comb_lattice_ms


def comb_residuals(times_ms: np.ndarray, layout: GridLayout) -> np.ndarray:
    times = np.asarray(times_ms, dtype=float)
    beat_ms = layout.beat_ms
    line_times = comb_line_times(layout)
    beat_positions = np.mod(times, beat_ms)
    differences = wrap_symmetric(beat_positions[:, None] - line_times[None, :], beat_ms)
    nearest = np.argmin(np.abs(differences), axis=1)
    return differences[np.arange(times.size), nearest]


def prior_support(
    event_phases: np.ndarray, values: np.ndarray, phase_ms: float, layout: GridLayout
) -> float:
    differences = wrap_symmetric(
        (event_phases - phase_ms)[:, None] - comb_line_times(layout)[None, :],
        layout.beat_ms,
    )
    sigma = min(SUPPORT_SIGMA_MS, layout.comb_lattice_ms * SUPPORT_SIGMA_LATTICE_RATIO)
    kernel = np.exp(-0.5 * (differences / sigma) ** 2) * LINE_PRIORS[None, :]
    return float(np.dot(values, kernel.max(axis=1)))


def select_phase_peak(
    peaks: list[PhasePeak],
    event_phases: np.ndarray,
    values: np.ndarray,
    layout: GridLayout,
) -> PhasePeak:
    ranked = sorted(peaks, key=lambda peak: peak.score, reverse=True)[
        :MAX_PHASE_CANDIDATES
    ]
    supports = [
        prior_support(event_phases, values, peak.phase_ms, layout) for peak in ranked
    ]
    floor = ranked[0].score * PHASE_BRANCH_SCORE_RATIO
    scored = [position for position, peak in enumerate(ranked) if peak.score >= floor]
    best_support = max(supports[position] for position in scored)
    eligible = [
        position
        for position in scored
        if supports[position] >= best_support * PHASE_BRANCH_SUPPORT_RATIO
    ]
    chosen = min(
        eligible,
        key=lambda position: (
            abs(ranked[position].phase_ms),
            ranked[position].phase_ms,
        ),
    )
    return ranked[chosen]


def phase_support(
    times_ms: np.ndarray, phase_ms: float, layout: GridLayout
) -> PhaseSupport:
    residuals = comb_residuals(np.asarray(times_ms, dtype=float) - phase_ms, layout)
    supporting = residuals[np.abs(residuals) <= fit_tolerance_ms(layout)]
    if supporting.size == 0:
        return PhaseSupport(int(residuals.size), 0, None)
    mad = float(np.median(np.abs(supporting - np.median(supporting))))
    return PhaseSupport(int(residuals.size), int(supporting.size), mad)


def estimate_comb_phase(
    times_ms: np.ndarray, weights: np.ndarray, layout: GridLayout
) -> CombPhaseEstimate | None:
    times = np.asarray(times_ms, dtype=float)
    values = np.asarray(weights, dtype=float)
    if times.size == 0 or values.size != times.size:
        return None
    valid = np.isfinite(times) & np.isfinite(values) & (values > 0.0)
    times = times[valid]
    values = values[valid]
    if times.size == 0:
        return None
    beat_ms = layout.beat_ms
    line_times = comb_line_times(layout)
    event_phases = np.mod(times, beat_ms)
    candidates = wrap_symmetric(event_phases[:, None] - line_times[None, :], beat_ms)
    bins = max(1024, int(math.ceil(beat_ms / PHASE_BIN_MS)))
    bin_width = beat_ms / bins
    histogram, edges = np.histogram(
        candidates.ravel(),
        bins=bins,
        range=(-beat_ms / 2.0, beat_ms / 2.0),
        weights=np.broadcast_to(values[:, None], candidates.shape).ravel(),
    )
    smoothed = ndimage.gaussian_filter1d(
        histogram.astype(float),
        sigma=max(1.0, PHASE_SMOOTH_MS / bin_width),
        mode="wrap",
    )
    peaks = np.flatnonzero(
        (smoothed >= np.roll(smoothed, 1)) & (smoothed > np.roll(smoothed, -1))
    )
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(smoothed))])
    centers = (edges[:-1] + edges[1:]) * 0.5

    def refine_peak(index: int) -> tuple[float, float]:
        left = smoothed[(index - 1) % bins]
        center = smoothed[index]
        right = smoothed[(index + 1) % bins]
        curvature = left - 2.0 * center + right
        delta = 0.0 if curvature >= 0.0 else float(
            np.clip(0.5 * (left - right) / curvature, -0.5, 0.5)
        )
        phase = float(wrap_symmetric(centers[index] + delta * bin_width, beat_ms))
        return phase, float(center)

    peak_rows = [
        PhasePeak(*refine_peak(int(index))) for index in peaks
    ]
    phase = select_phase_peak(peak_rows, event_phases, values, layout).phase_ms
    radius = min(
        PHASE_REFINE_RADIUS_MS, layout.comb_lattice_ms * PHASE_REFINE_LATTICE_RATIO
    )
    relative_candidates = wrap_symmetric(candidates - phase, beat_ms)
    selected_candidates = np.abs(relative_candidates) <= radius
    candidate_values = relative_candidates[selected_candidates]
    candidate_weights = np.broadcast_to(values[:, None], candidates.shape)[selected_candidates]
    if candidate_values.size:
        phase = float(
            wrap_symmetric(phase + weighted_median(candidate_values, candidate_weights), beat_ms)
        )
    return CombPhaseEstimate(phase)


def align_grid(
    times_ms: np.ndarray,
    weights: np.ndarray,
    reference_phase_ms: float,
    layout: GridLayout,
    diagnostic_weights: np.ndarray | None = None,
    phase_estimate: CombPhaseEstimate | None = None,
) -> GridAlignment | None:
    times = np.asarray(times_ms, dtype=float)
    values = np.asarray(weights, dtype=float)
    score_weights = values if diagnostic_weights is None else np.asarray(diagnostic_weights, dtype=float)
    if times.size == 0 or values.size != times.size or score_weights.size != times.size:
        return None
    valid = np.isfinite(times) & np.isfinite(values) & np.isfinite(score_weights)
    times = times[valid]
    values = values[valid]
    score_weights = score_weights[valid]
    if times.size == 0 or not np.any(values > 0.0):
        return None
    comb = phase_estimate or estimate_comb_phase(times, values, layout)
    if comb is None:
        return None
    period_steps = layout.period_steps
    index = lattice_index(times, reference_phase_ms, layout.lattice_ms)
    profile = np.bincount(index % period_steps, weights=score_weights, minlength=period_steps)
    scores = circular_scores(profile, layout.beat_score_points)
    shifts = np.arange(period_steps)
    steps = signed_steps(shifts, period_steps)
    tied = np.flatnonzero(scores >= scores.max() * (1.0 - TIE_TOLERANCE))
    chosen = int(tied[np.argmin(np.abs(steps[tied]))])
    rivals = (shifts - chosen) % layout.beat_steps != 0
    runner = int(np.flatnonzero(rivals)[np.argmax(scores[rivals])]) if rivals.any() else chosen
    if scores.max() > 0.0:
        competitive = scores >= scores.max() * (1.0 - MIN_BEAT_MARGIN)
        candidate_steps = tuple(int(step) for step in steps[competitive])
    else:
        candidate_steps = ()
    return GridAlignment(
        comb.phase_ms,
        int(steps[chosen]),
        int(steps[runner]),
        candidate_steps,
        margin_against(scores, chosen, rivals),
        layout.lattice_ms,
        layout.beat_ms,
        reference_phase_ms,
    )
