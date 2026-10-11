from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_detection import Heads
from onset_drift import DriftFit
from onset_odf import OnsetFunction
from onset_grid import estimate_comb_phase

CHECK_AGREE_MS = 3.0
CHECK_MIN_POINTS = 30


@dataclass(frozen=True)
class DetectorCheck:
    name: str
    phase_ms: float
    converted_ms: float
    difference_ms: float
    contrast: float
    uses_regression: bool


@dataclass(frozen=True)
class CrossCheck:
    rise_gap_ms: float
    checks: tuple[DetectorCheck, ...]

    @property
    def span_ms(self) -> float:
        return max(abs(check.difference_ms) for check in self.checks)

    @property
    def agreed(self) -> int:
        return sum(
            1 for check in self.checks if abs(check.difference_ms) <= CHECK_AGREE_MS
        )

    @property
    def all_agree(self) -> bool:
        return self.agreed == len(self.checks)


@dataclass(frozen=True)
class CheckBasis:
    drift: DriftFit | None
    config: AnalysisConfig
    rise_gap_ms: float
    conclusion_ms: float
    uses_regression: bool


def median_rise_gap(heads_ms: np.ndarray, peaks_ms: np.ndarray) -> float | None:
    if heads_ms.size == 0:
        return None
    return float(np.median(peaks_ms - heads_ms))


def detector_check(
    function: OnsetFunction, basis: CheckBasis
) -> DetectorCheck | None:
    beat_ms = basis.config.layout.comb_beat_ms
    times_ms, weights = function.times_ms, function.weights
    if times_ms.size < CHECK_MIN_POINTS:
        return None
    drift = basis.drift if basis.uses_regression else None
    slope = 0.0 if drift is None else drift.slope
    reference = 0.0 if drift is None else drift.reference_ms
    estimate = estimate_comb_phase(
        times_ms * (1.0 - slope), weights, basis.config.layout
    )
    if estimate is None:
        return None
    phase = float(wrap_symmetric(estimate.phase_ms + slope * reference, beat_ms))
    converted = float(wrap_symmetric(phase - basis.rise_gap_ms, beat_ms))
    difference = float(wrap_symmetric(converted - basis.conclusion_ms, beat_ms))
    return DetectorCheck(
        function.name, phase, converted, difference, estimate.phase_margin, drift is not None
    )


def primary_check_basis(
    selected: Heads,
    drift: DriftFit | None,
    config: AnalysisConfig,
    conclusion_ms: float,
    uses_regression: bool,
) -> CheckBasis | None:
    gap = median_rise_gap(selected.heads_ms, selected.peaks_ms)
    if gap is None:
        return None
    return CheckBasis(drift, config, gap, conclusion_ms, uses_regression)


def build_cross_check(
    functions: tuple[OnsetFunction, ...], basis: CheckBasis | None
) -> CrossCheck | None:
    if basis is None:
        return None
    results = (detector_check(function, basis) for function in functions)
    checks = tuple(check for check in results if check is not None)
    return CrossCheck(basis.rise_gap_ms, checks) if checks else None
