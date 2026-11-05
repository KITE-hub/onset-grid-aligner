from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_drift import DriftFit
from onset_odf import OnsetFunction
from onset_grid import estimate_comb_phase

CHECK_MIN_POINTS = 30


@dataclass(frozen=True)
class DetectorCheck:
    name: str
    phase_ms: float
    converted_ms: float
    difference_ms: float
    phase_margin: float
    uses_regression: bool
    runner_up_converted_ms: float | None


@dataclass(frozen=True)
class CrossCheck:
    checks: tuple[DetectorCheck, ...]


@dataclass(frozen=True)
class CheckBasis:
    drift: DriftFit | None
    config: AnalysisConfig
    conclusion_ms: float
    uses_regression: bool


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
    converted = phase
    difference = float(wrap_symmetric(converted - basis.conclusion_ms, beat_ms))
    runner_up = (
        None
        if estimate.runner_up_phase_ms is None
        else float(
            wrap_symmetric(estimate.runner_up_phase_ms + slope * reference, beat_ms)
        )
    )
    return DetectorCheck(
        function.name,
        phase,
        converted,
        difference,
        estimate.phase_margin,
        drift is not None,
        runner_up,
    )


def primary_check_basis(
    drift: DriftFit | None,
    config: AnalysisConfig,
    conclusion_ms: float,
    uses_regression: bool,
) -> CheckBasis | None:
    return CheckBasis(drift, config, conclusion_ms, uses_regression)


def build_cross_check(
    functions: tuple[OnsetFunction, ...], basis: CheckBasis | None
) -> CrossCheck | None:
    if basis is None:
        return None
    results = (detector_check(function, basis) for function in functions)
    checks = tuple(check for check in results if check is not None)
    return CrossCheck(checks) if checks else None
