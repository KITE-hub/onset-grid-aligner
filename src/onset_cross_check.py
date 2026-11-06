from __future__ import annotations

from dataclasses import dataclass

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_drift import DriftFit
from onset_odf import OnsetFunction
from onset_grid import estimate_comb_phase, phase_support, PhaseSupport
from onset_window_phase import timeline_stability

CHECK_MIN_POINTS = 30


@dataclass(frozen=True)
class DetectorCheck:
    name: str
    phase_ms: float
    uses_regression: bool
    support: PhaseSupport
    stability: float | None


@dataclass(frozen=True)
class CrossCheck:
    checks: tuple[DetectorCheck, ...]


@dataclass(frozen=True)
class CheckBasis:
    drift: DriftFit | None
    config: AnalysisConfig
    uses_regression: bool
    window_ms: float


def detector_check(
    function: OnsetFunction, basis: CheckBasis
) -> DetectorCheck | None:
    beat_ms = basis.config.layout.beat_ms
    times_ms, weights = function.times_ms, function.weights
    if times_ms.size < CHECK_MIN_POINTS:
        return None
    drift = basis.drift if basis.uses_regression else None
    slope = 0.0 if drift is None else drift.slope
    reference = 0.0 if drift is None else drift.reference_ms
    scaled_ms = times_ms * (1.0 - slope)
    estimate = estimate_comb_phase(scaled_ms, weights, basis.config.layout)
    if estimate is None:
        return None
    phase = float(wrap_symmetric(estimate.phase_ms + slope * reference, beat_ms))
    return DetectorCheck(
        function.name,
        phase,
        drift is not None,
        phase_support(scaled_ms, estimate.phase_ms, basis.config.layout),
        timeline_stability(
            times_ms,
            scaled_ms - estimate.phase_ms,
            basis.config.period_ms,
            basis.window_ms,
        ),
    )


def build_cross_check(
    functions: tuple[OnsetFunction, ...], basis: CheckBasis | None
) -> CrossCheck | None:
    if basis is None:
        return None
    results = (detector_check(function, basis) for function in functions)
    checks = tuple(check for check in results if check is not None)
    return CrossCheck(checks) if checks else None
