from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_detection import Detection, Heads
from onset_drift import DriftFit
from onset_groups import GroupAnalysis
from onset_kick import primary_keep_mask
from onset_labels import LABEL_ALL
from onset_odf import fold_phase, OnsetFunction
from onset_reference import ReferenceEstimate

CHECK_AGREE_MS = 3.0
CHECK_MIN_POINTS = 30
MIXTURE_AGREE_MS = 1.5


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
    detection: Detection
    label: str
    drift: DriftFit | None
    config: AnalysisConfig
    rise_gap_ms: float
    conclusion_ms: float
    uses_regression: bool


def median_rise_gap(
    heads_ms: np.ndarray, peaks_ms: np.ndarray, kick_ms: np.ndarray, label: str
) -> float | None:
    keep = primary_keep_mask(heads_ms, kick_ms, label)
    if not keep.any():
        return None
    return float(np.median(peaks_ms[keep] - heads_ms[keep]))


def detector_check(
    function: OnsetFunction, basis: CheckBasis
) -> DetectorCheck | None:
    period = basis.config.period_ms
    keep = primary_keep_mask(
        function.times_ms, basis.detection.kick_ms, basis.label
    )
    times_ms, weights = function.times_ms[keep], function.weights[keep]
    if times_ms.size < CHECK_MIN_POINTS:
        return None
    drift = basis.drift if basis.uses_regression else None
    slope = 0.0 if drift is None else drift.slope
    reference = 0.0 if drift is None else drift.reference_ms
    folded, contrast = fold_phase(times_ms * (1.0 - slope), weights, period)
    phase = float(wrap_symmetric(folded + slope * reference, period))
    converted = float(wrap_symmetric(phase - basis.rise_gap_ms, period))
    difference = float(wrap_symmetric(converted - basis.conclusion_ms, period))
    return DetectorCheck(
        function.name, phase, converted, difference, contrast, drift is not None
    )


def primary_check_basis(
    detection: Detection,
    selected: Heads,
    primary: GroupAnalysis,
    drift: DriftFit | None,
    config: AnalysisConfig,
    conclusion_ms: float,
    uses_regression: bool,
) -> CheckBasis | None:
    gap = median_rise_gap(
        selected.heads_ms, selected.peaks_ms, detection.kick_ms, primary.label
    )
    if gap is None:
        return None
    return CheckBasis(
        detection, primary.label, drift, config, gap, conclusion_ms, uses_regression
    )


def reference_check_basis(
    detection: Detection, config: AnalysisConfig, reference: ReferenceEstimate
) -> CheckBasis:
    return CheckBasis(
        detection,
        LABEL_ALL,
        reference.drift,
        config,
        reference.groups[reference.chosen].gap_ms,
        reference.conclusion_ms,
        reference.uses_regression,
    )


def build_cross_check(
    functions: tuple[OnsetFunction, ...], basis: CheckBasis | None
) -> CrossCheck | None:
    if basis is None:
        return None
    results = (detector_check(function, basis) for function in functions)
    checks = tuple(check for check in results if check is not None)
    return CrossCheck(basis.rise_gap_ms, checks) if checks else None
