from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_beat import align_beat
from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_detection import Detection
from onset_drift import DriftFit
from onset_grid import estimate_comb_phase, GridAlignment, phase_support, PhaseSupport
from onset_groups import analyze_group
from onset_odf import OnsetFunction
from onset_timing import TempoMap
from onset_warp import ANCHOR_UNAVAILABLE_MESSAGE, make_warp
from onset_window_phase import (
    estimate_drift,
    support_window_ms,
    timeline_stability,
    window_phases,
)

ENVELOPE_NAME = "envelope"
NO_EVENTS_MESSAGE = "オンセットを十分に検出できませんでした"
NO_KICKS = np.empty(0)


@dataclass(frozen=True)
class MethodEstimate:
    name: str
    phase_ms: float
    drift: DriftFit | None
    support: PhaseSupport
    stability: float | None
    alignment: GridAlignment | None = None

    def phase_at(self, time_ms: float) -> float:
        if self.drift is None:
            return self.phase_ms
        return self.phase_ms + self.drift.slope * (time_ms - self.drift.reference_ms)

    def beat_offset_ms(self, phase_ms: float) -> float | None:
        if self.alignment is None or not self.alignment.beat_decided:
            return None
        return self.alignment.beat_offset_ms(phase_ms)


def envelope_function(detection: Detection, head_level: float) -> OnsetFunction:
    heads = detection.heads[head_level]
    return OnsetFunction(
        ENVELOPE_NAME, heads.heads_ms, np.maximum(heads.strengths, 0.0)
    )


def method_sources(
    detection: Detection,
    head_level: float,
    functions: tuple[OnsetFunction, ...],
) -> tuple[OnsetFunction, ...]:
    return (envelope_function(detection, head_level), *functions)


def significant_drift(fit: DriftFit | None) -> DriftFit | None:
    return fit if fit is not None and fit.significant else None


def estimate_method(
    function: OnsetFunction,
    config: AnalysisConfig,
    kick_ms: np.ndarray = NO_KICKS,
) -> MethodEstimate | None:
    order = np.argsort(function.times_ms, kind="stable")
    times_ms = function.times_ms[order]
    weights = function.weights[order]
    period_ms = config.period_ms
    group = analyze_group(times_ms, period_ms)
    if group is None:
        return None
    windows = (
        window_phases(group, period_ms, config.drift_window_sec)
        if config.drift_window_sec > 0.0
        else None
    )
    drift = significant_drift(estimate_drift(times_ms, windows, config))
    slope = 0.0 if drift is None else drift.slope
    reference_ms = 0.0 if drift is None else drift.reference_ms
    scaled_ms = times_ms * (1.0 - slope)
    comb = estimate_comb_phase(scaled_ms, weights, config.layout)
    if comb is None:
        return None
    phase_ms = float(
        wrap_symmetric(comb.phase_ms + slope * reference_ms, config.beat_ms)
    )
    return MethodEstimate(
        function.name,
        phase_ms,
        drift,
        phase_support(scaled_ms, comb.phase_ms, config.layout),
        timeline_stability(
            times_ms,
            scaled_ms - comb.phase_ms,
            period_ms,
            support_window_ms(config),
        ),
        align_beat(times_ms, weights, kick_ms, phase_ms, config.layout),
    )


def analyze_methods(
    sources: tuple[OnsetFunction, ...],
    config: AnalysisConfig,
    kick_ms: np.ndarray = NO_KICKS,
    tempo: TempoMap | None = None,
) -> tuple[MethodEstimate, ...]:
    estimates: list[MethodEstimate] = []
    anchored = False
    for source in sources:
        warp = make_warp(tempo, config, source)
        if warp is None:
            continue
        anchored = True
        estimate = estimate_method(warp.function(source), config, warp.kicks(kick_ms))
        if estimate is not None:
            estimates.append(estimate)
    if estimates:
        return tuple(estimates)
    raise ValueError(
        ANCHOR_UNAVAILABLE_MESSAGE
        if sources and tempo is not None and not anchored
        else NO_EVENTS_MESSAGE
    )
