from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_cross_check import build_cross_check, CheckBasis, CrossCheck
from onset_detection import Detection, Heads
from onset_drift import DriftFit
from onset_groups import analyze_group, GroupAnalysis
from onset_odf import OnsetFunction
from onset_grid import (
    CombPhaseEstimate,
    estimate_comb_phase,
    phase_support,
    PhaseSupport,
)
from onset_phase import cluster_radius
from onset_window_phase import (
    estimate_drift,
    event_support,
    regression_trend_ms,
    resolve_phase,
    support_window_ms,
    window_phases,
)


@dataclass(frozen=True)
class Analysis:
    period_ms: float
    primary: GroupAnalysis
    drift: DriftFit | None
    conclusion_ms: float
    uses_regression: bool
    grid_phase: CombPhaseEstimate | None
    cross: CrossCheck | None
    stability: float | None
    grid_support: PhaseSupport | None

    def trend_ms(self, times_ms: np.ndarray) -> np.ndarray:
        return regression_trend_ms(times_ms, self.drift, self.uses_regression)

    def residuals_ms(self, values_ms: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
        return np.asarray(
            wrap_symmetric(
                values_ms - self.trend_ms(times_ms) - self.conclusion_ms,
                self.period_ms,
            )
        )



def aligned_members(analysis: Analysis, selected: Heads) -> np.ndarray:
    residual = analysis.residuals_ms(selected.heads_ms, selected.peaks_ms)
    return np.flatnonzero(np.abs(residual) < cluster_radius(analysis.period_ms))


def analyze(
    detection: Detection,
    config: AnalysisConfig,
    functions: tuple[OnsetFunction, ...],
) -> Analysis:
    period = config.period_ms
    selected = detection.heads[config.head_level]
    primary = analyze_group(selected.heads_ms, period)
    if primary is None:
        raise ValueError("オンセットを十分に検出できませんでした")
    windows = (
        window_phases(primary, period, config.drift_window_sec)
        if config.drift_window_sec > 0.0
        else None
    )
    drift = estimate_drift(primary.heads_ms, windows, config)
    conclusion_ms, uses_regression = resolve_phase(drift, primary.phase_ms, period)
    grid_phase = estimate_comb_phase(
        selected.heads_ms, np.maximum(selected.strengths, 0.0), config.layout
    )
    basis = (
        None
        if grid_phase is None
        else CheckBasis(drift, config, uses_regression, support_window_ms(config))
    )
    cross = build_cross_check(functions, basis)
    support = event_support(
        selected,
        conclusion_ms,
        drift,
        uses_regression,
        period,
        support_window_ms(config),
    )
    return Analysis(
        period,
        primary,
        drift,
        conclusion_ms,
        uses_regression,
        grid_phase,
        cross,
        support.stability,
        None
        if grid_phase is None
        else phase_support(selected.heads_ms, grid_phase.phase_ms, config.layout),
    )
