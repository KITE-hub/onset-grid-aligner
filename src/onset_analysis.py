from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_cross_check import build_cross_check, CrossCheck, primary_check_basis
from onset_detection import Detection, Heads
from onset_drift import DriftFit
from onset_groups import build_groups, GroupAnalysis, head_level_sensitivity
from onset_labels import LABEL_ALL
from onset_odf import OnsetFunction
from onset_grid import CombPhaseEstimate, estimate_comb_phase
from onset_phase import cluster_radius, standard_error_ms
from onset_window_phase import (
    estimate_drift,
    event_support,
    regression_trend_ms,
    resolve_phase,
    support_window_ms,
    window_phases,
    window_residuals,
    window_standard_error_ms,
    WindowPhases,
)


@dataclass(frozen=True)
class Analysis:
    period_ms: float
    groups: dict[str, GroupAnalysis]
    primary: GroupAnalysis
    windows: WindowPhases | None
    drift: DriftFit | None
    sensitivity: tuple[tuple[float, float | None], ...]
    conclusion_ms: float
    uses_regression: bool
    grid_phase: CombPhaseEstimate | None
    cross: CrossCheck | None
    stability: float | None
    grid_fit: float | None

    def trend_ms(self, times_ms: np.ndarray) -> np.ndarray:
        return regression_trend_ms(times_ms, self.drift, self.uses_regression)

    def residuals_ms(self, values_ms: np.ndarray, times_ms: np.ndarray) -> np.ndarray:
        return np.asarray(
            wrap_symmetric(
                values_ms - self.trend_ms(times_ms) - self.conclusion_ms,
                self.period_ms,
            )
        )

    @property
    def uncertainty_ms(self) -> float:
        events = standard_error_ms(self.primary.spread_ms, self.primary.cluster_count)
        residuals = window_residuals(
            self.windows,
            self.drift,
            self.conclusion_ms,
            self.uses_regression,
            self.period_ms,
        )
        return max(events, window_standard_error_ms(residuals))


def aligned_members(
    analysis: Analysis, selected: Heads
) -> tuple[np.ndarray, np.ndarray]:
    residual = analysis.residuals_ms(selected.heads_ms, selected.peaks_ms)
    members = np.flatnonzero(np.abs(residual) < cluster_radius(analysis.period_ms))
    return members, np.zeros(members.size)


def analyze(
    detection: Detection,
    config: AnalysisConfig,
    functions: tuple[OnsetFunction, ...],
) -> Analysis:
    period = config.period_ms
    selected = detection.heads[config.head_level]
    groups = build_groups(selected.heads_ms, period)
    if not groups:
        raise ValueError("オンセットを十分に検出できませんでした")
    primary = groups[LABEL_ALL]
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
        else primary_check_basis(
            drift, config, grid_phase.phase_ms, uses_regression
        )
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
        groups,
        primary,
        windows,
        drift,
        head_level_sensitivity(detection, period),
        conclusion_ms,
        uses_regression,
        grid_phase,
        cross,
        support.stability,
        support.grid_fit,
    )
