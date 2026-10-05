from __future__ import annotations

from dataclasses import dataclass

from onset_config import AnalysisConfig
from onset_cross_check import (
    build_cross_check,
    CrossCheck,
    primary_check_basis,
    reference_check_basis,
)
from onset_detection import Detection
from onset_drift import DriftFit
from onset_groups import (
    build_groups,
    GroupAnalysis,
    head_level_sensitivity,
    pick_primary,
)
from onset_labels import LABEL_REFERENCE
from onset_odf import OnsetFunction
from onset_phase import standard_error_ms
from onset_reference import estimate_reference, ReferenceEstimate
from onset_window_phase import (
    estimate_drift,
    event_support,
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
    cross: CrossCheck | None
    stability: float | None
    grid_fit: float | None
    reference: ReferenceEstimate | None

    @property
    def final_ms(self) -> float:
        if self.reference is None:
            return self.conclusion_ms
        return self.reference.conclusion_ms

    @property
    def final_uses_regression(self) -> bool:
        if self.reference is None:
            return self.uses_regression
        return self.reference.uses_regression

    @property
    def final_stability(self) -> float | None:
        if self.reference is None:
            return self.stability
        return self.reference.stability

    @property
    def final_grid_fit(self) -> float | None:
        if self.reference is None:
            return self.grid_fit
        return self.reference.grid_fit

    @property
    def final_label(self) -> str:
        return self.primary.label if self.reference is None else LABEL_REFERENCE

    @property
    def final_standard_error_ms(self) -> float:
        if self.reference is None:
            events = standard_error_ms(self.primary.spread_ms, self.primary.cluster_count)
            residuals = window_residuals(
                self.windows,
                self.drift,
                self.conclusion_ms,
                self.uses_regression,
                self.period_ms,
            )
            return max(events, window_standard_error_ms(residuals))
        reference = self.reference
        group = reference.groups[reference.chosen]
        residuals = window_residuals(
            reference.windows,
            reference.drift,
            reference.conclusion_ms,
            reference.uses_regression,
            self.period_ms,
        )
        return max(
            standard_error_ms(group.spread_ms, group.count),
            window_standard_error_ms(residuals),
        )


def analyze(
    detection: Detection,
    config: AnalysisConfig,
    functions: tuple[OnsetFunction, ...],
) -> Analysis:
    period = config.period_ms
    selected = detection.heads[config.head_level]
    groups = build_groups(selected.heads_ms, detection.kick_ms, period)
    primary = pick_primary(groups)
    if primary is None:
        raise ValueError("オンセットを十分に検出できませんでした")
    windows = (
        window_phases(primary, period, config.drift_window_sec)
        if config.drift_window_sec > 0.0
        else None
    )
    drift = estimate_drift(primary.heads_ms, windows, config)
    conclusion_ms, uses_regression = resolve_phase(drift, primary.phase_ms, period)
    reference = estimate_reference(detection, selected, config, conclusion_ms, drift)
    basis = (
        primary_check_basis(
            detection, selected, primary, drift, config, conclusion_ms, uses_regression
        )
        if reference is None
        else reference_check_basis(detection, config, reference)
    )
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
        head_level_sensitivity(detection, period, primary.label),
        conclusion_ms,
        uses_regression,
        build_cross_check(functions, basis),
        support.stability,
        support.grid_fit,
        reference,
    )
