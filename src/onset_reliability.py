from __future__ import annotations

from dataclasses import dataclass

from onset_analysis import Analysis
from onset_grid import GridAlignment, GridLayout, PhaseSupport
from onset_resolution import FIXED_GRID_NAME, grid_branch_phase

DRIFT_SLOPE_UNIT_MS_PER_MIN = 60000.0
BRANCH_SHIFT_EPSILON_MS = 0.005


@dataclass(frozen=True)
class MethodReliability:
    name: str
    support: PhaseSupport
    stability: float | None
    branch_shift_ms: float

    @property
    def branch_shifted(self) -> bool:
        return abs(self.branch_shift_ms) >= BRANCH_SHIFT_EPSILON_MS


@dataclass(frozen=True)
class Reliability:
    methods: tuple[MethodReliability, ...]
    drift_slope_ms_per_min: float | None

    def method(self, name: str) -> MethodReliability | None:
        return next((item for item in self.methods if item.name == name), None)


def branch_shift_ms(
    phase_ms: float, alignment: GridAlignment | None, layout: GridLayout
) -> float:
    snapped = grid_branch_phase(phase_ms, alignment, layout.comb_lattice_ms)
    return float(phase_ms - snapped)


def reliability_of(
    analysis: Analysis, alignment: GridAlignment | None, layout: GridLayout
) -> Reliability:
    methods: list[MethodReliability] = []
    if alignment is not None and analysis.grid_support is not None:
        methods.append(
            MethodReliability(
                FIXED_GRID_NAME, analysis.grid_support, analysis.stability, 0.0
            )
        )
    if analysis.cross is not None:
        methods.extend(
            MethodReliability(
                check.name,
                check.support,
                check.stability,
                branch_shift_ms(check.phase_ms, alignment, layout),
            )
            for check in analysis.cross.checks
        )
    drift = analysis.drift
    applied = analysis.uses_regression and drift is not None
    return Reliability(
        tuple(methods),
        drift.slope * DRIFT_SLOPE_UNIT_MS_PER_MIN if applied else None,
    )
