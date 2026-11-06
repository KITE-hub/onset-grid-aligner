from __future__ import annotations

from onset_analysis import Analysis
from onset_common import phase_to_delay_ms, wrap_symmetric
from onset_grid import GridAlignment, GridLayout

FIXED_GRID_NAME = "fixed_grid"

DelayCandidates = tuple[tuple[str, float], ...]


def midpoint_phase(
    phase_ms: float,
    analysis: Analysis,
    duration_ms: float,
    wrap_ms: float,
    use_regression: bool = True,
) -> float:
    drift = analysis.drift
    if use_regression and analysis.uses_regression and drift is not None:
        phase_ms += drift.slope * (duration_ms / 2.0 - drift.reference_ms)
    return float(wrap_symmetric(phase_ms, wrap_ms))


def grid_branch_phase(
    phase_ms: float,
    alignment: GridAlignment | None,
    lattice_ms: float | None = None,
) -> float:
    if alignment is None:
        return phase_ms
    step_ms = alignment.lattice_ms if lattice_ms is None else lattice_ms
    return float(
        alignment.phase_ms
        + wrap_symmetric(phase_ms - alignment.phase_ms, step_ms)
    )


def delay_ms_at_midpoint(
    phase_ms: float,
    analysis: Analysis,
    duration_ms: float,
    wrap_ms: float,
    use_regression: bool = True,
) -> float:
    midpoint = midpoint_phase(phase_ms, analysis, duration_ms, wrap_ms, use_regression)
    return float(wrap_symmetric(phase_to_delay_ms(midpoint), wrap_ms))


def delay_candidates(
    analysis: Analysis,
    alignment: GridAlignment | None,
    layout: GridLayout,
    duration_ms: float,
) -> DelayCandidates:
    wrap_ms = layout.beat_ms
    candidates: list[tuple[str, float]] = []
    if alignment is not None:
        candidates.append(
            (
                FIXED_GRID_NAME,
                delay_ms_at_midpoint(alignment.phase_ms, analysis, duration_ms, wrap_ms),
            )
        )
    if analysis.cross is not None:
        for check in analysis.cross.checks:
            branch_phase = grid_branch_phase(
                check.phase_ms, alignment, layout.comb_lattice_ms
            )
            candidates.append(
                (
                    check.name,
                    delay_ms_at_midpoint(
                        branch_phase,
                        analysis,
                        duration_ms,
                        wrap_ms,
                        check.uses_regression,
                    ),
                )
            )
    return tuple(candidates)
