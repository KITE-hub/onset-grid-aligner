from __future__ import annotations

from onset_common import phase_to_delay_ms, wrap_symmetric
from onset_grid import GridLayout
from onset_method import MethodEstimate

DelayCandidates = tuple[tuple[str, float], ...]


def delay_ms_at_midpoint(
    estimate: MethodEstimate, duration_ms: float, wrap_ms: float
) -> float:
    midpoint = wrap_symmetric(estimate.phase_at(duration_ms / 2.0), wrap_ms)
    return float(wrap_symmetric(phase_to_delay_ms(midpoint), wrap_ms))


def delay_candidates(
    estimates: tuple[MethodEstimate, ...], layout: GridLayout, duration_ms: float
) -> DelayCandidates:
    return tuple(
        (estimate.name, delay_ms_at_midpoint(estimate, duration_ms, layout.beat_ms))
        for estimate in estimates
    )
