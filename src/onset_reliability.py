from __future__ import annotations

from dataclasses import dataclass

from onset_grid import PhaseSupport
from onset_method import MethodEstimate

DRIFT_SLOPE_UNIT_MS_PER_MIN = 60000.0


@dataclass(frozen=True)
class MethodReliability:
    name: str
    support: PhaseSupport
    stability: float | None
    drift_slope_ms_per_min: float | None


@dataclass(frozen=True)
class Reliability:
    methods: tuple[MethodReliability, ...]

    def method(self, name: str) -> MethodReliability | None:
        return next((item for item in self.methods if item.name == name), None)


def method_reliability(estimate: MethodEstimate) -> MethodReliability:
    drift = estimate.drift
    return MethodReliability(
        estimate.name,
        estimate.support,
        estimate.stability,
        None if drift is None else drift.slope * DRIFT_SLOPE_UNIT_MS_PER_MIN,
    )


def reliability_of(estimates: tuple[MethodEstimate, ...]) -> Reliability:
    return Reliability(tuple(method_reliability(estimate) for estimate in estimates))
