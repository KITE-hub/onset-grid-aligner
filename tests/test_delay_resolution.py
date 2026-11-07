import pytest

from onset_drift import DriftFit
from onset_grid import GridLayout, PhaseSupport
from onset_method import MethodEstimate
from onset_resolution import delay_candidates

DURATION_MS = 10000.0
SUPPORT = PhaseSupport(10, 5, 0.5)


def make_layout(beat_ms):
    return GridLayout(60000.0 / beat_ms)


def make_drift(slope, reference_ms=0.0):
    return DriftFit(0.0, slope, slope, slope, reference_ms)


def make_estimate(name, phase_ms, drift=None):
    return MethodEstimate(name, phase_ms, drift, SUPPORT, None)


def candidates_of(estimates, beat_ms):
    return delay_candidates(tuple(estimates), make_layout(beat_ms), DURATION_MS)


def test_delay_is_the_negated_phase():
    candidates = dict(candidates_of([make_estimate("envelope", 12.0)], 100.0))

    assert candidates["envelope"] == pytest.approx(-12.0)


def test_delay_uses_phase_at_audio_midpoint():
    estimate = make_estimate("envelope", 0.0, make_drift(0.001))

    assert dict(candidates_of([estimate], 100.0))["envelope"] == pytest.approx(-5.0)


def test_drift_reference_time_shifts_the_midpoint_extrapolation():
    estimate = make_estimate("envelope", 0.0, make_drift(0.001, 2000.0))

    assert dict(candidates_of([estimate], 100.0))["envelope"] == pytest.approx(-3.0)


def test_no_estimates_yield_no_candidates():
    assert candidates_of([], 100.0) == ()


def test_every_method_keeps_its_own_phase_and_the_estimate_order():
    estimates = [
        make_estimate("envelope", 4.8),
        make_estimate("複素領域", -40.0),
        make_estimate("HFC", 4.7),
    ]

    candidates = candidates_of(estimates, 375.0)

    assert [name for name, _ in candidates] == ["envelope", "複素領域", "HFC"]
    assert dict(candidates) == {
        "envelope": pytest.approx(-4.8),
        "複素領域": pytest.approx(40.0),
        "HFC": pytest.approx(-4.7),
    }


def test_drift_is_applied_only_to_methods_that_carry_it():
    estimates = [
        make_estimate("HFC", 0.0, make_drift(0.001)),
        make_estimate("複素領域", 0.0),
    ]

    candidates = dict(candidates_of(estimates, 100.0))

    assert candidates["HFC"] == pytest.approx(-5.0)
    assert candidates["複素領域"] == pytest.approx(0.0)


def test_delay_wraps_into_half_a_beat():
    candidates = dict(candidates_of([make_estimate("envelope", 60.0)], 100.0))

    assert candidates["envelope"] == pytest.approx(40.0)
