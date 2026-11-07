import numpy as np
import pytest

from onset_drift import DriftFit
from onset_grid import GridLayout, phase_support, PhaseSupport
from onset_method import MethodEstimate
from onset_reliability import reliability_of
from onset_window_phase import timeline_stability

LAYOUT = GridLayout(120.0)


def test_phase_support_counts_only_attacks_on_the_grid():
    on_grid = 3.0 + np.arange(40) * LAYOUT.beat_ms
    off_grid = 3.0 + 10.4 + np.arange(10) * LAYOUT.beat_ms

    support = phase_support(np.concatenate([on_grid, off_grid]), 3.0, LAYOUT)

    assert support.attack_count == 50
    assert support.support_count == 40
    assert support.support_ratio == pytest.approx(0.8)
    assert support.support_mad_ms == pytest.approx(0.0)


def test_phase_support_has_no_mad_without_supporting_attacks():
    support = phase_support(10.4 + np.arange(5) * LAYOUT.beat_ms, 0.0, LAYOUT)

    assert support.support_count == 0
    assert support.support_mad_ms is None


def test_phase_support_ratio_is_unavailable_without_attacks():
    assert PhaseSupport(0, 0, None).support_ratio is None


def test_timeline_stability_is_full_when_every_window_is_aligned():
    times_ms = np.arange(0.0, 60000.0, 100.0)

    stability = timeline_stability(
        times_ms, np.zeros(times_ms.size), LAYOUT.lattice_ms, 10000.0
    )

    assert stability == pytest.approx(1.0)


def test_timeline_stability_is_unavailable_for_short_material():
    times_ms = np.arange(0.0, 5000.0, 100.0)

    assert (
        timeline_stability(
            times_ms, np.zeros(times_ms.size), LAYOUT.lattice_ms, 10000.0
        )
        is None
    )


def make_estimate(name, drift=None):
    return MethodEstimate(name, 0.0, drift, PhaseSupport(10, 5, 0.5), 0.5)


def test_reliability_lists_every_method_in_estimate_order():
    estimates = (
        make_estimate("envelope"),
        make_estimate("HFC"),
        make_estimate("複素領域"),
    )

    reliability = reliability_of(estimates)

    assert [method.name for method in reliability.methods] == [
        "envelope",
        "HFC",
        "複素領域",
    ]
    assert reliability.method("HFC").support == PhaseSupport(10, 5, 0.5)
    assert reliability.method("HFC").stability == 0.5
    assert reliability.method("missing") is None


def test_reliability_reports_each_methods_own_drift_slope_per_minute():
    drift = DriftFit(0.0, 0.001, 0.0009, 0.0011, 0.0)

    reliability = reliability_of((make_estimate("envelope", drift), make_estimate("HFC")))

    assert reliability.method("envelope").drift_slope_ms_per_min == pytest.approx(60.0)
    assert reliability.method("HFC").drift_slope_ms_per_min is None
