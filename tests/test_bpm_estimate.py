import numpy as np
import pytest

from onset_bpm_estimate import ESTIMATE_MIN_DOMINANCE, estimate_bpm
from onset_odf import OnsetFunction


def pulse(name, times_ms, offset_ms=0.0):
    return OnsetFunction(name, times_ms + offset_ms, np.ones(times_ms.size))


STEADY_MS = np.arange(0.0, 40000.0, 500.0)


def test_estimate_bpm_recovers_a_steady_pulse():
    estimate = estimate_bpm((pulse("a", STEADY_MS),), 60.0, 200.0)

    assert estimate.bpm == pytest.approx(120.0, abs=0.05)
    assert estimate.dominance is not None
    assert estimate.dominance > ESTIMATE_MIN_DOMINANCE


def test_estimate_bpm_pools_every_method_equally():
    sources = (pulse("a", STEADY_MS), pulse("b", STEADY_MS, 3.0), pulse("c", STEADY_MS, 2.0))

    estimate = estimate_bpm(sources, 60.0, 200.0)

    assert estimate.bpm == pytest.approx(120.0, abs=0.05)


def test_estimate_bpm_ignores_methods_with_too_few_events():
    sources = (pulse("sparse", np.arange(0.0, 5000.0, 500.0)), pulse("a", STEADY_MS))

    assert estimate_bpm(sources, 60.0, 200.0).bpm == pytest.approx(120.0, abs=0.05)


def test_estimate_bpm_ignores_events_before_the_start_of_the_audio():
    early = OnsetFunction(
        "early", np.concatenate(([-2.0], STEADY_MS[1:])), np.ones(STEADY_MS.size)
    )

    assert estimate_bpm((early,), 60.0, 200.0).bpm == pytest.approx(120.0, abs=0.05)


def test_estimate_bpm_offers_the_half_and_double_tempo():
    estimate = estimate_bpm((pulse("a", STEADY_MS),), 60.0, 200.0)

    assert [bpm for bpm, _ in estimate.octaves] == pytest.approx([60.0, 240.0], abs=0.01)


def test_estimate_bpm_requires_enough_attacks_in_some_method():
    sources = (pulse("sparse", np.arange(0.0, 5000.0, 500.0)),)

    with pytest.raises(ValueError):
        estimate_bpm(sources, 60.0, 200.0)
