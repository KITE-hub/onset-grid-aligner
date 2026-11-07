from dataclasses import replace

import numpy as np
import pytest

from onset_config import AnalysisConfig
from onset_detection import Detection, Heads
from onset_method import (
    analyze_methods,
    ENVELOPE_NAME,
    envelope_function,
    estimate_method,
    method_sources,
    NO_EVENTS_MESSAGE,
)
from onset_odf import OnsetFunction
from onset_timing import TempoMap, TempoPoint
from onset_warp import ANCHOR_UNAVAILABLE_MESSAGE

BPM = 120.0
CONFIG = AnalysisConfig(BPM, 1000.0, 12000.0, 0.15, 10.0)
SIXTEENTH_MS = 60000.0 / BPM / 4.0


def sixteenth_function(name, offset_ms, duration_ms, slope=0.0):
    grid = SIXTEENTH_MS * np.arange(int(duration_ms / SIXTEENTH_MS))
    times = (grid + offset_ms) / (1.0 - slope)
    return OnsetFunction(name, times, np.ones(times.size))


def make_detection(function):
    heads = Heads(function.times_ms, function.times_ms, function.weights)
    return Detection(44100, {CONFIG.head_level: heads})


def make_sources(envelope, *functions):
    detection = make_detection(envelope)
    return method_sources(detection, CONFIG.head_level, functions)


@pytest.mark.parametrize("offset_ms", [5.0, 12.3, -12.3])
def test_estimate_recovers_the_offset_of_a_steady_pulse(offset_ms):
    function = sixteenth_function("method", offset_ms, 40000.0)

    estimate = estimate_method(function, CONFIG)

    assert estimate is not None
    assert estimate.name == "method"
    assert estimate.phase_ms == pytest.approx(offset_ms, abs=0.3)


def test_estimate_follows_a_linear_drift_of_its_own_events():
    function = sixteenth_function("method", 5.0, 60000.0, slope=2e-4)

    estimate = estimate_method(function, CONFIG)

    assert estimate is not None
    assert estimate.drift is not None
    assert estimate.drift.slope == pytest.approx(2e-4, abs=2e-5)
    assert estimate.phase_at(30000.0) == pytest.approx(11.0, abs=1.0)


def test_estimate_needs_the_minimum_number_of_events():
    function = OnsetFunction("method", np.arange(7) * SIXTEENTH_MS, np.ones(7))

    assert estimate_method(function, CONFIG) is None


def test_estimate_does_not_depend_on_event_order():
    ordered = sixteenth_function("method", 7.0, 20000.0)
    order = np.random.default_rng(0).permutation(ordered.times_ms.size)
    shuffled = OnsetFunction("method", ordered.times_ms[order], ordered.weights[order])

    assert estimate_method(shuffled, CONFIG) == estimate_method(ordered, CONFIG)


def test_estimate_attaches_a_beat_alignment_to_every_method():
    function = sixteenth_function("method", 7.0, 30000.0)
    kick_ms = 500.0 * np.arange(60) + 7.0

    estimate = estimate_method(function, CONFIG, kick_ms)

    assert estimate is not None
    assert estimate.alignment is not None
    assert estimate.alignment.phase_ms == pytest.approx(estimate.phase_ms)


def test_envelope_is_estimated_by_the_same_path_as_every_other_method():
    other = sixteenth_function("other", 7.0, 30000.0)
    envelope = envelope_function(make_detection(other), CONFIG.head_level)

    from_envelope = estimate_method(envelope, CONFIG)
    from_other = estimate_method(other, CONFIG)

    assert from_envelope is not None and from_other is not None
    assert from_envelope.name == ENVELOPE_NAME
    assert replace(from_envelope, name="same") == replace(from_other, name="same")


def test_envelope_function_clips_negative_strengths():
    times = np.array([10.0, 20.0])
    detection = Detection(
        44100, {CONFIG.head_level: Heads(times, times, np.array([-1.0, 2.0]))}
    )

    function = envelope_function(detection, CONFIG.head_level)

    assert function.name == ENVELOPE_NAME
    np.testing.assert_array_equal(function.weights, [0.0, 2.0])


def test_analyze_methods_lists_the_envelope_first_then_every_function():
    base = sixteenth_function("base", 5.0, 20000.0)
    sources = make_sources(
        base,
        sixteenth_function("A", 5.0, 20000.0),
        sixteenth_function("B", 5.0, 20000.0),
    )

    estimates = analyze_methods(sources, CONFIG)

    assert [estimate.name for estimate in estimates] == [ENVELOPE_NAME, "A", "B"]


def test_analyze_methods_skips_a_method_without_enough_events():
    base = sixteenth_function("base", 5.0, 20000.0)
    sources = make_sources(
        base,
        sixteenth_function("good", 5.0, 20000.0),
        OnsetFunction("sparse", np.arange(3) * SIXTEENTH_MS, np.ones(3)),
    )

    estimates = analyze_methods(sources, CONFIG)

    assert [estimate.name for estimate in estimates] == [ENVELOPE_NAME, "good"]


def test_analyze_methods_fails_when_no_method_has_events():
    empty = OnsetFunction("base", np.empty(0), np.empty(0))

    with pytest.raises(ValueError, match=NO_EVENTS_MESSAGE):
        analyze_methods(make_sources(empty), CONFIG)


def test_analyze_methods_accepts_a_constant_tempo_map():
    base = sixteenth_function("base", 5.0, 30000.0)
    sources = make_sources(base, sixteenth_function("A", 5.0, 30000.0))
    tempo = TempoMap((TempoPoint(0.0, 0.0, BPM),), None)

    estimates = analyze_methods(sources, CONFIG, tempo=tempo)

    assert [estimate.name for estimate in estimates] == [ENVELOPE_NAME, "A"]
    for estimate in estimates:
        assert estimate.phase_ms == pytest.approx(5.0, abs=0.3)


def test_analyze_methods_fails_without_events_in_the_dominant_tempo():
    sources = make_sources(sixteenth_function("base", 5.0, 30000.0))
    tempo = TempoMap((TempoPoint(0.0, 0.0, 90.0),), None)

    with pytest.raises(ValueError, match=ANCHOR_UNAVAILABLE_MESSAGE):
        analyze_methods(sources, CONFIG, tempo=tempo)
