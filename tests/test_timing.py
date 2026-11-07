import numpy as np
import pytest

from onset_odf import OnsetFunction
from onset_timing import TempoMap, TempoPoint
from onset_warp import GridWarp, IDENTITY_WARP

TEMPO = TempoMap((TempoPoint(0.0, 0.0, 120.0), TempoPoint(8.0, 4000.0, 240.0)), 10000.0)


def test_beats_at_follows_each_tempo_segment():
    beats = TEMPO.beats_at(np.array([0.0, 2000.0, 4000.0, 5000.0]))

    np.testing.assert_allclose(beats, [0.0, 4.0, 8.0, 12.0])


def test_ms_at_follows_each_tempo_segment():
    times = TEMPO.ms_at(np.array([4.0, 8.0, 12.0]))

    np.testing.assert_allclose(times, [2000.0, 4000.0, 5000.0])


def test_beats_and_milliseconds_round_trip():
    times = np.linspace(0.0, 9000.0, 50)

    np.testing.assert_allclose(TEMPO.ms_at(TEMPO.beats_at(times)), times)


@pytest.mark.parametrize(("total_ms", "bpm"), [(10000.0, 240.0), (6000.0, 120.0)])
def test_dominant_bpm_is_the_tempo_with_the_longest_span(total_ms, bpm):
    assert TEMPO.dominant_bpm(total_ms) == bpm


def test_identity_warp_leaves_times_untouched():
    times = np.array([1.0, 2.5, 4000.0])

    np.testing.assert_array_equal(IDENTITY_WARP.times(times), times)


def test_warp_maps_a_tempo_change_onto_the_constant_grid():
    warp = GridWarp(TEMPO, 1, 500.0, 0.0)

    warped = warp.times(np.array([500.0, 4250.0, 4253.0]))

    np.testing.assert_allclose(warped, [500.0, 4500.0, 4503.0])


def test_warp_keeps_the_name_and_weights_of_a_function():
    warp = GridWarp(TEMPO, 1, 500.0, 0.0)
    function = OnsetFunction("method", np.array([500.0, 4250.0]), np.array([0.3, 0.9]))

    warped = warp.function(function)

    assert warped.name == "method"
    np.testing.assert_allclose(warped.times_ms, [500.0, 4500.0])
    np.testing.assert_array_equal(warped.weights, function.weights)
