import numpy as np
import pytest

from onset_config import DEFAULT_HEAD_LEVEL
from onset_odf import (
    adaptive_threshold_events,
    chunk_functions,
    compute_onset_functions,
    frame_times_ms,
    head_shift_frames,
    OnsetKind,
    rise_position,
)

SAMPLE_RATE = 44100
BAND = (1000.0, 12000.0)


def make_bursts(antiphase=False):
    rng = np.random.default_rng(4)
    left = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)
    burst_length = int(0.008 * SAMPLE_RATE)
    interval = int(0.1 * SAMPLE_RATE)
    for start in range(int(0.1 * SAMPLE_RATE), left.size - burst_length, interval):
        left[start : start + burst_length] = rng.standard_normal(burst_length)
    return np.column_stack((left, -left if antiphase else left))


def test_complex_domain_prediction_is_quiet_for_steady_complex_tones():
    phase = np.arange(4, dtype=float)[:, None] * 0.37
    spectra = np.broadcast_to(np.exp(1j * phase), (4, 4)).copy()

    functions = chunk_functions(spectra, np.arange(4, dtype=np.float32))

    assert np.all(functions[1] < 1e-10)


def test_odf_frame_time_is_the_stft_window_center():
    times_ms = frame_times_ms(np.asarray([0]), 8, 1, 1000)

    assert times_ms[0] == pytest.approx(4.0)


def test_complex_function_selects_local_peaks_above_moving_median():
    frames = np.arange(4000, dtype=float)
    values = 0.1 + 0.01 * np.sin(2.0 * np.pi * frames / 20.0)
    values[1000] = 0.8

    events = adaptive_threshold_events("複素領域", values, 8, 1, 1000)

    assert events is not None
    assert events.times_ms.size == 1
    before_peak = frame_times_ms(np.asarray([999]), 8, 1, 1000)[0]
    peak = frame_times_ms(np.asarray([1000]), 8, 1, 1000)[0]
    assert before_peak < events.times_ms[0] <= peak
    assert events.weights[0] > 0.5


def test_complex_events_survive_antiphase_stereo_channels():
    functions = compute_onset_functions(make_bursts(antiphase=True), SAMPLE_RATE, *BAND)
    complex_function = next(
        (function for function in functions if function.name == "複素領域"), None
    )

    assert complex_function is not None
    assert complex_function.times_ms.size >= 5


def test_percussive_bursts_yield_events_for_every_onset_kind():
    functions = compute_onset_functions(make_bursts(), SAMPLE_RATE, *BAND)

    assert {function.name for function in functions} == {kind.value for kind in OnsetKind}
    assert all(function.times_ms.size >= 5 for function in functions)


def test_silence_produces_no_onset_functions():
    silence = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)

    assert compute_onset_functions(silence, SAMPLE_RATE, *BAND) == ()


def test_audio_shorter_than_one_frame_produces_no_onset_functions():
    short = np.zeros((100, 2), dtype=np.float32)

    assert compute_onset_functions(short, SAMPLE_RATE, *BAND) == ()


def ramp_strength():
    values = np.zeros(200)
    values[100:111] = np.linspace(0.0, 1.0, 11)
    return values


@pytest.mark.parametrize(("level", "position"), [(0.5, 105.0), (0.25, 102.5)])
def test_rise_position_interpolates_the_level_crossing(level, position):
    assert rise_position(ramp_strength(), 110, 30, level) == pytest.approx(position)


def test_rise_position_of_a_flat_curve_is_the_peak():
    assert rise_position(np.ones(50), 40, 30, 0.5) == 40.0


def test_head_shift_is_zero_at_the_default_level():
    shift = head_shift_frames(ramp_strength(), np.array([110]), 30, DEFAULT_HEAD_LEVEL)

    np.testing.assert_array_equal(shift, [0.0])


def test_head_shift_moves_later_for_a_higher_level():
    shift = head_shift_frames(ramp_strength(), np.array([110]), 30, 0.5)

    assert shift == pytest.approx([3.5])
