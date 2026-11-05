import numpy as np
import pytest

from onset_config import DEFAULT_HIGH_HZ, DEFAULT_LOW_HZ
from onset_odf import (
    active_events,
    chunk_functions,
    compute_onset_functions,
    frame_times_ms,
    ODF_HIGH_HZ,
    ODF_LOW_HZ,
)


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

    events = active_events("複素領域", values, 8, 1, 1000)

    assert events is not None
    assert events.times_ms.size == 1
    expected = frame_times_ms(np.asarray([1000]), 8, 1, 1000)[0]
    assert events.times_ms[0] == pytest.approx(expected)
    assert events.weights[0] > 0.5


def test_complex_events_survive_antiphase_stereo_channels():
    sample_rate = 44100
    rng = np.random.default_rng(4)
    left = np.zeros(2 * sample_rate, dtype=np.float32)
    burst_length = int(0.008 * sample_rate)
    interval = int(0.1 * sample_rate)
    for start in range(int(0.1 * sample_rate), left.size - burst_length, interval):
        left[start : start + burst_length] = rng.standard_normal(burst_length)
    samples = np.column_stack((left, -left))

    functions = compute_onset_functions(samples, sample_rate)
    complex_function = next(
        (function for function in functions if function.name == "複素領域"), None
    )

    assert complex_function is not None
    assert complex_function.times_ms.size >= 5


def test_envelope_and_spectral_detectors_share_the_analysis_band():
    assert (DEFAULT_LOW_HZ, DEFAULT_HIGH_HZ) == (ODF_LOW_HZ, ODF_HIGH_HZ)
