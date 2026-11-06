import numpy as np
import pytest

from onset_render_summary import render_delay_candidates
from onset_wavelet import (
    compute_wavelet_functions,
    COARSE_SCALE_NAME,
    MULTI_SCALE_NAME,
    usable_centers,
    WAVELET_CENTERS_HZ,
    WAVELET_NAMES,
)

SAMPLE_RATE = 44100
BURST_MS = 8.0
DURATION_SEC = 3.0
STARTS_MS = np.asarray([100.0 + 150.0 * index for index in range(18)])
TOLERANCE_MS = 0.5


def make_bursts(gains=None, antiphase=False):
    rng = np.random.default_rng(4)
    left = np.zeros(int(DURATION_SEC * SAMPLE_RATE), dtype=np.float32)
    length = int(BURST_MS * SAMPLE_RATE / 1000.0)
    scale = np.ones(STARTS_MS.size) if gains is None else gains
    for start_ms, gain in zip(STARTS_MS, scale):
        start = int(round(start_ms * SAMPLE_RATE / 1000.0))
        left[start : start + length] = gain * rng.standard_normal(length)
    return np.column_stack((left, -left if antiphase else left))


def attack_errors(function):
    distances = np.abs(function.times_ms[None, :] - STARTS_MS[:, None])
    return function.times_ms[distances.argmin(axis=1)] - STARTS_MS


def by_name(functions):
    return {function.name: function for function in functions}


def test_usable_centers_follow_the_nyquist_ratio():
    assert usable_centers(44100) == WAVELET_CENTERS_HZ
    assert usable_centers(8000) == (2000.0,)
    assert usable_centers(4000) == ()


@pytest.mark.parametrize("name", [MULTI_SCALE_NAME, COARSE_SCALE_NAME])
@pytest.mark.parametrize(
    "samples",
    [
        make_bursts(),
        make_bursts(antiphase=True),
        make_bursts(gains=np.linspace(0.05, 1.0, STARTS_MS.size)),
    ],
    ids=["same-phase", "antiphase", "velocity-range"],
)
def test_events_land_on_attack_starts_without_early_events(samples, name):
    function = by_name(compute_wavelet_functions(samples, SAMPLE_RATE))[name]

    assert function.times_ms.size == STARTS_MS.size
    assert np.abs(attack_errors(function)).max() < TOLERANCE_MS


def test_mono_input_is_accepted():
    functions = compute_wavelet_functions(make_bursts()[:, 0], SAMPLE_RATE)

    assert tuple(function.name for function in functions) == WAVELET_NAMES


def test_silence_and_empty_audio_produce_no_functions():
    assert compute_wavelet_functions(np.zeros((SAMPLE_RATE, 2)), SAMPLE_RATE) == ()
    assert compute_wavelet_functions(np.zeros((0, 2)), SAMPLE_RATE) == ()


def test_wavelet_candidates_are_listed_in_the_report():
    candidates = tuple((name, 1.0) for name in WAVELET_NAMES)

    report = "\n".join(render_delay_candidates(candidates))

    for name in WAVELET_NAMES:
        assert f"  {name}: +1.00 ms" in report
