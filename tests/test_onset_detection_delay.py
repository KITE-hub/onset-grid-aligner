import numpy as np

from onset_detection import build_heads, locate_head


def test_build_heads_subtracts_the_delay_from_heads_only():
    sample_rate = 48000
    envelope = np.zeros(4800)
    envelope[1000:1100] = np.linspace(0.0, 1.0, 100)
    envelope[1100:] = 1.0
    peaks = np.array([1050])
    strengths = np.array([1.0])

    plain = build_heads(envelope, peaks, strengths, sample_rate, (0.15,))[0.15]
    delayed = build_heads(envelope, peaks, strengths, sample_rate, (0.15,), 3.0)[0.15]

    assert plain.heads_ms.size == 1
    np.testing.assert_allclose(delayed.heads_ms, plain.heads_ms - 3.0)
    np.testing.assert_allclose(delayed.peaks_ms, plain.peaks_ms)


def ramp_envelope():
    envelope = np.zeros(400)
    envelope[100:200] = np.linspace(0.0, 1.0, 100, endpoint=False)
    envelope[200:] = 1.0
    return envelope


def test_locate_head_returns_the_first_sample_above_the_head_level():
    head = locate_head(ramp_envelope(), 210, 120, 20, 0.255)

    assert head == 126


def test_higher_head_level_locates_a_later_head():
    envelope = ramp_envelope()

    early = locate_head(envelope, 210, 120, 20, 0.15)
    late = locate_head(envelope, 210, 120, 20, 0.6)

    assert early < late


def test_locate_head_rejects_a_flat_envelope():
    assert locate_head(np.ones(400), 200, 120, 20, 0.15) is None


def test_locate_head_rejects_a_peak_without_lookback():
    assert locate_head(ramp_envelope(), 0, 120, 20, 0.15) is None
