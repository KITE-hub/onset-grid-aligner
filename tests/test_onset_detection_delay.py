import numpy as np

from onset_detection import build_heads


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
