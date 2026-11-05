import pytest

from benchmark.real_drum_validation import candidate_spread_ms, circular_error_ms
from onset_common import phase_to_delay_ms


def test_positive_phase_requires_advancing_audio():
    assert phase_to_delay_ms(12.0) == -12.0


def test_negative_phase_requires_delaying_audio():
    assert phase_to_delay_ms(-12.0) == 12.0


def test_circular_error_uses_shortest_phase_distance():
    assert circular_error_ms(198.0, -199.0, 400.0) == pytest.approx(-3.0)


def test_candidate_spread_uses_shortest_circular_distance():
    candidates = {"a": 198.0, "b": -199.0}
    assert candidate_spread_ms(candidates, 400.0) == pytest.approx(3.0)
