import numpy as np
import pytest

from onset_grid import estimate_comb_phase, GridLayout

DURATION_MS = 24000.0


def sixteenth_events(bpm, offset_ms):
    interval_ms = 60000.0 / bpm / 4.0
    times = offset_ms + interval_ms * np.arange(int(DURATION_MS / interval_ms))
    return times, np.ones(times.size)


@pytest.mark.parametrize(
    ("bpm", "offset_ms"),
    [
        (120.0, 12.3),
        (120.0, -12.3),
        (90.0, 23.7),
        (140.0, -27.5),
        (160.0, 29.0),
        (160.0, -29.0),
        (200.0, 29.0),
    ],
)
def test_symmetric_pattern_resolves_to_the_true_offset_within_thirty_ms(
    bpm, offset_ms
):
    times, weights = sixteenth_events(bpm, offset_ms)

    estimate = estimate_comb_phase(times, weights, GridLayout(bpm))

    assert estimate is not None
    assert estimate.phase_ms == pytest.approx(offset_ms, abs=0.3)
