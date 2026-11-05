from pathlib import Path
import sys

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))

from verify_synthetic import cli_error, tempo_error


@pytest.mark.parametrize(
    ("pattern", "bpm", "offset_ms"),
    [
        ("rock8", 120.0, 23.71),
        ("four_floor", 90.0, -23.71),
        ("funk16", 174.0, 12.0),
        ("shuffle", 150.0, -10.0),
        ("triplet_mix", 120.0, 40.0),
        ("sparse_kick", 150.0, 23.71),
        ("dnb", 174.0, -45.0),
    ],
)
def test_synthetic_drums_recover_known_offset(pattern, bpm, offset_ms):
    error = cli_error(pattern, bpm, offset_ms, 24.0, 0.0)
    assert abs(error) <= 3.0


def test_synthetic_tempo_change_recovers_known_offset():
    assert abs(tempo_error("funk16", -23.71)) <= 3.0
