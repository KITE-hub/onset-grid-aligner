import pytest

from verify_synthetic import cli_errors, drift_errors, tempo_errors

TOLERANCE_MS = 3.0


def exceeding_tolerance(errors):
    return {
        name: round(error, 2)
        for name, error in errors.items()
        if abs(error) > TOLERANCE_MS
    }


@pytest.mark.parametrize(
    ("pattern", "bpm", "offset_ms"),
    [
        ("rock8", 120.0, 23.71),
        ("rock8", 120.0, 0.0),
        ("four_floor", 90.0, -23.71),
        ("funk16", 174.0, 12.0),
        ("shuffle", 150.0, -10.0),
        ("triplet_mix", 120.0, 40.0),
        ("sparse_kick", 150.0, 23.71),
        ("dnb", 174.0, -45.0),
    ],
)
def test_every_method_recovers_known_offset(pattern, bpm, offset_ms):
    errors = cli_errors(pattern, bpm, offset_ms, 24.0, 0.0)

    assert exceeding_tolerance(errors) == {}


def test_every_method_recovers_known_offset_under_tempo_change():
    errors = tempo_errors("funk16", -23.71)

    assert exceeding_tolerance(errors) == {}


@pytest.mark.parametrize(("pattern", "offset_ms"), [("rock8", 12.0), ("funk16", -12.0)])
def test_every_method_recovers_the_midpoint_offset_under_constant_drift(pattern, offset_ms):
    errors = drift_errors(pattern, 120.05, 120.0, offset_ms, 24.0)

    assert exceeding_tolerance(errors) == {}
