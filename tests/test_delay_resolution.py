from types import SimpleNamespace

import pytest

from onset_grid import GridLayout
from onset_resolution import delay_candidates, grid_branch_phase

DURATION_MS = 10000.0


def make_layout(beat_ms):
    return GridLayout(60000.0 / beat_ms)


def make_analysis(checks=(), uses_regression=False, slope=0.0, reference_ms=0.0):
    return SimpleNamespace(
        uses_regression=uses_regression,
        drift=SimpleNamespace(slope=slope, reference_ms=reference_ms),
        cross=SimpleNamespace(checks=checks) if checks else None,
    )


def make_check(name, phase_ms, uses_regression=False):
    return SimpleNamespace(name=name, phase_ms=phase_ms, uses_regression=uses_regression)


def make_alignment(phase_ms, lattice_ms=1000.0):
    return SimpleNamespace(phase_ms=phase_ms, lattice_ms=lattice_ms)


def candidates_of(analysis, alignment, beat_ms):
    return dict(delay_candidates(analysis, alignment, make_layout(beat_ms), DURATION_MS))


def test_delay_uses_phase_at_audio_midpoint():
    analysis = make_analysis(uses_regression=True, slope=0.001, reference_ms=0.0)

    candidates = candidates_of(analysis, make_alignment(0.0), 100.0)

    assert candidates["fixed_grid"] == pytest.approx(-5.0)


def test_grid_period_candidate_uses_nearest_equivalent_branch():
    alignment = SimpleNamespace(phase_ms=23.0, lattice_ms=15.625)

    phase = grid_branch_phase(-5.0, alignment)

    assert phase == pytest.approx(26.25)
    assert abs(phase - alignment.phase_ms) < alignment.lattice_ms / 2.0


def test_no_alignment_and_no_cross_check_yield_no_candidates():
    assert candidates_of(make_analysis(), None, 100.0) == {}


def test_spectral_candidate_without_alignment_keeps_its_own_phase():
    analysis = make_analysis(checks=(make_check("複素領域", -71.31),))

    candidates = candidates_of(analysis, None, 375.0)

    assert candidates == {"複素領域": pytest.approx(71.31)}


def test_fixed_grid_candidate_is_the_negated_phase():
    candidates = candidates_of(make_analysis(), make_alignment(12.0), 100.0)

    assert candidates["fixed_grid"] == pytest.approx(-12.0)


def test_spectral_candidate_follows_grid_branch():
    analysis = make_analysis(
        checks=(
            make_check("SuperFlux風", 4.8),
            make_check("複素領域", -40.0),
            make_check("HFC", 4.7),
        )
    )

    candidates = candidates_of(analysis, make_alignment(4.8), 100.0)

    assert candidates["複素領域"] == pytest.approx(-5.8333, abs=1e-3)


def test_wavelet_candidates_follow_grid_branch():
    analysis = make_analysis(
        checks=(
            make_check("Wavelet多重スケール", 4.6),
            make_check("Wavelet粗スケール", -45.4),
        )
    )

    candidates = candidates_of(analysis, make_alignment(4.8, 50.0), 100.0)

    assert candidates["Wavelet多重スケール"] == pytest.approx(-4.6)
    assert candidates["Wavelet粗スケール"] == pytest.approx(-4.6)


def test_spectral_candidate_on_another_branch_follows_the_fixed_grid_branch():
    analysis = make_analysis(checks=(make_check("HFC", -45.4),))

    candidates = candidates_of(analysis, make_alignment(4.8, 50.0), 100.0)

    assert candidates["HFC"] == pytest.approx(-4.6)


def test_regression_is_applied_only_to_checks_that_use_it():
    analysis = make_analysis(
        checks=(make_check("HFC", 0.0, True), make_check("複素領域", 0.0, False)),
        uses_regression=True,
        slope=0.001,
    )

    candidates = candidates_of(analysis, None, 100.0)

    assert candidates["HFC"] == pytest.approx(-5.0)
    assert candidates["複素領域"] == pytest.approx(0.0)
