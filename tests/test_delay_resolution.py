from types import SimpleNamespace

import pytest

from onset_resolution import grid_branch_phase, resolve_delay


def make_analysis(
    conclusion_ms=0.0,
    uses_regression=False,
    slope=0.0,
    reference_ms=0.0,
):
    drift = SimpleNamespace(slope=slope, reference_ms=reference_ms)
    return SimpleNamespace(
        period_ms=10.0,
        conclusion_ms=conclusion_ms,
        uses_regression=uses_regression,
        drift=drift,
        cross=None,
    )


def test_delay_uses_phase_at_audio_midpoint():
    analysis = make_analysis(uses_regression=True, slope=0.001, reference_ms=0.0)
    resolution = resolve_delay(analysis, None, None, None, 10000.0, 100.0)
    assert resolution.delay_ms == pytest.approx(-5.0)
    assert resolution.status == "provisional"


def test_delay_requires_family_agreement_for_consensus():
    analysis = make_analysis(conclusion_ms=11.0)
    alignment = SimpleNamespace(phase_ms=10.0, phase_margin=0.1)
    cross = SimpleNamespace(
        checks=(SimpleNamespace(name="SuperFlux", converted_ms=12.0, uses_regression=False, phase_margin=0.1),)
    )
    bands = SimpleNamespace(
        bands=(SimpleNamespace(phase_ms=9.0),),
        ringing=None,
        kick=None,
    )
    analysis.cross = cross
    resolution = resolve_delay(analysis, alignment, bands, None, 10000.0, 100.0)
    assert resolution.status == "consensus"
    assert resolution.delay_ms == pytest.approx(-10.0)
    assert resolution.spread_ms == pytest.approx(3.0)


def test_grid_period_candidate_uses_nearest_equivalent_branch():
    alignment = SimpleNamespace(phase_ms=23.0, lattice_ms=15.625)

    phase = grid_branch_phase(-5.0, alignment)

    assert phase == pytest.approx(26.25)
    assert abs(phase - alignment.phase_ms) < alignment.lattice_ms / 2.0


def test_disagreeing_estimator_families_withhold_scalar_delay():
    analysis = make_analysis(conclusion_ms=20.0)
    alignment = SimpleNamespace(phase_ms=20.0, phase_margin=0.1)
    cross = SimpleNamespace(
        checks=(SimpleNamespace(name="SuperFlux", converted_ms=-20.0, uses_regression=False, phase_margin=0.1),)
    )
    analysis.cross = cross
    resolution = resolve_delay(analysis, alignment, None, None, 10000.0, 100.0)
    assert resolution.status == "ambiguous"
    assert resolution.delay_ms is None


def test_low_frequency_kick_diagnostic_does_not_vote_in_delay():
    analysis = make_analysis(conclusion_ms=2.0)
    bands = SimpleNamespace(
        bands=(),
        ringing=None,
        kick=SimpleNamespace(phase_ms=45.0),
    )
    resolution = resolve_delay(analysis, None, bands, None, 10000.0, 100.0)
    assert resolution.family_count == 1
    assert resolution.delay_ms == pytest.approx(-2.0)
    assert all("kick" not in name for name, _ in resolution.candidates)


def test_low_margin_spectral_candidate_remains_in_spectral_family():
    analysis = make_analysis()
    analysis.cross = SimpleNamespace(
        checks=(
            SimpleNamespace(
                name="複素領域",
                converted_ms=-71.31,
                uses_regression=False,
                phase_margin=0.0,
            ),
        )
    )
    resolution = resolve_delay(analysis, None, None, None, 10000.0, 375.0)

    assert resolution.status == "provisional"
    assert resolution.delay_ms == pytest.approx(0.0)
    assert resolution.family_count == 1
    candidate = dict(resolution.candidates)["複素領域"]
    assert candidate == pytest.approx(71.31)
    assert resolution.excluded_candidates == ("複素領域",)


def test_fixed_grid_candidate_remains_visible_when_phase_margin_is_low():
    analysis = make_analysis()
    alignment = SimpleNamespace(phase_ms=12.0, phase_margin=0.0)
    resolution = resolve_delay(analysis, alignment, None, None, 10000.0, 100.0)

    assert resolution.status == "provisional"
    assert resolution.delay_ms == pytest.approx(-12.0)
    assert resolution.family_count == 1
    candidate = dict(resolution.candidates)["fixed_grid"]
    assert candidate == pytest.approx(-12.0)


def test_ambiguous_complex_candidate_does_not_shift_consensus():
    analysis = make_analysis(conclusion_ms=4.9)
    analysis.cross = SimpleNamespace(
        checks=(
            SimpleNamespace(
                name="SuperFlux風",
                converted_ms=4.8,
                uses_regression=False,
                phase_margin=0.1,
            ),
            SimpleNamespace(
                name="複素領域",
                converted_ms=-40.0,
                uses_regression=False,
                phase_margin=0.0,
            ),
            SimpleNamespace(
                name="HFC",
                converted_ms=4.7,
                uses_regression=False,
                phase_margin=0.1,
            ),
        )
    )
    alignment = SimpleNamespace(phase_ms=4.8, phase_margin=0.1)
    bands = SimpleNamespace(
        bands=(SimpleNamespace(phase_ms=4.9),),
        ringing=None,
    )
    resolution = resolve_delay(analysis, alignment, bands, None, 10000.0, 100.0)

    assert resolution.status == "consensus"
    assert resolution.delay_ms == pytest.approx(-4.8)
    assert dict(resolution.candidates)["複素領域"] == pytest.approx(40.0)




def test_wavelet_candidates_form_their_own_family_on_the_grid_branch():
    analysis = make_analysis(conclusion_ms=4.8)
    analysis.cross = SimpleNamespace(
        checks=(
            SimpleNamespace(
                name="Wavelet多重スケール",
                converted_ms=4.6,
                uses_regression=False,
                phase_margin=0.0,
            ),
            SimpleNamespace(
                name="Wavelet粗スケール",
                converted_ms=-45.4,
                uses_regression=False,
                phase_margin=0.0,
            ),
        )
    )
    alignment = SimpleNamespace(phase_ms=4.8, phase_margin=0.1, lattice_ms=50.0)
    resolution = resolve_delay(analysis, alignment, None, None, 10000.0, 100.0)

    candidates = dict(resolution.candidates)
    assert candidates["Wavelet多重スケール"] == pytest.approx(-4.6)
    assert candidates["Wavelet粗スケール"] == pytest.approx(-4.6)
    assert resolution.excluded_candidates == ()
    assert resolution.family_count == 2
    assert resolution.status == "consensus"


def test_spectral_candidate_on_another_branch_follows_the_fixed_grid_branch():
    analysis = make_analysis(conclusion_ms=4.8)
    analysis.cross = SimpleNamespace(
        checks=(
            SimpleNamespace(
                name="HFC",
                converted_ms=-45.4,
                uses_regression=False,
                phase_margin=0.5,
            ),
        )
    )
    alignment = SimpleNamespace(phase_ms=4.8, phase_margin=0.1, lattice_ms=50.0)
    resolution = resolve_delay(analysis, alignment, None, None, 10000.0, 100.0)

    assert dict(resolution.candidates)["HFC"] == pytest.approx(-4.6)
    assert "HFC" not in resolution.excluded_candidates
