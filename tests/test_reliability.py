import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "batch"))

from batch_csv import reliability_cells
from onset_grid import GridLayout, phase_support, PhaseSupport
from onset_reliability import MethodReliability, reliability_of
from onset_window_phase import timeline_stability

LAYOUT = GridLayout(120.0)


def test_phase_support_counts_only_attacks_on_the_grid():
    on_grid = 3.0 + np.arange(40) * LAYOUT.beat_ms
    off_grid = 3.0 + 10.4 + np.arange(10) * LAYOUT.beat_ms

    support = phase_support(np.concatenate([on_grid, off_grid]), 3.0, LAYOUT)

    assert support.attack_count == 50
    assert support.support_count == 40
    assert support.support_ratio == pytest.approx(0.8)
    assert support.support_mad_ms == pytest.approx(0.0)


def test_phase_support_has_no_mad_without_supporting_attacks():
    support = phase_support(10.4 + np.arange(5) * LAYOUT.beat_ms, 0.0, LAYOUT)

    assert support.support_count == 0
    assert support.support_mad_ms is None


def test_phase_support_ratio_is_unavailable_without_attacks():
    assert PhaseSupport(0, 0, None).support_ratio is None


def test_timeline_stability_is_full_when_every_window_is_aligned():
    times_ms = np.arange(0.0, 60000.0, 100.0)

    stability = timeline_stability(
        times_ms, np.zeros(times_ms.size), LAYOUT.lattice_ms, 10000.0
    )

    assert stability == pytest.approx(1.0)


def test_timeline_stability_is_unavailable_for_short_material():
    times_ms = np.arange(0.0, 5000.0, 100.0)

    assert (
        timeline_stability(
            times_ms, np.zeros(times_ms.size), LAYOUT.lattice_ms, 10000.0
        )
        is None
    )


def make_analysis(checks, grid_support, stability=1.0):
    return SimpleNamespace(
        grid_support=grid_support,
        stability=stability,
        cross=SimpleNamespace(checks=checks) if checks else None,
        drift=None,
        uses_regression=False,
    )


def make_check(name, phase_ms):
    return SimpleNamespace(
        name=name,
        phase_ms=phase_ms,
        support=PhaseSupport(10, 5, 0.5),
        stability=0.5,
    )


def test_reliability_reports_how_far_each_method_was_moved_to_the_shared_branch():
    alignment = SimpleNamespace(phase_ms=0.0, lattice_ms=LAYOUT.comb_lattice_ms)
    analysis = make_analysis(
        (make_check("HFC", 0.2), make_check("複素領域", -20.0)),
        PhaseSupport(100, 90, 0.1),
    )

    reliability = reliability_of(analysis, alignment, LAYOUT)

    assert [method.name for method in reliability.methods] == [
        "fixed_grid",
        "HFC",
        "複素領域",
    ]
    assert reliability.method("fixed_grid").branch_shifted is False
    assert reliability.method("HFC").branch_shifted is False
    assert reliability.method("複素領域").branch_shift_ms == pytest.approx(
        -20.8333, abs=1e-3
    )


def test_reliability_omits_fixed_grid_without_an_alignment():
    analysis = make_analysis((make_check("HFC", 0.2),), PhaseSupport(100, 90, 0.1))

    reliability = reliability_of(analysis, None, LAYOUT)

    assert [method.name for method in reliability.methods] == ["HFC"]
    assert reliability.methods[0].branch_shift_ms == pytest.approx(0.0)


def test_csv_cells_join_methods_in_candidate_order():
    methods = {
        "fixed_grid": MethodReliability("fixed_grid", PhaseSupport(100, 80, 0.5), 1.0, 0.0),
        "複素領域": MethodReliability("複素領域", PhaseSupport(50, 20, None), None, -20.83),
    }

    cells = reliability_cells(methods)

    assert cells["support_mad_ms"] == "0.50/NA/NA/NA/NA/NA"
    assert cells["support_ratio"] == "0.80/NA/0.40/NA/NA/NA"
    assert cells["attack_counts"] == "100/NA/50/NA/NA/NA"
    assert cells["stabilities"] == "1.00/NA/NA/NA/NA/NA"
    assert cells["branch_note"] == "STFT_complex は -20.83 ms 補正"


def test_csv_branch_note_lists_every_shifted_method():
    methods = {
        "複素領域": MethodReliability("複素領域", PhaseSupport(1, 1, 0.0), None, 20.33),
        "HFC": MethodReliability("HFC", PhaseSupport(1, 1, 0.0), None, 20.33),
    }

    note = reliability_cells(methods)["branch_note"]

    assert note == "STFT_complex は +20.33 ms 補正, STFT_HFC は +20.33 ms 補正"
