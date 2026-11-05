from types import SimpleNamespace

import numpy as np
import pytest

import onset_cross_check
from onset_config import AnalysisConfig
from onset_cross_check import CheckBasis, detector_check
from onset_odf import OnsetFunction


def test_detector_phase_is_not_shifted_by_primary_envelope_rise_gap(monkeypatch):
    config = AnalysisConfig(120.0, 1500.0, 6000.0, 0.15, 10.0)
    basis = CheckBasis(None, config, 0.0, False)
    function = OnsetFunction("複素領域", np.arange(30.0), np.ones(30))
    estimate = SimpleNamespace(phase_ms=4.0, phase_margin=0.1)
    monkeypatch.setattr(
        onset_cross_check, "estimate_comb_phase", lambda *args: estimate
    )

    result = detector_check(function, basis)

    assert result is not None
    assert result.converted_ms == pytest.approx(4.0)
