from types import SimpleNamespace

import numpy as np
import pytest

import onset_band_check
from onset_config import AnalysisConfig
from onset_grid import GridLayout
from onset_labels import LABEL_ALL


def test_causal_filter_candidate_removes_estimated_filter_delay(monkeypatch):
    config = AnalysisConfig(160.0, 1500.0, 6000.0, 0.15, 10.0)
    period = GridLayout(config.bpm).lattice_ms
    base = SimpleNamespace(phase_ms=2.0)
    analysis = SimpleNamespace(groups={LABEL_ALL: base})

    def detected_heads(samples, sample_rate, low_hz, high_hz, level, causal):
        phase = 5.0 if causal else 2.0
        return np.arange(80) * period + phase

    monkeypatch.setattr(onset_band_check, "detected_heads", detected_heads)
    monkeypatch.setattr(onset_band_check, "causal_delay_ms", lambda *args: 3.0)

    result = onset_band_check.band_check(
        np.zeros((100, 2)), 48000, config, analysis, False
    )

    assert result is not None
    assert result.ringing is not None
    assert result.ringing.phase_ms == pytest.approx(result.bands[0].phase_ms)
