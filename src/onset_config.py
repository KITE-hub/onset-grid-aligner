from __future__ import annotations

from dataclasses import dataclass

from onset_grid import GridLayout

MIN_BPM = 30.0
MAX_BPM = 400.0

ANALYSIS_LOW_HZ = 1000.0
ANALYSIS_HIGH_HZ = 12000.0
DEFAULT_LOW_HZ = ANALYSIS_LOW_HZ
DEFAULT_HIGH_HZ = ANALYSIS_HIGH_HZ
MIN_ONSET_GAP_MS = 30.0
DEFAULT_HEAD_LEVEL = 0.15
DEFAULT_DRIFT_WINDOW_SEC = 10.0
SENSITIVITY_LEVELS = (0.05, 0.15, 0.3)
FILTER_ZERO = "zero"
FILTER_CAUSAL = "causal"
FILTER_CHOICES = (FILTER_ZERO, FILTER_CAUSAL)


@dataclass(frozen=True)
class AnalysisConfig:
    bpm: float
    low_hz: float
    high_hz: float
    head_level: float
    drift_window_sec: float

    @property
    def layout(self) -> GridLayout:
        return GridLayout(self.bpm)

    @property
    def period_ms(self) -> float:
        return self.layout.lattice_ms

    @property
    def grid_beat_ms(self) -> float:
        return self.layout.grid_beat_ms

    @property
    def beat_ms(self) -> float:
        return self.layout.beat_ms
