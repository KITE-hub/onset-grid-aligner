from __future__ import annotations

from dataclasses import dataclass

SUBDIVISIONS = (2, 3, 4, 6, 8)
MIN_BPM = 30.0
MAX_BPM = 400.0

DEFAULT_LOW_HZ = 1500.0
DEFAULT_HIGH_HZ = 6000.0
DEFAULT_HEAD_LEVEL = 0.15
DEFAULT_DRIFT_WINDOW_SEC = 10.0
SENSITIVITY_LEVELS = (0.05, 0.15, 0.3)
FILTER_ZERO = "zero"
FILTER_CAUSAL = "causal"
FILTER_CHOICES = (FILTER_ZERO, FILTER_CAUSAL)
REFERENCE_ALL = "all"
REFERENCE_SLOW = "slow"
REFERENCE_FAST = "fast"
REFERENCE_LARGEST = "largest"
REFERENCE_MIXED = "mixed"
REFERENCE_CHOICES = (
    REFERENCE_ALL,
    REFERENCE_LARGEST,
    REFERENCE_FAST,
    REFERENCE_SLOW,
    REFERENCE_MIXED,
)
DEFAULT_REFERENCE_GROUP = REFERENCE_ALL


@dataclass(frozen=True)
class AnalysisConfig:
    bpm: float
    subdivision: int
    low_hz: float
    high_hz: float
    head_level: float
    drift_window_sec: float
    reference_group: str = DEFAULT_REFERENCE_GROUP

    @property
    def period_ms(self) -> float:
        return 60000.0 / self.bpm / self.subdivision
