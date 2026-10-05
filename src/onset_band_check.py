from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_analysis import Analysis
from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_detection import detected_heads
from onset_groups import analyze_group
from onset_kick import kick_band_heads, KICK_HIGH_HZ, KICK_LOW_HZ, KickEnvelope
from onset_labels import LABEL_ALL
from onset_warp import GridWarp, IDENTITY_WARP

ALT_LOW_BAND = (500.0, 1500.0)
ALT_HIGH_START_HZ = 6000.0
ALT_HIGH_END_HZ = 12000.0
ALT_NYQUIST_RATIO = 0.45
ALT_MIN_BAND_RATIO = 1.5


@dataclass(frozen=True)
class BandPhase:
    label: str
    phase_ms: float
    difference_ms: float


@dataclass(frozen=True)
class BandCheck:
    reference_ms: float
    bands: tuple[BandPhase, ...]
    ringing: BandPhase | None
    kick: BandPhase | None

    @property
    def span_ms(self) -> float:
        values = [0.0, *(band.difference_ms for band in self.bands)]
        return max(values) - min(values)


def alternate_bands(sample_rate: int) -> tuple[tuple[float, float], ...]:
    high = (ALT_HIGH_START_HZ, min(ALT_HIGH_END_HZ, ALT_NYQUIST_RATIO * sample_rate))
    return tuple(
        (low, upper)
        for low, upper in (ALT_LOW_BAND, high)
        if upper >= ALT_MIN_BAND_RATIO * low
    )


def phase_entry(
    label: str, heads_ms: np.ndarray, base_phase: float, period: float
) -> BandPhase | None:
    group = analyze_group(LABEL_ALL, heads_ms, period)
    if group is None:
        return None
    difference = float(wrap_symmetric(group.phase_ms - base_phase, period))
    return BandPhase(label, group.phase_ms, difference)


def band_check(
    samples: np.ndarray,
    sample_rate: int,
    config: AnalysisConfig,
    analysis: Analysis,
    kick: KickEnvelope,
    causal: bool,
    warp: GridWarp = IDENTITY_WARP,
) -> BandCheck | None:
    base = analysis.groups.get(LABEL_ALL)
    if base is None:
        return None
    period = config.period_ms
    level = config.head_level
    entries = (
        phase_entry(
            f"{low:g}~{high:g} Hz",
            warp.times(detected_heads(samples, sample_rate, low, high, level, causal)),
            base.phase_ms,
            period,
        )
        for low, high in alternate_bands(sample_rate)
    )
    bands = tuple(entry for entry in entries if entry is not None)
    variant = "零位相" if causal else "因果"
    ringing = phase_entry(
        f"{config.low_hz:g}~{config.high_hz:g} Hz {variant}フィルタ",
        warp.times(
            detected_heads(
                samples, sample_rate, config.low_hz, config.high_hz, level, not causal
            )
        ),
        base.phase_ms,
        period,
    )
    kick_entry = phase_entry(
        f"{KICK_LOW_HZ:g}~{KICK_HIGH_HZ:g} Hz キック帯域(独立検出)",
        warp.times(kick_band_heads(kick, period, level)),
        base.phase_ms,
        period,
    )
    if not bands and ringing is None and kick_entry is None:
        return None
    return BandCheck(base.phase_ms, bands, ringing, kick_entry)
