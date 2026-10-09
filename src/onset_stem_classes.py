from __future__ import annotations

import numpy as np
from scipy import signal

CLASS_KICK = 0
CLASS_SNARE = 1
CLASS_HAT = 2
CLASS_LABELS = {
    CLASS_KICK: "キック系",
    CLASS_SNARE: "スネア・タム系",
    CLASS_HAT: "ハイハット・シンバル系",
}
CLASS_BANDS_HZ = (
    (30.0, 120.0),
    (120.0, 2500.0),
    (2500.0, 6000.0),
    (6000.0, 16000.0),
)
INCREMENT_WINDOW_MS = 12.0
BAND_NYQUIST_RATIO = 0.45
KICK_LOW_SHARE = 0.4
SNARE_MID_SHARE = 0.1
SHARE_FLOOR = 1e-30


def window_bounds(
    heads_ms: np.ndarray, sample_rate: int, total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def index(offset_ms: float) -> np.ndarray:
        samples = np.round((heads_ms + offset_ms) * sample_rate / 1000.0)
        return np.clip(samples.astype(int), 0, total)

    return index(-INCREMENT_WINDOW_MS), index(0.0), index(INCREMENT_WINDOW_MS)


def band_power_prefix(
    samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> np.ndarray:
    sos = signal.butter(
        4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    filtered = signal.sosfiltfilt(sos, samples, axis=0)
    power = np.square(filtered, dtype=np.float64).sum(axis=1)
    return np.concatenate(([0.0], np.cumsum(power)))


def band_increments(
    samples: np.ndarray, sample_rate: int, heads_ms: np.ndarray
) -> np.ndarray:
    before, at, after = window_bounds(heads_ms, sample_rate, samples.shape[0])
    increments = np.zeros((heads_ms.size, len(CLASS_BANDS_HZ)))
    ceiling = BAND_NYQUIST_RATIO * sample_rate
    for column, (low_hz, high_hz) in enumerate(CLASS_BANDS_HZ):
        high_hz = min(high_hz, ceiling)
        if low_hz >= high_hz:
            continue
        prefix = band_power_prefix(samples, sample_rate, low_hz, high_hz)
        gained = (prefix[after] - prefix[at]) - (prefix[at] - prefix[before])
        increments[:, column] = np.maximum(gained, 0.0)
    return increments


def classify_events(increments: np.ndarray) -> np.ndarray:
    shares = increments / (increments.sum(axis=1, keepdims=True) + SHARE_FLOOR)
    return np.where(
        shares[:, 0] >= KICK_LOW_SHARE,
        CLASS_KICK,
        np.where(shares[:, 1] >= SNARE_MID_SHARE, CLASS_SNARE, CLASS_HAT),
    )
