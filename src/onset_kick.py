from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from onset_common import ms_to_samples
from onset_detection import bandpass_envelope

KICK_LOW_HZ = 30.0
KICK_HIGH_HZ = 120.0
KICK_SMOOTH_MS = 2.0
KICK_PEAK_RATIO = 0.5
KICK_PEAK_PERCENTILE = 99.5
KICK_GAP_BEAT_RATIO = 0.15
KICK_TARGET_RATE_HZ = 2000
KICK_WINDOW_BEFORE_MS = 15.0
KICK_WINDOW_AFTER_MS = 35.0


@dataclass(frozen=True)
class KickEnvelope:
    envelope: np.ndarray
    sample_rate: float
    peak_height: float


def kick_envelope(samples: np.ndarray, sample_rate: int) -> KickEnvelope:
    factor = max(1, sample_rate // KICK_TARGET_RATE_HZ)
    reduced = signal.resample_poly(samples, 1, factor, axis=0)
    reduced_rate = sample_rate / factor
    envelope = np.max(
        [
            bandpass_envelope(
                channel, reduced_rate, KICK_LOW_HZ, KICK_HIGH_HZ, KICK_SMOOTH_MS
            )
            for channel in reduced.T
        ],
        axis=0,
    )
    height = KICK_PEAK_RATIO * float(np.percentile(envelope, KICK_PEAK_PERCENTILE))
    return KickEnvelope(envelope, reduced_rate, height)


def kick_gap_ms(beat_ms: float) -> float:
    return KICK_GAP_BEAT_RATIO * beat_ms


def kick_peaks_ms(kick: KickEnvelope, beat_ms: float) -> np.ndarray:
    gap_ms = kick_gap_ms(beat_ms)
    peaks, _ = signal.find_peaks(
        kick.envelope,
        height=kick.peak_height,
        distance=ms_to_samples(gap_ms, kick.sample_rate),
    )
    return peaks * 1000.0 / kick.sample_rate


def kick_coincident_mask(times_ms: np.ndarray, kick_ms: np.ndarray) -> np.ndarray:
    if kick_ms.size == 0:
        return np.zeros(times_ms.shape, dtype=bool)
    left = np.searchsorted(kick_ms, times_ms - KICK_WINDOW_BEFORE_MS, side="right")
    right = np.searchsorted(kick_ms, times_ms + KICK_WINDOW_AFTER_MS, side="left")
    return right > left
