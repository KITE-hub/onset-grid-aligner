from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import signal

from onset_common import ms_to_samples
from onset_detection import (
    bandpass_envelope,
    Detection,
    heads_ms_from_peaks,
    SLOPE_PERCENTILE,
    SLOPE_THRESHOLD_RATIO,
)
from onset_labels import LABEL_KICK, LABEL_OTHER

KICK_LOW_HZ = 30.0
KICK_HIGH_HZ = 120.0
KICK_SMOOTH_MS = 2.0
KICK_PEAK_RATIO = 0.5
KICK_PEAK_PERCENTILE = 99.5
KICK_MIN_GAP_MS = 120.0
KICK_GAP_PERIOD_RATIO = 0.6
KICK_TARGET_RATE_HZ = 2000
KICK_WINDOW_BEFORE_MS = 15.0
KICK_WINDOW_AFTER_MS = 35.0
KICK_SLOPE_WINDOW_MS = 4.0


@dataclass(frozen=True)
class KickEnvelope:
    envelope: np.ndarray
    sample_rate: float
    peak_height: float


def kick_envelope(samples: np.ndarray, sample_rate: int) -> KickEnvelope:
    factor = max(1, sample_rate // KICK_TARGET_RATE_HZ)
    reduced = signal.resample_poly(samples.mean(axis=1), 1, factor)
    reduced_rate = sample_rate / factor
    envelope = bandpass_envelope(
        reduced, reduced_rate, KICK_LOW_HZ, KICK_HIGH_HZ, KICK_SMOOTH_MS
    )
    height = KICK_PEAK_RATIO * float(np.percentile(envelope, KICK_PEAK_PERCENTILE))
    return KickEnvelope(envelope, reduced_rate, height)


def kick_peaks_ms(kick: KickEnvelope, period_ms: float) -> np.ndarray:
    gap_ms = min(KICK_MIN_GAP_MS, KICK_GAP_PERIOD_RATIO * period_ms)
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


def split_by_kick(
    heads_ms: np.ndarray, kick_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coincident = kick_coincident_mask(heads_ms, kick_ms)
    return heads_ms[~coincident], heads_ms[coincident]


def primary_keep_mask(
    times_ms: np.ndarray, kick_ms: np.ndarray, label: str
) -> np.ndarray:
    if label == LABEL_OTHER:
        return ~kick_coincident_mask(times_ms, kick_ms)
    if label == LABEL_KICK:
        return kick_coincident_mask(times_ms, kick_ms)
    return np.ones(times_ms.shape, dtype=bool)


def kick_band_heads(kick: KickEnvelope, period: float, head_level: float) -> np.ndarray:
    window = ms_to_samples(KICK_SLOPE_WINDOW_MS, kick.sample_rate) | 1
    slope = signal.savgol_filter(kick.envelope, window, 2, deriv=1)
    cap = float(np.percentile(slope, SLOPE_PERCENTILE))
    if cap <= 0.0:
        return np.empty(0)
    gap_ms = min(KICK_MIN_GAP_MS, KICK_GAP_PERIOD_RATIO * period)
    peaks, _ = signal.find_peaks(
        slope,
        height=SLOPE_THRESHOLD_RATIO * cap,
        distance=ms_to_samples(gap_ms, kick.sample_rate),
    )
    return heads_ms_from_peaks(kick.envelope, peaks, kick.sample_rate, head_level)[0]


def attach_kicks(
    detection: Detection, kick: KickEnvelope, period_ms: float
) -> Detection:
    return replace(detection, kick_ms=kick_peaks_ms(kick, period_ms))
