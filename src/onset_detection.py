from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage, signal

from onset_common import causal_envelope, ms_to_samples
from onset_config import DEFAULT_HEAD_LEVEL, SENSITIVITY_LEVELS
from onset_odf import block_caps

ENVELOPE_SMOOTH_MS = 0.7
SLOPE_WINDOW_MS = 1.5
SLOPE_THRESHOLD_RATIO = 0.15
SLOPE_PERCENTILE = 99.5
CAP_BLOCK_SEC = 4.0
CAP_FLOOR_RATIO = 0.1
MIN_ONSET_GAP_MS = 30.0
LOOKBACK_MS = 20.0
PEAK_WINDOW_MS = 12.0
MIN_ATTACK_RATIO = 1.5


@dataclass(frozen=True)
class Heads:
    heads_ms: np.ndarray
    peaks_ms: np.ndarray


@dataclass(frozen=True)
class LocalCap:
    values: np.ndarray
    block: int

    @property
    def usable(self) -> bool:
        return bool(self.values.min() > 0.0)

    def at(self, indices: np.ndarray) -> np.ndarray:
        return self.values[np.minimum(indices // self.block, self.values.size - 1)]


@dataclass(frozen=True)
class Detection:
    envelope: np.ndarray
    slope: np.ndarray
    caps: LocalCap
    peaks: np.ndarray
    kick_ms: np.ndarray
    sample_rate: int
    heads: dict[float, Heads]


def bandpass_envelope(
    samples: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    smooth_ms: float,
    causal: bool = False,
) -> np.ndarray:
    if causal:
        return causal_envelope(samples, sample_rate, low_hz, high_hz, smooth_ms)
    sos = signal.butter(
        4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    filtered = signal.sosfiltfilt(sos, samples)
    analytic = signal.hilbert(filtered, N=fft.next_fast_len(filtered.size))
    envelope = np.abs(analytic[: filtered.size])
    width = ms_to_samples(smooth_ms, sample_rate)
    return ndimage.uniform_filter1d(envelope, width, mode="nearest")


def peak_envelope(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    smooth_ms: float,
    causal: bool = False,
) -> np.ndarray:
    merged = np.zeros(samples.shape[0])
    for channel in samples.T:
        current = bandpass_envelope(
            channel, sample_rate, low_hz, high_hz, smooth_ms, causal
        )
        np.maximum(merged, current, out=merged)
    return merged


def envelope_slope(envelope: np.ndarray, sample_rate: int) -> np.ndarray:
    window = ms_to_samples(SLOPE_WINDOW_MS, sample_rate) | 1
    return signal.savgol_filter(envelope, window, 2, deriv=1)


def locate_head(
    envelope: np.ndarray,
    peak: int,
    lookback: int,
    forward: int,
    head_level: float,
) -> int | None:
    start = max(0, peak - lookback)
    if peak - start < 1:
        return None
    base = envelope[start:peak].min()
    top = envelope[peak : peak + forward].max()
    if top <= base * MIN_ATTACK_RATIO:
        return None
    level = base + head_level * (top - base)
    below = np.flatnonzero(envelope[start : peak + 1] < level)
    return start + (int(below[-1]) + 1 if below.size else 0)


def attack_peaks(
    slope: np.ndarray, caps: LocalCap, sample_rate: int
) -> np.ndarray:
    if not caps.usable:
        return np.empty(0, dtype=np.intp)
    peaks, _ = signal.find_peaks(
        slope,
        height=SLOPE_THRESHOLD_RATIO * float(caps.values.min()),
        distance=ms_to_samples(MIN_ONSET_GAP_MS, sample_rate),
    )
    return peaks[slope[peaks] >= SLOPE_THRESHOLD_RATIO * caps.at(peaks)]


def heads_ms_from_peaks(
    envelope: np.ndarray, peaks: np.ndarray, sample_rate: int, head_level: float
) -> tuple[np.ndarray, np.ndarray]:
    lookback = ms_to_samples(LOOKBACK_MS, sample_rate)
    forward = ms_to_samples(PEAK_WINDOW_MS, sample_rate)
    heads = [
        locate_head(envelope, int(peak), lookback, forward, head_level)
        for peak in peaks
    ]
    valid = np.fromiter((head is not None for head in heads), dtype=bool, count=len(heads))
    found = np.fromiter((head for head in heads if head is not None), dtype=float)
    scale = 1000.0 / sample_rate
    return found * scale, peaks[valid] * scale


def head_levels(head_level: float) -> tuple[float, ...]:
    return tuple(sorted({*SENSITIVITY_LEVELS, DEFAULT_HEAD_LEVEL, head_level}))


def build_heads(
    envelope: np.ndarray,
    peaks: np.ndarray,
    sample_rate: int,
    levels: tuple[float, ...],
) -> dict[float, Heads]:
    return {
        level: Heads(*heads_ms_from_peaks(envelope, peaks, sample_rate, level))
        for level in levels
    }


def detected_heads(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    level: float,
    causal: bool,
) -> np.ndarray:
    detection = detect_attacks(samples, sample_rate, low_hz, high_hz, (level,), causal)
    return detection.heads[level].heads_ms


def detect_attacks(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    levels: tuple[float, ...],
    causal: bool = False,
) -> Detection:
    envelope = peak_envelope(
        samples, sample_rate, low_hz, high_hz, ENVELOPE_SMOOTH_MS, causal
    )
    slope = envelope_slope(envelope, sample_rate)
    block = ms_to_samples(CAP_BLOCK_SEC * 1000.0, sample_rate)
    caps = LocalCap(
        block_caps(slope, block, SLOPE_PERCENTILE, CAP_FLOOR_RATIO), block
    )
    peaks = attack_peaks(slope, caps, sample_rate)
    return Detection(
        envelope,
        slope,
        caps,
        peaks,
        np.empty(0),
        sample_rate,
        build_heads(envelope, peaks, sample_rate, levels),
    )
