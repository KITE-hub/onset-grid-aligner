from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft, ndimage, signal

from onset_config import DEFAULT_HEAD_LEVEL, MIN_ONSET_GAP_MS, USABLE_NYQUIST_RATIO

ODF_WINDOW_MS = 6.0
ODF_HOP_MS = 1.0
ODF_LAG_FRAMES = 2
ODF_HISTORY_FRAMES = max(ODF_LAG_FRAMES, 2)
ODF_CHUNK_FRAMES = 4096
ODF_COMPRESSION = 10.0
ODF_FREQUENCY_FILTER_BINS = 3
ODF_MIN_BINS = 4
ODF_BLOCK_SEC = 4.0
ODF_PERCENTILE = 99.5
ODF_FLOOR_RATIO = 0.1
ODF_ACTIVE_RATIO = 0.15
ODF_COMPLEX_THRESHOLD_WINDOW_MS = 100.0
ODF_COMPLEX_THRESHOLD_RATIO = 1.5
ODF_HEAD_LOOKBACK_MS = MIN_ONSET_GAP_MS

FOLD_BIN_MS = 0.1
FOLD_SIGMA_MS = 1.0
FOLD_MIN_BINS = 8


class OnsetKind(Enum):
    SUPERFLUX = "SuperFlux風"
    COMPLEX = "複素領域"
    HFC = "HFC"


@dataclass(frozen=True)
class OnsetFunction:
    name: str
    times_ms: np.ndarray
    weights: np.ndarray


def block_caps(
    values: np.ndarray, block: int, percentile: float, floor_ratio: float
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(1)
    count = -(-values.size // block)
    floor = floor_ratio * float(np.percentile(values, percentile))
    caps = np.fromiter(
        (
            np.percentile(values[index * block : (index + 1) * block], percentile)
            for index in range(count)
        ),
        dtype=float,
        count=count,
    )
    return np.maximum(caps, floor)


def fold_profiles(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    bin_ms: float = FOLD_BIN_MS,
) -> tuple[np.ndarray, float]:
    rows = times_ms.shape[0]
    bins = max(FOLD_MIN_BINS, int(round(period_ms / bin_ms)))
    scaled = np.mod(times_ms, period_ms) * (bins / period_ms)
    lower = np.floor(scaled)
    fraction = scaled - lower
    first = lower.astype(np.int64) % bins
    second = (first + 1) % bins
    offsets = (np.arange(rows, dtype=np.int64) * bins)[:, None]
    row_weights = np.broadcast_to(weights, times_ms.shape)
    profiles = np.bincount(
        (first + offsets).ravel(),
        (row_weights * (1.0 - fraction)).ravel(),
        minlength=rows * bins,
    )
    profiles += np.bincount(
        (second + offsets).ravel(),
        (row_weights * fraction).ravel(),
        minlength=rows * bins,
    )
    return profiles.reshape(rows, bins), period_ms / bins


def fold_profile(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    bin_ms: float = FOLD_BIN_MS,
) -> tuple[np.ndarray, float]:
    profiles, step = fold_profiles(times_ms[None, :], weights, period_ms, bin_ms)
    return profiles[0], step


def fold_phase(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    sigma_ms: float = FOLD_SIGMA_MS,
    bin_ms: float = FOLD_BIN_MS,
) -> tuple[float, float]:
    profile, step = fold_profile(times_ms, weights, period_ms, bin_ms)
    smoothed = ndimage.gaussian_filter1d(profile, sigma_ms / step, mode="wrap")
    best = int(np.argmax(smoothed))
    left = smoothed[best - 1]
    right = smoothed[(best + 1) % smoothed.size]
    curvature = left - 2.0 * smoothed[best] + right
    shift = 0.0 if curvature >= 0.0 else float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    phase = (((best + shift) * step + period_ms / 2.0) % period_ms) - period_ms / 2.0
    mean = float(smoothed.mean())
    contrast = float(smoothed[best]) / mean if mean > 0.0 else 0.0
    return float(phase), contrast


def frame_layout(sample_rate: int) -> tuple[int, int]:
    size = fft.next_fast_len(
        max(8, int(round(ODF_WINDOW_MS * sample_rate / 1000.0))), real=True
    )
    hop = max(1, int(round(ODF_HOP_MS * sample_rate / 1000.0)))
    return size, hop


def frame_times_ms(
    indices: np.ndarray, size: int, hop: int, sample_rate: int
) -> np.ndarray:
    position = indices * hop + size / 2.0
    return position * 1000.0 / sample_rate


def lookback_frames(sample_rate: int, hop: int) -> int:
    return max(1, int(round(ODF_HEAD_LOOKBACK_MS * sample_rate / (1000.0 * hop))))


def rise_position(
    values: np.ndarray, peak: int, lookback: int, level: float
) -> float:
    start = max(0, peak - lookback)
    window = values[start : peak + 1]
    base = float(window.min())
    top = float(values[peak])
    if top <= base:
        return float(peak)
    target = base + level * (top - base)
    below = np.flatnonzero(window < target)
    if below.size == 0:
        return float(start)
    lower = start + int(below[-1])
    first = float(values[lower])
    second = float(values[lower + 1])
    return lower + (target - first) / (second - first)


def head_shift_frames(
    values: np.ndarray, peaks: np.ndarray, lookback: int, level: float
) -> np.ndarray:
    if level == DEFAULT_HEAD_LEVEL:
        return np.zeros(peaks.size)
    return np.fromiter(
        (
            rise_position(values, int(peak), lookback, level)
            - rise_position(values, int(peak), lookback, DEFAULT_HEAD_LEVEL)
            for peak in peaks
        ),
        dtype=float,
        count=peaks.size,
    )


def threshold_crossing_frames(
    peaks: np.ndarray,
    values: np.ndarray,
    threshold: np.ndarray,
    lookback: int,
) -> np.ndarray:
    crossings = peaks.astype(float)
    for index, peak in enumerate(peaks):
        start = max(0, int(peak) - lookback)
        relative = values[start : peak + 1] - threshold[start : peak + 1]
        below = np.flatnonzero(relative < 0.0)
        if below.size == 0:
            continue
        lower = start + int(below[-1])
        if lower >= peak:
            continue
        first = values[lower] - threshold[lower]
        second = values[lower + 1] - threshold[lower + 1]
        if second <= first:
            continue
        crossings[index] = lower - first / (second - first)
    return crossings


def chunk_functions(spectra: np.ndarray, band_weights: np.ndarray) -> np.ndarray:
    history = ODF_HISTORY_FRAMES
    lag = ODF_LAG_FRAMES
    total = spectra.shape[0]
    magnitude = np.abs(spectra)
    compressed = np.log1p(ODF_COMPRESSION * magnitude)
    trailing = ndimage.maximum_filter1d(compressed, ODF_FREQUENCY_FILTER_BINS, axis=1)
    flux = np.maximum(
        compressed[history:] - trailing[history - lag : total - lag], 0.0
    ).sum(axis=1)
    contour = (magnitude * band_weights).sum(axis=1)
    hfc = np.maximum(contour[history:] - contour[history - lag : total - lag], 0.0)
    phase = np.angle(spectra)
    predicted = magnitude[history - 1 : total - 1] * np.exp(
        1j * (2.0 * phase[history - 1 : total - 1] - phase[history - 2 : total - 2])
    )
    deviation = np.abs(spectra[history:] - predicted)
    complex_domain = deviation.sum(axis=1)
    return np.stack((flux, complex_domain, hfc))


def spectral_series(
    samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> tuple[np.ndarray, int, int] | None:
    size, hop = frame_layout(sample_rate)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[1] == 0:
        return None
    frame_count = (audio.shape[0] - size) // hop + 1
    frequencies = fft.rfftfreq(size, 1.0 / sample_rate)
    band = np.flatnonzero((frequencies >= low_hz) & (frequencies <= high_hz))
    if frame_count <= ODF_HISTORY_FRAMES or band.size < ODF_MIN_BINS:
        return None
    window = signal.get_window("hann", size).astype(np.float32)
    frames = sliding_window_view(audio, size, axis=0)[::hop]
    band_weights = band.astype(np.float32)
    series = np.empty((len(OnsetKind), frame_count), dtype=np.float32)
    for start in range(0, frame_count, ODF_CHUNK_FRAMES):
        stop = min(frame_count, start + ODF_CHUNK_FRAMES)
        first = max(0, start - ODF_HISTORY_FRAMES)
        spectra = fft.rfft(
            frames[first:stop] * window[None, None, :], axis=-1
        )[..., band]
        padding = ODF_HISTORY_FRAMES - (start - first)
        if padding > 0:
            spectra = np.concatenate((np.repeat(spectra[:1], padding, axis=0), spectra))
        channel_functions = np.stack(
            [
                chunk_functions(spectra[:, channel, :], band_weights)
                for channel in range(audio.shape[1])
            ]
        )
        series[:, start:stop] = np.mean(channel_functions, axis=0)
    return series, size, hop


def normalized_strength(values: np.ndarray, sample_rate: int, hop: int) -> np.ndarray:
    block = max(1, int(round(ODF_BLOCK_SEC * sample_rate / hop)))
    caps = np.repeat(
        block_caps(values, block, ODF_PERCENTILE, ODF_FLOOR_RATIO), block
    )[: values.size]
    return np.divide(
        values, caps, out=np.zeros(values.shape, dtype=np.float64), where=caps > 0.0
    )


def peak_frames(
    strength: np.ndarray, threshold: float | np.ndarray, sample_rate: int, hop: int
) -> np.ndarray:
    distance = max(1, int(round(MIN_ONSET_GAP_MS * sample_rate / (1000.0 * hop))))
    height = (
        np.pad(threshold, (1, 1), constant_values=ODF_ACTIVE_RATIO)
        if isinstance(threshold, np.ndarray)
        else threshold
    )
    padded = np.pad(strength, (1, 1), constant_values=-np.inf)
    peaks, _ = signal.find_peaks(padded, height=height, distance=distance)
    active = peaks - 1
    return active[(active >= 0) & (active < strength.size)]


def fixed_threshold_events(
    name: str,
    values: np.ndarray,
    size: int,
    hop: int,
    sample_rate: int,
    head_level: float = DEFAULT_HEAD_LEVEL,
) -> OnsetFunction | None:
    strength = np.clip(normalized_strength(values, sample_rate, hop), 0.0, 1.0)
    active = peak_frames(strength, ODF_ACTIVE_RATIO, sample_rate, hop)
    if active.size == 0:
        return None
    frames = active + head_shift_frames(
        strength, active, lookback_frames(sample_rate, hop), head_level
    )
    return OnsetFunction(
        name, frame_times_ms(frames, size, hop, sample_rate), strength[active]
    )


def adaptive_threshold_events(
    name: str,
    values: np.ndarray,
    size: int,
    hop: int,
    sample_rate: int,
    head_level: float = DEFAULT_HEAD_LEVEL,
) -> OnsetFunction | None:
    strength = normalized_strength(values, sample_rate, hop)
    window = max(
        3,
        int(round(ODF_COMPLEX_THRESHOLD_WINDOW_MS * sample_rate / (1000.0 * hop))),
    )
    if window % 2 == 0:
        window += 1
    baseline = ndimage.median_filter(strength, size=window, mode="nearest")
    threshold = np.maximum(ODF_ACTIVE_RATIO, ODF_COMPLEX_THRESHOLD_RATIO * baseline)
    active = peak_frames(strength, threshold, sample_rate, hop)
    if active.size == 0:
        return None
    weights = np.maximum(strength[active] - threshold[active], 0.0)
    lookback = lookback_frames(sample_rate, hop)
    onset_frames = threshold_crossing_frames(
        active, strength, threshold, lookback
    ) + head_shift_frames(strength, active, lookback, head_level)
    return OnsetFunction(
        name, frame_times_ms(onset_frames, size, hop, sample_rate), weights
    )


def compute_onset_functions(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    head_level: float = DEFAULT_HEAD_LEVEL,
) -> tuple[OnsetFunction, ...]:
    ceiling_hz = min(high_hz, USABLE_NYQUIST_RATIO * sample_rate)
    result = spectral_series(samples, sample_rate, low_hz, ceiling_hz)
    if result is None:
        return ()
    series, size, hop = result
    functions = (
        (
            adaptive_threshold_events
            if kind is OnsetKind.COMPLEX
            else fixed_threshold_events
        )(kind.value, values, size, hop, sample_rate, head_level)
        for kind, values in zip(OnsetKind, series)
    )
    return tuple(function for function in functions if function is not None)
