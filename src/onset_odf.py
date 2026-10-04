from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft, ndimage, signal

ODF_LOW_HZ = 1000.0
ODF_HIGH_HZ = 12000.0
ODF_NYQUIST_RATIO = 0.45
ODF_WINDOW_MS = 6.0
ODF_HOP_MS = 1.0
ODF_LAG_FRAMES = 2
ODF_HANN_STEEPEST_FALL = 0.75
ODF_HISTORY_FRAMES = max(ODF_LAG_FRAMES, 2)
ODF_CHUNK_FRAMES = 4096
ODF_COMPRESSION = 10.0
ODF_FREQUENCY_FILTER_BINS = 3
ODF_MIN_BINS = 4
ODF_BLOCK_SEC = 4.0
ODF_PERCENTILE = 99.5
ODF_FLOOR_RATIO = 0.1
ODF_ACTIVE_RATIO = 0.15
ODF_NAMES = ("SuperFlux風", "複素領域", "HFC")

FOLD_BIN_MS = 0.1
FOLD_SIGMA_MS = 1.0
FOLD_MIN_BINS = 8


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


def fold_profile(
    times_ms: np.ndarray,
    weights: np.ndarray,
    period_ms: float,
    bin_ms: float = FOLD_BIN_MS,
) -> tuple[np.ndarray, float]:
    bins = max(FOLD_MIN_BINS, int(round(period_ms / bin_ms)))
    scaled = np.mod(times_ms, period_ms) * (bins / period_ms)
    lower = np.floor(scaled)
    fraction = scaled - lower
    first = lower.astype(np.int64) % bins
    second = (first + 1) % bins
    profile = np.bincount(first, weights * (1.0 - fraction), minlength=bins)
    profile += np.bincount(second, weights * fraction, minlength=bins)
    return profile, period_ms / bins


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
    position = (
        indices * hop
        + size * ODF_HANN_STEEPEST_FALL
        - ODF_LAG_FRAMES * hop / 2.0
    )
    return position * 1000.0 / sample_rate


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
    rising = magnitude[history:] > magnitude[history - 1 : total - 1]
    complex_domain = (deviation * rising).sum(axis=1)
    return np.stack((flux, complex_domain, hfc))


def spectral_series(
    mono: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> tuple[np.ndarray, int, int] | None:
    size, hop = frame_layout(sample_rate)
    frame_count = (mono.size - size) // hop + 1
    frequencies = fft.rfftfreq(size, 1.0 / sample_rate)
    band = np.flatnonzero((frequencies >= low_hz) & (frequencies <= high_hz))
    if frame_count <= ODF_HISTORY_FRAMES or band.size < ODF_MIN_BINS:
        return None
    window = signal.get_window("hann", size).astype(np.float32)
    frames = sliding_window_view(mono, size)[::hop]
    band_weights = band.astype(np.float32)
    series = np.empty((len(ODF_NAMES), frame_count), dtype=np.float32)
    for start in range(0, frame_count, ODF_CHUNK_FRAMES):
        stop = min(frame_count, start + ODF_CHUNK_FRAMES)
        first = max(0, start - ODF_HISTORY_FRAMES)
        spectra = fft.rfft(frames[first:stop] * window, axis=1)[:, band]
        padding = ODF_HISTORY_FRAMES - (start - first)
        if padding > 0:
            spectra = np.concatenate((np.repeat(spectra[:1], padding, axis=0), spectra))
        series[:, start:stop] = chunk_functions(spectra, band_weights)
    return series, size, hop


def active_events(
    name: str, values: np.ndarray, size: int, hop: int, sample_rate: int
) -> OnsetFunction | None:
    block = max(1, int(round(ODF_BLOCK_SEC * sample_rate / hop)))
    caps = np.repeat(
        block_caps(values, block, ODF_PERCENTILE, ODF_FLOOR_RATIO), block
    )[: values.size]
    strength = np.divide(
        values, caps, out=np.zeros(values.shape, dtype=np.float64), where=caps > 0.0
    )
    strength = np.clip(strength, 0.0, 1.0)
    active = np.flatnonzero(strength > ODF_ACTIVE_RATIO)
    if active.size == 0:
        return None
    return OnsetFunction(
        name, frame_times_ms(active, size, hop, sample_rate), strength[active]
    )


def compute_onset_functions(
    samples: np.ndarray, sample_rate: int
) -> tuple[OnsetFunction, ...]:
    high_hz = min(ODF_HIGH_HZ, ODF_NYQUIST_RATIO * sample_rate)
    mono = samples.mean(axis=1, dtype=np.float32)
    result = spectral_series(mono, sample_rate, ODF_LOW_HZ, high_hz)
    if result is None:
        return ()
    series, size, hop = result
    functions = (
        active_events(name, values, size, hop, sample_rate)
        for name, values in zip(ODF_NAMES, series)
    )
    return tuple(function for function in functions if function is not None)
