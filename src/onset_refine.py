from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_common import causal_delay_ms, causal_envelope, ms_to_samples

REFINE_BANDS = ((1000.0, 3000.0), (3000.0, 8000.0), (8000.0, 16000.0))
REFINE_NYQUIST_RATIO = 0.45
REFINE_MIN_BAND_RATIO = 1.5
REFINE_SMOOTH_MS = 0.5
REFINE_LOOKBACK_MS = 20.0
REFINE_FORWARD_MS = 12.0
REFINE_MIN_ATTACK_RATIO = 1.5
REFINE_CHUNK_EVENTS = 2048


@dataclass(frozen=True)
class RefinedHeads:
    bands: tuple[tuple[float, float], ...]
    band_heads_ms: np.ndarray
    heads_ms: np.ndarray


def usable_bands(sample_rate: int) -> tuple[tuple[float, float], ...]:
    ceiling = REFINE_NYQUIST_RATIO * sample_rate
    clipped = ((low, min(high, ceiling)) for low, high in REFINE_BANDS)
    return tuple(
        (low, high)
        for low, high in clipped
        if high >= REFINE_MIN_BAND_RATIO * low
    )


def merged_causal_envelope(
    samples: np.ndarray, sample_rate: int, low_hz: float, high_hz: float
) -> np.ndarray:
    merged = np.zeros(samples.shape[0])
    for channel in samples.T:
        np.maximum(
            merged,
            causal_envelope(channel, sample_rate, low_hz, high_hz, REFINE_SMOOTH_MS),
            out=merged,
        )
    return merged


def chunk_heads_ms(
    envelope: np.ndarray,
    peaks: np.ndarray,
    sample_rate: int,
    head_level: float,
) -> np.ndarray:
    lookback = ms_to_samples(REFINE_LOOKBACK_MS, sample_rate)
    forward = ms_to_samples(REFINE_FORWARD_MS, sample_rate)
    heads = np.full(peaks.size, np.nan)
    inside = (peaks >= lookback) & (peaks + forward <= envelope.size)
    if not inside.any():
        return heads
    centers = peaks[inside]
    segments = envelope[centers[:, None] + np.arange(-lookback, forward)[None, :]]
    base = segments[:, :lookback].min(axis=1)
    top = segments[:, lookback:].max(axis=1)
    level = base + head_level * (top - base)
    below = segments[:, : lookback + 1] < level[:, None]
    last = lookback - np.argmax(below[:, ::-1], axis=1)
    rows = np.arange(centers.size)
    lower = segments[rows, last]
    rise = segments[rows, last + 1] - lower
    fraction = np.clip(
        np.divide(level - lower, rise, out=np.ones(centers.size), where=rise > 0.0),
        0.0,
        1.0,
    )
    attack = (top > base * REFINE_MIN_ATTACK_RATIO) & below.any(axis=1)
    positions = centers + (last - lookback) + fraction
    heads[np.flatnonzero(inside)[attack]] = positions[attack] * 1000.0 / sample_rate
    return heads


def band_heads_ms(
    envelope: np.ndarray,
    peaks: np.ndarray,
    sample_rate: int,
    head_level: float,
) -> np.ndarray:
    heads = np.empty(peaks.size)
    for start in range(0, peaks.size, REFINE_CHUNK_EVENTS):
        stop = start + REFINE_CHUNK_EVENTS
        heads[start:stop] = chunk_heads_ms(
            envelope, peaks[start:stop], sample_rate, head_level
        )
    return heads


def median_ignoring_nan(stack: np.ndarray) -> np.ndarray:
    count = (~np.isnan(stack)).sum(axis=0)
    ordered = np.sort(stack, axis=0)
    high = np.take_along_axis(ordered, (count // 2)[None, :], axis=0)[0]
    low = np.take_along_axis(
        ordered, (np.maximum(count - 1, 0) // 2)[None, :], axis=0
    )[0]
    return np.where(count > 0, (low + high) / 2.0, np.nan)


def refine_heads(
    samples: np.ndarray,
    sample_rate: int,
    peaks_ms: np.ndarray,
    head_level: float,
) -> RefinedHeads | None:
    bands = usable_bands(sample_rate)
    if not bands or peaks_ms.size == 0:
        return None
    peaks = np.round(peaks_ms * sample_rate / 1000.0).astype(np.int64)
    rows = []
    for low_hz, high_hz in bands:
        envelope = merged_causal_envelope(samples, sample_rate, low_hz, high_hz)
        delay_ms = causal_delay_ms(sample_rate, low_hz, high_hz, REFINE_SMOOTH_MS)
        rows.append(band_heads_ms(envelope, peaks, sample_rate, head_level) - delay_ms)
    stack = np.vstack(rows)
    return RefinedHeads(bands, stack, median_ignoring_nan(stack))
