from __future__ import annotations

import numpy as np
from scipy import fft, ndimage

from onset_common import ms_to_samples
from onset_odf import active_events, OnsetFunction

WAVELET_CENTERS_HZ = (2000.0, 3000.0, 4500.0, 6500.0, 9000.0)
WAVELET_QUALITY = 2.0
WAVELET_NYQUIST_RATIO = 0.45
WAVELET_EDGE_MS = 100.0
WAVELET_PEAK_WINDOW_MS = 12.0
WAVELET_FLOOR_RATIO = 1e-2
WAVELET_ABSOLUTE_FLOOR_RATIO = 1e-3
WAVELET_ABSOLUTE_PERCENTILE = 99.5
WAVELET_FRAME_SIZE = 0
WAVELET_FRAME_HOP = 1
WAVELET_FAMILY = "wavelet"
MULTI_SCALE_NAME = "Wavelet多重スケール"
COARSE_SCALE_NAME = "Wavelet粗スケール"
WAVELET_VARIANTS = (
    (MULTI_SCALE_NAME, (0.3, 0.6, 1.2)),
    (COARSE_SCALE_NAME, (1.2,)),
)
WAVELET_NAMES = tuple(name for name, _ in WAVELET_VARIANTS)
WAVELET_SCALES_MS = tuple(
    sorted({scale for _, scales in WAVELET_VARIANTS for scale in scales})
)


def usable_centers(sample_rate: int) -> tuple[float, ...]:
    reach = 1.0 + 1.0 / WAVELET_QUALITY
    ceiling = WAVELET_NYQUIST_RATIO * sample_rate
    return tuple(
        center for center in WAVELET_CENTERS_HZ if reach * center <= ceiling
    )


def analytic_envelope(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    center_hz: float,
    length: int,
    fft_size: int,
) -> np.ndarray:
    sigma_hz = center_hz / WAVELET_QUALITY
    response = np.exp(-0.5 * np.square((frequencies - center_hz) / sigma_hz))
    return np.abs(fft.ifft(spectrum * response, fft_size)[:length])


def relative_growth(envelope: np.ndarray, sample_rate: int) -> np.ndarray:
    window = ms_to_samples(WAVELET_PEAK_WINDOW_MS, sample_rate) | 1
    local_peak = ndimage.maximum_filter1d(
        envelope, window, origin=-(window // 2), mode="nearest"
    )
    floor = np.maximum(
        WAVELET_FLOOR_RATIO * local_peak,
        WAVELET_ABSOLUTE_FLOOR_RATIO
        * np.percentile(envelope, WAVELET_ABSOLUTE_PERCENTILE),
    )
    per_ms = sample_rate / 1000.0
    return (
        np.gradient(envelope)
        * per_ms
        / (envelope + floor + np.finfo(np.float32).tiny)
    )


def scale_growths(envelope: np.ndarray, sample_rate: int) -> dict[float, np.ndarray]:
    growth = relative_growth(envelope, sample_rate)
    per_ms = sample_rate / 1000.0
    return {
        scale: np.maximum(
            ndimage.gaussian_filter1d(growth, scale * per_ms, mode="nearest"), 0.0
        )
        for scale in WAVELET_SCALES_MS
    }


def add_channel_series(
    series: np.ndarray,
    channel: np.ndarray,
    sample_rate: int,
    centers: tuple[float, ...],
) -> None:
    length = channel.shape[0]
    fft_size = fft.next_fast_len(length + ms_to_samples(WAVELET_EDGE_MS, sample_rate))
    spectrum = fft.rfft(channel, fft_size)
    frequencies = fft.rfftfreq(fft_size, 1.0 / sample_rate).astype(np.float32)
    for center in centers:
        growths = scale_growths(
            analytic_envelope(spectrum, frequencies, center, length, fft_size),
            sample_rate,
        )
        for row, (_, scales) in zip(series, WAVELET_VARIANTS):
            row += np.power(
                np.prod([growths[scale] for scale in scales], axis=0),
                1.0 / len(scales),
            )


def compute_wavelet_functions(
    samples: np.ndarray, sample_rate: int
) -> tuple[OnsetFunction, ...]:
    centers = usable_centers(sample_rate)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if not centers or audio.ndim != 2 or 0 in audio.shape:
        return ()
    series = np.zeros((len(WAVELET_VARIANTS), audio.shape[0]), dtype=np.float32)
    for channel in audio.T:
        add_channel_series(series, channel, sample_rate, centers)
    functions = (
        active_events(name, values, WAVELET_FRAME_SIZE, WAVELET_FRAME_HOP, sample_rate)
        for (name, _), values in zip(WAVELET_VARIANTS, series)
    )
    return tuple(function for function in functions if function is not None)
