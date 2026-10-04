from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

FILTER_ORDER = 4
DELAY_POINTS = 256


def ms_to_samples(milliseconds: float, sample_rate: float) -> int:
    return max(1, int(milliseconds * sample_rate / 1000.0))


def wrap_symmetric(values: np.ndarray | float, period: float) -> np.ndarray | float:
    return (values + period / 2.0) % period - period / 2.0


def bandpass_sos(sample_rate: float, low_hz: float, high_hz: float) -> np.ndarray:
    return signal.butter(
        FILTER_ORDER, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )


def causal_delay_ms(
    sample_rate: float, low_hz: float, high_hz: float, smooth_ms: float
) -> float:
    frequencies = np.linspace(low_hz, high_hz, DELAY_POINTS)
    _, response = signal.sosfreqz(
        bandpass_sos(sample_rate, low_hz, high_hz), worN=frequencies, fs=sample_rate
    )
    phase = np.unwrap(np.angle(response))
    group_delay_sec = -np.gradient(phase, 2.0 * np.pi * frequencies)
    smoothing_samples = (ms_to_samples(smooth_ms, sample_rate) - 1) / 2.0
    return float(np.median(group_delay_sec)) * 1000.0 + smoothing_samples * 1000.0 / sample_rate


def causal_envelope(
    samples: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    smooth_ms: float,
) -> np.ndarray:
    sos = bandpass_sos(sample_rate, low_hz, high_hz)
    rectified = np.abs(signal.sosfilt(sos, samples))
    width = ms_to_samples(smooth_ms, sample_rate)
    return ndimage.uniform_filter1d(
        rectified, width, mode="nearest", origin=width - 1 - width // 2
    )
