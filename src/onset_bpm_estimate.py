from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft

from onset_config import DEFAULT_HEAD_LEVEL, MAX_BPM, MIN_BPM
from onset_detection import Detection

ESTIMATE_MIN_BPM = 60.0
ESTIMATE_MAX_BPM = 260.0
ESTIMATE_MIN_EVENTS = 32
ESTIMATE_BIN_MS = 2.0
ESTIMATE_PAD_FACTOR = 8
ESTIMATE_COARSE_STEP_BPM = 0.05
ESTIMATE_COARSE_HARMONICS = 8
ESTIMATE_FINE_HARMONICS = 12
ESTIMATE_FINE_SPAN_BPM = 0.3
ESTIMATE_FINE_STEP_BPM = 0.002
ESTIMATE_FINE_CHUNK = 32
ESTIMATE_OCTAVE_TOLERANCE = 0.03
ESTIMATE_OCTAVE_NEIGHBORHOOD = 0.003
ESTIMATE_MIN_DOMINANCE = 1.4
BPM_DECIMALS = 2


@dataclass(frozen=True)
class OnsetEvents:
    times_ms: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class BpmEstimate:
    bpm: float
    dominance: float | None
    octaves: tuple[tuple[float, float], ...]


def attack_events(detection: Detection) -> OnsetEvents:
    default_heads = detection.heads[DEFAULT_HEAD_LEVEL]
    heads_ms, peaks_ms = default_heads.heads_ms, default_heads.peaks_ms
    if heads_ms.size == 0 or not detection.caps.usable:
        return OnsetEvents(np.empty(0), np.empty(0))
    indices = np.round(peaks_ms * detection.sample_rate / 1000.0).astype(int)
    strengths = np.clip(detection.slope[indices] / detection.caps.at(indices), 0.0, 1.0)
    return OnsetEvents(heads_ms, strengths)


def onset_spectrum(events: OnsetEvents) -> tuple[np.ndarray, np.ndarray]:
    indices = np.round(events.times_ms / ESTIMATE_BIN_MS).astype(int)
    train = np.bincount(indices, weights=events.weights)
    train = train - train.mean()
    size = fft.next_fast_len(train.size * ESTIMATE_PAD_FACTOR, real=True)
    magnitude = np.abs(fft.rfft(train, size))
    frequencies = fft.rfftfreq(size, ESTIMATE_BIN_MS / 1000.0)
    return frequencies, magnitude


def harmonic_salience(
    frequencies: np.ndarray, magnitude: np.ndarray, bpms: np.ndarray
) -> np.ndarray:
    fundamental = bpms / 60.0
    total = np.zeros_like(fundamental)
    for harmonic in range(1, ESTIMATE_COARSE_HARMONICS + 1):
        total += np.interp(harmonic * fundamental, frequencies, magnitude)
    return total


def local_salience(
    frequencies: np.ndarray, magnitude: np.ndarray, bpm: float
) -> float:
    offsets = np.linspace(
        -ESTIMATE_OCTAVE_NEIGHBORHOOD, ESTIMATE_OCTAVE_NEIGHBORHOOD, 25
    )
    return float(harmonic_salience(frequencies, magnitude, bpm * (1.0 + offsets)).max())


def octave_distance(ratios: np.ndarray) -> np.ndarray:
    exponent = np.log2(ratios)
    return np.abs(exponent - np.round(exponent))


def dominance_ratio(
    bpms: np.ndarray, salience: np.ndarray, best_index: int
) -> float | None:
    distance = octave_distance(bpms / bpms[best_index])
    rivals = salience[distance > ESTIMATE_OCTAVE_TOLERANCE]
    if rivals.size == 0 or rivals.max() <= 0.0:
        return None
    return float(salience[best_index] / rivals.max())


def refine_bpm(events: OnsetEvents, center: float) -> float:
    grid = np.arange(
        center - ESTIMATE_FINE_SPAN_BPM,
        center + ESTIMATE_FINE_SPAN_BPM + ESTIMATE_FINE_STEP_BPM / 2.0,
        ESTIMATE_FINE_STEP_BPM,
    )
    seconds = events.times_ms / 1000.0
    scores = np.empty(grid.size)
    for start in range(0, grid.size, ESTIMATE_FINE_CHUNK):
        chunk = grid[start : start + ESTIMATE_FINE_CHUNK]
        base = np.exp(-2j * np.pi * np.outer(chunk / 60.0, seconds))
        current = base
        score = np.zeros(chunk.size)
        for _ in range(ESTIMATE_FINE_HARMONICS):
            score += np.abs(current @ events.weights)
            current = current * base
        scores[start : start + chunk.size] = score
    return float(grid[int(np.argmax(scores))])


def octave_candidates(
    frequencies: np.ndarray, magnitude: np.ndarray, bpm: float
) -> tuple[tuple[float, float], ...]:
    reference = local_salience(frequencies, magnitude, bpm)
    candidates = []
    for factor in (0.5, 2.0):
        value = bpm * factor
        if MIN_BPM <= value <= MAX_BPM:
            ratio = local_salience(frequencies, magnitude, value) / reference
            candidates.append((round(value, BPM_DECIMALS), float(ratio)))
    return tuple(candidates)


def estimate_bpm(
    detection: Detection, min_bpm: float, max_bpm: float
) -> BpmEstimate:
    events = attack_events(detection)
    if events.times_ms.size < ESTIMATE_MIN_EVENTS:
        raise ValueError("BPMを推定できるだけのオンセットを検出できませんでした")
    frequencies, magnitude = onset_spectrum(events)
    bpms = np.arange(min_bpm, max_bpm, ESTIMATE_COARSE_STEP_BPM)
    salience = harmonic_salience(frequencies, magnitude, bpms)
    best_index = int(np.argmax(salience))
    bpm = round(refine_bpm(events, float(bpms[best_index])), BPM_DECIMALS)
    return BpmEstimate(
        bpm,
        dominance_ratio(bpms, salience, best_index),
        octave_candidates(frequencies, magnitude, bpm),
    )
