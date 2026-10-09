from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from onset_audio_io import load_audio, readable_audio, validate_audio
from onset_bpm_estimate import BpmEstimate, estimate_bpm
from onset_config import AnalysisConfig
from onset_detection import detect_attacks, Detection, head_levels
from onset_stem_classes import band_increments, classify_events
from onset_stem_phase import estimate_stem, StemEstimate
from onset_timing import TempoMap
from onset_warp import make_warp

STEM_LOW_HZ = 1500.0
STEM_HIGH_HZ = 12000.0


@dataclass(frozen=True)
class StemOptions:
    bpm: float | None
    bpm_min: float
    bpm_max: float
    subdivision: int
    low_hz: float
    high_hz: float
    head_level: float
    drift_window_sec: float


@dataclass(frozen=True)
class StemRun:
    config: AnalysisConfig
    estimate: StemEstimate
    bpm_estimate: BpmEstimate | None


def resolve_stem_bpm(
    options: StemOptions, attacks: Detection
) -> tuple[float, BpmEstimate | None]:
    if options.bpm is not None:
        return options.bpm, None
    estimate = estimate_bpm(attacks, options.bpm_min, options.bpm_max)
    return estimate.bpm, estimate


def analyze_stem(
    samples: np.ndarray,
    sample_rate: int,
    options: StemOptions,
    tempo: TempoMap | None = None,
) -> StemRun:
    attacks = detect_attacks(
        samples,
        sample_rate,
        options.low_hz,
        options.high_hz,
        head_levels(options.head_level),
    )
    bpm, bpm_estimate = resolve_stem_bpm(options, attacks)
    config = AnalysisConfig(
        bpm=bpm,
        subdivision=options.subdivision,
        low_hz=options.low_hz,
        high_hz=options.high_hz,
        head_level=options.head_level,
        drift_window_sec=options.drift_window_sec,
    )
    real = attacks.heads[options.head_level]
    warped = make_warp(tempo, config, attacks).heads(real)
    labels = classify_events(band_increments(samples, sample_rate, real.heads_ms))
    estimate = estimate_stem(warped.heads_ms, labels, config)
    if estimate is None:
        raise ValueError("グリッドに乗ったオンセットを持つクラスを検出できませんでした")
    return StemRun(config, estimate, bpm_estimate)


def run_stem(audio_path: Path, options: StemOptions) -> StemRun:
    with readable_audio(audio_path) as readable_path:
        validate_audio(readable_path, options.high_hz)
        samples, sample_rate = load_audio(readable_path)
    return analyze_stem(samples, sample_rate, options)
