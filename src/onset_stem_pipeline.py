from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from onset_audio_io import load_audio, readable_audio, validate_audio
from onset_bpm_estimate import BpmEstimate, estimate_bpm
from onset_config import AnalysisConfig
from onset_detection import detect_attacks, Detection, head_levels, Heads
from onset_reference import group_trend_ms
from onset_stem_classes import attack_mask, band_increments, classify_events
from onset_stem_grid import (
    align_grid,
    build_layout,
    comb_mask,
    GridAlignment,
    GridLayout,
    STEPS_PER_BEAT,
)
from onset_stem_phase import BandEstimate, estimate_stem, StemResult
from onset_subdivision import fold_bpm
from onset_timing import TempoMap
from onset_warp import make_warp

STEM_LOW_HZ = 1500.0
STEM_SPLIT_HZ = 6000.0
STEM_HIGH_HZ = 16000.0


@dataclass(frozen=True)
class StemOptions:
    bpm: float | None
    bpm_min: float
    bpm_max: float
    low_hz: float
    split_hz: float
    high_hz: float
    head_level: float
    drift_window_sec: float

    @property
    def bands_hz(self) -> tuple[tuple[float, float], ...]:
        return (
            (self.low_hz, self.split_hz),
            (self.split_hz, self.high_hz),
        )


@dataclass(frozen=True)
class StemRun:
    config: AnalysisConfig
    result: StemResult
    bpm: float
    bpm_estimate: BpmEstimate | None
    layout: GridLayout
    alignment: GridAlignment | None

    @property
    def delay_ms(self) -> float:
        if self.alignment is None:
            return self.result.delay_ms
        return -self.alignment.offset_ms


def resolve_stem_bpm(
    options: StemOptions, attacks: Detection
) -> tuple[float, BpmEstimate | None]:
    if options.bpm is not None:
        return options.bpm, None
    estimate = estimate_bpm(attacks, options.bpm_min, options.bpm_max)
    return estimate.bpm, estimate


def grid_config(options: StemOptions, grid_bpm: float) -> AnalysisConfig:
    return AnalysisConfig(
        bpm=grid_bpm,
        subdivision=STEPS_PER_BEAT,
        low_hz=options.low_hz,
        high_hz=options.high_hz,
        head_level=options.head_level,
        drift_window_sec=options.drift_window_sec,
    )


def split_increments(
    samples: np.ndarray, sample_rate: int, reals: list[Heads]
) -> list[np.ndarray]:
    heads_ms = np.concatenate([real.heads_ms for real in reals])
    increments = band_increments(samples, sample_rate, heads_ms)
    boundaries = np.cumsum([real.heads_ms.size for real in reals])[:-1]
    return np.split(increments, boundaries)


@dataclass(frozen=True)
class BandEvents:
    band: tuple[float, float]
    times_ms: np.ndarray
    labels: np.ndarray
    strengths: np.ndarray


def collect_band_events(
    band: tuple[float, float],
    attacks: Detection,
    real: Heads,
    increments: np.ndarray,
    config: AnalysisConfig,
    tempo: TempoMap | None,
) -> BandEvents:
    band_config = replace(config, low_hz=band[0], high_hz=band[1])
    warped = make_warp(tempo, band_config, attacks).heads(real)
    kept = attack_mask(increments)
    totals = increments[kept].sum(axis=1)
    strengths = totals / np.median(totals) if totals.size else totals
    return BandEvents(
        band,
        warped.heads_ms[kept],
        classify_events(increments[kept]),
        strengths,
    )


def estimate_band(
    events: BandEvents, config: AnalysisConfig, on_comb: np.ndarray | None
) -> BandEstimate | None:
    band_config = replace(config, low_hz=events.band[0], high_hz=events.band[1])
    estimate = estimate_stem(events.times_ms, events.labels, band_config, on_comb)
    if estimate is None:
        return None
    return BandEstimate(
        events.band[0],
        events.band[1],
        estimate,
        events.times_ms - group_trend_ms(events.times_ms, estimate.drift),
        events.strengths,
    )


def align_bands(
    found: tuple[BandEstimate, ...], result: StemResult, layout: GridLayout
) -> GridAlignment | None:
    return align_grid(
        np.concatenate([band.times_ms for band in found]),
        np.concatenate([band.strengths for band in found]),
        result.phase_ms,
        layout,
    )


def comb_estimates(
    events: list[BandEvents],
    first: list[BandEstimate | None],
    config: AnalysisConfig,
    alignment: GridAlignment,
    layout: GridLayout,
) -> list[BandEstimate | None]:
    return [
        None
        if estimate is None
        else estimate_band(
            item, config, comb_mask(estimate.times_ms, alignment, layout)
        )
        for item, estimate in zip(events, first)
    ]


def analyze_stem(
    samples: np.ndarray,
    sample_rate: int,
    options: StemOptions,
    tempo: TempoMap | None = None,
) -> StemRun:
    bands = options.bands_hz
    detections = [
        detect_attacks(
            samples, sample_rate, low_hz, high_hz, head_levels(options.head_level)
        )
        for low_hz, high_hz in bands
    ]
    bpm, bpm_estimate = resolve_stem_bpm(options, detections[0])
    grid_bpm = fold_bpm(bpm)
    config = grid_config(options, grid_bpm)
    layout = build_layout(bpm, grid_bpm)
    reals = [detection.heads[options.head_level] for detection in detections]
    increments = split_increments(samples, sample_rate, reals)
    events = [
        collect_band_events(band, detection, real, band_increments, config, tempo)
        for band, detection, real, band_increments in zip(
            bands, detections, reals, increments
        )
    ]
    first = [estimate_band(item, config, None) for item in events]
    found = tuple(item for item in first if item is not None)
    if not found:
        raise ValueError("グリッドに乗ったオンセットを持つクラスを検出できませんでした")
    alignment = align_bands(found, StemResult(config.period_ms, found), layout)
    if alignment is None:
        return StemRun(
            config, StemResult(config.period_ms, found), bpm, bpm_estimate, layout, None
        )
    second = comb_estimates(events, first, config, alignment, layout)
    found = tuple(item for item in second if item is not None)
    if not found:
        raise ValueError("12本の線に乗ったオンセットを持つクラスを検出できませんでした")
    result = StemResult(config.period_ms, found)
    return StemRun(
        config, result, bpm, bpm_estimate, layout, align_bands(found, result, layout)
    )


def run_stem(audio_path: Path, options: StemOptions) -> StemRun:
    with readable_audio(audio_path) as readable_path:
        validate_audio(readable_path, options.high_hz)
        samples, sample_rate = load_audio(readable_path)
    return analyze_stem(samples, sample_rate, options)
