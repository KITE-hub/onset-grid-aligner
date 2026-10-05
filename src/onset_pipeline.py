from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from onset_analysis import Analysis, analyze
from onset_audio_io import load_audio, readable_audio, validate_audio
from onset_band_check import band_check, BandCheck
from onset_bpm_estimate import BpmEstimate, estimate_bpm
from onset_config import AnalysisConfig, FILTER_CAUSAL
from onset_decoder import decode_info, DecodeInfo
from onset_detection import detect_attacks, Detection, head_levels
from onset_kick import attach_kicks, kick_envelope
from onset_odf import compute_onset_functions
from onset_refine_check import refine_check, RefineCheck
from onset_warp import make_warp, warped_detection, warped_functions


@dataclass(frozen=True)
class RunOptions:
    bpm: float | None
    bpm_min: float
    bpm_max: float
    subdivision: int
    low_hz: float
    high_hz: float
    head_level: float
    drift_window_sec: float
    reference_group: str
    filter_mode: str
    run_band_check: bool
    run_refine: bool


@dataclass(frozen=True)
class AnalysisRun:
    config: AnalysisConfig
    analysis: Analysis
    estimate: BpmEstimate | None
    decode: DecodeInfo | None
    bands: BandCheck | None
    refine: RefineCheck | None


def config_from_options(options: RunOptions, bpm: float) -> AnalysisConfig:
    return AnalysisConfig(
        bpm=bpm,
        subdivision=options.subdivision,
        low_hz=options.low_hz,
        high_hz=options.high_hz,
        head_level=options.head_level,
        drift_window_sec=options.drift_window_sec,
        reference_group=options.reference_group,
    )


def resolve_bpm(
    options: RunOptions, detection: Detection
) -> tuple[float, BpmEstimate | None]:
    if options.bpm is not None:
        return options.bpm, None
    estimate = estimate_bpm(detection, options.bpm_min, options.bpm_max)
    return estimate.bpm, estimate


def run_pipeline(audio_path: Path, options: RunOptions) -> AnalysisRun:
    with readable_audio(audio_path) as readable_path:
        validate_audio(readable_path, options.high_hz)
        samples, sample_rate = load_audio(readable_path)
    decode = decode_info(audio_path, sample_rate)
    causal = options.filter_mode == FILTER_CAUSAL
    attacks = detect_attacks(
        samples,
        sample_rate,
        options.low_hz,
        options.high_hz,
        head_levels(options.head_level),
        causal,
    )
    functions = compute_onset_functions(samples, sample_rate)
    bpm, estimate = resolve_bpm(options, attacks)
    config = config_from_options(options, bpm)
    kick = kick_envelope(samples, sample_rate)
    detection = attach_kicks(attacks, kick, config.period_ms)
    warp = make_warp(None, config, detection)
    analysis = analyze(
        warped_detection(detection, warp),
        config,
        warped_functions(functions, warp),
    )
    bands = (
        band_check(samples, sample_rate, config, analysis, kick, causal, warp)
        if options.run_band_check
        else None
    )
    refine = (
        refine_check(samples, sample_rate, attacks, kick, config, analysis, warp)
        if options.run_refine
        else None
    )
    return AnalysisRun(config, analysis, estimate, decode, bands, refine)
