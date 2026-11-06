from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from onset_analysis import Analysis
from onset_audio_io import (
    load_audio,
    readable_audio,
    validate_audio,
    validate_loaded_duration,
)
from onset_beat import analyze_beat, BeatAnalysis
from onset_bpm_estimate import BpmEstimate, estimate_bpm
from onset_config import AnalysisConfig, FILTER_CAUSAL
from onset_decoder import decode_info, DecodeInfo
from onset_detection import detect_attacks, Detection, head_levels
from onset_grid import GridAlignment
from onset_kick import attach_kicks, kick_envelope
from onset_odf import compute_onset_functions
from onset_resolution import DelayCandidates, delay_candidates
from onset_wavelet import compute_wavelet_functions


@dataclass(frozen=True)
class RunOptions:
    bpm: float | None
    bpm_min: float
    bpm_max: float
    low_hz: float
    high_hz: float
    head_level: float
    drift_window_sec: float
    filter_mode: str


@dataclass(frozen=True)
class AnalysisRun:
    config: AnalysisConfig
    beat: BeatAnalysis
    estimate: BpmEstimate | None
    decode: DecodeInfo | None
    candidates: DelayCandidates

    @property
    def analysis(self) -> Analysis:
        return self.beat.analysis

    @property
    def alignment(self) -> GridAlignment | None:
        return self.beat.alignment


def config_from_options(options: RunOptions, bpm: float) -> AnalysisConfig:
    return AnalysisConfig(
        bpm=bpm,
        low_hz=options.low_hz,
        high_hz=options.high_hz,
        head_level=options.head_level,
        drift_window_sec=options.drift_window_sec,
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
    validate_loaded_duration(samples.shape[0], sample_rate)
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
    functions = (
        *compute_onset_functions(samples, sample_rate),
        *compute_wavelet_functions(samples, sample_rate),
    )
    bpm, estimate = resolve_bpm(options, attacks)
    config = config_from_options(options, bpm)
    kick = kick_envelope(samples, sample_rate)
    beat = analyze_beat(attach_kicks(attacks, kick, config.grid_beat_ms), config, functions)
    candidates = delay_candidates(
        beat.analysis,
        beat.alignment,
        config.layout,
        samples.shape[0] * 1000.0 / sample_rate,
    )
    return AnalysisRun(config, beat, estimate, decode, candidates)
