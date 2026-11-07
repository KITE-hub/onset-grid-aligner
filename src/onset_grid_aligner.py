from __future__ import annotations

import sys

from onset_audio_io import decode_with_ffmpeg, load_audio, readable_audio, validate_audio
from onset_cli import main
from onset_config import (
    AnalysisConfig,
    DEFAULT_DRIFT_WINDOW_SEC,
    DEFAULT_HEAD_LEVEL,
    DEFAULT_HIGH_HZ,
    DEFAULT_LOW_HZ,
    FILTER_CAUSAL,
    FILTER_CHOICES,
    FILTER_ZERO,
)
from onset_detection import detect_attacks, Detection, head_levels, Heads
from onset_grid import GridAlignment, GridLayout
from onset_kick import kick_envelope, kick_peaks_ms, KickEnvelope
from onset_method import (
    analyze_methods,
    ENVELOPE_NAME,
    envelope_function,
    estimate_method,
    method_sources,
    MethodEstimate,
)
from onset_phase import MIN_ONSETS
from onset_reliability import MethodReliability, Reliability, reliability_of
from onset_resolution import DelayCandidates, delay_candidates
from onset_warp import GridWarp, make_warp

__all__ = [
    "analyze_methods", "ENVELOPE_NAME", "envelope_function", "estimate_method",
    "method_sources", "MethodEstimate",
    "decode_with_ffmpeg", "load_audio", "readable_audio",
    "validate_audio",
    "AnalysisConfig", "DEFAULT_DRIFT_WINDOW_SEC", "DEFAULT_HEAD_LEVEL", "DEFAULT_HIGH_HZ",
    "DEFAULT_LOW_HZ", "FILTER_CAUSAL", "FILTER_CHOICES",
    "FILTER_ZERO", "detect_attacks",
    "Detection", "head_levels", "Heads", "GridAlignment", "GridLayout",
    "kick_envelope", "kick_peaks_ms", "KickEnvelope", "MIN_ONSETS",
    "GridWarp", "make_warp", "DelayCandidates", "delay_candidates",
    "MethodReliability", "Reliability", "reliability_of",
]

if __name__ == "__main__":
    sys.exit(main())
