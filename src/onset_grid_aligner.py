from __future__ import annotations

import sys

from onset_analysis import Analysis, analyze
from onset_audio_io import decode_with_ffmpeg, load_audio, readable_audio, validate_audio
from onset_band_check import BandCheck, band_check
from onset_beat import analyze_beat, BeatAnalysis
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
from onset_cross_check import CrossCheck
from onset_detection import detect_attacks, Detection, head_levels, Heads
from onset_grid import GridAlignment, GridLayout
from onset_kick import attach_kicks, kick_envelope, KickEnvelope
from onset_phase import is_stable_value, MIN_ONSETS, SE_CONFIDENCE_Z
from onset_refine_check import refine_check, RefineCheck
from onset_warp import GridWarp, make_warp

__all__ = [
    "Analysis", "analyze", "decode_with_ffmpeg", "load_audio", "readable_audio",
    "validate_audio", "BandCheck", "band_check", "analyze_beat", "BeatAnalysis",
    "AnalysisConfig", "DEFAULT_DRIFT_WINDOW_SEC", "DEFAULT_HEAD_LEVEL", "DEFAULT_HIGH_HZ",
    "DEFAULT_LOW_HZ", "FILTER_CAUSAL", "FILTER_CHOICES",
    "FILTER_ZERO", "CrossCheck", "detect_attacks",
    "Detection", "head_levels", "Heads", "GridAlignment", "GridLayout", "attach_kicks",
    "kick_envelope", "KickEnvelope", "is_stable_value", "MIN_ONSETS", "SE_CONFIDENCE_Z",
    "refine_check", "RefineCheck", "GridWarp", "make_warp",
]

if __name__ == "__main__":
    sys.exit(main())
