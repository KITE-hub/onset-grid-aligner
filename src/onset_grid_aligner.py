from __future__ import annotations

import sys

from onset_analysis import Analysis, analyze
from onset_audio_io import decode_with_ffmpeg, load_audio, readable_audio, validate_audio
from onset_band_check import BandCheck, band_check
from onset_cli import main
from onset_config import (
    AnalysisConfig,
    DEFAULT_DRIFT_WINDOW_SEC,
    DEFAULT_HEAD_LEVEL,
    DEFAULT_HIGH_HZ,
    DEFAULT_LOW_HZ,
    DEFAULT_REFERENCE_GROUP,
    FILTER_CAUSAL,
    FILTER_CHOICES,
    FILTER_ZERO,
    REFERENCE_CHOICES,
    SUBDIVISIONS,
)
from onset_cross_check import CrossCheck, MIXTURE_AGREE_MS
from onset_detection import detect_attacks, Detection, head_levels, Heads, LocalCap
from onset_kick import attach_kicks, kick_envelope, KickEnvelope
from onset_phase import is_stable_value, MIN_ONSETS, SE_CONFIDENCE_Z, screening_fit
from onset_reference import ReferenceEstimate
from onset_refine_check import refine_check, RefineCheck
from onset_warp import (
    GridWarp,
    make_warp,
    resolve_tempo,
    warped_detection,
    warped_functions,
)

if __name__ == "__main__":
    sys.exit(main())
