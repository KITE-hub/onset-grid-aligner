from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onset_analysis import Analysis, analyze
from onset_common import phase_to_delay_ms
from onset_config import AnalysisConfig
from onset_detection import Detection, Heads
from onset_grid import align_grid, GridAlignment
from onset_kick import kick_coincident_mask
from onset_odf import OnsetFunction
from onset_phase import MIN_ONSETS
from onset_warp import GridWarp, IDENTITY_WARP, warped_detection, warped_functions

KICK_BEAT_WEIGHT = 1.0


@dataclass(frozen=True)
class BeatAnalysis:
    detection: Detection
    analysis: Analysis
    alignment: GridAlignment | None

    def beat_offset_ms(self, phase_ms: float) -> float | None:
        if self.alignment is None or not self.alignment.beat_decided:
            return None
        return self.alignment.beat_offset_ms(phase_ms)

    @property
    def offset_ms(self) -> float | None:
        return None if self.alignment is None else self.alignment.phase_ms

    @property
    def delay_ms(self) -> float | None:
        return None if self.offset_ms is None else phase_to_delay_ms(self.offset_ms)


def beat_weights(heads: Heads, kick_ms: np.ndarray) -> np.ndarray:
    kicks = kick_coincident_mask(heads.heads_ms, kick_ms)
    return heads.strengths + KICK_BEAT_WEIGHT * kicks


def align_beat(
    detection: Detection, analysis: Analysis, config: AnalysisConfig
) -> GridAlignment | None:
    selected = detection.heads[config.head_level]
    if selected.heads_ms.size < MIN_ONSETS:
        return None
    times_ms = selected.heads_ms
    weights = np.maximum(selected.strengths, 0.0)
    diagnostic_weights = beat_weights(selected, detection.kick_ms)
    return align_grid(
        times_ms,
        weights,
        analysis.conclusion_ms,
        config.layout,
        diagnostic_weights,
        analysis.grid_phase,
    )


def analyze_beat(
    detection: Detection,
    config: AnalysisConfig,
    functions: tuple[OnsetFunction, ...],
    warp: GridWarp = IDENTITY_WARP,
) -> BeatAnalysis:
    warped = warped_detection(detection, warp)
    functions = warped_functions(functions, warp)
    analysis = analyze(warped, config, functions)
    alignment = align_beat(warped, analysis, config)
    return BeatAnalysis(detection, analysis, alignment)
