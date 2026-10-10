from __future__ import annotations

from typing import Callable

import numpy as np

from onset_config import AnalysisConfig, SUBDIVISIONS
from onset_detection import Detection
from onset_phase import MIN_ONSETS, screening_fit
from onset_timing import TempoMap
from onset_warp import make_warp

SUBDIVISION_SELECTION_TOLERANCE = 0.05
FOLD_MIN_BPM = 150.0


def fold_bpm(bpm: float) -> float:
    return float(bpm * 2.0 ** np.ceil(np.log2(FOLD_MIN_BPM / bpm)))


def subdivision_scores(
    attacks: Detection,
    tempo: TempoMap | None,
    make_config: Callable[[int], AnalysisConfig],
    candidates: tuple[int, ...] = SUBDIVISIONS,
) -> dict[int, float]:
    scores = {}
    for subdivision in candidates:
        config = make_config(subdivision)
        warp = make_warp(tempo, config, attacks)
        peaks_ms = warp.heads(attacks.heads[config.head_level]).peaks_ms
        if peaks_ms.size >= MIN_ONSETS:
            scores[subdivision] = screening_fit(peaks_ms, config.period_ms)
    return scores


def choose_subdivision(scores: dict[int, float]) -> int:
    if not scores:
        raise ValueError("分割数を選べるだけのオンセットがありません")
    best = max(scores.values())
    return next(
        subdivision
        for subdivision in SUBDIVISIONS
        if scores.get(subdivision, -1.0) >= best - SUBDIVISION_SELECTION_TOLERANCE
    )
