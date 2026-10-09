from __future__ import annotations

from onset_analysis import Analysis
from onset_band_check import BandCheck
from onset_bpm_estimate import BpmEstimate
from onset_config import AnalysisConfig
from onset_decoder import DecodeInfo
from onset_refine_check import RefineCheck
from onset_render_checks import (
    render_band_check,
    render_cross_check,
    render_precision,
    render_refine,
)
from onset_render_drift import render_drift, render_reference, render_stability
from onset_render_summary import (
    describe,
    render_conclusion,
    render_decoder_variants,
    render_estimate,
    render_group_gap,
    render_header,
    render_sensitivity,
)
from onset_timing import TempoMap


def render_report(
    audio_name: str,
    config: AnalysisConfig,
    analysis: Analysis,
    estimate: BpmEstimate | None,
    decode: DecodeInfo | None,
    bands: BandCheck | None,
    refine: RefineCheck | None,
    tempo: TempoMap | None = None,
) -> list[str]:
    period = analysis.period_ms
    lines = [
        f"ファイル: {audio_name}",
        *render_estimate(estimate),
        *render_header(config, period, tempo),
    ]
    for group in analysis.groups.values():
        lines += describe(group, period)
    lines += render_group_gap(analysis.groups, period)
    lines += render_sensitivity(analysis.sensitivity, period)
    lines += render_drift(analysis, config.bpm)
    lines += render_cross_check(analysis.cross)
    lines += render_band_check(bands, config)
    lines += render_refine(refine, analysis)
    if analysis.reference is None:
        lines += render_conclusion(analysis, True)
        lines += render_stability(analysis.stability, analysis.grid_fit)
    else:
        lines += render_reference(analysis, config)
        lines += render_conclusion(analysis, False)
    lines += render_precision(analysis, bands)
    lines += render_decoder_variants(decode, analysis.final_ms, period)
    return lines
