from __future__ import annotations

from onset_pipeline import AnalysisRun
from onset_render_checks import (
    render_band_check,
    render_cross_check,
    render_precision,
    render_refine,
)
from onset_render_drift import render_drift, render_stability
from onset_render_summary import (
    describe,
    render_alignment,
    render_decoder_variants,
    render_estimate,
    render_header,
    render_recommendation,
    render_sensitivity,
)
from onset_bpm_estimate import ESTIMATE_MIN_DOMINANCE
import numpy as np

from onset_format import signed
from onset_grid import comb_residuals
from onset_timing import TempoMap


def render_report(
    audio_name: str, run: AnalysisRun, tempo: TempoMap | None = None
) -> list[str]:
    analysis = run.analysis
    config = run.config
    period = analysis.period_ms
    lines = [
        f"ファイル: {audio_name}",
        *render_estimate(run.estimate),
        *render_header(config, tempo),
        *render_recommendation(run),
        *render_alignment(run),
    ]
    for group in analysis.groups.values():
        lines += describe(group, period)
    lines += render_sensitivity(analysis.sensitivity, period)
    lines += render_drift(run)
    lines += render_cross_check(analysis.cross)
    lines += render_band_check(run.bands, config)
    lines += render_refine(run.refine, run)
    lines += render_stability(analysis.stability, analysis.grid_fit)
    lines += render_precision(analysis, run.bands)
    lines += render_decoder_variants(run)
    return lines


def render_grid_diagnostics(run: AnalysisRun) -> list[str]:
    times = np.asarray(run.analysis.primary.heads_ms, dtype=float)
    alignment = run.alignment
    if times.size == 0:
        return [
            "検出アタック数: 0件",
            "アタック検出範囲: 算出不可",
            "アタック間隔中央値: 算出不可",
            "アタック間隔MAD: 算出不可",
            "補正後残差中央値: 算出不可",
            "補正後残差MAD: 算出不可",
            "補正後残差P95: 算出不可",
            "固定線の許容幅・範囲内件数: 算出不可",
            "次点のグローバル位相候補: 算出不可",
            "位相候補の一意性: 判定不可",
        ]
    intervals = np.diff(times)
    interval_median = float(np.median(intervals)) if intervals.size else None
    interval_mad = (
        float(np.median(np.abs(intervals - interval_median)))
        if interval_median is not None
        else None
    )
    offset = run.offset_ms
    if offset is None:
        residuals = np.empty(0, dtype=float)
        residual_median = residual_mad = residual_p95 = None
    else:
        residuals = comb_residuals(times - offset, run.config.layout)
        residual_median = float(np.median(residuals))
        residual_mad = float(np.median(np.abs(residuals - residual_median)))
        residual_p95 = float(np.percentile(np.abs(residuals), 95))
    tolerance = min(2.0, run.config.layout.comb_lattice_ms * 0.15)
    if residuals.size:
        within = int(np.count_nonzero(np.abs(residuals) <= tolerance))
        fit_text = f"±{tolerance:.2f} ms / {within}/{residuals.size}件"
    else:
        fit_text = "算出不可"
    if alignment is None or alignment.runner_up_phase_ms is None:
        runner_text = "なし"
        uniqueness_text = "判定不可"
    else:
        runner = alignment.runner_up_phase_ms
        beat_ms = run.config.layout.comb_beat_ms
        separation = abs(float((runner - alignment.phase_ms + beat_ms / 2.0) % beat_ms - beat_ms / 2.0))
        runner_text = f"{signed(runner)} ms (主候補との差 {separation:.2f} ms)"
        verdict = "明瞭" if alignment.phase_margin >= 0.05 else "曖昧"
        uniqueness_text = f"{verdict} (次点に対する優位差 {alignment.phase_margin:.1%})"
    return [
        f"検出アタック数: {times.size}件",
        f"アタック検出範囲: {times[0] / 1000.0:.3f}–{times[-1] / 1000.0:.3f} s ({(times[-1] - times[0]) / 1000.0:.3f} s)",
        f"アタック間隔中央値: {interval_median:.2f} ms" if interval_median is not None else "アタック間隔中央値: 算出不可",
        f"アタック間隔MAD: {interval_mad:.2f} ms" if interval_mad is not None else "アタック間隔MAD: 算出不可",
        f"補正後残差中央値: {signed(residual_median)} ms" if residual_median is not None else "補正後残差中央値: 算出不可",
        f"補正後残差MAD: {residual_mad:.2f} ms" if residual_mad is not None else "補正後残差MAD: 算出不可",
        f"補正後残差P95: {residual_p95:.2f} ms" if residual_p95 is not None else "補正後残差P95: 算出不可",
        f"固定線の許容幅・範囲内件数: {fit_text}",
        f"次点のグローバル位相候補: {runner_text}",
        f"位相候補の一意性: {uniqueness_text}",
    ]


def render_compact_report(run: AnalysisRun) -> list[str]:
    lines: list[str] = []
    alignment = run.alignment
    if run.offset_ms is None or run.delay_ms is None:
        lines.extend(
            [
                "グリッド開始オフセット: 算出不可",
                "音源を遅らせる量: 算出不可",
            ]
        )
    else:
        lines.extend(
            [
                f"グリッド開始オフセット: {signed(run.offset_ms)} ms",
                f"音源を遅らせる量: {signed(run.delay_ms)} ms (負なら前にずらす)",
            ]
        )
    if alignment is None:
        lines.append("信頼度: 算出不可")
    else:
        uniqueness = "明瞭" if alignment.phase_margin >= 0.05 else "曖昧"
        lines.append(
            f"信頼度: 固定12位置適合率 {alignment.comb_fit:.0%}"
            f" / 残差MAD {alignment.comb_mad_ms:.2f} ms"
            f" / 誤差目安 ±{alignment.comb_se95_ms:.2f} ms (95%)"
            f" / 位相候補 {uniqueness}"
        )
    lines.extend(render_cross_check(run.analysis.cross))
    lines.extend(render_grid_diagnostics(run))
    if (
        run.estimate is not None
        and run.estimate.dominance is not None
        and run.estimate.dominance < ESTIMATE_MIN_DOMINANCE
    ):
        lines.append("注意: BPM自動推定が不明瞭です。必要なら --bpm で指定してください")
    return lines
