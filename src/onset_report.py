from __future__ import annotations

import numpy as np

from onset_bpm_estimate import ESTIMATE_MIN_DOMINANCE
from onset_decoder import DECODER_EXPLANATIONS, decoder_variants
from onset_format import signed
from onset_grid import MIN_PHASE_MARGIN, comb_residuals
from onset_pipeline import AnalysisRun
from onset_render_summary import render_delay_candidates


def render_grid_diagnostics(run: AnalysisRun) -> list[str]:
    times = np.asarray(run.analysis.primary.heads_ms, dtype=float)
    alignment = run.alignment
    if times.size == 0:
        return [
            "検出アタック数: 0件",
            "固定12位置候補に対する残差中央値: 算出不可",
            "固定12位置候補に対する残差MAD: 算出不可",
            "固定12位置候補に対する残差P95: 算出不可",
            "固定線の許容幅・範囲内件数: 算出不可",
            "位相候補の一意性: 判定不可",
        ]
    offset = None if alignment is None else alignment.phase_ms
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
        uniqueness_text = "判定不可"
    else:
        verdict = "明瞭" if alignment.phase_margin >= MIN_PHASE_MARGIN else "曖昧"
        uniqueness_text = f"{verdict} (次点に対する優位差 {alignment.phase_margin:.1%})"
    return [
        f"検出アタック数: {times.size}件",
        f"固定12位置候補に対する残差中央値: {signed(residual_median)} ms" if residual_median is not None else "固定12位置候補に対する残差中央値: 算出不可",
        f"固定12位置候補に対する残差MAD: {residual_mad:.2f} ms" if residual_mad is not None else "固定12位置候補に対する残差MAD: 算出不可",
        f"固定12位置候補に対する残差P95: {residual_p95:.2f} ms" if residual_p95 is not None else "固定12位置候補に対する残差P95: 算出不可",
        f"固定線の許容幅・範囲内件数: {fit_text}",
        f"位相候補の一意性: {uniqueness_text}",
    ]


def render_decoder_variants(run: AnalysisRun) -> list[str]:
    decode = run.decode
    if decode is None:
        return []
    fixed_grid = dict(run.resolution.candidates).get("fixed_grid")
    lines = [
        "[参考: デコーダ遅延の仮定別の補正量(上の補正量を上書きしない)]",
        f"  {DECODER_EXPLANATIONS[decode.suffix]}",
    ]
    for variant in decoder_variants(decode):
        shift_ms = (
            (variant.skip_samples - decode.applied_skip) * 1000.0 / decode.sample_rate
        )
        reference = (
            "" if fixed_grid is None else f" / fixed_grid {signed(fixed_grid + shift_ms)} ms"
        )
        marker = " <- 現在のデコード補正" if variant.skip_samples == decode.applied_skip else ""
        lines.append(
            f"  {variant.label}: 先頭 {variant.skip_samples} サンプルを削除"
            f" / 全候補に {signed(shift_ms)} ms を加算{reference}{marker}"
        )
    return lines


def render_compact_report(run: AnalysisRun) -> list[str]:
    lines = render_delay_candidates(run.resolution, run)
    alignment = run.alignment
    if alignment is None:
        lines.append("固定12位置候補の診断: 算出不可")
    else:
        uniqueness = "明瞭" if alignment.phase_margin >= MIN_PHASE_MARGIN else "曖昧"
        lines.append(
            f"固定12位置候補の診断: 適合率 {alignment.comb_fit:.0%}"
            f" / 残差MAD {alignment.comb_mad_ms:.2f} ms"
            f" / グリッド残差の標準誤差換算値 ±{alignment.comb_se95_ms:.2f} ms (95%係数、方式間差・候補曖昧さを含まない)"
            f" / 位相候補 {uniqueness}"
        )
    lines.extend(render_grid_diagnostics(run))
    lines.extend(render_decoder_variants(run))
    if (
        run.estimate is not None
        and run.estimate.dominance is not None
        and run.estimate.dominance < ESTIMATE_MIN_DOMINANCE
    ):
        lines.append("注意: BPM自動推定が不明瞭です。必要なら --bpm で指定してください")
    return lines
