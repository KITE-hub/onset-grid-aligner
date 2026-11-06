from __future__ import annotations

from onset_bpm_estimate import ESTIMATE_MIN_DOMINANCE
from onset_decoder import DECODER_EXPLANATIONS, decoder_variants
from onset_format import signed
from onset_pipeline import AnalysisRun
from onset_reliability import MethodReliability, Reliability, reliability_of
from onset_render_summary import render_delay_candidates


def render_method_reliability(method: MethodReliability) -> str:
    support = method.support
    mad = support.support_mad_ms
    ratio = support.support_ratio
    stability = method.stability
    mad_text = "算出不可" if mad is None else f"{mad:.2f} ms"
    ratio_text = "算出不可" if ratio is None else f"{ratio:.0%}"
    stability_text = "算出不可" if stability is None else f"{stability:.0%}"
    return (
        f"  {method.name}: 根拠アタック内の残差MAD {mad_text}"
        f" / 根拠アタック割合 {ratio_text}"
        f" / 検出アタック数 {support.attack_count}件"
        f" / 窓単位の安定性 {stability_text}"
    )


def render_reliability(reliability: Reliability) -> list[str]:
    lines = ["[信頼性の目安: 方式ごと]"]
    lines.extend(render_method_reliability(method) for method in reliability.methods)
    if reliability.drift_slope_ms_per_min is not None:
        lines.append(
            f"ドリフト補正: 適用 (傾き {signed(reliability.drift_slope_ms_per_min)} ms/分、"
            "各候補は音源の中央時点に換算)"
        )
    return lines


def render_decoder_variants(run: AnalysisRun) -> list[str]:
    decode = run.decode
    if decode is None:
        return []
    fixed_grid = dict(run.candidates).get("fixed_grid")
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
    reliability = reliability_of(run.analysis, run.alignment, run.config.layout)
    shifts = {method.name: method.branch_shift_ms for method in reliability.methods}
    lines = render_delay_candidates(run.candidates, shifts)
    lines.extend(render_reliability(reliability))
    lines.extend(render_decoder_variants(run))
    if (
        run.estimate is not None
        and run.estimate.dominance is not None
        and run.estimate.dominance < ESTIMATE_MIN_DOMINANCE
    ):
        lines.append("注意: BPM自動推定が不明瞭です。必要なら --bpm で指定してください")
    return lines
