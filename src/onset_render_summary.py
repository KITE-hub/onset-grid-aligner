from __future__ import annotations

import numpy as np

from onset_analysis import Analysis
from onset_bpm_estimate import BpmEstimate, ESTIMATE_MIN_DOMINANCE
from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_decoder import DECODER_EXPLANATIONS, decoder_variants
from onset_format import signed
from onset_grid import COMB_STEPS, GRID_LATTICES, estimate_comb_phase
from onset_groups import GroupAnalysis
from onset_pipeline import AnalysisRun
from onset_timing import TempoMap


def describe(group: GroupAnalysis, period: float) -> list[str]:
    offsets = group.offsets_ms
    median = float(wrap_symmetric(np.median(offsets), period))
    q25, q75 = np.percentile(offsets, [25, 75])
    return [
        "",
        f"[{group.label}] n={offsets.size}",
        f"  主クラスタ中央値 {signed(group.phase_ms)} ms (MAD {group.spread_ms:.2f} ms, n={group.cluster_count})",
        f"  全体中央値(参考) {signed(median)} ms (四分位幅 {q75 - q25:.2f} ms)",
    ]


def render_sensitivity(
    sensitivity: tuple[tuple[float, float | None], ...], period: float
) -> list[str]:
    valid = [(level, phase) for level, phase in sensitivity if phase is not None]
    if not valid:
        return []
    phases = np.asarray([phase for _, phase in valid])
    relative = wrap_symmetric(phases - phases[0], period)
    lines = ["", "[定義感度: head-level(頭の定義の影響。検出器の頑健性の検証ではない) / 主クラスタ中央値]"]
    lines += [f"  head-level {level:.2f}: {signed(phase)} ms" for level, phase in valid]
    lines.append(f"  最大差 {relative.max() - relative.min():.2f} ms")
    return lines


def render_estimate(estimate: BpmEstimate | None) -> list[str]:
    if estimate is None:
        return []
    if estimate.dominance is None:
        verdict = "比較対象がなく判定不可"
    elif estimate.dominance >= ESTIMATE_MIN_DOMINANCE:
        verdict = f"次点の {estimate.dominance:.2f} 倍 -> 明瞭"
    else:
        verdict = f"次点の {estimate.dominance:.2f} 倍 -> 不明瞭(--bpm での指定を推奨)"
    lines = [f"BPM自動推定: {estimate.bpm:.2f} ({verdict})"]
    if estimate.octaves:
        octaves = " / ".join(
            f"{bpm:g} (相対強度 {ratio:.2f})" for bpm, ratio in estimate.octaves
        )
        lines.append(f"オクターブ違いの候補: {octaves} / 拍の数え方が違う場合は --bpm で指定")
    return lines


def render_decoder_variants(run: AnalysisRun) -> list[str]:
    decode = run.decode
    if decode is None:
        return []
    lines = ["", "[参考診断: デコーダ遅延の仮定別（上部の補正量を上書きしない）]", f"  {DECODER_EXPLANATIONS[decode.suffix]}"]
    selected = run.beat.detection.heads[run.config.head_level]
    base_times_ms = selected.heads_ms
    weights = np.maximum(selected.strengths, 0.0)
    for variant in decoder_variants(decode):
        shift_ms = (decode.applied_skip - variant.skip_samples) * 1000.0 / decode.sample_rate
        estimate = estimate_comb_phase(base_times_ms + shift_ms, weights, run.config.layout)
        skipped_ms = variant.skip_samples * 1000.0 / decode.sample_rate
        marker = " <- 現在のデコード補正" if variant.skip_samples == decode.applied_skip else ""
        lines.append(f"  {variant.label}: 先頭 {variant.skip_samples} サンプル({skipped_ms:.2f} ms)を削除")
        if estimate is None:
            lines.append("    固定12位置の補正量を推定できません")
        else:
            lines.append(
                f"    固定12位置の位相 {signed(estimate.phase_ms)} ms / この仮定での参考移動量 {signed(-estimate.phase_ms)} ms{marker}"
            )
    return lines


def grid_lines(config: AnalysisConfig) -> list[str]:
    layout = config.layout
    diagnostic_fold = (
        "拍頭候補の参考診断にはBPMを折り返さず使用"
        if layout.octave == 0
        else f"拍頭候補の参考診断のみBPMを×{2.0 ** layout.octave:g}で{layout.grid_bpm:.2f} BPMに折返し"
    )
    lattices = ",".join(str(value) for value in GRID_LATTICES)
    steps = ",".join(str(step) for step in COMB_STEPS)
    return [
        f"BPM {config.bpm:g} / 4分音符1拍={layout.comb_beat_ms:.3f} ms, 24ステップ幅={layout.comb_lattice_ms:.3f} ms"
        f" / 固定線: {lattices}分割の合併(1拍24ステップ中{len(COMB_STEPS)}本: {steps})",
        "符号: 負 = アタックが最寄りの固定線より前。音源移動量はこの符号と逆",
        "グリッド開始オフセットは、全アタックに共通する位相シフトを固定12位置への全体適合から推定。個別の最寄り線残差を先に中央値化しない。拍頭候補とVOXの拍子記号は算出に使わない。",
        f"{diagnostic_fold}。7/8や9/16の小節頭を判定するものではなく、固定12位置のパターンも変更しない。",
    ]


def render_header(config: AnalysisConfig, tempo: TempoMap | None) -> list[str]:
    if tempo is None:
        return grid_lines(config)
    return [
        f"テンポマップ {len(tempo.points)} 区間を一括解析 / 基準BPM {config.bpm:g}(最長区間)",
        *grid_lines(config),
        "VOXのテンポマップに合わせて時間軸を正規化し、正規化後の各アタックを基準BPMの固定12位置と照合する",
        "判定は固定12位置までの時間差に基づき、拍子記号(7/8、9/16など)は補正位相の式に混ぜない",
    ]


def render_recommendation(run: AnalysisRun) -> list[str]:
    lines = ["", "[補正量]"]
    if run.offset_ms is None or run.delay_ms is None:
        lines.extend(["  グリッド開始オフセット: 算出不可", "  音源を遅らせる量: 算出不可"])
        return lines
    lines.extend(
        [
            f"  グリッド開始オフセット: {signed(run.offset_ms)} ms",
            f"  音源を遅らせる量: {signed(run.delay_ms)} ms (負なら前にずらす)",
            "  計算: 全アタックに共通する位相シフトを固定12位置への全体適合から推定。音源移動量は逆符号。",
        ]
    )
    return lines


def render_alignment(run: AnalysisRun) -> list[str]:
    alignment = run.alignment
    lines = [
        "",
        "[参考診断・補正に不使用: BPM拍頭候補]",
    ]
    if alignment is None:
        lines.append("  判定できませんでした。拍頭候補と固定12位置の適合度を診断するためのオンセットが不足しています")
        return lines
    verdict = "明瞭" if alignment.beat_decided else "未確定"
    lines.append(
        f"  判定: {verdict} (マージン {alignment.beat_margin:.2f}) / 暫定最良候補 {alignment.shift_steps:+d} ステップ"
    )
    lines.append(
        f"  次点の拍頭候補オフセット（参考値）: {signed(alignment.runner_up_ms)} ms"
    )
    if not alignment.beat_decided:
        candidates = alignment.candidate_offsets_ms
        if candidates:
            offsets = ", ".join(f"{signed(value)}" for value in candidates)
            delays = ", ".join(f"{signed(-value)}" for value in candidates)
            lines.append(f"  拍頭候補オフセット（参考値）: {offsets} ms")
            lines.append(f"  拍頭候補から換算した移動量（参考値・採用しない）: {delays} ms")
        else:
            lines.append("  BPM拍頭候補を絞れませんでした")
    lines.extend(
        [
            "  これらは4分音符単位の拍頭候補であり、記譜上の小節頭や分母音符の拍頭を示しません。音源補正には「音源を遅らせる量」を使用してください。",
            "",
            "[適合度の診断: 固定12位置]",
            f"  固定12位置への適合率: {alignment.comb_fit:.1%}",
            f"  残差MAD: {alignment.comb_mad_ms:.2f} ms / 対象アタック: {alignment.comb_count} 件",
            f"  統計誤差の目安 (95%): ±{alignment.comb_se95_ms:.2f} ms",
        ]
    )
    return lines
