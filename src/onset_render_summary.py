from __future__ import annotations

import numpy as np

from onset_analysis import Analysis
from onset_bpm_estimate import BpmEstimate, ESTIMATE_MIN_DOMINANCE
from onset_common import wrap_symmetric
from onset_config import AnalysisConfig
from onset_decoder import DecodeInfo, DECODER_EXPLANATIONS, decoder_variants
from onset_format import signed
from onset_groups import GroupAnalysis
from onset_labels import LABEL_KICK, LABEL_OTHER
from onset_timing import TempoMap

DISPLAY_BIN_MS = 3.0
DISPLAY_WIDTH = 40


def describe(group: GroupAnalysis, period: float) -> list[str]:
    offsets = group.offsets_ms
    median = float(wrap_symmetric(np.median(offsets), period))
    q25, q75 = np.percentile(offsets, [25, 75])
    return [
        "",
        f"[{group.label}] n={offsets.size}",
        f"  主クラスタ中央値 {signed(group.phase_ms)} ms (MAD {group.spread_ms:.2f} ms, n={group.cluster_count})",
        f"  全体中央値(参考) {signed(median)} ms (四分位幅 {q75 - q25:.2f} ms)",
        f"  音源を遅らせる量 {signed(-group.phase_ms)} ms / グリッド開始オフセット {signed(group.phase_ms)} ms",
    ]


def render_histogram(group: GroupAnalysis, period: float) -> list[str]:
    half = period / 2.0
    edges = np.arange(
        group.center_ms - half, group.center_ms + half + DISPLAY_BIN_MS, DISPLAY_BIN_MS
    )
    counts, _ = np.histogram(group.offsets_ms, bins=edges)
    labels = wrap_symmetric(edges[:-1], period)
    peak = max(int(counts.max()), 1)
    return [
        f"  {label:+7.1f} | {'#' * round(DISPLAY_WIDTH * count / peak)} {count}"
        for label, count in zip(labels, counts)
    ]


def render_group_gap(groups: dict[str, GroupAnalysis], period: float) -> list[str]:
    other = groups.get(LABEL_OTHER)
    kick = groups.get(LABEL_KICK)
    if other is None or kick is None:
        return []
    gap = float(wrap_symmetric(kick.phase_ms - other.phase_ms, period))
    return [
        "",
        f"[キックとその他の位相差] キック同時 - 非同時 = {signed(gap)} ms",
        "  差が大きい場合は、どちらの立ち上がりに合わせるかを判断すること",
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


def render_conclusion(analysis: Analysis, is_main: bool) -> list[str]:
    basis = "回帰の曲中央時点の値" if analysis.uses_regression else "主クラスタ中央値"
    title = (
        f"[結論: {analysis.primary.label} / 根拠: {basis}]"
        if is_main
        else f"[参考: 全オンセットの混合値による結論 / 根拠: {basis}]"
    )
    return [
        "",
        title,
        f"  グリッド開始オフセット {signed(analysis.conclusion_ms)} ms",
        f"  音源を遅らせる量 {signed(-analysis.conclusion_ms)} ms",
    ]


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


def render_decoder_variants(
    decode: DecodeInfo | None, conclusion_ms: float, period: float
) -> list[str]:
    if decode is None:
        return []
    lines = ["", "[デコーダの遅延補正ごとの結果]", f"  {DECODER_EXPLANATIONS[decode.suffix]}"]
    for variant in decoder_variants(decode):
        shift_ms = (decode.applied_skip - variant.skip_samples) * 1000.0 / decode.sample_rate
        offset = float(wrap_symmetric(conclusion_ms + shift_ms, period))
        skipped_ms = variant.skip_samples * 1000.0 / decode.sample_rate
        marker = " <- [結論]と同じ(本スクリプトのデコード結果)" if variant.skip_samples == decode.applied_skip else ""
        lines += [
            f"  {variant.label}: 先頭 {variant.skip_samples} サンプル({skipped_ms:.2f} ms)を削除",
            f"    グリッド開始オフセット {signed(offset)} ms / 音源を遅らせる量 {signed(-offset)} ms{marker}",
        ]
    return lines


def render_header(
    config: AnalysisConfig, period: float, tempo: TempoMap | None
) -> list[str]:
    if tempo is None:
        return [
            f"BPM {config.bpm:g} / 1拍を{config.subdivision}分割 / グリッド間隔 {period:.3f} ms",
            "符号: 負 = オンセットがグリッド線より前",
            f"結果はグリッド間隔の剰余でのみ決まる(±{period:.3f} ms ずらしても一致)。拍頭の位置はDAWで確認すること",
        ]
    return [
        f"テンポマップ {len(tempo.points)} 区間を一括解析 / 基準BPM {config.bpm:g}(最長区間) / 1拍を{config.subdivision}分割 / 基準グリッド間隔 {period:.3f} ms",
        "各オンセットを、その時刻のテンポマップ上の最寄りグリッド線との残差(実時間 ms)に換算し、全区間をまとめて評価する",
        "時刻表示はグリッド換算時間で、BPMの異なる区間では実時間と一致しない",
        "符号: 負 = オンセットがグリッド線より前",
        "拍頭の位置はDAWで確認すること",
    ]
