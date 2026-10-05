from __future__ import annotations

from onset_analysis import Analysis
from onset_band_check import BandCheck
from onset_config import AnalysisConfig
from onset_cross_check import CHECK_AGREE_MS, CrossCheck
from onset_format import signed
from onset_phase import SE_CONFIDENCE_Z
from onset_refine_check import REFINE_RADIUS_MS, RefineCheck
from onset_window_phase import MIN_DRIFT_WINDOWS


def render_cross_check(cross: CrossCheck | None) -> list[str]:
    if cross is None:
        return []
    lines = [
        "",
        "[独立検出器による検算(粗い一致確認)]",
        f"  包絡線の頭から最大傾きまでの遅れ(中央値) {cross.rise_gap_ms:.2f} ms をアタック頭への換算に使用",
    ]
    for check in cross.checks:
        agreed = abs(check.difference_ms) <= CHECK_AGREE_MS
        verdict = "一致" if agreed else "乖離"
        basis = "回帰の曲中央時点" if check.uses_regression else "全体"
        lines.append(
            f"  {check.name}: 位相 {signed(check.phase_ms)} ms ({basis})"
            f" / 頭基準に換算 {signed(check.converted_ms)} ms"
            f" / 結論との差 {signed(check.difference_ms)} ms"
            f" / 鋭さ {check.contrast:.1f} -> {verdict}"
        )
    summary = "一致" if cross.all_agree else "乖離あり(検出ミス・頭の定義・グルーヴの揺れを確認)"
    lines += [
        f"  判定: {cross.agreed}/{len(cross.checks)} 検出器が ±{CHECK_AGREE_MS:g} ms 以内 -> {summary}",
        "  各検出器は窓長に由来する時間定義の差を含むため、数 ms 単位の一致確認であり、それ以下の精度の根拠にはならない",
    ]
    return lines


def render_band_check(bands: BandCheck | None, config: AnalysisConfig) -> list[str]:
    if bands is None:
        return []
    lines = [
        "",
        "[別帯域・別フィルタ方式・キック帯域での頭位相(全オンセットの主クラスタ中央値)]",
        f"  基準 {config.low_hz:g}~{config.high_hz:g} Hz: {signed(bands.reference_ms)} ms",
    ]
    lines += [
        f"  {band.label}: {signed(band.phase_ms)} ms (基準との差 {signed(band.difference_ms)} ms)"
        for band in bands.bands
    ]
    if bands.bands:
        lines.append(
            f"  帯域間の最大差 {bands.span_ms:.2f} ms / 帯域により立ち上がりの形が異なるため、数 ms 以内の差は通常の範囲"
        )
    if bands.ringing is not None:
        lines.append(
            f"  {bands.ringing.label}: {signed(bands.ringing.phase_ms)} ms (基準との差 {signed(bands.ringing.difference_ms)} ms)"
            " <- 零位相フィルタの前方リンギングによる早まりの目安。因果側は群遅延ぶん遅めに出るため、真値は両者の間"
        )
    if bands.kick is not None:
        lines.append(
            f"  {bands.kick.label}: {signed(bands.kick.phase_ms)} ms (基準との差 {signed(bands.kick.difference_ms)} ms)"
            " <- キック同時の分類窓に依存しない測定。低域は時間分解能が粗い"
        )
    return lines


def render_refine(refine: RefineCheck | None, analysis: Analysis) -> list[str]:
    if refine is None:
        return []
    error95 = SE_CONFIDENCE_Z * refine.standard_error_ms
    lines = [
        "",
        "[局所精密化: 因果フィルタ・複数帯域・サブサンプル補間でグリッド近傍のアタック頭を再測定]",
        f"  対象 {refine.count} 件(結論から ±{REFINE_RADIUS_MS:g} ms 以内) / MAD {refine.spread_ms:.2f} ms / 統計誤差(95%) ±{error95:.2f} ms",
        f"  結論 {signed(analysis.final_ms)} ms -> 精密化後 {signed(refine.offset_ms)} ms (差 {signed(refine.shift_ms)} ms)",
        f"  精密化後の値で合わせる場合に音源を遅らせる量: {signed(0.0 - refine.offset_ms)} ms",
    ]
    lines += [
        f"  {band.low_hz:g}~{band.high_hz:g} Hz: 差 {signed(band.shift_ms)} ms (n={band.count})"
        for band in refine.bands
    ]
    lines.append(
        "  因果フィルタの群遅延と平滑の遅れは補正済み。残る差は帯域・フィルタ方式・平滑による立ち上がり形状の定義違い"
    )
    return lines


def render_precision(analysis: Analysis, bands: BandCheck | None) -> list[str]:
    error95 = SE_CONFIDENCE_Z * analysis.final_standard_error_ms
    lines = [
        "",
        "[推定精度の目安]",
        f"  統計誤差(95%) ±{error95:.2f} ms (イベント単位と時間窓単位の大きいほう。時間窓が{MIN_DRIFT_WINDOWS}未満ならイベント単位のみ)",
    ]
    if analysis.cross is not None:
        lines.append(f"  独立検出器間の最大乖離 {analysis.cross.span_ms:.2f} ms")
    if bands is not None:
        lines.append(f"  帯域間の最大差 {bands.span_ms:.2f} ms")
        if bands.ringing is not None:
            lines.append(f"  フィルタ方式による差 {abs(bands.ringing.difference_ms):.2f} ms")
    lines.append(
        "  表示は0.01 ms単位だが測定精度ではない。上記の乖離や head-level 差が数 ms ある場合は、そちらが精度の上限になる"
    )
    return lines
