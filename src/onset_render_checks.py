from __future__ import annotations

from onset_analysis import Analysis
from onset_band_check import BandCheck
from onset_config import AnalysisConfig
from onset_cross_check import CHECK_AGREE_MS, CrossCheck
from onset_format import signed
from onset_phase import SE_CONFIDENCE_Z
from onset_pipeline import AnalysisRun
from onset_refine_check import REFINE_RADIUS_MS, RefineCheck
from onset_window_phase import MIN_DRIFT_WINDOWS


def render_cross_check(cross: CrossCheck | None) -> list[str]:
    if cross is None:
        return ["独立検出器による検算: 判定不可（有効な検算データなし）"]
    descriptions = {
        "SuperFlux風": "スペクトル差分",
        "複素領域": "複素スペクトル予測誤差",
        "HFC": "高域成分の増加量",
    }
    lines = [
        "独立検出器による検算:",
        f"  共通補正: 検出器時刻からアタック頭への換算 {signed(-cross.rise_gap_ms)} ms",
    ]
    for check in cross.checks:
        agreed = abs(check.difference_ms) <= CHECK_AGREE_MS
        verdict = "一致" if agreed else "不一致"
        method = descriptions.get(check.name, "独立オンセット検出")
        lines.append(
            f"  {check.name}（{method}）: 生オフセット {signed(check.phase_ms)} ms"
            f" -> 頭補正後 {signed(check.converted_ms)} ms"
            f" / 主推定との差 {signed(check.difference_ms)} ms"
            f" / 候補優位差 {check.contrast:.0%} / {verdict}"
        )
    summary = "全手法一致" if cross.all_agree else "手法間に差あり"
    lines.append(
        f"  検算結果: {cross.agreed}/{len(cross.checks)} 手法が ±{CHECK_AGREE_MS:g} ms以内"
        f" / 最大乖離 {cross.span_ms:.2f} ms / {summary}"
    )
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


def render_refine(refine: RefineCheck | None, run: AnalysisRun) -> list[str]:
    analysis = run.analysis
    if refine is None:
        return []
    error95 = SE_CONFIDENCE_Z * refine.standard_error_ms
    lines = [
        "",
        "[局所精密化: 因果フィルタ・複数帯域・サブサンプル補間でグリッド近傍のアタック頭を再測定]",
        f"  対象 {refine.count} 件(既存周期位相から ±{REFINE_RADIUS_MS:g} ms 以内) / MAD {refine.spread_ms:.2f} ms / 統計誤差(95%) ±{error95:.2f} ms",
        f"  既存のオンセット周期位相 {signed(analysis.conclusion_ms)} ms -> 精密化後 {signed(refine.offset_ms)} ms (差 {signed(refine.shift_ms)} ms)",
    ]
    lines.append(
        f"  精密化後の既存周期位相に合わせる参考移動量: {signed(0.0 - refine.offset_ms)} ms"
    )
    lines += [
        f"  {band.low_hz:g}~{band.high_hz:g} Hz: 差 {signed(band.shift_ms)} ms (n={band.count})"
        for band in refine.bands
    ]
    lines.append(
        "  因果フィルタの群遅延と平滑の遅れは補正済み。残る差は帯域・フィルタ方式・平滑による立ち上がり形状の定義違い"
    )
    return lines


def render_precision(analysis: Analysis, bands: BandCheck | None) -> list[str]:
    error95 = SE_CONFIDENCE_Z * analysis.uncertainty_ms
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
