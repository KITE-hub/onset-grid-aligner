from __future__ import annotations

from onset_analysis import Analysis
from onset_common import wrap_symmetric
from onset_config import (
    AnalysisConfig,
    REFERENCE_FAST,
    REFERENCE_LARGEST,
    REFERENCE_MIXED,
    REFERENCE_SLOW,
)
from onset_drift import DriftFit
from onset_format import signed
from onset_labels import LABEL_REFERENCE
from onset_phase import is_stable_value, MIN_GRID_FIT, STABLE_WINDOW_RATIO
from onset_window_phase import WindowPhases

REFERENCE_DESCRIPTIONS = {
    REFERENCE_LARGEST: "件数が最大の群",
    REFERENCE_FAST: "立ち上がりが最も速い群",
    REFERENCE_SLOW: "立ち上がりが最も遅い群",
    REFERENCE_MIXED: "群に分けず全オンセットの混合値",
}


def render_fit(fit: DriftFit, bpm: float, period: float, end_ms: float) -> list[str]:
    tempo = bpm * (1.0 - fit.slope)
    tempo_low = bpm * (1.0 - fit.slope_high)
    tempo_high = bpm * (1.0 - fit.slope_low)
    verdict = "有意なテンポずれあり" if fit.significant else "テンポずれは有意でない"
    center_phase = float(wrap_symmetric(fit.phase_at(fit.reference_ms), period))
    start_phase = float(wrap_symmetric(fit.phase_ms, period))
    start_gap = fit.slope * (0.0 - fit.reference_ms)
    end_gap = fit.slope * (end_ms - fit.reference_ms)
    lines = [
        f"  回帰[{fit.method}]: 傾き {fit.slope * 60000.0:+.2f} ms/分 -> 推定BPM {tempo:.4f} (95%区間 {tempo_low:.4f} ~ {tempo_high:.4f}) / {verdict}",
        f"  曲中央({fit.reference_ms / 1000.0:.1f}s)での位相 {signed(center_phase)} ms",
        f"  曲頭(0 ms)での位相(切片) {signed(start_phase)} ms",
        f"  BPM {bpm:g} 固定で曲中央に揃えた場合のズレ: 曲頭 {signed(start_gap)} ms / 終端({end_ms / 1000.0:.0f}s) {signed(end_gap)} ms",
    ]
    if fit.significant:
        lines.append(
            f"  推定BPM {tempo:.4f} を小数で指定できる場合は、曲頭の位相(切片) {signed(start_phase)} ms を採用する"
        )
    return lines


def window_lines(windows: WindowPhases) -> list[str]:
    return [
        f"  {start / 1000.0:6.0f}s~ {signed(phase):>8} ms (n={count})"
        for start, phase, count in zip(
            windows.starts_ms, windows.phases_ms, windows.counts
        )
    ]


def render_drift(analysis: Analysis, bpm: float) -> list[str]:
    windows = analysis.windows
    if windows is None:
        return []
    lines = ["", f"[時間窓ごとの位相(連続化済み): {analysis.primary.label}]"]
    lines += window_lines(windows)
    if analysis.drift is None:
        lines.append("  回帰は不採用(窓数不足、または窓ごとの位相が一貫しないため)")
    else:
        end_ms = float(analysis.primary.heads_ms[-1])
        lines += render_fit(analysis.drift, bpm, analysis.period_ms, end_ms)
    return lines


def render_stability(stability: float | None, grid_fit: float | None) -> list[str]:
    if stability is None or grid_fit is None:
        return []
    verdict = (
        "安定"
        if is_stable_value(stability, grid_fit)
        else "不安定(結論を鵜呑みにせず、--subdivision の変更やステム音源での再測定を推奨)"
    )
    return [
        "",
        f"[信頼度] 結論の位相が偶然より有意に支持される時間窓 {stability * 100:.0f}%"
        f"(基準 {STABLE_WINDOW_RATIO * 100:.0f}%以上)",
        f"  グリッド上にあるオンセットの割合(偶然を除く) {grid_fit * 100:.0f}%"
        f"(基準 {MIN_GRID_FIT * 100:.0f}%以上) -> {verdict}",
    ]


def render_reference(analysis: Analysis, config: AnalysisConfig) -> list[str]:
    reference = analysis.reference
    if reference is None:
        return []
    period = analysis.period_ms
    total = sum(group.count for group in reference.groups)
    lines = ["", f"[位相の群分解(最大傾き位置のピーク) / 基準群の選択: {config.reference_group}]"]
    for index, group in enumerate(reference.groups):
        marker = (
            f" <- 基準({REFERENCE_DESCRIPTIONS[config.reference_group]})"
            if index == reference.chosen
            else ""
        )
        lines.append(
            f"  群{index + 1}: 最大傾き {signed(float(wrap_symmetric(group.peak_ms, period)))} ms"
            f" / 頭 {signed(float(wrap_symmetric(group.head_ms, period)))} ms"
            f" / 遅れ {group.gap_ms:.2f} ms / 件数 {group.count} ({group.count / total * 100:.0f}%){marker}"
        )
    if len(reference.groups) > 1:
        lines.append(
            "  群が複数あるため、現行の結論は群の混合値です。件数比が時間で変わると見かけのドリフトが出ます"
        )
    if reference.windows is not None and reference.window_counts is not None:
        lines.append("  時間窓ごとの基準群の頭位相 / 各群の件数")
        lines += [
            f"  {start / 1000.0:6.0f}s~ {signed(phase):>8} ms / 件数 {' / '.join(str(int(value)) for value in row)}"
            for start, phase, row in zip(
                reference.windows.starts_ms,
                reference.windows.phases_ms,
                reference.window_counts,
            )
        ]
    if reference.drift is not None:
        end_ms = float(analysis.primary.heads_ms[-1])
        lines += render_fit(reference.drift, config.bpm, period, end_ms)
    elif reference.windows is not None:
        lines.append("  回帰は不採用(窓数不足、または窓ごとの位相が一貫しないため)")
    basis = "回帰の曲中央時点の値" if reference.uses_regression else "基準群の頭位相の中央値"
    lines += [
        "",
        f"[結論: {LABEL_REFERENCE} / 根拠: {basis}]",
        f"  グリッド開始オフセット {signed(reference.conclusion_ms)} ms",
        f"  音源を遅らせる量 {signed(-reference.conclusion_ms)} ms",
        f"  混合値の結論との差 {signed(reference.difference_ms)} ms",
    ]
    lines += render_stability(reference.stability, reference.grid_fit)
    if reference.sensitivity:
        lines += ["", "[定義感度: head-level(頭の定義の影響。検出器の頑健性の検証ではない) / 基準群の頭位相の中央値]"]
        lines += [
            f"  head-level {level:g}: {signed(phase)} ms" for level, phase in reference.sensitivity
        ]
        phases = [phase for _, phase in reference.sensitivity]
        lines.append(f"  最大差 {max(phases) - min(phases):.2f} ms")
    return lines
