from __future__ import annotations

from onset_common import wrap_symmetric
from onset_drift import DriftFit
from onset_format import signed
from onset_phase import is_stable_value, MIN_GRID_FIT, STABLE_WINDOW_RATIO
from onset_pipeline import AnalysisRun
from onset_window_phase import WindowPhases


def render_fit(fit: DriftFit, run: AnalysisRun, end_ms: float) -> list[str]:
    bpm = run.config.bpm
    period = run.analysis.period_ms
    tempo = bpm * (1.0 - fit.slope)
    tempo_low = bpm * (1.0 - fit.slope_high)
    tempo_high = bpm * (1.0 - fit.slope_low)
    verdict = "有意なテンポずれあり" if fit.significant else "テンポずれは有意でない"
    center_phase = float(wrap_symmetric(fit.phase_at(fit.reference_ms), period))
    start_gap = fit.slope * (0.0 - fit.reference_ms)
    end_gap = fit.slope * (end_ms - fit.reference_ms)
    start_phase = float(wrap_symmetric(center_phase + start_gap, period))
    lines = [
        f"  回帰[{fit.method}]: 傾き {fit.slope * 60000.0:+.2f} ms/分 -> 推定BPM {tempo:.4f} (95%区間 {tempo_low:.4f} ~ {tempo_high:.4f}) / {verdict}",
        f"  BPM {bpm:g} 固定で曲中央に揃えた場合のズレ: 曲頭 {signed(start_gap)} ms / 終端({end_ms / 1000.0:.0f}s) {signed(end_gap)} ms",
        f"  曲中央({fit.reference_ms / 1000.0:.1f}s)でのオンセット格子位相 {signed(center_phase)} ms",
        f"  曲頭(0 ms)でのオンセット格子位相(切片) {signed(start_phase)} ms",
    ]
    if fit.significant:
        lines.append(
            f"  推定BPM {tempo:.4f} を小数で指定できる場合は、曲頭の格子位相 {signed(start_phase)} ms を採用する"
            f"(音源を遅らせる量 {signed(-start_phase)} ms)"
        )
    return lines


def window_lines(windows: WindowPhases) -> list[str]:
    return [
        f"  {start / 1000.0:6.0f}s~ {signed(phase):>8} ms (n={count})"
        for start, phase, count in zip(
            windows.starts_ms, windows.phases_ms, windows.counts
        )
    ]


def render_drift(run: AnalysisRun) -> list[str]:
    analysis = run.analysis
    windows = analysis.windows
    if windows is None:
        return []
    lines = ["", "[時間窓ごとの位相(連続化済み)]"]
    lines += window_lines(windows)
    if analysis.drift is None:
        lines.append("  回帰は不採用(窓数不足、または窓ごとの位相が一貫しないため)")
    else:
        end_ms = float(analysis.primary.heads_ms[-1])
        lines += render_fit(analysis.drift, run, end_ms)
    return lines


def render_stability(stability: float | None, grid_fit: float | None) -> list[str]:
    if stability is None or grid_fit is None:
        return []
    verdict = (
        "安定"
        if is_stable_value(stability, grid_fit)
        else "不安定(ドラムステム音源での再測定や --bpm の確認を推奨)"
    )
    return [
        "",
        f"[信頼度の参考値] 既存周期法の位相が支持される時間窓 {stability * 100:.0f}%"
        f"(基準 {STABLE_WINDOW_RATIO * 100:.0f}%以上)",
        f"  固定12位置への適合率 {grid_fit * 100:.0f}%"
        f"(基準 {MIN_GRID_FIT * 100:.0f}%以上) -> {verdict}",
    ]
