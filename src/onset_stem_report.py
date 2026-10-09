from __future__ import annotations

from onset_stem_classes import CLASS_LABELS
from onset_stem_phase import ClassPhase
from onset_stem_pipeline import StemRun


def class_line(index: int, item: ClassPhase, used: tuple[int, ...]) -> str:
    status = "使用" if index in used else "除外(グリッドに乗っていない)"
    return (
        f"  {CLASS_LABELS[item.label]:<12} n={item.count:<5} 周期内={item.inliers:<5} "
        f"{-item.phase_ms:+.2f} ms  集中度 {item.fit:.2f}  {status}"
    )


def drift_lines(run: StemRun) -> list[str]:
    drift = run.estimate.drift
    if drift is None or not drift.significant:
        return ["テンポずれ: 有意でない"]
    return [
        f"テンポずれ: 傾き {drift.slope * 1e6:+.0f} ppm(95%区間 {drift.slope_low * 1e6:+.0f} ~ {drift.slope_high * 1e6:+.0f} ppm)",
        "  遅延量は曲の基準時刻での値です",
    ]


def render_stem_report(name: str, run: StemRun) -> list[str]:
    estimate = run.estimate
    source = "指定" if run.bpm_estimate is None else "推定"
    lines = [
        f"ファイル: {name}",
        f"BPM: {run.config.bpm:.2f}({source}) / グリッド周期 {run.config.period_ms:.2f} ms",
        "",
        f"音源を遅らせる量: {estimate.delay_ms:+.2f} ms (不確かさ ±{estimate.uncertainty_ms:.2f} ms)",
        f"信頼度: {estimate.confidence}",
        f"  統計誤差(95%) ±{estimate.se95_ms:.2f} ms / クラス間の幅 {estimate.span_ms:.2f} ms",
        "",
        "クラス別(音源を遅らせる量):",
    ]
    lines.extend(
        class_line(index, item, estimate.used)
        for index, item in enumerate(estimate.classes)
    )
    lines.append("")
    lines.extend(drift_lines(run))
    lines.append(
        f"注意: 値はグリッド周期 {run.config.period_ms:.2f} ms の剰余としてのみ意味を持ちます"
    )
    return lines
