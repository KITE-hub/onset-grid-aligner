from __future__ import annotations

from onset_stem_classes import CLASS_LABELS
from onset_stem_grid import comb_steps, GRID_LATTICES, GridAlignment, STEPS_PER_BEAT
from onset_stem_phase import BandEstimate, ClassPhase, StemResult
from onset_stem_pipeline import StemRun

BAND_COUNT = 2


def band_title(band: BandEstimate) -> str:
    return f"{band.low_hz:g}~{band.high_hz:g} Hz"


def class_line(index: int, item: ClassPhase, used: tuple[int, ...]) -> str:
    status = "使用" if index in used else "除外(グリッドに乗っていない)"
    return (
        f"  {CLASS_LABELS[item.label]:<12} n={item.count:<5} 線上={item.inliers:<5} "
        f"{-item.phase_ms:+.2f} ms  集中度 {item.fit:.2f}  {status}"
    )


def band_lines(band: BandEstimate) -> list[str]:
    estimate = band.estimate
    lines = ["", f"[{band_title(band)} のクラス別(音源を遅らせる量 / 線上=12本の線に乗ったイベント数 / 集中度=偶然を除いた割合)]"]
    lines.extend(
        class_line(index, item, estimate.used)
        for index, item in enumerate(estimate.classes)
    )
    return lines


def drift_lines(result: StemResult) -> list[str]:
    lines = []
    for band in result.bands:
        drift = band.estimate.drift
        if drift is not None and drift.significant:
            lines.append(
                f"  {band_title(band)}: 傾き {drift.slope * 1e6:+.0f} ppm"
                f"(95%区間 {drift.slope_low * 1e6:+.0f} ~ {drift.slope_high * 1e6:+.0f} ppm)"
            )
    if not lines:
        return ["テンポずれ: 有意でない"]
    return ["テンポずれ(遅延量は曲の基準時刻での値):", *lines]


def grid_lines(run: StemRun) -> list[str]:
    layout = run.layout
    lattices = ",".join(str(value) for value in GRID_LATTICES)
    return [
        f"グリッド: {run.config.bpm:.2f} BPM(折り返し後) / 1拍={STEPS_PER_BEAT}分割の格子 {layout.lattice_ms:.2f} ms",
        f"  線: {lattices}分割の合併(1拍{len(comb_steps())}本) / 1拍={layout.beat_steps}ステップ={layout.beat_ms:.2f} ms",
    ]


def decision_text(name: str, decided: bool, margin: float) -> str:
    return f"{name}: {'決定' if decided else '未確定'}(マージン {margin:.2f})"


def alignment_lines(alignment: GridAlignment | None) -> list[str]:
    if alignment is None:
        return ["拍の位置: 判定できませんでした(格子に乗るイベントがありません)"]
    lines = [
        "  " + decision_text("拍の頭", alignment.beat_decided, alignment.beat_margin),
        f"  次点の候補: {-alignment.runner_up_ms:+.2f} ms",
    ]
    if not alignment.beat_decided:
        lines.append("  拍の頭が未確定です。別の拍位置の候補と区別できていません。DAWで確認してください")
    return lines


def render_stem_report(name: str, run: StemRun) -> list[str]:
    result = run.result
    source = "指定" if run.bpm_estimate is None else "推定"
    lines = [
        f"ファイル: {name}",
        f"BPM: {run.bpm:.2f}({source})",
        *grid_lines(run),
        "",
        f"音源を遅らせる量: {run.delay_ms:+.2f} ms (不確かさ ±{result.uncertainty_ms:.2f} ms)",
        *alignment_lines(run.alignment),
        f"信頼度: {result.confidence}",
        f"  統計誤差(95%) ±{result.se95_ms:.2f} ms / クラス間の幅 {result.span_ms:.2f} ms / 帯域間の差 {result.band_gap_ms:.2f} ms",
    ]
    if len(result.bands) < BAND_COUNT:
        lines.append("  片方の帯域でグリッドに乗ったクラスが見つからず、帯域間の差は算出できません")
    lines += ["", "帯域別(音源を遅らせる量。格子の周期内の位相のみ):"]
    lines.extend(
        f"  {band_title(band):<16} {band.estimate.delay_ms:+.2f} ms"
        for band in result.bands
    )
    for band in result.bands:
        lines += band_lines(band)
    lines.append("")
    lines.extend(drift_lines(result))
    lines.append(
        f"注意: 値は1拍({run.layout.beat_ms:.2f} ms)単位でのみ意味を持ちます。小節の頭は判定しません"
    )
    return lines
