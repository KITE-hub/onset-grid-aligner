from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from onset_common import phase_to_delay_ms, wrap_symmetric

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "benchmark" / "real_drums.json"
ALLOWED_CATEGORIES = {"acoustic", "electronic", "hybrid"}
ALLOWED_SPLITS = {"calibration", "holdout"}


@dataclass(frozen=True)
class DrumRecord:
    audio: Path
    category: str
    genre: str
    source_id: str
    split: str
    bpm: float
    reference_delay_ms: float
    tolerance_ms: float


def load_records() -> tuple[DrumRecord, ...]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = []
    for item in payload.get("records", []):
        audio = (MANIFEST.parent / item["audio"]).resolve()
        if not audio.is_relative_to(MANIFEST.parent.resolve()):
            raise ValueError(f"audio path escapes benchmark directory: {item['audio']}")
        category = item["category"]
        split = item["split"]
        source_id = item["source_id"].strip()
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported recording category: {category}")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"unsupported benchmark split: {split}")
        if not source_id or not item.get("genre", "").strip():
            raise ValueError("source_id and genre are required")
        if float(item["bpm"]) <= 0.0 or float(item["tolerance_ms"]) <= 0.0:
            raise ValueError("bpm and tolerance_ms must be positive")
        records.append(
            DrumRecord(
                audio=audio,
                category=category,
                genre=item["genre"].strip(),
                source_id=source_id,
                split=split,
                bpm=float(item["bpm"]),
                reference_delay_ms=float(item["reference_delay_ms"]),
                tolerance_ms=float(item["tolerance_ms"]),
            )
        )
    sources_by_split = {
        split: {record.source_id for record in records if record.split == split}
        for split in ALLOWED_SPLITS
    }
    if sources_by_split["calibration"] & sources_by_split["holdout"]:
        raise ValueError("source_id values must not overlap between calibration and holdout")
    return tuple(records)


def phase_at_midpoint(run, duration_ms: float) -> float:
    analysis = run.analysis
    phase = analysis.conclusion_ms
    drift = analysis.drift
    if analysis.uses_regression and drift is not None:
        phase += drift.slope * (duration_ms / 2.0 - drift.reference_ms)
    return float(wrap_symmetric(phase, run.config.layout.beat_ms))


def candidate_delays(run, duration_ms: float) -> dict[str, float | None]:
    analysis = run.analysis
    drift = analysis.drift
    uses_regression = analysis.uses_regression and drift is not None
    midpoint_delta = duration_ms / 2.0 - drift.reference_ms if uses_regression else 0.0
    period = run.config.layout.beat_ms
    candidates = {
        "current_comb_delay_ms": run.beat.delay_ms,
        "comb_midpoint_extrapolated_delay_ms": None,
        "cluster_drift_midpoint_delay_ms": phase_to_delay_ms(
            phase_at_midpoint(run, duration_ms)
        ),
    }
    if run.beat.delay_ms is not None:
        phase = -run.beat.delay_ms
        if uses_regression:
            phase += drift.slope * midpoint_delta
        candidates["comb_midpoint_extrapolated_delay_ms"] = phase_to_delay_ms(
            float(wrap_symmetric(phase, period))
        )
    if analysis.cross is not None:
        for check in analysis.cross.checks:
            phase = check.phase_ms
            if check.uses_regression and uses_regression:
                phase += drift.slope * midpoint_delta
            candidates[f"{check.name}_midpoint_delay_ms"] = phase_to_delay_ms(
                float(wrap_symmetric(phase, period))
            )
    return candidates


def circular_error_ms(actual: float, expected: float, period_ms: float) -> float:
    return float(wrap_symmetric(actual - expected, period_ms))


def candidate_spread_ms(
    candidates: dict[str, float | None], period_ms: float
) -> float | None:
    values = [value for value in candidates.values() if value is not None]
    if len(values) < 2:
        return None
    distances = [
        abs(circular_error_ms(first, second, period_ms))
        for index, first in enumerate(values)
        for second in values[index + 1 :]
    ]
    return max(distances)


def duration_ms(audio: Path) -> float:
    return sf.info(audio).duration * 1000.0
