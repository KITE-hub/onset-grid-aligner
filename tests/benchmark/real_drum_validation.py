from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from onset_common import wrap_symmetric

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


def circular_error_ms(actual: float, expected: float, period_ms: float) -> float:
    return float(wrap_symmetric(actual - expected, period_ms))


def candidate_spread_ms(
    candidates: dict[str, float], period_ms: float
) -> float | None:
    values = list(candidates.values())
    if len(values) < 2:
        return None
    distances = [
        abs(circular_error_ms(first, second, period_ms))
        for index, first in enumerate(values)
        for second in values[index + 1 :]
    ]
    return max(distances)
