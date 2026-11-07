import json

import pytest

from benchmark import real_drum_validation
from benchmark.real_drum_validation import (
    candidate_spread_ms,
    circular_error_ms,
    load_records,
)
from onset_config import (
    DEFAULT_DRIFT_WINDOW_SEC,
    DEFAULT_HEAD_LEVEL,
    DEFAULT_HIGH_HZ,
    DEFAULT_LOW_HZ,
    FILTER_ZERO,
)
from onset_pipeline import RunOptions, run_pipeline

RECORDS = load_records()


def make_record(**overrides):
    record = {
        "audio": "audio/a.wav",
        "source_id": "a",
        "split": "calibration",
        "category": "acoustic",
        "genre": "funk",
        "bpm": 96.0,
        "reference_delay_ms": -7.5,
        "tolerance_ms": 3.0,
    }
    return {**record, **overrides}


def register(tmp_path, monkeypatch, records):
    manifest = tmp_path / "real_drums.json"
    manifest.write_text(json.dumps({"version": 1, "records": records}), encoding="utf-8")
    monkeypatch.setattr(real_drum_validation, "MANIFEST", manifest)


def test_load_records_reads_a_valid_manifest(tmp_path, monkeypatch):
    register(tmp_path, monkeypatch, [make_record()])

    records = load_records()

    assert len(records) == 1
    assert records[0].audio == (tmp_path / "audio" / "a.wav").resolve()
    assert records[0].reference_delay_ms == -7.5


def test_load_records_rejects_audio_outside_the_benchmark_directory(tmp_path, monkeypatch):
    register(tmp_path, monkeypatch, [make_record(audio="../outside.wav")])

    with pytest.raises(ValueError, match="escapes"):
        load_records()


@pytest.mark.parametrize(
    "overrides",
    [{"category": "vinyl"}, {"split": "train"}, {"genre": " "}, {"bpm": 0}],
)
def test_load_records_rejects_invalid_fields(tmp_path, monkeypatch, overrides):
    register(tmp_path, monkeypatch, [make_record(**overrides)])

    with pytest.raises(ValueError):
        load_records()


def test_load_records_rejects_a_source_shared_by_both_splits(tmp_path, monkeypatch):
    register(
        tmp_path,
        monkeypatch,
        [make_record(), make_record(audio="audio/b.wav", split="holdout")],
    )

    with pytest.raises(ValueError, match="overlap"):
        load_records()


def test_real_drum_set_contains_acoustic_and_varied_songs():
    if not RECORDS:
        pytest.skip("real drum recordings are not registered yet")
    assert any(record.category in {"acoustic", "hybrid"} for record in RECORDS)
    assert len({record.genre for record in RECORDS}) >= 3
    assert {record.split for record in RECORDS} == {"calibration", "holdout"}


@pytest.mark.parametrize("record", RECORDS, ids=lambda item: item.audio.stem)
def test_real_drum_midpoint_candidates(record):
    if not record.audio.is_file():
        pytest.fail(f"missing benchmark audio: {record.audio}")
    options = RunOptions(
        bpm=record.bpm,
        bpm_min=60.0,
        bpm_max=200.0,
        low_hz=DEFAULT_LOW_HZ,
        high_hz=DEFAULT_HIGH_HZ,
        head_level=DEFAULT_HEAD_LEVEL,
        drift_window_sec=DEFAULT_DRIFT_WINDOW_SEC,
        filter_mode=FILTER_ZERO,
    )
    run = run_pipeline(record.audio, options)
    candidates = dict(run.candidates)
    period_ms = run.config.layout.beat_ms
    errors = {
        name: circular_error_ms(value, record.reference_delay_ms, period_ms)
        for name, value in candidates.items()
    }
    print(
        json.dumps(
            {
                "source_id": record.source_id,
                "split": record.split,
                "errors_ms": errors,
                "spread_ms": candidate_spread_ms(candidates, period_ms),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    assert candidates
    if record.split == "holdout":
        failing = {
            name: round(error, 2)
            for name, error in errors.items()
            if abs(error) > record.tolerance_ms
        }
        assert failing == {}
