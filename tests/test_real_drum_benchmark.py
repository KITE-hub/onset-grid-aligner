import json

import pytest

from benchmark.real_drum_validation import candidate_delays, candidate_spread_ms, circular_error_ms, duration_ms, load_records
from onset_config import (
    DEFAULT_DRIFT_WINDOW_SEC,
    DEFAULT_HEAD_LEVEL,
    DEFAULT_HIGH_HZ,
    DEFAULT_LOW_HZ,
    FILTER_ZERO,
)
from onset_pipeline import RunOptions, run_pipeline

RECORDS = load_records()


def test_real_drum_benchmark_is_registered_or_explicitly_skipped():
    if not RECORDS:
        pytest.skip("real drum recordings are not registered yet")


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
        run_band_check=True,
        run_refine=True,
    )
    run = run_pipeline(record.audio, options)
    candidates = candidate_delays(run, duration_ms(record.audio))
    period_ms = run.config.layout.comb_beat_ms
    errors = {
        name: None
        if value is None
        else circular_error_ms(value, record.reference_delay_ms, period_ms)
        for name, value in candidates.items()
    }
    spread_ms = candidate_spread_ms(candidates, period_ms)
    print(
        json.dumps(
            {
                "source_id": record.source_id,
                "split": record.split,
                "errors_ms": errors,
                "spread_ms": spread_ms,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    assert candidates["cluster_drift_midpoint_delay_ms"] is not None

