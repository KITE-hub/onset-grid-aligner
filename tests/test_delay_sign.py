import pytest

from verify_synthetic import analyze_cli, synth


@pytest.mark.parametrize("source_offset_ms", [12.0, -12.0])
def test_reported_delay_has_audio_movement_sign(source_offset_ms):
    audio = synth.render(
        "rock8",
        synth.constant_segments(120.0, 12.0),
        source_offset_ms,
    )
    _, delay_ms = analyze_cli(audio, 120.0)
    assert delay_ms is not None
    assert delay_ms == pytest.approx(-source_offset_ms, abs=3.0)
