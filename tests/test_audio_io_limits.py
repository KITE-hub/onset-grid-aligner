import numpy as np
import pytest
import soundfile as sf

import onset_audio_io


def write_stereo(path):
    sf.write(path, np.zeros((100, 2), dtype=np.float32), 48000)


def test_validate_audio_rejects_too_many_total_samples(tmp_path, monkeypatch):
    path = tmp_path / "stereo.wav"
    write_stereo(path)
    monkeypatch.setattr(onset_audio_io, "MAX_TOTAL_SAMPLES", 150)

    with pytest.raises(ValueError):
        onset_audio_io.validate_audio(path, 12000.0)


def test_validate_audio_accepts_samples_within_the_limit(tmp_path, monkeypatch):
    path = tmp_path / "stereo.wav"
    write_stereo(path)
    monkeypatch.setattr(onset_audio_io, "MAX_TOTAL_SAMPLES", 200)

    onset_audio_io.validate_audio(path, 12000.0)


def test_audio_suffix_has_no_aliases(tmp_path):
    assert onset_audio_io.audio_suffix(tmp_path / "a.S3V") == ".s3v"
    assert ".s3v" not in onset_audio_io.ALLOWED_SUFFIXES


def test_validate_loaded_duration_rejects_too_short_audio():
    with pytest.raises(ValueError, match="短すぎます"):
        onset_audio_io.validate_loaded_duration(300, 44100)


def test_validate_loaded_duration_accepts_minimum_length():
    onset_audio_io.validate_loaded_duration(22050, 44100)


def test_validate_audio_rejects_audio_longer_than_the_limit(tmp_path, monkeypatch):
    path = tmp_path / "stereo.wav"
    write_stereo(path)
    monkeypatch.setattr(onset_audio_io, "MAX_DURATION_SEC", 0.001)

    with pytest.raises(ValueError, match="長すぎます"):
        onset_audio_io.validate_audio(path, 12000.0)


@pytest.mark.parametrize(
    ("limit_name", "limit"), [("MAX_SAMPLE_RATE", 44100), ("MAX_CHANNELS", 1)]
)
def test_validate_audio_rejects_formats_above_the_limits(
    tmp_path, monkeypatch, limit_name, limit
):
    path = tmp_path / "stereo.wav"
    write_stereo(path)
    monkeypatch.setattr(onset_audio_io, limit_name, limit)

    with pytest.raises(ValueError, match="サンプルレート"):
        onset_audio_io.validate_audio(path, 12000.0)


def test_validate_audio_rejects_a_cutoff_at_the_nyquist_frequency(tmp_path):
    path = tmp_path / "stereo.wav"
    write_stereo(path)

    with pytest.raises(ValueError, match="--high-hz"):
        onset_audio_io.validate_audio(path, 24000.0)
