from pathlib import Path
from types import SimpleNamespace

import onset_decoder

FFMPEG_STDERR = """Input #0, mp3, from 'song.mp3':
  Metadata:
    comment         : start: 123.5
  Duration: 00:03:00.00, start: 0.025057, bitrate: 128 kb/s
  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 128 kb/s
"""


def fake_run(stderr):
    return lambda *args, **kwargs: SimpleNamespace(stderr=stderr)


def test_leading_skip_ignores_start_text_in_metadata(monkeypatch):
    monkeypatch.setattr(onset_decoder, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(onset_decoder.subprocess, "run", fake_run(FFMPEG_STDERR))

    assert onset_decoder.leading_skip_samples(Path("song.mp3"), 44100) == 1105


def test_leading_skip_is_zero_without_a_duration_line(monkeypatch):
    monkeypatch.setattr(onset_decoder, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        onset_decoder.subprocess, "run", fake_run("  Metadata:\n    comment : start: 9\n")
    )

    assert onset_decoder.leading_skip_samples(Path("song.mp3"), 44100) == 0
