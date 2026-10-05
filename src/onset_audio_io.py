from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG_SUFFIXES = {".wma", ".mp3"}
SUFFIX_ALIASES = {".s3v": ".wma"}
ALLOWED_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".ogg"} | FFMPEG_SUFFIXES
FFMPEG_TIMEOUT_SEC = 300.0
MAX_DURATION_SEC = 900.0
MAX_SAMPLE_RATE = 192000
MAX_CHANNELS = 8


def audio_suffix(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    return SUFFIX_ALIASES.get(suffix, suffix)


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found is not None:
        return found
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise FileNotFoundError(
            "ffmpeg が見つかりません。ffmpeg を PATH に追加するか "
            "`python -m pip install imageio-ffmpeg` を実行してください"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_with_ffmpeg(source: Path, destination: Path) -> None:
    command = [
        find_ffmpeg(),
        "-v", "error",
        "-nostdin",
        "-y",
        "-i", str(source),
        "-map", "0:a:0",
        "-t", f"{MAX_DURATION_SEC + 1.0:g}",
        "-c:a", "pcm_f32le",
        str(destination),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=FFMPEG_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg のデコードに失敗しました: {result.stderr.strip()}")


@contextmanager
def readable_audio(audio_path: Path) -> Iterator[Path]:
    if audio_suffix(audio_path) not in FFMPEG_SUFFIXES:
        yield audio_path
        return
    with tempfile.TemporaryDirectory() as directory:
        decoded = Path(directory) / "decoded.wav"
        decode_with_ffmpeg(audio_path, decoded)
        yield decoded


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    return sf.read(path, dtype="float32", always_2d=True)


def validate_audio(audio_path: Path, high_hz: float) -> None:
    info = sf.info(audio_path)
    if info.duration > MAX_DURATION_SEC:
        raise ValueError(f"音源が長すぎます(上限 {MAX_DURATION_SEC:g} 秒)")
    if info.samplerate > MAX_SAMPLE_RATE or info.channels > MAX_CHANNELS:
        raise ValueError(
            f"サンプルレートは {MAX_SAMPLE_RATE} Hz 以下、チャンネル数は {MAX_CHANNELS} 以下にしてください"
        )
    if high_hz >= info.samplerate / 2.0:
        raise ValueError("--high-hz はサンプルレートの半分未満にしてください")
