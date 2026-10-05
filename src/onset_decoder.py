from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from onset_audio_io import (
    audio_suffix,
    FFMPEG_SUFFIXES,
    FFMPEG_TIMEOUT_SEC,
    find_ffmpeg,
)

MP3_SUFFIX = ".mp3"
MP3_DECODER_DELAY_SAMPLES = 529
MP3_ENCODER_DELAY_SAMPLES = 576
MP3_FULL_DELAY_SAMPLES = MP3_ENCODER_DELAY_SAMPLES + MP3_DECODER_DELAY_SAMPLES
START_PATTERN = re.compile(r"start:\s*(-?\d+(?:\.\d+)?)")
DECODER_EXPLANATIONS = {
    ".mp3": "MP3は先頭にエンコーダ遅延とデコーダ遅延が入り、再生側ごとに削る量が異なる。どれが使われるかはファイルから決まらないため、想定される削除量ごとの結果を全て表示する。使う再生環境に当てはまる行を採用すること",
    ".wma": "WMAは先頭から削る量を示す情報がファイルになく、削除量を特定できないため、先頭を削らない場合のみ表示する",
}


@dataclass(frozen=True)
class DecodeInfo:
    suffix: str
    sample_rate: int
    applied_skip: int


@dataclass(frozen=True)
class DecoderVariant:
    label: str
    skip_samples: int


def leading_skip_samples(audio_path: Path, sample_rate: int) -> int:
    result = subprocess.run(
        [find_ffmpeg(), "-hide_banner", "-nostdin", "-i", str(audio_path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=FFMPEG_TIMEOUT_SEC,
        check=False,
    )
    match = START_PATTERN.search(result.stderr)
    if match is None:
        return 0
    return max(0, round(float(match.group(1)) * sample_rate))


def decode_info(audio_path: Path, sample_rate: int) -> DecodeInfo | None:
    suffix = audio_suffix(audio_path)
    if suffix not in FFMPEG_SUFFIXES:
        return None
    return DecodeInfo(suffix, sample_rate, leading_skip_samples(audio_path, sample_rate))


def variant_label(skip_samples: int, suffix: str) -> str:
    if skip_samples == 0:
        return "先頭を削らずに再生する場合"
    if suffix == MP3_SUFFIX and skip_samples == MP3_DECODER_DELAY_SAMPLES:
        return f"デコーダ遅延({MP3_DECODER_DELAY_SAMPLES})だけ削る場合"
    if suffix == MP3_SUFFIX and skip_samples == MP3_FULL_DELAY_SAMPLES:
        return (
            f"エンコーダ遅延(LAME標準の{MP3_ENCODER_DELAY_SAMPLES})と"
            f"デコーダ遅延({MP3_DECODER_DELAY_SAMPLES})を削る場合"
        )
    return "ファイル内のヘッダ情報どおりに削る場合"


def decoder_variants(decode: DecodeInfo) -> tuple[DecoderVariant, ...]:
    skips = {0, decode.applied_skip}
    if decode.suffix == MP3_SUFFIX:
        skips.update((MP3_DECODER_DELAY_SAMPLES, MP3_FULL_DELAY_SAMPLES))
    return tuple(
        DecoderVariant(variant_label(skip, decode.suffix), skip) for skip in sorted(skips)
    )
