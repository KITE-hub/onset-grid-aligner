from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from onset_audio_io import ALLOWED_SUFFIXES, audio_suffix, MAX_DURATION_SEC
from onset_bpm_estimate import ESTIMATE_MAX_BPM, ESTIMATE_MIN_BPM
from onset_config import (
    DEFAULT_DRIFT_WINDOW_SEC,
    DEFAULT_HEAD_LEVEL,
    DEFAULT_HIGH_HZ,
    DEFAULT_LOW_HZ,
    DEFAULT_REFERENCE_GROUP,
    FILTER_CHOICES,
    FILTER_ZERO,
    MAX_BPM,
    MIN_BPM,
    REFERENCE_CHOICES,
    SUBDIVISIONS,
)
from onset_pipeline import run_pipeline, RunOptions
from onset_report import render_report
from onset_window_phase import MIN_DRIFT_WINDOW_SEC


def validate_arguments(args: argparse.Namespace) -> Path:
    audio_path = args.audio.expanduser().resolve()
    if audio_suffix(audio_path) not in ALLOWED_SUFFIXES:
        raise ValueError(f"未対応の拡張子です: {sorted(ALLOWED_SUFFIXES)}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {audio_path}")
    if args.bpm is not None and not MIN_BPM <= args.bpm <= MAX_BPM:
        raise ValueError(f"--bpm は {MIN_BPM:g}~{MAX_BPM:g} の範囲で指定してください")
    if not MIN_BPM <= args.bpm_min < args.bpm_max <= MAX_BPM:
        raise ValueError(
            f"--bpm-min と --bpm-max は {MIN_BPM:g}~{MAX_BPM:g} の範囲で min < max にしてください"
        )
    if not 0.0 < args.head_level < 1.0:
        raise ValueError("--head-level は 0 より大きく 1 より小さい値にしてください")
    if not 0.0 < args.low_hz < args.high_hz:
        raise ValueError("--low-hz は正の値で --high-hz より小さくしてください")
    window = args.drift_window_sec
    if not (window == 0.0 or MIN_DRIFT_WINDOW_SEC <= window <= MAX_DURATION_SEC):
        raise ValueError(
            f"--drift-window-sec は 0(無効) または {MIN_DRIFT_WINDOW_SEC:g}~{MAX_DURATION_SEC:g} にしてください"
        )
    return audio_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BPM固定音源のオンセット位相(グリッドに対するずれ)を推定する"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--bpm-min", type=float, default=ESTIMATE_MIN_BPM)
    parser.add_argument("--bpm-max", type=float, default=ESTIMATE_MAX_BPM)
    parser.add_argument("--subdivision", type=int, default=4, choices=SUBDIVISIONS)
    parser.add_argument("--low-hz", type=float, default=DEFAULT_LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=DEFAULT_HIGH_HZ)
    parser.add_argument("--head-level", type=float, default=DEFAULT_HEAD_LEVEL)
    parser.add_argument(
        "--drift-window-sec", type=float, default=DEFAULT_DRIFT_WINDOW_SEC
    )
    parser.add_argument(
        "--reference-group",
        choices=REFERENCE_CHOICES,
        default=DEFAULT_REFERENCE_GROUP,
    )
    parser.add_argument("--no-band-check", dest="band_check", action="store_false")
    parser.add_argument("--no-refine", dest="refine", action="store_false")
    parser.add_argument("--filter-mode", choices=FILTER_CHOICES, default=FILTER_ZERO)
    return parser


def options_from_args(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        bpm=args.bpm,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        subdivision=args.subdivision,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        head_level=args.head_level,
        drift_window_sec=args.drift_window_sec,
        reference_group=args.reference_group,
        filter_mode=args.filter_mode,
        run_band_check=args.band_check,
        run_refine=args.refine,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audio_path = validate_arguments(args)
        run = run_pipeline(audio_path, options_from_args(args))
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 2
    print(
        "\n".join(
            render_report(
                audio_path.name,
                run.config,
                run.analysis,
                run.estimate,
                run.decode,
                run.bands,
                run.refine,
                None,
            )
        )
    )
    return 0
