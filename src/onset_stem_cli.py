from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from onset_bpm_estimate import ESTIMATE_MAX_BPM, ESTIMATE_MIN_BPM
from onset_cli import validate_arguments
from onset_config import DEFAULT_DRIFT_WINDOW_SEC, DEFAULT_HEAD_LEVEL, SUBDIVISIONS
from onset_stem_pipeline import run_stem, STEM_HIGH_HZ, STEM_LOW_HZ, StemOptions
from onset_stem_report import render_stem_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ドラムステムのオンセット位相(グリッドに対するずれ)を楽器クラス等重みで推定する"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--bpm-min", type=float, default=ESTIMATE_MIN_BPM)
    parser.add_argument("--bpm-max", type=float, default=ESTIMATE_MAX_BPM)
    parser.add_argument("--subdivision", type=int, default=4, choices=SUBDIVISIONS)
    parser.add_argument("--low-hz", type=float, default=STEM_LOW_HZ)
    parser.add_argument("--high-hz", type=float, default=STEM_HIGH_HZ)
    parser.add_argument("--head-level", type=float, default=DEFAULT_HEAD_LEVEL)
    parser.add_argument(
        "--drift-window-sec", type=float, default=DEFAULT_DRIFT_WINDOW_SEC
    )
    return parser


def options_from_args(args: argparse.Namespace) -> StemOptions:
    return StemOptions(
        bpm=args.bpm,
        bpm_min=args.bpm_min,
        bpm_max=args.bpm_max,
        subdivision=args.subdivision,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        head_level=args.head_level,
        drift_window_sec=args.drift_window_sec,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audio_path = validate_arguments(args)
        run = run_stem(audio_path, options_from_args(args))
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 2
    print("\n".join(render_stem_report(audio_path.name, run)))
    return 0
