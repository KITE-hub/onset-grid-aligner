from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import synth_drums as synth

CLI_PATTERNS = ("rock8", "four_floor", "funk16", "shuffle", "triplet_mix", "dnb", "sparse_kick")
CLI_BPMS = (90.0, 120.0, 150.0, 174.0)
CLI_OFFSETS = (-45.0, -23.71, -10.0, 0.0, 12.0, 23.71, 40.0, 55.0)
JITTER_MS = 1.0
LOOP_CASES = (("rock8", 120.0, 23.71), ("funk16", 174.0, -23.71), ("dnb", 150.0, 40.0))
TEMPO_PATTERNS = ("rock8", "funk16")
TEMPO_SEGMENTS = ((10, 150.0), (8, 180.0), (10, 150.0))
TEMPO_OFFSETS = (-60.0, -40.0, -23.71, 0.0, 12.0, 23.71, 40.0, 60.0)
SUITES = ("cli", "jitter", "loop", "tempo")


def configure_paths(root: Path) -> None:
    sys.path.insert(0, str(root / "batch"))
    sys.path.insert(0, str(root / "src"))


def analyze_cli(audio: np.ndarray, bpm: float) -> tuple[float, float]:
    from onset_config import DEFAULT_DRIFT_WINDOW_SEC, DEFAULT_HEAD_LEVEL, DEFAULT_HIGH_HZ, DEFAULT_LOW_HZ, FILTER_ZERO
    from onset_pipeline import run_pipeline, RunOptions

    options = RunOptions(
        bpm=bpm,
        bpm_min=60.0,
        bpm_max=200.0,
        low_hz=DEFAULT_LOW_HZ,
        high_hz=DEFAULT_HIGH_HZ,
        head_level=DEFAULT_HEAD_LEVEL,
        drift_window_sec=DEFAULT_DRIFT_WINDOW_SEC,
        filter_mode=FILTER_ZERO,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.wav"
        sf.write(path, audio, synth.SAMPLE_RATE, subtype="PCM_24")
        run = run_pipeline(path, options)
    delay = dict(run.candidates).get("fixed_grid")
    return (None if delay is None else -delay), delay


def analyze_tempo(audio: np.ndarray, segments: tuple[tuple[int, float], ...]) -> float:
    import onset_grid_aligner as core
    from onset_odf import compute_onset_functions
    from onset_timing import TempoMap, TempoPoint

    points = []
    beats = 0.0
    elapsed = 0.0
    for bars, bpm in segments:
        points.append(TempoPoint(beats, elapsed, bpm))
        beats += bars * 4.0
        elapsed += bars * 4.0 * 60000.0 / bpm
    tempo = TempoMap(tuple(points), elapsed)
    duration_ms = audio.shape[0] * 1000.0 / synth.SAMPLE_RATE
    detection = core.detect_attacks(
        audio,
        synth.SAMPLE_RATE,
        core.DEFAULT_LOW_HZ,
        core.DEFAULT_HIGH_HZ,
        core.head_levels(core.DEFAULT_HEAD_LEVEL),
        False,
    )
    config = core.AnalysisConfig(
        tempo.dominant_bpm(duration_ms),
        core.DEFAULT_LOW_HZ,
        core.DEFAULT_HIGH_HZ,
        core.DEFAULT_HEAD_LEVEL,
        core.DEFAULT_DRIFT_WINDOW_SEC,
    )
    functions = compute_onset_functions(audio, synth.SAMPLE_RATE)
    kick = core.kick_envelope(audio, synth.SAMPLE_RATE)
    detection = core.attach_kicks(detection, kick, config.grid_beat_ms)
    warp = core.make_warp(tempo, config, detection)
    return core.analyze_beat(detection, config, functions, warp).offset_ms


def cli_error(pattern: str, bpm: float, offset: float, seconds: float, jitter: float) -> float:
    audio = synth.render(pattern, synth.constant_segments(bpm, seconds), offset, jitter)
    measured = analyze_cli(audio, bpm)[0]
    if measured is None:
        raise AssertionError(
            f"{pattern} {bpm:g} BPM: 合成音源のオフセット推定が保留されました"
        )
    return measured - offset


def tempo_error(pattern: str, offset: float) -> float:
    audio = synth.render(pattern, TEMPO_SEGMENTS, offset)
    return analyze_tempo(audio, TEMPO_SEGMENTS) - offset


def loop_error(pattern: str, bpm: float, offset: float, seconds: float) -> tuple[float, float]:
    audio = synth.render(pattern, synth.constant_segments(bpm, seconds), offset)
    measured, delay = analyze_cli(audio, bpm)
    residual, _ = analyze_cli(synth.shift_audio(audio, delay), bpm)
    return measured - offset, residual


def format_error(error: float, tolerance: float) -> str:
    return f"{error:+.1f}{'*' if abs(error) > tolerance else ''}"


def run_grid(suite: str, args: argparse.Namespace) -> int:
    jitter = JITTER_MS if suite == "jitter" else 0.0
    failures = 0
    print(f"[{suite}] 列: 真値オフセット {', '.join(f'{v:g}' for v in CLI_OFFSETS)} ms / 値は 推定-真値 (ms), * は許容超過")
    for pattern in CLI_PATTERNS:
        for bpm in CLI_BPMS:
            errors = [cli_error(pattern, bpm, offset, args.seconds, jitter) for offset in CLI_OFFSETS]
            failures += sum(abs(error) > args.tolerance_ms for error in errors)
            cells = " ".join(f"{format_error(error, args.tolerance_ms):>7s}" for error in errors)
            print(f"{pattern:12s} {bpm:5.1f}BPM | {cells}", flush=True)
    print(f"[{suite}] 失敗 {failures} / {len(CLI_PATTERNS) * len(CLI_BPMS) * len(CLI_OFFSETS)}")
    return failures


def run_loop(args: argparse.Namespace) -> int:
    failures = 0
    print("[loop] 推奨移動量を適用して再測定 (適用後の残差が0に近ければ符号と量が正しい)")
    for pattern, bpm, offset in LOOP_CASES:
        error, residual = loop_error(pattern, bpm, offset, args.seconds)
        bad = abs(error) > args.tolerance_ms or abs(residual) > args.tolerance_ms
        failures += bad
        print(f"{pattern:12s} {bpm:5.1f}BPM 真値 {offset:+7.2f} | 推定誤差 {error:+.2f} ms / 適用後の残差 {residual:+.2f} ms{' *' if bad else ''}", flush=True)
    print(f"[loop] 失敗 {failures} / {len(LOOP_CASES)}")
    return failures


def run_tempo(args: argparse.Namespace) -> int:
    failures = 0
    print(f"[tempo] テンポ変化 {TEMPO_SEGMENTS} / 列: 真値 {', '.join(f'{v:g}' for v in TEMPO_OFFSETS)} ms")
    for pattern in TEMPO_PATTERNS:
        errors = [tempo_error(pattern, offset) for offset in TEMPO_OFFSETS]
        failures += sum(abs(error) > args.tolerance_ms for error in errors)
        cells = " ".join(f"{format_error(error, args.tolerance_ms):>7s}" for error in errors)
        print(f"{pattern:12s} | {cells}", flush=True)
    print(f"[tempo] 失敗 {failures} / {len(TEMPO_PATTERNS) * len(TEMPO_OFFSETS)}")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="合成ドラムでオフセット推定を検証する")
    parser.add_argument("--root", type=Path, default=HERE.parent)
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--tolerance-ms", type=float, default=3.0)
    parser.add_argument("--seconds", type=float, default=24.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_paths(args.root.resolve())
    suites = SUITES if args.suite == "all" else (args.suite,)
    runners = {
        "cli": lambda: run_grid("cli", args),
        "jitter": lambda: run_grid("jitter", args),
        "loop": lambda: run_loop(args),
        "tempo": lambda: run_tempo(args),
    }
    started = time.time()
    failures = sum(runners[suite]() for suite in suites)
    print(f"総失敗 {failures} / 経過 {time.time() - started:.0f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
