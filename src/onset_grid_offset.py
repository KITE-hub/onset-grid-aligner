from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import fft, ndimage, signal, stats

from onset_common import causal_envelope, ms_to_samples, wrap_symmetric
from onset_drift import DriftFit, fit_direct
from onset_odf import OnsetFunction, block_caps, compute_onset_functions, fold_phase
from onset_refine import refine_heads
from onset_timing import MS_PER_MINUTE, TempoMap

FFMPEG_SUFFIXES = {".wma", ".mp3"}
MP3_SUFFIX = ".mp3"
SUFFIX_ALIASES = {".s3v": ".wma"}
MP3_DECODER_DELAY_SAMPLES = 529
MP3_ENCODER_DELAY_SAMPLES = 576
MP3_FULL_DELAY_SAMPLES = MP3_ENCODER_DELAY_SAMPLES + MP3_DECODER_DELAY_SAMPLES
START_PATTERN = re.compile(r"start:\s*(-?\d+(?:\.\d+)?)")
DECODER_EXPLANATIONS = {
    ".mp3": "MP3は先頭にエンコーダ遅延とデコーダ遅延が入り、再生側ごとに削る量が異なる。どれが使われるかはファイルから決まらないため、想定される削除量ごとの結果を全て表示する。使う再生環境に当てはまる行を採用すること",
    ".wma": "WMAは先頭から削る量を示す情報がファイルになく、削除量を特定できないため、先頭を削らない場合のみ表示する",
}
ALLOWED_SUFFIXES = {".wav", ".flac", ".aif", ".aiff", ".ogg"} | FFMPEG_SUFFIXES
FFMPEG_TIMEOUT_SEC = 300.0
SUBDIVISIONS = (1, 2, 3, 4, 6, 8)
MIN_BPM = 30.0
MAX_BPM = 400.0
MAX_DURATION_SEC = 900.0
MAX_SAMPLE_RATE = 192000
MAX_CHANNELS = 8
MIN_ONSETS = 8
MIN_WINDOW_ONSETS = 15
MIN_DRIFT_WINDOWS = 5
MIN_DRIFT_WINDOW_SEC = 1.0
MIN_TREND_INLIER_RATIO = 0.8
STABLE_WINDOW_RATIO = 0.7
MIN_GRID_FIT = 0.1
WINDOW_SUPPORT_Z = 3.0
MIN_SUPPORT_WINDOWS = 3

DEFAULT_LOW_HZ = 1500.0
DEFAULT_HIGH_HZ = 6000.0
DEFAULT_HEAD_LEVEL = 0.15
DEFAULT_DRIFT_WINDOW_SEC = 10.0
SENSITIVITY_LEVELS = (0.05, 0.15, 0.3)

ENVELOPE_SMOOTH_MS = 0.7
SLOPE_WINDOW_MS = 1.5
SLOPE_THRESHOLD_RATIO = 0.15
SLOPE_PERCENTILE = 99.5
CAP_BLOCK_SEC = 4.0
CAP_FLOOR_RATIO = 0.1
MIN_ONSET_GAP_MS = 30.0
LOOKBACK_MS = 20.0
PEAK_WINDOW_MS = 12.0
MIN_ATTACK_RATIO = 1.5

KICK_LOW_HZ = 30.0
KICK_HIGH_HZ = 120.0
KICK_SMOOTH_MS = 2.0
KICK_PEAK_RATIO = 0.5
KICK_PEAK_PERCENTILE = 99.5
KICK_MIN_GAP_MS = 120.0
KICK_GAP_PERIOD_RATIO = 0.6
KICK_TARGET_RATE_HZ = 2000
KICK_WINDOW_BEFORE_MS = 15.0
KICK_WINDOW_AFTER_MS = 35.0

CLUSTER_BIN_MS = 1.0
CLUSTER_SMOOTH_BINS = 5
CLUSTER_RADIUS_MS = 6.0
CLUSTER_RADIUS_PERIOD_RATIO = 0.2
DISPLAY_BIN_MS = 3.0
DISPLAY_WIDTH = 40

CHECK_AGREE_MS = 3.0
CHECK_MIN_POINTS = 30
KICK_SLOPE_WINDOW_MS = 4.0
REFINE_RADIUS_MS = 15.0
FILTER_ZERO = "zero"
FILTER_CAUSAL = "causal"
FILTER_CHOICES = (FILTER_ZERO, FILTER_CAUSAL)
MIXTURE_AGREE_MS = 1.5
ALT_LOW_BAND = (500.0, 1500.0)
ALT_HIGH_START_HZ = 6000.0
ALT_HIGH_END_HZ = 12000.0
ALT_NYQUIST_RATIO = 0.45
ALT_MIN_BAND_RATIO = 1.5
SE_MEDIAN_FACTOR = 1.2533
MAD_TO_SIGMA = 1.4826
SE_CONFIDENCE_Z = 1.96

GROUP_REGION_MS = 8.0
GROUP_BIN_MS = 0.25
GROUP_SMOOTH_SIGMA_MS = 0.6
GROUP_MIN_HEIGHT_RATIO = 0.3
GROUP_MIN_PROMINENCE_RATIO = 0.2
GROUP_MIN_SEPARATION_MS = 1.5
GROUP_ASSIGN_RADIUS_MS = 2.0
GROUP_MIN_EVENTS = 30
GROUP_MIN_SHARE = 0.15
GROUP_MIN_WINDOW_EVENTS = 8
REFERENCE_SLOW = "slow"
REFERENCE_FAST = "fast"
REFERENCE_LARGEST = "largest"
REFERENCE_MIXED = "mixed"
REFERENCE_CHOICES = (
    REFERENCE_LARGEST,
    REFERENCE_FAST,
    REFERENCE_SLOW,
    REFERENCE_MIXED,
)
DEFAULT_REFERENCE_GROUP = REFERENCE_LARGEST
REFERENCE_DESCRIPTIONS = {
    REFERENCE_LARGEST: "件数が最大の群",
    REFERENCE_FAST: "立ち上がりが最も速い群",
    REFERENCE_SLOW: "立ち上がりが最も遅い群",
    REFERENCE_MIXED: "群に分けず全オンセットの混合値",
}

ESTIMATE_MIN_BPM = 60.0
ESTIMATE_MAX_BPM = 260.0
ESTIMATE_MIN_EVENTS = 32
ESTIMATE_BIN_MS = 2.0
ESTIMATE_PAD_FACTOR = 8
ESTIMATE_COARSE_STEP_BPM = 0.05
ESTIMATE_COARSE_HARMONICS = 8
ESTIMATE_FINE_HARMONICS = 12
ESTIMATE_FINE_SPAN_BPM = 0.3
ESTIMATE_FINE_STEP_BPM = 0.002
ESTIMATE_FINE_CHUNK = 32
ESTIMATE_OCTAVE_TOLERANCE = 0.03
ESTIMATE_OCTAVE_NEIGHBORHOOD = 0.003
ESTIMATE_MIN_DOMINANCE = 1.4
BPM_DECIMALS = 2

LABEL_ALL = "全オンセット"
LABEL_OTHER = "キック非同時(ハイハット/スネア等)"
LABEL_KICK = "キック同時"
LABEL_REFERENCE = "基準群"


@dataclass(frozen=True)
class AnalysisConfig:
    bpm: float
    subdivision: int
    low_hz: float
    high_hz: float
    head_level: float
    drift_window_sec: float
    reference_group: str = DEFAULT_REFERENCE_GROUP

    @property
    def period_ms(self) -> float:
        return 60000.0 / self.bpm / self.subdivision


@dataclass(frozen=True)
class GroupAnalysis:
    label: str
    heads_ms: np.ndarray
    offsets_ms: np.ndarray
    center_ms: float
    phase_ms: float
    spread_ms: float
    cluster_count: int
    concentration: float


@dataclass(frozen=True)
class WindowPhases:
    starts_ms: np.ndarray
    centers_ms: np.ndarray
    phases_ms: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class DetectorCheck:
    name: str
    phase_ms: float
    converted_ms: float
    difference_ms: float
    contrast: float
    uses_regression: bool


@dataclass(frozen=True)
class CrossCheck:
    rise_gap_ms: float
    checks: tuple[DetectorCheck, ...]

    @property
    def span_ms(self) -> float:
        return max(abs(check.difference_ms) for check in self.checks)

    @property
    def agreed(self) -> int:
        return sum(
            1 for check in self.checks if abs(check.difference_ms) <= CHECK_AGREE_MS
        )

    @property
    def all_agree(self) -> bool:
        return self.agreed == len(self.checks)


@dataclass(frozen=True)
class CheckBasis:
    detection: Detection
    label: str
    drift: DriftFit | None
    config: AnalysisConfig
    rise_gap_ms: float
    conclusion_ms: float
    uses_regression: bool


@dataclass(frozen=True)
class BandPhase:
    label: str
    phase_ms: float
    difference_ms: float


@dataclass(frozen=True)
class BandCheck:
    reference_ms: float
    bands: tuple[BandPhase, ...]
    ringing: BandPhase | None
    kick: BandPhase | None

    @property
    def span_ms(self) -> float:
        values = [0.0, *(band.difference_ms for band in self.bands)]
        return max(values) - min(values)


@dataclass(frozen=True)
class RefineBand:
    low_hz: float
    high_hz: float
    shift_ms: float
    count: int


@dataclass(frozen=True)
class RefineCheck:
    offset_ms: float
    shift_ms: float
    spread_ms: float
    count: int
    standard_error_ms: float
    bands: tuple[RefineBand, ...]


@dataclass(frozen=True)
class WindowSlices:
    starts_ms: np.ndarray
    lows: np.ndarray
    highs: np.ndarray

    @property
    def counts(self) -> np.ndarray:
        return self.highs - self.lows


@dataclass(frozen=True)
class PhaseGroup:
    peak_ms: float
    head_ms: float
    gap_ms: float
    spread_ms: float
    members: np.ndarray

    @property
    def count(self) -> int:
        return int(self.members.size)


@dataclass(frozen=True)
class ReferenceEstimate:
    groups: tuple[PhaseGroup, ...]
    chosen: int
    windows: WindowPhases | None
    window_counts: np.ndarray | None
    drift: DriftFit | None
    conclusion_ms: float
    uses_regression: bool
    stability: float | None
    grid_fit: float | None
    difference_ms: float
    sensitivity: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Heads:
    heads_ms: np.ndarray
    peaks_ms: np.ndarray


@dataclass(frozen=True)
class GridWarp:
    tempo: TempoMap | None = None
    subdivision: int = 1
    period_ms: float = 0.0
    anchor_ms: float = 0.0

    def shifts(self, times_ms: np.ndarray) -> np.ndarray:
        if self.tempo is None:
            return np.zeros(np.shape(times_ms))
        index = self.tempo.grid_index(times_ms - self.anchor_ms, self.subdivision)
        return index * self.period_ms - self.tempo.line_ms(index, self.subdivision)

    def times(self, times_ms: np.ndarray) -> np.ndarray:
        return times_ms + self.shifts(times_ms)

    def kicks(self, kick_ms: np.ndarray) -> np.ndarray:
        return np.sort(self.times(kick_ms))

    def heads(self, heads: Heads) -> Heads:
        shift = self.shifts(heads.peaks_ms)
        return Heads(heads.heads_ms + shift, heads.peaks_ms + shift)


IDENTITY_WARP = GridWarp()


@dataclass(frozen=True)
class KickEnvelope:
    envelope: np.ndarray
    sample_rate: float
    peak_height: float


@dataclass(frozen=True)
class LocalCap:
    values: np.ndarray
    block: int

    @property
    def usable(self) -> bool:
        return bool(self.values.min() > 0.0)

    def at(self, indices: np.ndarray) -> np.ndarray:
        return self.values[np.minimum(indices // self.block, self.values.size - 1)]


@dataclass(frozen=True)
class Detection:
    envelope: np.ndarray
    slope: np.ndarray
    caps: LocalCap
    peaks: np.ndarray
    kick_ms: np.ndarray
    sample_rate: int
    heads: dict[float, Heads]


@dataclass(frozen=True)
class Analysis:
    period_ms: float
    groups: dict[str, GroupAnalysis]
    primary: GroupAnalysis
    windows: WindowPhases | None
    drift: DriftFit | None
    sensitivity: tuple[tuple[float, float | None], ...]
    conclusion_ms: float
    uses_regression: bool
    cross: CrossCheck | None
    stability: float | None
    grid_fit: float | None
    reference: ReferenceEstimate | None

    @property
    def final_ms(self) -> float:
        if self.reference is None:
            return self.conclusion_ms
        return self.reference.conclusion_ms

    @property
    def final_uses_regression(self) -> bool:
        if self.reference is None:
            return self.uses_regression
        return self.reference.uses_regression

    @property
    def final_stability(self) -> float | None:
        if self.reference is None:
            return self.stability
        return self.reference.stability

    @property
    def final_grid_fit(self) -> float | None:
        if self.reference is None:
            return self.grid_fit
        return self.reference.grid_fit

    @property
    def final_label(self) -> str:
        return self.primary.label if self.reference is None else LABEL_REFERENCE

    @property
    def final_standard_error_ms(self) -> float:
        if self.reference is None:
            events = standard_error_ms(self.primary.spread_ms, self.primary.cluster_count)
            residuals = window_residuals(
                self.windows,
                self.drift,
                self.conclusion_ms,
                self.uses_regression,
                self.period_ms,
            )
            return max(events, window_standard_error_ms(residuals))
        reference = self.reference
        group = reference.groups[reference.chosen]
        residuals = window_residuals(
            reference.windows,
            reference.drift,
            reference.conclusion_ms,
            reference.uses_regression,
            self.period_ms,
        )
        return max(
            standard_error_ms(group.spread_ms, group.count),
            window_standard_error_ms(residuals),
        )


@dataclass(frozen=True)
class EventSupport:
    stability: float | None
    grid_fit: float


@dataclass(frozen=True)
class OnsetEvents:
    times_ms: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class BpmEstimate:
    bpm: float
    dominance: float | None
    octaves: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DecodeInfo:
    suffix: str
    sample_rate: int
    applied_skip: int


@dataclass(frozen=True)
class DecoderVariant:
    label: str
    skip_samples: int


def audio_suffix(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    return SUFFIX_ALIASES.get(suffix, suffix)


def signed(value: float) -> str:
    return f"{round(float(value), 2) + 0.0:+.2f}"


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


def bandpass_envelope(
    samples: np.ndarray,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
    smooth_ms: float,
    causal: bool = False,
) -> np.ndarray:
    if causal:
        return causal_envelope(samples, sample_rate, low_hz, high_hz, smooth_ms)
    sos = signal.butter(
        4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    filtered = signal.sosfiltfilt(sos, samples)
    analytic = signal.hilbert(filtered, N=fft.next_fast_len(filtered.size))
    envelope = np.abs(analytic[: filtered.size])
    width = ms_to_samples(smooth_ms, sample_rate)
    return ndimage.uniform_filter1d(envelope, width, mode="nearest")


def peak_envelope(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    smooth_ms: float,
    causal: bool = False,
) -> np.ndarray:
    merged = np.zeros(samples.shape[0])
    for channel in samples.T:
        current = bandpass_envelope(
            channel, sample_rate, low_hz, high_hz, smooth_ms, causal
        )
        np.maximum(merged, current, out=merged)
    return merged


def envelope_slope(envelope: np.ndarray, sample_rate: int) -> np.ndarray:
    window = ms_to_samples(SLOPE_WINDOW_MS, sample_rate) | 1
    return signal.savgol_filter(envelope, window, 2, deriv=1)


def locate_head(
    envelope: np.ndarray,
    peak: int,
    lookback: int,
    forward: int,
    head_level: float,
) -> int | None:
    start = max(0, peak - lookback)
    if peak - start < 1:
        return None
    base = envelope[start:peak].min()
    top = envelope[peak : peak + forward].max()
    if top <= base * MIN_ATTACK_RATIO:
        return None
    level = base + head_level * (top - base)
    below = np.flatnonzero(envelope[start : peak + 1] < level)
    return start + (int(below[-1]) + 1 if below.size else 0)


def attack_peaks(
    slope: np.ndarray, caps: LocalCap, sample_rate: int
) -> np.ndarray:
    if not caps.usable:
        return np.empty(0, dtype=np.intp)
    peaks, _ = signal.find_peaks(
        slope,
        height=SLOPE_THRESHOLD_RATIO * float(caps.values.min()),
        distance=ms_to_samples(MIN_ONSET_GAP_MS, sample_rate),
    )
    return peaks[slope[peaks] >= SLOPE_THRESHOLD_RATIO * caps.at(peaks)]


def heads_ms_from_peaks(
    envelope: np.ndarray, peaks: np.ndarray, sample_rate: int, head_level: float
) -> tuple[np.ndarray, np.ndarray]:
    lookback = ms_to_samples(LOOKBACK_MS, sample_rate)
    forward = ms_to_samples(PEAK_WINDOW_MS, sample_rate)
    heads = [
        locate_head(envelope, int(peak), lookback, forward, head_level)
        for peak in peaks
    ]
    valid = np.fromiter((head is not None for head in heads), dtype=bool, count=len(heads))
    found = np.fromiter((head for head in heads if head is not None), dtype=float)
    scale = 1000.0 / sample_rate
    return found * scale, peaks[valid] * scale


def head_levels(head_level: float) -> tuple[float, ...]:
    return tuple(sorted({*SENSITIVITY_LEVELS, DEFAULT_HEAD_LEVEL, head_level}))


def build_heads(
    envelope: np.ndarray,
    peaks: np.ndarray,
    sample_rate: int,
    levels: tuple[float, ...],
) -> dict[float, Heads]:
    return {
        level: Heads(*heads_ms_from_peaks(envelope, peaks, sample_rate, level))
        for level in levels
    }


def kick_envelope(samples: np.ndarray, sample_rate: int) -> KickEnvelope:
    factor = max(1, sample_rate // KICK_TARGET_RATE_HZ)
    reduced = signal.resample_poly(samples.mean(axis=1), 1, factor)
    reduced_rate = sample_rate / factor
    envelope = bandpass_envelope(
        reduced, reduced_rate, KICK_LOW_HZ, KICK_HIGH_HZ, KICK_SMOOTH_MS
    )
    height = KICK_PEAK_RATIO * float(np.percentile(envelope, KICK_PEAK_PERCENTILE))
    return KickEnvelope(envelope, reduced_rate, height)


def kick_peaks_ms(kick: KickEnvelope, period_ms: float) -> np.ndarray:
    gap_ms = min(KICK_MIN_GAP_MS, KICK_GAP_PERIOD_RATIO * period_ms)
    peaks, _ = signal.find_peaks(
        kick.envelope,
        height=kick.peak_height,
        distance=ms_to_samples(gap_ms, kick.sample_rate),
    )
    return peaks * 1000.0 / kick.sample_rate


def kick_coincident_mask(times_ms: np.ndarray, kick_ms: np.ndarray) -> np.ndarray:
    if kick_ms.size == 0:
        return np.zeros(times_ms.shape, dtype=bool)
    left = np.searchsorted(kick_ms, times_ms - KICK_WINDOW_BEFORE_MS, side="right")
    right = np.searchsorted(kick_ms, times_ms + KICK_WINDOW_AFTER_MS, side="left")
    return right > left


def split_by_kick(
    heads_ms: np.ndarray, kick_ms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coincident = kick_coincident_mask(heads_ms, kick_ms)
    return heads_ms[~coincident], heads_ms[coincident]


def primary_keep_mask(
    times_ms: np.ndarray, kick_ms: np.ndarray, label: str
) -> np.ndarray:
    if label == LABEL_OTHER:
        return ~kick_coincident_mask(times_ms, kick_ms)
    if label == LABEL_KICK:
        return kick_coincident_mask(times_ms, kick_ms)
    return np.ones(times_ms.shape, dtype=bool)


def circular_mode(offsets: np.ndarray, period: float) -> float:
    bins = max(CLUSTER_SMOOTH_BINS, int(round(period / CLUSTER_BIN_MS)))
    counts, edges = np.histogram(offsets, bins=bins, range=(-period / 2.0, period / 2.0))
    smoothed = ndimage.uniform_filter1d(
        counts.astype(float), CLUSTER_SMOOTH_BINS, mode="wrap"
    )
    best = int(np.argmax(smoothed))
    return float((edges[best] + edges[best + 1]) / 2.0)


def recentered_offsets(
    heads_ms: np.ndarray, period: float
) -> tuple[np.ndarray, float]:
    offsets = wrap_symmetric(heads_ms, period)
    center = circular_mode(offsets, period)
    return wrap_symmetric(offsets - center, period) + center, center


def cluster_radius(period: float) -> float:
    return min(CLUSTER_RADIUS_MS, CLUSTER_RADIUS_PERIOD_RATIO * period)


def chance_stability(period: float) -> float:
    return min(1.0, 2.0 * cluster_radius(period) / period)


def excess_stability(stability: float, period: float) -> float:
    chance = chance_stability(period)
    return (stability - chance) / (1.0 - chance)


def is_stable_value(stability: float | None, grid_fit: float | None) -> bool:
    return (
        stability is not None
        and grid_fit is not None
        and stability >= STABLE_WINDOW_RATIO
        and grid_fit >= MIN_GRID_FIT
    )


def screening_fit(peaks_ms: np.ndarray, period: float) -> float:
    offsets, center = recentered_offsets(peaks_ms, period)
    inside = float(np.mean(np.abs(offsets - center) < cluster_radius(period)))
    return max(0.0, excess_stability(inside, period))


def standard_error_ms(spread_ms: float, count: int) -> float:
    return float(SE_MEDIAN_FACTOR * MAD_TO_SIGMA * spread_ms / np.sqrt(max(count, 1)))


def dominant_cluster(
    offsets: np.ndarray, center: float, period: float
) -> tuple[float, float, int]:
    members = offsets[np.abs(offsets - center) < cluster_radius(period)]
    median = float(np.median(members))
    spread = float(np.median(np.abs(members - median)))
    return median, spread, int(members.size)


def concentration_score(count: int, total: int, period: float) -> float:
    chance = min(1.0, 2.0 * cluster_radius(period) / period)
    deviation = np.sqrt(max(total * chance * (1.0 - chance), 1.0))
    return float((count - total * chance) / deviation)


def analyze_group(
    label: str, heads_ms: np.ndarray, period: float
) -> GroupAnalysis | None:
    if heads_ms.size < MIN_ONSETS:
        return None
    offsets, center = recentered_offsets(heads_ms, period)
    phase, spread, count = dominant_cluster(offsets, center, period)
    return GroupAnalysis(
        label,
        heads_ms,
        offsets,
        center,
        float(wrap_symmetric(phase, period)),
        spread,
        count,
        concentration_score(count, heads_ms.size, period),
    )


def build_groups(
    heads_ms: np.ndarray, kick_ms: np.ndarray, period: float
) -> dict[str, GroupAnalysis]:
    other_ms, kick_coincident_ms = split_by_kick(heads_ms, kick_ms)
    candidates = (
        (LABEL_ALL, heads_ms),
        (LABEL_OTHER, other_ms),
        (LABEL_KICK, kick_coincident_ms),
    )
    analyzed = (analyze_group(label, values, period) for label, values in candidates)
    return {group.label: group for group in analyzed if group is not None}


def pick_primary(groups: dict[str, GroupAnalysis]) -> GroupAnalysis | None:
    if not groups:
        return None
    return max(groups.values(), key=lambda group: group.concentration)


def window_phases(
    group: GroupAnalysis, period: float, window_sec: float
) -> WindowPhases | None:
    window_ms = window_sec * 1000.0
    starts = np.arange(0.0, group.heads_ms[-1], window_ms)
    lows = np.searchsorted(group.heads_ms, starts)
    highs = np.searchsorted(group.heads_ms, starts + window_ms)
    counts = highs - lows
    keep = counts >= MIN_WINDOW_ONSETS
    centers, phases = [], []
    for low, high in zip(lows[keep], highs[keep]):
        heads = group.heads_ms[low:high]
        offsets, center = recentered_offsets(heads, period)
        phases.append(dominant_cluster(offsets, center, period)[0])
        centers.append(float(np.median(heads)))
    if not phases:
        return None
    continuous = np.unwrap(wrap_symmetric(np.asarray(phases), period), period=period)
    return WindowPhases(starts[keep], np.asarray(centers), continuous, counts[keep])


def inlier_ratio(residuals: np.ndarray, period: float) -> float:
    return float(np.mean(np.abs(residuals) < cluster_radius(period)))


def fit_drift(windows: WindowPhases | None, period: float) -> DriftFit | None:
    if windows is None or windows.centers_ms.size < MIN_DRIFT_WINDOWS:
        return None
    slope, intercept, low, high = stats.theilslopes(
        windows.phases_ms, windows.centers_ms
    )
    residuals = windows.phases_ms - (intercept + slope * windows.centers_ms)
    if inlier_ratio(residuals, period) < MIN_TREND_INLIER_RATIO:
        return None
    reference = float(np.average(windows.centers_ms, weights=windows.counts))
    return DriftFit(float(intercept), float(slope), float(low), float(high), reference)


def window_residuals(
    windows: WindowPhases | None,
    drift: DriftFit | None,
    conclusion_ms: float,
    uses_regression: bool,
    period: float,
) -> np.ndarray | None:
    if windows is None:
        return None
    trend = (
        drift.slope * (windows.centers_ms - drift.reference_ms)
        if uses_regression and drift is not None
        else 0.0
    )
    return np.asarray(wrap_symmetric(windows.phases_ms - conclusion_ms - trend, period))


def regression_trend_ms(
    times_ms: np.ndarray, drift: DriftFit | None, uses_regression: bool
) -> np.ndarray:
    if not uses_regression or drift is None:
        return np.zeros(times_ms.shape)
    return drift.slope * (times_ms - drift.reference_ms)


def support_window_ms(config: AnalysisConfig) -> float:
    seconds = (
        config.drift_window_sec
        if config.drift_window_sec > 0.0
        else DEFAULT_DRIFT_WINDOW_SEC
    )
    return seconds * 1000.0


def supported_window_ratio(
    times_ms: np.ndarray, inside: np.ndarray, period: float, window_ms: float
) -> float | None:
    edges = np.arange(0.0, times_ms[-1] + window_ms, window_ms)
    bounds = np.searchsorted(times_ms, edges)
    scores = [
        concentration_score(int(inside[low:high].sum()), int(high - low), period)
        for low, high in zip(bounds[:-1], bounds[1:])
        if high - low >= MIN_WINDOW_ONSETS
    ]
    if len(scores) < MIN_SUPPORT_WINDOWS:
        return None
    return float(np.mean(np.asarray(scores) >= WINDOW_SUPPORT_Z))


def event_support(
    selected: Heads,
    conclusion_ms: float,
    drift: DriftFit | None,
    uses_regression: bool,
    period: float,
    window_ms: float,
) -> EventSupport:
    order = np.argsort(selected.peaks_ms)
    times_ms = selected.peaks_ms[order]
    trend = regression_trend_ms(times_ms, drift, uses_regression)
    residuals = wrap_symmetric(selected.heads_ms[order] - trend - conclusion_ms, period)
    inside = np.abs(residuals) < cluster_radius(period)
    return EventSupport(
        supported_window_ratio(times_ms, inside, period, window_ms),
        max(0.0, excess_stability(float(inside.mean()), period)),
    )


def window_standard_error_ms(residuals: np.ndarray | None) -> float:
    if residuals is None or residuals.size < MIN_DRIFT_WINDOWS:
        return 0.0
    spread = float(np.median(np.abs(residuals - np.median(residuals))))
    return standard_error_ms(spread, residuals.size)


def estimate_drift(
    heads_ms: np.ndarray, windows: WindowPhases | None, config: AnalysisConfig
) -> DriftFit | None:
    if config.drift_window_sec <= 0.0:
        return None
    period = config.period_ms
    direct = fit_direct(
        heads_ms,
        np.ones(heads_ms.size),
        period,
        cluster_radius(period),
        config.drift_window_sec * 1000.0,
    )
    return direct if direct is not None else fit_drift(windows, period)


def regression_phase(fit: DriftFit, period: float) -> float:
    return float(wrap_symmetric(fit.phase_at(fit.reference_ms), period))


def resolve_phase(
    fit: DriftFit | None, fallback_ms: float, period: float
) -> tuple[float, bool]:
    if fit is not None and fit.significant:
        return regression_phase(fit, period), True
    return fallback_ms, False


def median_rise_gap(
    heads_ms: np.ndarray, peaks_ms: np.ndarray, kick_ms: np.ndarray, label: str
) -> float | None:
    keep = primary_keep_mask(heads_ms, kick_ms, label)
    if not keep.any():
        return None
    return float(np.median(peaks_ms[keep] - heads_ms[keep]))


def detector_check(
    function: OnsetFunction, basis: CheckBasis
) -> DetectorCheck | None:
    period = basis.config.period_ms
    keep = primary_keep_mask(
        function.times_ms, basis.detection.kick_ms, basis.label
    )
    times_ms, weights = function.times_ms[keep], function.weights[keep]
    if times_ms.size < CHECK_MIN_POINTS:
        return None
    drift = basis.drift if basis.uses_regression else None
    slope = 0.0 if drift is None else drift.slope
    reference = 0.0 if drift is None else drift.reference_ms
    folded, contrast = fold_phase(times_ms * (1.0 - slope), weights, period)
    phase = float(wrap_symmetric(folded + slope * reference, period))
    converted = float(wrap_symmetric(phase - basis.rise_gap_ms, period))
    difference = float(wrap_symmetric(converted - basis.conclusion_ms, period))
    return DetectorCheck(
        function.name, phase, converted, difference, contrast, drift is not None
    )


def primary_check_basis(
    detection: Detection,
    selected: Heads,
    primary: GroupAnalysis,
    drift: DriftFit | None,
    config: AnalysisConfig,
    conclusion_ms: float,
    uses_regression: bool,
) -> CheckBasis | None:
    gap = median_rise_gap(
        selected.heads_ms, selected.peaks_ms, detection.kick_ms, primary.label
    )
    if gap is None:
        return None
    return CheckBasis(
        detection, primary.label, drift, config, gap, conclusion_ms, uses_regression
    )


def reference_check_basis(
    detection: Detection, config: AnalysisConfig, reference: ReferenceEstimate
) -> CheckBasis:
    return CheckBasis(
        detection,
        LABEL_ALL,
        reference.drift,
        config,
        reference.groups[reference.chosen].gap_ms,
        reference.conclusion_ms,
        reference.uses_regression,
    )


def build_cross_check(
    functions: tuple[OnsetFunction, ...], basis: CheckBasis | None
) -> CrossCheck | None:
    if basis is None:
        return None
    results = (detector_check(function, basis) for function in functions)
    checks = tuple(check for check in results if check is not None)
    return CrossCheck(basis.rise_gap_ms, checks) if checks else None


def alternate_bands(sample_rate: int) -> tuple[tuple[float, float], ...]:
    high = (ALT_HIGH_START_HZ, min(ALT_HIGH_END_HZ, ALT_NYQUIST_RATIO * sample_rate))
    return tuple(
        (low, upper)
        for low, upper in (ALT_LOW_BAND, high)
        if upper >= ALT_MIN_BAND_RATIO * low
    )


def detected_heads(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    level: float,
    causal: bool,
) -> np.ndarray:
    detection = detect_attacks(samples, sample_rate, low_hz, high_hz, (level,), causal)
    return detection.heads[level].heads_ms


def phase_entry(
    label: str, heads_ms: np.ndarray, base_phase: float, period: float
) -> BandPhase | None:
    group = analyze_group(LABEL_ALL, heads_ms, period)
    if group is None:
        return None
    difference = float(wrap_symmetric(group.phase_ms - base_phase, period))
    return BandPhase(label, group.phase_ms, difference)


def kick_band_heads(kick: KickEnvelope, period: float, head_level: float) -> np.ndarray:
    window = ms_to_samples(KICK_SLOPE_WINDOW_MS, kick.sample_rate) | 1
    slope = signal.savgol_filter(kick.envelope, window, 2, deriv=1)
    cap = float(np.percentile(slope, SLOPE_PERCENTILE))
    if cap <= 0.0:
        return np.empty(0)
    gap_ms = min(KICK_MIN_GAP_MS, KICK_GAP_PERIOD_RATIO * period)
    peaks, _ = signal.find_peaks(
        slope,
        height=SLOPE_THRESHOLD_RATIO * cap,
        distance=ms_to_samples(gap_ms, kick.sample_rate),
    )
    return heads_ms_from_peaks(kick.envelope, peaks, kick.sample_rate, head_level)[0]


def band_check(
    samples: np.ndarray,
    sample_rate: int,
    config: AnalysisConfig,
    analysis: Analysis,
    kick: KickEnvelope,
    causal: bool,
    warp: GridWarp = IDENTITY_WARP,
) -> BandCheck | None:
    base = analysis.groups.get(LABEL_ALL)
    if base is None:
        return None
    period = config.period_ms
    level = config.head_level
    entries = (
        phase_entry(
            f"{low:g}~{high:g} Hz",
            warp.times(detected_heads(samples, sample_rate, low, high, level, causal)),
            base.phase_ms,
            period,
        )
        for low, high in alternate_bands(sample_rate)
    )
    bands = tuple(entry for entry in entries if entry is not None)
    variant = "零位相" if causal else "因果"
    ringing = phase_entry(
        f"{config.low_hz:g}~{config.high_hz:g} Hz {variant}フィルタ",
        warp.times(
            detected_heads(
                samples, sample_rate, config.low_hz, config.high_hz, level, not causal
            )
        ),
        base.phase_ms,
        period,
    )
    kick_entry = phase_entry(
        f"{KICK_LOW_HZ:g}~{KICK_HIGH_HZ:g} Hz キック帯域(独立検出)",
        warp.times(kick_band_heads(kick, period, level)),
        base.phase_ms,
        period,
    )
    if not bands and ringing is None and kick_entry is None:
        return None
    return BandCheck(base.phase_ms, bands, ringing, kick_entry)


def event_trend_ms(analysis: Analysis, times_ms: np.ndarray) -> np.ndarray:
    drift = analysis.reference.drift if analysis.reference is not None else analysis.drift
    return regression_trend_ms(times_ms, drift, analysis.final_uses_regression)


def event_residuals(
    values_ms: np.ndarray, times_ms: np.ndarray, analysis: Analysis, period: float
) -> np.ndarray:
    trend = event_trend_ms(analysis, times_ms)
    return np.asarray(wrap_symmetric(values_ms - trend - analysis.final_ms, period))


def refine_members(
    analysis: Analysis, selected: Heads, kick_ms: np.ndarray, period: float
) -> np.ndarray:
    if analysis.reference is not None:
        return analysis.reference.groups[analysis.reference.chosen].members
    residual = event_residuals(selected.heads_ms, selected.peaks_ms, analysis, period)
    keep = primary_keep_mask(selected.heads_ms, kick_ms, analysis.primary.label)
    return np.flatnonzero(keep & (np.abs(residual) < cluster_radius(period)))


def within_radius(residuals: np.ndarray) -> np.ndarray:
    return residuals[np.abs(residuals) <= REFINE_RADIUS_MS]


def refine_check(
    samples: np.ndarray,
    sample_rate: int,
    attacks: Detection,
    kick: KickEnvelope,
    config: AnalysisConfig,
    analysis: Analysis,
    warp: GridWarp = IDENTITY_WARP,
) -> RefineCheck | None:
    period = config.period_ms
    kick_ms = warp.kicks(kick_peaks_ms(kick, period))
    real = attacks.heads[config.head_level]
    selected = warp.heads(real)
    members = refine_members(analysis, selected, kick_ms, period)
    if members.size < MIN_ONSETS:
        return None
    times_ms = real.peaks_ms[members]
    shift = selected.peaks_ms[members] - times_ms
    refined = refine_heads(samples, sample_rate, times_ms, config.head_level)
    if refined is None:
        return None
    warped_ms = times_ms + shift
    combined = within_radius(
        event_residuals(refined.heads_ms + shift, warped_ms, analysis, period)
    )
    if combined.size < MIN_ONSETS:
        return None
    shift = float(np.median(combined))
    spread = float(np.median(np.abs(combined - shift)))
    bands = []
    for (low, high), heads in zip(refined.bands, refined.band_heads_ms):
        residual = within_radius(event_residuals(heads + shift, warped_ms, analysis, period))
        if residual.size >= MIN_ONSETS:
            bands.append(RefineBand(low, high, float(np.median(residual)), residual.size))
    return RefineCheck(
        float(wrap_symmetric(analysis.final_ms + shift, period)),
        shift,
        spread,
        combined.size,
        standard_error_ms(spread, combined.size),
        tuple(bands),
    )


def head_level_sensitivity(
    detection: Detection, period: float, label: str
) -> tuple[tuple[float, float | None], ...]:
    results = []
    for level in SENSITIVITY_LEVELS:
        heads_ms = detection.heads[level].heads_ms
        group = build_groups(heads_ms, detection.kick_ms, period).get(label)
        results.append((level, None if group is None else group.phase_ms))
    return tuple(results)


def window_slices(
    heads_ms: np.ndarray, window_ms: float, window_count: int
) -> WindowSlices:
    starts = np.arange(window_count) * window_ms
    return WindowSlices(
        starts,
        np.searchsorted(heads_ms, starts),
        np.searchsorted(heads_ms, starts + window_ms),
    )


def group_modes(offsets: np.ndarray, center: float) -> np.ndarray:
    edges = np.arange(
        center - GROUP_REGION_MS, center + GROUP_REGION_MS + GROUP_BIN_MS, GROUP_BIN_MS
    )
    counts, _ = np.histogram(offsets, bins=edges)
    smoothed = ndimage.gaussian_filter1d(
        counts.astype(float), GROUP_SMOOTH_SIGMA_MS / GROUP_BIN_MS, mode="nearest"
    )
    top = float(smoothed.max())
    if top <= 0.0:
        return np.empty(0)
    peaks, _ = signal.find_peaks(
        smoothed,
        height=GROUP_MIN_HEIGHT_RATIO * top,
        prominence=GROUP_MIN_PROMINENCE_RATIO * top,
        distance=max(1, int(round(GROUP_MIN_SEPARATION_MS / GROUP_BIN_MS))),
    )
    return (edges[:-1] + edges[1:])[peaks] / 2.0


def assign_groups(offsets: np.ndarray, modes: np.ndarray) -> np.ndarray:
    distances = np.abs(offsets[:, None] - modes[None, :])
    nearest = np.argmin(distances, axis=1)
    within = distances[np.arange(offsets.size), nearest] <= GROUP_ASSIGN_RADIUS_MS
    return np.where(within, nearest, -1)


def build_phase_groups(
    offsets: np.ndarray, gaps_ms: np.ndarray, center: float
) -> tuple[PhaseGroup, ...]:
    modes = group_modes(offsets, center)
    if modes.size == 0:
        return ()
    labels = assign_groups(offsets, modes)
    groups = []
    for index in range(modes.size):
        members = np.flatnonzero(labels == index)
        if members.size == 0:
            continue
        heads = offsets[members] - gaps_ms[members]
        groups.append(
            PhaseGroup(
                float(np.median(offsets[members])),
                float(np.median(heads)),
                float(np.median(gaps_ms[members])),
                float(np.median(np.abs(heads - np.median(heads)))),
                members,
            )
        )
    return tuple(groups)


def choose_group(groups: tuple[PhaseGroup, ...], rule: str) -> int | None:
    grouped = sum(group.count for group in groups)
    eligible = [
        index
        for index, group in enumerate(groups)
        if group.count >= GROUP_MIN_EVENTS and group.count >= GROUP_MIN_SHARE * grouped
    ]
    if not eligible:
        return None
    if rule == REFERENCE_FAST:
        return min(eligible, key=lambda index: groups[index].gap_ms)
    if rule == REFERENCE_SLOW:
        return max(eligible, key=lambda index: groups[index].gap_ms)
    return max(eligible, key=lambda index: groups[index].count)


def group_window_counts(
    times_ms: np.ndarray, starts_ms: np.ndarray, window_ms: float
) -> np.ndarray:
    return np.searchsorted(times_ms, starts_ms + window_ms) - np.searchsorted(
        times_ms, starts_ms
    )


def reference_windows(
    peaks_ms: np.ndarray,
    offsets: np.ndarray,
    groups: tuple[PhaseGroup, ...],
    chosen: int,
    window_ms: float,
    window_count: int,
) -> tuple[WindowPhases | None, np.ndarray | None]:
    group = groups[chosen]
    times = peaks_ms[group.members]
    values = offsets[group.members]
    slices = window_slices(times, window_ms, window_count)
    kept = slices.counts >= GROUP_MIN_WINDOW_EVENTS
    if not kept.any():
        return None, None
    bounds = list(zip(slices.lows[kept], slices.highs[kept]))
    phases = np.array([np.median(values[low:high]) for low, high in bounds]) - group.gap_ms
    centers = np.array([np.median(times[low:high]) for low, high in bounds])
    starts = slices.starts_ms[kept]
    counts = np.stack(
        [
            group_window_counts(peaks_ms[other.members], starts, window_ms)
            for other in groups
        ],
        axis=1,
    )
    return WindowPhases(starts, centers, phases, slices.counts[kept]), counts


def reference_sensitivity(
    detection: Detection,
    selected: Heads,
    offsets: np.ndarray,
    group: PhaseGroup,
    period: float,
) -> tuple[tuple[float, float], ...]:
    group_peaks = selected.peaks_ms[group.members]
    results = []
    for level in SENSITIVITY_LEVELS:
        heads = detection.heads[level]
        inside = np.isin(heads.peaks_ms, group_peaks)
        if not inside.any():
            continue
        indices = np.searchsorted(selected.peaks_ms, heads.peaks_ms[inside])
        phases = offsets[indices] - (heads.peaks_ms[inside] - heads.heads_ms[inside])
        results.append((level, float(wrap_symmetric(np.median(phases), period))))
    return tuple(results)


def event_window_layout(peaks_ms: np.ndarray, window_sec: float) -> tuple[float, int]:
    window_ms = window_sec * 1000.0
    return window_ms, max(1, int(np.ceil((float(peaks_ms[-1]) + 1.0) / window_ms)))


def group_trend_ms(peaks_ms: np.ndarray, drift: DriftFit | None) -> np.ndarray:
    if drift is None or not drift.significant:
        return np.zeros(peaks_ms.shape)
    return drift.slope * (peaks_ms - drift.reference_ms)


def estimate_reference(
    detection: Detection,
    selected: Heads,
    config: AnalysisConfig,
    reference_ms: float,
    mixed_drift: DriftFit | None,
) -> ReferenceEstimate | None:
    period = config.period_ms
    if config.reference_group == REFERENCE_MIXED:
        return None
    if selected.peaks_ms.size < MIN_ONSETS:
        return None
    trend = group_trend_ms(selected.peaks_ms, mixed_drift)
    offsets, center = recentered_offsets(selected.peaks_ms - trend, period)
    groups = build_phase_groups(offsets, selected.peaks_ms - selected.heads_ms, center)
    chosen = choose_group(groups, config.reference_group)
    if chosen is None:
        return None
    windows, window_counts = (None, None)
    if config.drift_window_sec > 0.0:
        window_ms, window_count = event_window_layout(
            selected.peaks_ms, config.drift_window_sec
        )
        windows, window_counts = reference_windows(
            selected.peaks_ms, offsets + trend, groups, chosen, window_ms, window_count
        )
    drift = estimate_drift(selected.heads_ms[groups[chosen].members], windows, config)
    fallback_ms = float(wrap_symmetric(groups[chosen].head_ms, period))
    conclusion_ms, uses_regression = resolve_phase(drift, fallback_ms, period)
    support = event_support(
        selected,
        conclusion_ms,
        drift,
        uses_regression,
        period,
        support_window_ms(config),
    )
    return ReferenceEstimate(
        groups,
        chosen,
        windows,
        window_counts,
        drift,
        conclusion_ms,
        uses_regression,
        support.stability,
        support.grid_fit,
        float(wrap_symmetric(conclusion_ms - reference_ms, period)),
        reference_sensitivity(detection, selected, offsets, groups[chosen], period),
    )


def detect_attacks(
    samples: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    levels: tuple[float, ...],
    causal: bool = False,
) -> Detection:
    envelope = peak_envelope(
        samples, sample_rate, low_hz, high_hz, ENVELOPE_SMOOTH_MS, causal
    )
    slope = envelope_slope(envelope, sample_rate)
    block = ms_to_samples(CAP_BLOCK_SEC * 1000.0, sample_rate)
    caps = LocalCap(
        block_caps(slope, block, SLOPE_PERCENTILE, CAP_FLOOR_RATIO), block
    )
    peaks = attack_peaks(slope, caps, sample_rate)
    return Detection(
        envelope,
        slope,
        caps,
        peaks,
        np.empty(0),
        sample_rate,
        build_heads(envelope, peaks, sample_rate, levels),
    )


def attach_kicks(
    detection: Detection, kick: KickEnvelope, period_ms: float
) -> Detection:
    return replace(detection, kick_ms=kick_peaks_ms(kick, period_ms))


def attack_events(detection: Detection) -> OnsetEvents:
    default_heads = detection.heads[DEFAULT_HEAD_LEVEL]
    heads_ms, peaks_ms = default_heads.heads_ms, default_heads.peaks_ms
    if heads_ms.size == 0 or not detection.caps.usable:
        return OnsetEvents(np.empty(0), np.empty(0))
    indices = np.round(peaks_ms * detection.sample_rate / 1000.0).astype(int)
    strengths = np.clip(detection.slope[indices] / detection.caps.at(indices), 0.0, 1.0)
    return OnsetEvents(heads_ms, strengths)


def onset_spectrum(events: OnsetEvents) -> tuple[np.ndarray, np.ndarray]:
    indices = np.round(events.times_ms / ESTIMATE_BIN_MS).astype(int)
    train = np.bincount(indices, weights=events.weights)
    train = train - train.mean()
    size = fft.next_fast_len(train.size * ESTIMATE_PAD_FACTOR, real=True)
    magnitude = np.abs(fft.rfft(train, size))
    frequencies = fft.rfftfreq(size, ESTIMATE_BIN_MS / 1000.0)
    return frequencies, magnitude


def harmonic_salience(
    frequencies: np.ndarray, magnitude: np.ndarray, bpms: np.ndarray
) -> np.ndarray:
    fundamental = bpms / 60.0
    total = np.zeros_like(fundamental)
    for harmonic in range(1, ESTIMATE_COARSE_HARMONICS + 1):
        total += np.interp(harmonic * fundamental, frequencies, magnitude)
    return total


def local_salience(
    frequencies: np.ndarray, magnitude: np.ndarray, bpm: float
) -> float:
    offsets = np.linspace(
        -ESTIMATE_OCTAVE_NEIGHBORHOOD, ESTIMATE_OCTAVE_NEIGHBORHOOD, 25
    )
    return float(harmonic_salience(frequencies, magnitude, bpm * (1.0 + offsets)).max())


def octave_distance(ratios: np.ndarray) -> np.ndarray:
    exponent = np.log2(ratios)
    return np.abs(exponent - np.round(exponent))


def dominance_ratio(
    bpms: np.ndarray, salience: np.ndarray, best_index: int
) -> float | None:
    distance = octave_distance(bpms / bpms[best_index])
    rivals = salience[distance > ESTIMATE_OCTAVE_TOLERANCE]
    if rivals.size == 0 or rivals.max() <= 0.0:
        return None
    return float(salience[best_index] / rivals.max())


def refine_bpm(events: OnsetEvents, center: float) -> float:
    grid = np.arange(
        center - ESTIMATE_FINE_SPAN_BPM,
        center + ESTIMATE_FINE_SPAN_BPM + ESTIMATE_FINE_STEP_BPM / 2.0,
        ESTIMATE_FINE_STEP_BPM,
    )
    seconds = events.times_ms / 1000.0
    scores = np.empty(grid.size)
    for start in range(0, grid.size, ESTIMATE_FINE_CHUNK):
        chunk = grid[start : start + ESTIMATE_FINE_CHUNK]
        base = np.exp(-2j * np.pi * np.outer(chunk / 60.0, seconds))
        current = base
        score = np.zeros(chunk.size)
        for _ in range(ESTIMATE_FINE_HARMONICS):
            score += np.abs(current @ events.weights)
            current = current * base
        scores[start : start + chunk.size] = score
    return float(grid[int(np.argmax(scores))])


def octave_candidates(
    frequencies: np.ndarray, magnitude: np.ndarray, bpm: float
) -> tuple[tuple[float, float], ...]:
    reference = local_salience(frequencies, magnitude, bpm)
    candidates = []
    for factor in (0.5, 2.0):
        value = bpm * factor
        if MIN_BPM <= value <= MAX_BPM:
            ratio = local_salience(frequencies, magnitude, value) / reference
            candidates.append((round(value, BPM_DECIMALS), float(ratio)))
    return tuple(candidates)


def estimate_bpm(
    detection: Detection, min_bpm: float, max_bpm: float
) -> BpmEstimate:
    events = attack_events(detection)
    if events.times_ms.size < ESTIMATE_MIN_EVENTS:
        raise ValueError("BPMを推定できるだけのオンセットを検出できませんでした")
    frequencies, magnitude = onset_spectrum(events)
    bpms = np.arange(min_bpm, max_bpm, ESTIMATE_COARSE_STEP_BPM)
    salience = harmonic_salience(frequencies, magnitude, bpms)
    best_index = int(np.argmax(salience))
    bpm = round(refine_bpm(events, float(bpms[best_index])), BPM_DECIMALS)
    return BpmEstimate(
        bpm,
        dominance_ratio(bpms, salience, best_index),
        octave_candidates(frequencies, magnitude, bpm),
    )


def analyze(
    detection: Detection,
    config: AnalysisConfig,
    functions: tuple[OnsetFunction, ...],
) -> Analysis:
    period = config.period_ms
    selected = detection.heads[config.head_level]
    groups = build_groups(selected.heads_ms, detection.kick_ms, period)
    primary = pick_primary(groups)
    if primary is None:
        raise ValueError("オンセットを十分に検出できませんでした")
    windows = (
        window_phases(primary, period, config.drift_window_sec)
        if config.drift_window_sec > 0.0
        else None
    )
    drift = estimate_drift(primary.heads_ms, windows, config)
    conclusion_ms, uses_regression = resolve_phase(drift, primary.phase_ms, period)
    reference = estimate_reference(detection, selected, config, conclusion_ms, drift)
    basis = (
        primary_check_basis(
            detection, selected, primary, drift, config, conclusion_ms, uses_regression
        )
        if reference is None
        else reference_check_basis(detection, config, reference)
    )
    support = event_support(
        selected,
        conclusion_ms,
        drift,
        uses_regression,
        period,
        support_window_ms(config),
    )
    return Analysis(
        period,
        groups,
        primary,
        windows,
        drift,
        head_level_sensitivity(detection, period, primary.label),
        conclusion_ms,
        uses_regression,
        build_cross_check(functions, basis),
        support.stability,
        support.grid_fit,
        reference,
    )


def fold_factor(bpm: float) -> float:
    factor = 1.0
    while bpm * factor > MAX_BPM:
        factor /= 2.0
    while bpm * factor < MIN_BPM:
        factor *= 2.0
    return factor


def resolve_tempo(tempo: TempoMap, total_ms: float) -> tuple[TempoMap, float]:
    folded = tempo.scaled(fold_factor(tempo.dominant_bpm(total_ms)))
    return folded, folded.dominant_bpm(total_ms)


def grid_anchor_ms(tempo: TempoMap, subdivision: int, peaks_ms: np.ndarray) -> float:
    index = tempo.grid_index(peaks_ms, subdivision)
    residual = peaks_ms - tempo.line_ms(index, subdivision)
    widest_ms = MS_PER_MINUTE / tempo.slowest_bpm / subdivision
    return circular_mode(residual, widest_ms)


def make_warp(
    tempo: TempoMap | None, config: AnalysisConfig, detection: Detection
) -> GridWarp:
    if tempo is None:
        return IDENTITY_WARP
    peaks_ms = detection.heads[config.head_level].peaks_ms
    anchor_ms = (
        grid_anchor_ms(tempo, config.subdivision, peaks_ms)
        if peaks_ms.size >= MIN_ONSETS
        else 0.0
    )
    return GridWarp(tempo, config.subdivision, config.period_ms, anchor_ms)


def warped_detection(detection: Detection, warp: GridWarp) -> Detection:
    return replace(
        detection,
        heads={level: warp.heads(item) for level, item in detection.heads.items()},
        kick_ms=warp.kicks(detection.kick_ms),
    )


def warped_functions(
    functions: tuple[OnsetFunction, ...], warp: GridWarp
) -> tuple[OnsetFunction, ...]:
    return tuple(
        replace(function, times_ms=warp.times(function.times_ms))
        for function in functions
    )


def describe(group: GroupAnalysis, period: float) -> list[str]:
    offsets = group.offsets_ms
    median = float(wrap_symmetric(np.median(offsets), period))
    q25, q75 = np.percentile(offsets, [25, 75])
    return [
        "",
        f"[{group.label}] n={offsets.size}",
        f"  主クラスタ中央値 {signed(group.phase_ms)} ms (MAD {group.spread_ms:.2f} ms, n={group.cluster_count})",
        f"  全体中央値(参考) {signed(median)} ms (四分位幅 {q75 - q25:.2f} ms)",
        f"  音源を遅らせる量 {signed(-group.phase_ms)} ms / グリッド開始オフセット {signed(group.phase_ms)} ms",
    ]


def render_histogram(group: GroupAnalysis, period: float) -> list[str]:
    half = period / 2.0
    edges = np.arange(
        group.center_ms - half, group.center_ms + half + DISPLAY_BIN_MS, DISPLAY_BIN_MS
    )
    counts, _ = np.histogram(group.offsets_ms, bins=edges)
    labels = wrap_symmetric(edges[:-1], period)
    peak = max(int(counts.max()), 1)
    return [
        f"  {label:+7.1f} | {'#' * round(DISPLAY_WIDTH * count / peak)} {count}"
        for label, count in zip(labels, counts)
    ]


def render_group_gap(groups: dict[str, GroupAnalysis], period: float) -> list[str]:
    other = groups.get(LABEL_OTHER)
    kick = groups.get(LABEL_KICK)
    if other is None or kick is None:
        return []
    gap = float(wrap_symmetric(kick.phase_ms - other.phase_ms, period))
    return [
        "",
        f"[キックとその他の位相差] キック同時 - 非同時 = {signed(gap)} ms",
        "  差が大きい場合は、どちらの立ち上がりに合わせるかを判断すること",
    ]


def render_sensitivity(
    sensitivity: tuple[tuple[float, float | None], ...], period: float
) -> list[str]:
    valid = [(level, phase) for level, phase in sensitivity if phase is not None]
    if not valid:
        return []
    phases = np.asarray([phase for _, phase in valid])
    relative = wrap_symmetric(phases - phases[0], period)
    lines = ["", "[定義感度: head-level(頭の定義の影響。検出器の頑健性の検証ではない) / 主クラスタ中央値]"]
    lines += [f"  head-level {level:.2f}: {signed(phase)} ms" for level, phase in valid]
    lines.append(f"  最大差 {relative.max() - relative.min():.2f} ms")
    return lines


def render_fit(fit: DriftFit, bpm: float, period: float, end_ms: float) -> list[str]:
    tempo = bpm * (1.0 - fit.slope)
    tempo_low = bpm * (1.0 - fit.slope_high)
    tempo_high = bpm * (1.0 - fit.slope_low)
    verdict = "有意なテンポずれあり" if fit.significant else "テンポずれは有意でない"
    center_phase = float(wrap_symmetric(fit.phase_at(fit.reference_ms), period))
    start_phase = float(wrap_symmetric(fit.phase_ms, period))
    start_gap = fit.slope * (0.0 - fit.reference_ms)
    end_gap = fit.slope * (end_ms - fit.reference_ms)
    lines = [
        f"  回帰[{fit.method}]: 傾き {fit.slope * 60000.0:+.2f} ms/分 -> 推定BPM {tempo:.4f} (95%区間 {tempo_low:.4f} ~ {tempo_high:.4f}) / {verdict}",
        f"  曲中央({fit.reference_ms / 1000.0:.1f}s)での位相 {signed(center_phase)} ms",
        f"  曲頭(0 ms)での位相(切片) {signed(start_phase)} ms",
        f"  BPM {bpm:g} 固定で曲中央に揃えた場合のズレ: 曲頭 {signed(start_gap)} ms / 終端({end_ms / 1000.0:.0f}s) {signed(end_gap)} ms",
    ]
    if fit.significant:
        lines.append(
            f"  推定BPM {tempo:.4f} を小数で指定できる場合は、曲頭の位相(切片) {signed(start_phase)} ms を採用する"
        )
    return lines


def window_lines(windows: WindowPhases) -> list[str]:
    return [
        f"  {start / 1000.0:6.0f}s~ {signed(phase):>8} ms (n={count})"
        for start, phase, count in zip(
            windows.starts_ms, windows.phases_ms, windows.counts
        )
    ]


def render_drift(analysis: Analysis, bpm: float) -> list[str]:
    windows = analysis.windows
    if windows is None:
        return []
    lines = ["", f"[時間窓ごとの位相(連続化済み): {analysis.primary.label}]"]
    lines += window_lines(windows)
    if analysis.drift is None:
        lines.append("  回帰は不採用(窓数不足、または窓ごとの位相が一貫しないため)")
    else:
        end_ms = float(analysis.primary.heads_ms[-1])
        lines += render_fit(analysis.drift, bpm, analysis.period_ms, end_ms)
    return lines


def render_stability(stability: float | None, grid_fit: float | None) -> list[str]:
    if stability is None or grid_fit is None:
        return []
    verdict = (
        "安定"
        if is_stable_value(stability, grid_fit)
        else "不安定(結論を鵜呑みにせず、--subdivision の変更やステム音源での再測定を推奨)"
    )
    return [
        "",
        f"[信頼度] 結論の位相が偶然より有意に支持される時間窓 {stability * 100:.0f}%"
        f"(基準 {STABLE_WINDOW_RATIO * 100:.0f}%以上)",
        f"  グリッド上にあるオンセットの割合(偶然を除く) {grid_fit * 100:.0f}%"
        f"(基準 {MIN_GRID_FIT * 100:.0f}%以上) -> {verdict}",
    ]


def render_cross_check(cross: CrossCheck | None) -> list[str]:
    if cross is None:
        return []
    lines = [
        "",
        "[独立検出器による検算(粗い一致確認)]",
        f"  包絡線の頭から最大傾きまでの遅れ(中央値) {cross.rise_gap_ms:.2f} ms をアタック頭への換算に使用",
    ]
    for check in cross.checks:
        agreed = abs(check.difference_ms) <= CHECK_AGREE_MS
        verdict = "一致" if agreed else "乖離"
        basis = "回帰の曲中央時点" if check.uses_regression else "全体"
        lines.append(
            f"  {check.name}: 位相 {signed(check.phase_ms)} ms ({basis})"
            f" / 頭基準に換算 {signed(check.converted_ms)} ms"
            f" / 結論との差 {signed(check.difference_ms)} ms"
            f" / 鋭さ {check.contrast:.1f} -> {verdict}"
        )
    summary = "一致" if cross.all_agree else "乖離あり(検出ミス・頭の定義・グルーヴの揺れを確認)"
    lines += [
        f"  判定: {cross.agreed}/{len(cross.checks)} 検出器が ±{CHECK_AGREE_MS:g} ms 以内 -> {summary}",
        "  各検出器は窓長に由来する時間定義の差を含むため、数 ms 単位の一致確認であり、それ以下の精度の根拠にはならない",
    ]
    return lines


def render_band_check(bands: BandCheck | None, config: AnalysisConfig) -> list[str]:
    if bands is None:
        return []
    lines = [
        "",
        "[別帯域・別フィルタ方式・キック帯域での頭位相(全オンセットの主クラスタ中央値)]",
        f"  基準 {config.low_hz:g}~{config.high_hz:g} Hz: {signed(bands.reference_ms)} ms",
    ]
    lines += [
        f"  {band.label}: {signed(band.phase_ms)} ms (基準との差 {signed(band.difference_ms)} ms)"
        for band in bands.bands
    ]
    if bands.bands:
        lines.append(
            f"  帯域間の最大差 {bands.span_ms:.2f} ms / 帯域により立ち上がりの形が異なるため、数 ms 以内の差は通常の範囲"
        )
    if bands.ringing is not None:
        lines.append(
            f"  {bands.ringing.label}: {signed(bands.ringing.phase_ms)} ms (基準との差 {signed(bands.ringing.difference_ms)} ms)"
            " <- 零位相フィルタの前方リンギングによる早まりの目安。因果側は群遅延ぶん遅めに出るため、真値は両者の間"
        )
    if bands.kick is not None:
        lines.append(
            f"  {bands.kick.label}: {signed(bands.kick.phase_ms)} ms (基準との差 {signed(bands.kick.difference_ms)} ms)"
            " <- キック同時の分類窓に依存しない測定。低域は時間分解能が粗い"
        )
    return lines


def render_refine(refine: RefineCheck | None, analysis: Analysis) -> list[str]:
    if refine is None:
        return []
    error95 = SE_CONFIDENCE_Z * refine.standard_error_ms
    lines = [
        "",
        "[局所精密化: 因果フィルタ・複数帯域・サブサンプル補間でグリッド近傍のアタック頭を再測定]",
        f"  対象 {refine.count} 件(結論から ±{REFINE_RADIUS_MS:g} ms 以内) / MAD {refine.spread_ms:.2f} ms / 統計誤差(95%) ±{error95:.2f} ms",
        f"  結論 {signed(analysis.final_ms)} ms -> 精密化後 {signed(refine.offset_ms)} ms (差 {signed(refine.shift_ms)} ms)",
        f"  精密化後の値で合わせる場合に音源を遅らせる量: {signed(0.0 - refine.offset_ms)} ms",
    ]
    lines += [
        f"  {band.low_hz:g}~{band.high_hz:g} Hz: 差 {signed(band.shift_ms)} ms (n={band.count})"
        for band in refine.bands
    ]
    lines.append(
        "  因果フィルタの群遅延と平滑の遅れは補正済み。残る差は帯域・フィルタ方式・平滑による立ち上がり形状の定義違い"
    )
    return lines


def render_precision(analysis: Analysis, bands: BandCheck | None) -> list[str]:
    error95 = SE_CONFIDENCE_Z * analysis.final_standard_error_ms
    lines = [
        "",
        "[推定精度の目安]",
        f"  統計誤差(95%) ±{error95:.2f} ms (イベント単位と時間窓単位の大きいほう。時間窓が{MIN_DRIFT_WINDOWS}未満ならイベント単位のみ)",
    ]
    if analysis.cross is not None:
        lines.append(f"  独立検出器間の最大乖離 {analysis.cross.span_ms:.2f} ms")
    if bands is not None:
        lines.append(f"  帯域間の最大差 {bands.span_ms:.2f} ms")
        if bands.ringing is not None:
            lines.append(f"  フィルタ方式による差 {abs(bands.ringing.difference_ms):.2f} ms")
    lines.append(
        "  表示は0.01 ms単位だが測定精度ではない。上記の乖離や head-level 差が数 ms ある場合は、そちらが精度の上限になる"
    )
    return lines


def render_reference(analysis: Analysis, config: AnalysisConfig) -> list[str]:
    reference = analysis.reference
    if reference is None:
        return []
    period = analysis.period_ms
    total = sum(group.count for group in reference.groups)
    lines = ["", f"[位相の群分解(最大傾き位置のピーク) / 基準群の選択: {config.reference_group}]"]
    for index, group in enumerate(reference.groups):
        marker = (
            f" <- 基準({REFERENCE_DESCRIPTIONS[config.reference_group]})"
            if index == reference.chosen
            else ""
        )
        lines.append(
            f"  群{index + 1}: 最大傾き {signed(float(wrap_symmetric(group.peak_ms, period)))} ms"
            f" / 頭 {signed(float(wrap_symmetric(group.head_ms, period)))} ms"
            f" / 遅れ {group.gap_ms:.2f} ms / 件数 {group.count} ({group.count / total * 100:.0f}%){marker}"
        )
    if len(reference.groups) > 1:
        lines.append(
            "  群が複数あるため、現行の結論は群の混合値です。件数比が時間で変わると見かけのドリフトが出ます"
        )
    if reference.windows is not None and reference.window_counts is not None:
        lines.append("  時間窓ごとの基準群の頭位相 / 各群の件数")
        lines += [
            f"  {start / 1000.0:6.0f}s~ {signed(phase):>8} ms / 件数 {' / '.join(str(int(value)) for value in row)}"
            for start, phase, row in zip(
                reference.windows.starts_ms,
                reference.windows.phases_ms,
                reference.window_counts,
            )
        ]
    if reference.drift is not None:
        end_ms = float(analysis.primary.heads_ms[-1])
        lines += render_fit(reference.drift, config.bpm, period, end_ms)
    elif reference.windows is not None:
        lines.append("  回帰は不採用(窓数不足、または窓ごとの位相が一貫しないため)")
    basis = "回帰の曲中央時点の値" if reference.uses_regression else "基準群の頭位相の中央値"
    lines += [
        "",
        f"[結論: {LABEL_REFERENCE} / 根拠: {basis}]",
        f"  グリッド開始オフセット {signed(reference.conclusion_ms)} ms",
        f"  音源を遅らせる量 {signed(-reference.conclusion_ms)} ms",
        f"  混合値の結論との差 {signed(reference.difference_ms)} ms",
    ]
    lines += render_stability(reference.stability, reference.grid_fit)
    if reference.sensitivity:
        lines += ["", "[定義感度: head-level(頭の定義の影響。検出器の頑健性の検証ではない) / 基準群の頭位相の中央値]"]
        lines += [
            f"  head-level {level:g}: {signed(phase)} ms" for level, phase in reference.sensitivity
        ]
        phases = [phase for _, phase in reference.sensitivity]
        lines.append(f"  最大差 {max(phases) - min(phases):.2f} ms")
    return lines


def render_conclusion(analysis: Analysis, is_main: bool) -> list[str]:
    basis = "回帰の曲中央時点の値" if analysis.uses_regression else "主クラスタ中央値"
    title = (
        f"[結論: {analysis.primary.label} / 根拠: {basis}]"
        if is_main
        else f"[参考: 全オンセットの混合値による結論 / 根拠: {basis}]"
    )
    return [
        "",
        title,
        f"  グリッド開始オフセット {signed(analysis.conclusion_ms)} ms",
        f"  音源を遅らせる量 {signed(-analysis.conclusion_ms)} ms",
    ]


def render_estimate(estimate: BpmEstimate | None) -> list[str]:
    if estimate is None:
        return []
    if estimate.dominance is None:
        verdict = "比較対象がなく判定不可"
    elif estimate.dominance >= ESTIMATE_MIN_DOMINANCE:
        verdict = f"次点の {estimate.dominance:.2f} 倍 -> 明瞭"
    else:
        verdict = f"次点の {estimate.dominance:.2f} 倍 -> 不明瞭(--bpm での指定を推奨)"
    lines = [f"BPM自動推定: {estimate.bpm:.2f} ({verdict})"]
    if estimate.octaves:
        octaves = " / ".join(
            f"{bpm:g} (相対強度 {ratio:.2f})" for bpm, ratio in estimate.octaves
        )
        lines.append(f"オクターブ違いの候補: {octaves} / 拍の数え方が違う場合は --bpm で指定")
    return lines


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


def render_decoder_variants(
    decode: DecodeInfo | None, conclusion_ms: float, period: float
) -> list[str]:
    if decode is None:
        return []
    lines = ["", "[デコーダの遅延補正ごとの結果]", f"  {DECODER_EXPLANATIONS[decode.suffix]}"]
    for variant in decoder_variants(decode):
        shift_ms = (decode.applied_skip - variant.skip_samples) * 1000.0 / decode.sample_rate
        offset = float(wrap_symmetric(conclusion_ms + shift_ms, period))
        skipped_ms = variant.skip_samples * 1000.0 / decode.sample_rate
        marker = " <- [結論]と同じ(本スクリプトのデコード結果)" if variant.skip_samples == decode.applied_skip else ""
        lines += [
            f"  {variant.label}: 先頭 {variant.skip_samples} サンプル({skipped_ms:.2f} ms)を削除",
            f"    グリッド開始オフセット {signed(offset)} ms / 音源を遅らせる量 {signed(-offset)} ms{marker}",
        ]
    return lines


def render_header(
    config: AnalysisConfig, period: float, tempo: TempoMap | None
) -> list[str]:
    if tempo is None:
        return [
            f"BPM {config.bpm:g} / 1拍を{config.subdivision}分割 / グリッド間隔 {period:.3f} ms",
            "符号: 負 = オンセットがグリッド線より前",
            f"結果はグリッド間隔の剰余でのみ決まる(±{period:.3f} ms ずらしても一致)。拍頭の位置はDAWで確認すること",
        ]
    return [
        f"テンポマップ {len(tempo.points)} 区間を一括解析 / 基準BPM {config.bpm:g}(最長区間) / 1拍を{config.subdivision}分割 / 基準グリッド間隔 {period:.3f} ms",
        "各オンセットを、その時刻のテンポマップ上の最寄りグリッド線との残差(実時間 ms)に換算し、全区間をまとめて評価する",
        "時刻表示はグリッド換算時間で、BPMの異なる区間では実時間と一致しない",
        "符号: 負 = オンセットがグリッド線より前",
        "拍頭の位置はDAWで確認すること",
    ]


def render_report(
    audio_name: str,
    config: AnalysisConfig,
    analysis: Analysis,
    estimate: BpmEstimate | None,
    decode: DecodeInfo | None,
    bands: BandCheck | None,
    refine: RefineCheck | None,
    tempo: TempoMap | None = None,
) -> list[str]:
    period = analysis.period_ms
    lines = [
        f"ファイル: {audio_name}",
        *render_estimate(estimate),
        *render_header(config, period, tempo),
    ]
    for group in analysis.groups.values():
        lines += describe(group, period) + render_histogram(group, period)
    lines += render_group_gap(analysis.groups, period)
    lines += render_sensitivity(analysis.sensitivity, period)
    lines += render_drift(analysis, config.bpm)
    lines += render_cross_check(analysis.cross)
    lines += render_band_check(bands, config)
    lines += render_refine(refine, analysis)
    if analysis.reference is None:
        lines += render_conclusion(analysis, True)
        lines += render_stability(analysis.stability, analysis.grid_fit)
    else:
        lines += render_reference(analysis, config)
        lines += render_conclusion(analysis, False)
    lines += render_precision(analysis, bands)
    lines += render_decoder_variants(decode, analysis.final_ms, period)
    return lines


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


def config_from_args(args: argparse.Namespace, bpm: float) -> AnalysisConfig:
    return AnalysisConfig(
        bpm=bpm,
        subdivision=args.subdivision,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        head_level=args.head_level,
        drift_window_sec=args.drift_window_sec,
        reference_group=args.reference_group,
    )


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


def resolve_bpm(
    args: argparse.Namespace, detection: Detection
) -> tuple[float, BpmEstimate | None]:
    if args.bpm is not None:
        return args.bpm, None
    estimate = estimate_bpm(detection, args.bpm_min, args.bpm_max)
    return estimate.bpm, estimate


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audio_path = validate_arguments(args)
        with readable_audio(audio_path) as readable_path:
            validate_audio(readable_path, args.high_hz)
            samples, sample_rate = load_audio(readable_path)
        decode = decode_info(audio_path, sample_rate)
        causal = args.filter_mode == FILTER_CAUSAL
        attacks = detect_attacks(
            samples,
            sample_rate,
            args.low_hz,
            args.high_hz,
            head_levels(args.head_level),
            causal,
        )
        functions = compute_onset_functions(samples, sample_rate)
        bpm, estimate = resolve_bpm(args, attacks)
        config = config_from_args(args, bpm)
        kick = kick_envelope(samples, sample_rate)
        detection = attach_kicks(attacks, kick, config.period_ms)
        warp = make_warp(None, config, detection)
        analysis = analyze(
            warped_detection(detection, warp),
            config,
            warped_functions(functions, warp),
        )
        bands = (
            band_check(samples, sample_rate, config, analysis, kick, causal, warp)
            if args.band_check
            else None
        )
        refine = (
            refine_check(samples, sample_rate, attacks, kick, config, analysis, warp)
            if args.refine
            else None
        )
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"エラー: {error}", file=sys.stderr)
        return 2
    print(
        "\n".join(
            render_report(
                audio_path.name, config, analysis, estimate, decode, bands, refine, None
            )
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
