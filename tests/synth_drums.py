from __future__ import annotations

import numpy as np

SAMPLE_RATE = 44100
STEPS_PER_BEAT = 24
STEPS_PER_BAR = 96
FADE_MS = 8.0
PATTERNS = {
    "rock8": {
        "kick": (0, 48, 60),
        "snare": (24, 72),
        "hat": tuple(range(0, 96, 12)),
    },
    "four_floor": {
        "kick": (0, 24, 48, 72),
        "snare": (),
        "hat": tuple(range(12, 96, 24)),
    },
    "funk16": {
        "kick": (0, 18, 36, 54, 60, 78),
        "snare": (24, 72, 90),
        "hat": tuple(range(0, 96, 6)),
    },
    "shuffle": {
        "kick": (0, 48),
        "snare": (24, 72),
        "hat": tuple(beat * 24 + offset for beat in range(4) for offset in (0, 16)),
    },
    "triplet_mix": {
        "kick": (0, 32, 48, 72),
        "snare": (24, 72, 88),
        "hat": tuple(beat * 24 + offset for beat in range(4) for offset in (0, 8, 16)),
    },
    "sparse_kick": {
        "kick": (0, 48),
        "snare": (),
        "hat": (),
    },
    "dnb": {
        "kick": (0, 60),
        "snare": (24, 72),
        "hat": tuple(range(0, 96, 12)),
    },
}


def decay(length: int, tau_ms: float, attack_ms: float) -> np.ndarray:
    time_ms = np.arange(length) / SAMPLE_RATE * 1000.0
    return np.clip(time_ms / attack_ms, 0.0, 1.0) * np.exp(-time_ms / tau_ms)


def fade_tail(signal: np.ndarray) -> np.ndarray:
    count = int(FADE_MS * SAMPLE_RATE / 1000.0)
    faded = signal.copy()
    faded[-count:] *= np.linspace(1.0, 0.0, count)
    return faded


def kick_sample(rng: np.random.Generator) -> np.ndarray:
    length = int(0.25 * SAMPLE_RATE)
    time = np.arange(length) / SAMPLE_RATE
    frequency = 55.0 + 90.0 * np.exp(-time / 0.02)
    body = np.sin(2.0 * np.pi * np.cumsum(frequency) / SAMPLE_RATE) * decay(length, 70.0, 0.6)
    click = rng.standard_normal(length) * decay(length, 3.0, 0.1) * 0.35
    return fade_tail((body + click) * 0.9)


def snare_sample(rng: np.random.Generator) -> np.ndarray:
    length = int(0.22 * SAMPLE_RATE)
    time = np.arange(length) / SAMPLE_RATE
    tone = np.sin(2.0 * np.pi * 190.0 * time) * decay(length, 45.0, 0.3) * 0.5
    noise = rng.standard_normal(length) * decay(length, 60.0, 0.15)
    return fade_tail((tone + noise) * 0.7)


def hat_sample(rng: np.random.Generator) -> np.ndarray:
    length = int(0.12 * SAMPLE_RATE)
    spectrum = np.fft.rfft(rng.standard_normal(length))
    spectrum[np.fft.rfftfreq(length, 1.0 / SAMPLE_RATE) < 4500.0] *= 0.05
    noise = np.fft.irfft(spectrum, length)
    noise /= np.max(np.abs(noise)) + 1e-9
    return fade_tail(noise * decay(length, 14.0, 0.1) * 0.35)


def beat_to_ms(beats: float, segments: tuple[tuple[int, float], ...]) -> float:
    elapsed = 0.0
    remaining = beats
    for bars, bpm in segments:
        span = bars * 4.0
        if remaining <= span:
            return elapsed + remaining * 60000.0 / bpm
        elapsed += span * 60000.0 / bpm
        remaining -= span
    return elapsed + remaining * 60000.0 / segments[-1][1]


def total_bars(segments: tuple[tuple[int, float], ...]) -> int:
    return sum(bars for bars, _ in segments)


def render(
    pattern: str,
    segments: tuple[tuple[int, float], ...],
    offset_ms: float,
    jitter_ms: float = 0.0,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = {
        "kick": kick_sample(rng),
        "snare": snare_sample(rng),
        "hat": hat_sample(rng),
    }
    duration_ms = beat_to_ms(total_bars(segments) * 4.0, segments)
    length = int(duration_ms * SAMPLE_RATE / 1000.0)
    mix = np.zeros(length + SAMPLE_RATE)
    for bar in range(total_bars(segments)):
        for name, positions in PATTERNS[pattern].items():
            for position in positions:
                beats = (bar * STEPS_PER_BAR + position) / STEPS_PER_BEAT
                time_ms = beat_to_ms(beats, segments) + offset_ms
                time_ms += rng.normal(0.0, jitter_ms) if jitter_ms > 0.0 else 0.0
                start = int(round(time_ms * SAMPLE_RATE / 1000.0))
                if not 0 <= start < length:
                    continue
                gain = 0.8 + 0.2 * rng.random() if name == "hat" else 1.0
                sample = samples[name]
                end = min(mix.size, start + sample.size)
                mix[start:end] += sample[: end - start] * gain
    mix = mix[:length]
    stereo = np.stack([mix, mix], axis=1)
    return (stereo / np.max(np.abs(stereo)) * 0.8).astype(np.float32)


def constant_segments(bpm: float, seconds: float) -> tuple[tuple[int, float], ...]:
    bars = max(1, int(seconds * 1000.0 / (4.0 * 60000.0 / bpm)))
    return ((bars, bpm),)


def shift_audio(audio: np.ndarray, delay_ms: float) -> np.ndarray:
    count = int(round(abs(delay_ms) * SAMPLE_RATE / 1000.0))
    if delay_ms >= 0.0:
        return np.concatenate([np.zeros((count, audio.shape[1]), dtype=audio.dtype), audio])
    return audio[count:]
