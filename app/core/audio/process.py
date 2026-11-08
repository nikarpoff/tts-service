"""
Sound of a sentence (TTS-4): from the model output to the PCM the client plays.

1. Resampling to the requested rate.
2. The leading silence is cut to a few milliseconds: it adds to the latency directly.
3. Peak normalization to -1 dBFS: sentences sound equally loud and do not clip.
4. 5 ms fades on both edges: no clicks where the sentences join.
"""

from math import gcd

import numpy as np

from scipy.signal import resample_poly

from app.core.config import AUDIO_SETTINGS as CFG

PCM_MAX = 32767
BYTES_PER_SAMPLE = 2


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """The signal at `target_rate`; polyphase filtering keeps aliasing out."""
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    common = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // common, source_rate // common).astype(np.float32)


def trim_leading_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """The signal from KEEP_LEADING_SILENCE_MS before its first sound."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < CFG.MIN_PEAK:
        return audio

    threshold = peak * 10 ** (CFG.SILENCE_DB / 20)
    first_sound = int(np.argmax(np.abs(audio) >= threshold))
    start = max(0, first_sound - int(sample_rate * CFG.KEEP_LEADING_SILENCE_MS / 1000))
    return audio[start:]


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    """The signal with its peak at PEAK_DBFS; silence stays silence."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < CFG.MIN_PEAK:
        return audio
    return (audio * (10 ** (CFG.PEAK_DBFS / 20) / peak)).astype(np.float32)


def apply_fades(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Half-cosine fade in and out of FADE_MS: the edges start and end at zero."""
    length = min(int(sample_rate * CFG.FADE_MS / 1000), audio.size // 2)
    if length < 2:
        return audio

    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, length))).astype(np.float32)
    audio = audio.copy()
    audio[:length] *= ramp
    audio[-length:] *= ramp[::-1]
    return audio


def to_pcm16(audio: np.ndarray) -> bytes:
    """Signed 16-bit little-endian PCM."""
    samples = np.round(np.clip(audio, -1.0, 1.0) * PCM_MAX).astype("<i2")
    return samples.tobytes()


def process(audio: np.ndarray, source_rate: int, target_rate: int) -> bytes:
    """Model output of a sentence as the PCM the client plays."""
    audio = resample(np.asarray(audio, dtype=np.float32), source_rate, target_rate)
    audio = trim_leading_silence(audio, target_rate)
    audio = normalize_peak(audio)
    audio = apply_fades(audio, target_rate)
    return to_pcm16(audio)


def pcm_duration_ms(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / BYTES_PER_SAMPLE / sample_rate * 1000
