"""
Sound (TTS-4): peak not above -1 dBFS, 5 ms edges, leading silence up to 50 ms, resampling.
"""

import io
import wave

import numpy as np
import pytest

from app.core.audio import process, to_wav
from app.core.audio.process import apply_fades, resample
from conftest import leading_silence_ms, pcm_samples, peak_dbfs

RATE = 48000


def tone(seconds: float, rate: int = RATE, amplitude: float = 0.5, frequency: float = 440.0) -> np.ndarray:
    return (amplitude * np.sin(2 * np.pi * frequency * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


def silence(seconds: float, rate: int = RATE, noise_db: float = -70.0) -> np.ndarray:
    # A model "silence" is a quiet noise, not zeros
    noise = np.random.default_rng(1).standard_normal(int(seconds * rate)) * 10 ** (noise_db / 20)
    return noise.astype(np.float32)


@pytest.mark.parametrize("amplitude", [0.05, 0.5, 1.0, 1.7], ids=["quiet", "half", "full", "clipping"])
def test_peak_normalized(amplitude):
    samples = pcm_samples(process(tone(0.5, amplitude=amplitude), RATE, RATE))
    assert -1.05 <= peak_dbfs(samples) <= -1.0 + 1e-3


def test_silence_not_amplified():
    samples = pcm_samples(process(np.zeros(4800, dtype=np.float32), RATE, RATE))
    assert np.max(np.abs(samples)) == 0


def test_fades():
    rate = 24000
    samples = pcm_samples(process(tone(0.5, rate, amplitude=1.0, frequency=1000), rate, rate))
    fade = int(rate * 0.005)

    # The edges start and end at zero and grow over 5 ms
    assert samples[0] == 0 and samples[-1] == 0
    envelope = [np.max(np.abs(samples[start:start + fade // 5])) for start in range(0, fade, fade // 5)]
    assert envelope == sorted(envelope)
    assert np.max(np.abs(samples[:fade // 5])) < 0.3 * np.max(np.abs(samples))
    assert np.max(np.abs(samples[fade:2 * fade])) > 0.85

    # Nothing to fade in a one-sample signal
    assert apply_fades(np.ones(1, dtype=np.float32), rate).tolist() == [1.0]


@pytest.mark.parametrize("lead_seconds", [0.0, 0.03, 0.3, 1.0])
def test_leading_silence_up_to_50_ms(lead_seconds):
    audio = np.concatenate([silence(lead_seconds), tone(0.4)])
    samples = pcm_samples(process(audio, RATE, 24000))

    assert leading_silence_ms(samples, 24000) <= 50
    # The sound itself is kept
    assert len(samples) / 24000 >= 0.4 - 0.001


@pytest.mark.parametrize("source", [16000, 22050, 24000, 48000])
@pytest.mark.parametrize("target", [8000, 16000, 22050, 24000, 48000])
def test_resample(source, target):
    audio = tone(1.0, source, frequency=440.0)
    resampled = resample(audio, source, target)

    # The duration and the pitch stay
    assert abs(len(resampled) - target) <= 1
    spectrum = np.abs(np.fft.rfft(resampled))
    frequency = np.argmax(spectrum) * target / len(resampled)
    assert abs(frequency - 440.0) < 2.0


def test_resample_keeps_out_aliasing():
    # 10 kHz is above the 4 kHz Nyquist of 8 kHz: it is filtered, not folded into the band
    resampled = resample(tone(1.0, RATE, frequency=10000.0), RATE, 8000)
    # The filter edges ring for a few samples: the band is checked inside
    assert np.max(np.abs(resampled[100:-100])) < 0.01


def test_wav():
    pcm = process(tone(0.25), RATE, 22050)
    data = to_wav(pcm, 22050)
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 22050)
        assert wav.readframes(wav.getnframes()) == pcm
