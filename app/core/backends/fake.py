"""
Fake backend: a tone instead of speech.

The tone lasts a fixed time per character and every sentence waits FAKE_LATENCY_MS,
so the streaming, the cache and the timings behave as with a model, without weights.
A lead of silence imitates the models, which start with one.
"""

import time

from typing import Any, Dict, List

import numpy as np

from loguru import logger

from app.core.backends.base import Backend
from app.core.config import Settings
from app.core.entities import Voice

SAMPLE_RATE = 48000                 # Silero speaks at 48 kHz: the requests go through resampling
MS_PER_CHAR = 60.0                  # tone duration per character at speed 1.0
LEADING_SILENCE_MS = 100.0          # models start with silence; the audio processing cuts it
AMPLITUDE = 0.5                     # below the target peak: the normalization raises it

VOICES = (
    Voice("fake_f", "f", SAMPLE_RATE),
    Voice("fake_m", "m", SAMPLE_RATE),
)
TONES_HZ = {"fake_f": 440.0, "fake_m": 220.0}


class FakeBackend(Backend):
    """Deterministic stand-in for the model: no weights, no GPU; FAKE_LATENCY_MS imitates the inference time of a sentence"""
    name = "fake"
    default_model = "fake"
    models = frozenset({"fake"})

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        self.latency_ms = settings.FAKE_LATENCY_MS
        logger.warning("Fake backend is on: the answers are synthetic, do not measure or deploy them")
        logger.info("Fake model ready: latency_ms={}", self.latency_ms)

    def voices(self) -> List[Voice]:
        return list(VOICES)

    def synthesize(self, text: str, voice: Voice, speed: float) -> np.ndarray:
        """Blocking: call it from a worker thread."""
        self._imitate_inference()

        lead = np.zeros(int(voice.sample_rate * LEADING_SILENCE_MS / 1000), dtype=np.float32)
        length = int(voice.sample_rate * len(text) * MS_PER_CHAR / speed / 1000)
        phase = 2 * np.pi * TONES_HZ[voice.id] * np.arange(length) / voice.sample_rate
        tone = (AMPLITUDE * np.sin(phase)).astype(np.float32)
        return np.concatenate([lead, tone])

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "latency_ms": self.latency_ms}

    def _imitate_inference(self) -> None:
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000)
