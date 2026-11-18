"""
Silero backend: Silero TTS for Russian, on CPU.

A model is a torch.package file with the acoustic model, the vocoder and the accentor
inside, so torch alone runs it. The model puts the stress and `ё` itself (TTS-2); a `+`
before a vowel is its own stress mark and overrides the automatic one.

The Russian models are CC BY-NC-SA 4.0: no commercial use.
"""

import re
import threading

from pathlib import Path
from typing import Dict, List
from xml.sax.saxutils import escape

import numpy as np
import torch

from loguru import logger

from app.core.backends.base import Backend
from app.core.backends.weights import fetch, weights_dir
from app.core.config import Settings
from app.core.entities import Voice

MODEL_URL = "https://models.silero.ai/models/tts/ru/{model}.pt"
SAMPLE_RATE = 48000                 # the highest rate of the models; the requests go through resampling

# Voices of TTS-3 in the order of /v1/voices: the default one first
GENDERS: Dict[str, str] = {"xenia": "f", "kseniya": "f", "baya": "f", "aidar": "m", "eugene": "m"}

# The models read Russian letters only and drop the rest
RUSSIAN_LETTER = re.compile(r"[а-яё]", re.IGNORECASE)


class SileroBackend(Backend):
    """Silero TTS: several voices in one model, automatic stress"""
    name = "silero"
    default_model = "v5_ru"
    models = frozenset({"v4_ru", "v5_ru", "v5_1_ru", "v5_2_ru", "v5_3_ru", "v5_4_ru", "v5_5_ru", "v5_6_ru"})
    supports_stress = True

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        path = self.fetch(settings, model)

        self._model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        self._model.to(torch.device(self.device))
        # The packaged model keeps state between its steps: one synthesis at a time
        self._lock = threading.Lock()

        self._voices = [Voice(speaker, gender, SAMPLE_RATE) for speaker, gender in GENDERS.items() if speaker in self._model.speakers]
        logger.info("Silero model ready: model={} device={} voices={}", model, self.device, [voice.id for voice in self._voices])

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Path:
        return fetch(MODEL_URL.format(model=model), weights_dir(settings, cls.name) / f"{model}.pt")

    def voices(self) -> List[Voice]:
        return list(self._voices)

    def synthesize(self, text: str, voice: Voice, speed: float) -> np.ndarray:
        """Blocking: call it from a worker thread."""
        if not RUSSIAN_LETTER.search(text):
            # The model raises on a text it has nothing to read in; a pause keeps the answer going
            logger.warning("Sentence without Russian letters is not read: {} chars", len(text))
            return self.pause(voice.sample_rate)

        # The speed goes through SSML; the plain text path is kept for the usual speed
        if speed == 1.0:
            request = {"text": text}
        else:
            request = {"ssml_text": f'<speak><prosody rate="{round(speed * 100)}%">{escape(text)}</prosody></speak>'}

        with self._lock:
            audio = self._model.apply_tts(speaker=voice.id, sample_rate=voice.sample_rate, **request)
        return audio.cpu().numpy().astype(np.float32, copy=False)

    def close(self) -> None:
        self._model = None
