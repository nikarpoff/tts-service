"""
Piper backend: fast VITS voices in ONNX, on CPU.

A model is one voice: `ru_RU-<voice>-<quality>.onnx` with its JSON config. The text is
phonemized by espeak-ng built into the piper library, which also puts the stress
(TTS-2); a `+` before a vowel becomes the combining acute accent espeak-ng reads as
the stress mark.

The library is GPL-3.0; every voice has a license of its own, see README.
"""

import re

from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime

from loguru import logger
from piper import PiperVoice, SynthesisConfig

from app.core.backends.base import Backend
from app.core.backends.weights import fetch, huggingface_url, weights_dir
from app.core.config import Settings
from app.core.entities import Voice

VOICES_REPO = "rhasspy/piper-voices"
VOICE_PATH = "ru/ru_RU/{voice}/{quality}/{model}{extension}"
MODEL_FILES = (".onnx", ".onnx.json")

# Russian voices of the repository; the model name is ru_RU-<voice>-<quality>
GENDERS: Dict[str, str] = {
    "ru_RU-denis-medium": "m",
    "ru_RU-dmitri-medium": "m",
    "ru_RU-irina-medium": "f",
    "ru_RU-ruslan-medium": "m",
}

ACUTE_ACCENT = "\u0301"           # combining acute accent
STRESSED_VOWEL = re.compile(r"\+([аеёиоуыэюя])", re.IGNORECASE)


class PiperBackend(Backend):
    """Piper: one voice per model, espeak-ng phonemes"""
    name = "piper"
    default_model = "ru_RU-denis-medium"
    models = frozenset(GENDERS)
    supports_stress = True

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        path = self.fetch(settings, model)

        use_cuda = self.device == "cuda"
        if use_cuda and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            logger.warning("onnxruntime has no CUDA provider: Piper runs on CPU")
            use_cuda = False
            self.device = "cpu"

        self._piper = PiperVoice.load(path, use_cuda=use_cuda)
        self._voice = Voice(model.split("-")[1], GENDERS[model], self._piper.config.sample_rate)
        logger.info("Piper model ready: model={} device={} sample_rate={}", model, self.device, self._voice.sample_rate)

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Path:
        _, voice, quality = model.split("-")
        directory = weights_dir(settings, cls.name)
        for extension in MODEL_FILES:
            path = VOICE_PATH.format(voice=voice, quality=quality, model=model, extension=extension)
            fetch(huggingface_url(VOICES_REPO, path), directory / f"{model}{extension}")
        return directory / f"{model}.onnx"

    def voices(self) -> List[Voice]:
        return [self._voice]

    def synthesize(self, text: str, voice: Voice, speed: float) -> np.ndarray:
        """Blocking: call it from a worker thread."""
        text = self.espeak_text(text)
        # A longer phoneme is a slower speech
        config = SynthesisConfig(length_scale=self._piper.config.length_scale / speed)
        chunks = [chunk.audio_float_array for chunk in self._piper.synthesize(text, config)]
        if not chunks:
            logger.warning("Sentence without phonemes is not read: {} chars", len(text))
            return self.pause(voice.sample_rate)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    @staticmethod
    def espeak_text(text: str) -> str:
        """The `+` stress marks as espeak-ng reads them: the acute accent after the vowel; a stray `+` it would read aloud."""
        return STRESSED_VOWEL.sub(rf"\1{ACUTE_ACCENT}", text).replace("+", " ")

    def close(self) -> None:
        self._piper = None
