"""
Model backends: interchangeable implementations of the service model.

A backend is chosen by MODEL_BACKEND, built once at startup with its weights loaded,
and called from the engine worker threads only.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, FrozenSet, List

import numpy as np

from app.core.config import Settings, SYNTHESIS_SETTINGS
from app.core.entities import Voice


class Backend(ABC):
    """A model behind the service API"""
    name: ClassVar[str]                                     # the MODEL_BACKEND value
    default_model: ClassVar[str]                            # the model when MODEL_NAME is unset
    models: ClassVar[FrozenSet[str]] = frozenset()          # known MODEL_NAME values; empty takes any
    supports_stress: ClassVar[bool] = False                 # reads `+` before a vowel as the stress (TTS-2)

    def __init__(self, settings: Settings, model: str) -> None:
        # Models are loaded here, at startup: the first request must not pay for loading them
        self.settings = settings
        self.model = model
        self.device = settings.TARGET_PLATFORM

    @abstractmethod
    def voices(self) -> List[Voice]:
        """Voices of the loaded model, the default one first."""
        ...

    @abstractmethod
    def synthesize(self, text: str, voice: Voice, speed: float) -> np.ndarray:
        """Speech of one sentence: float32 mono in [-1, 1] at the voice sample rate. Blocking: call it from a worker thread."""
        ...

    @property
    def default_voice(self) -> Voice:
        return self.voices()[0]

    def warmup(self) -> None:
        """An idle call on a typical input. Blocking: call it from a worker thread."""
        self.synthesize(SYNTHESIS_SETTINGS.WARMUP_TEXT, self.default_voice, 1.0)

    def params(self) -> Dict[str, Any]:
        """Backend parameters that affect the answers, for /v1/info."""
        return {"voice": self.default_voice.id, "sample_rate": self.default_voice.sample_rate}

    def close(self) -> None:
        """Releases the model at shutdown."""
