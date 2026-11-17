"""
F5-TTS backend: a flow matching model fine-tuned on Russian speech, on GPU.

The model clones the voice of a reference recording, so a voice is a short recording
with its text: `<weights>/f5tts/voices/voices.yaml` lists them, the default one first:

    anna:
      gender: f
      audio: anna.wav       # WAV, 16-bit or float, 5-12 s of clean speech, next to the file
      text: "Что сказано в записи, с +ударениями."

The recordings are read once at startup and kept as tensors: the package API would decode
the file on every call, through ffmpeg.

The Russian weights are trained on texts with the stress marked by `+` and read an
unmarked text worse; the service passes the marks of the request through and adds none.

The weights are CC BY-NC 4.0: no commercial use.
"""

import threading

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import chunk_text, infer_batch_process
from f5_tts.model.utils import seed_everything
from loguru import logger
from scipy.io import wavfile

from app.core.backends.base import Backend
from app.core.backends.weights import fetch, huggingface_url, weights_dir
from app.core.config import Settings
from app.core.entities import Voice
from app.core.exceptions import BackendError

MODEL_REPO = "Misha24-10/F5-TTS_RUSSIAN"
VOCODER_REPO = "charactr/vocos-mel-24khz"
VOCODER_FILES = ("config.yaml", "pytorch_model.bin")
VOCAB_PATH = "F5TTS_v1_Base/vocab.txt"          # one vocabulary for all the checkpoints
ARCHITECTURE = "F5TTS_v1_Base"                  # config of the f5-tts package the checkpoints are trained with

# Checkpoints of the repository by MODEL_NAME
CHECKPOINTS: Dict[str, str] = {
    "F5TTS_v1_Base": "F5TTS_v1_Base/model_240000_inference.safetensors",
    "F5TTS_v1_Base_accent_tune": "F5TTS_v1_Base_accent_tune/model_last_inference.safetensors",
    "F5TTS_v1_Base_v2": "F5TTS_v1_Base_v2/model_last_inference.safetensors",
}

VOICES_FILE = "voices/voices.yaml"
SAMPLE_RATE = 24000                 # the vocoder rate, whatever the reference recording is
NFE_STEPS = 16                      # solver steps: half the package default, the latency goes first
SEED = 0                            # the same text sounds the same: the cache and the measurements rely on it


REFERENCE_TAIL_MS = 50.0           # silence after the reference: the package adds it too, the model expects a pause there
MAX_BATCH_SECONDS = 22.0            # reference and generated audio of one model call, as in the package


@dataclass(frozen=True)
class Reference:
    """The recording a voice is cloned from: mono samples at their own rate and what is said in them"""
    audio: torch.Tensor
    sample_rate: int
    text: str

    @property
    def seconds(self) -> float:
        return self.audio.shape[-1] / self.sample_rate


class F5TTSBackend(Backend):
    """F5-TTS: voices cloned from reference recordings"""
    name = "f5tts"
    default_model = "F5TTS_v1_Base_v2"
    models = frozenset(CHECKPOINTS)
    supports_stress = True

    def __init__(self, settings: Settings, model: str) -> None:
        super().__init__(settings, model)
        directory = self.fetch(settings, model)
        self._voices, self._references = self._load_voices(directory / VOICES_FILE)

        self._model = F5TTS(
            model=ARCHITECTURE,
            ckpt_file=str(directory / CHECKPOINTS[model]),
            vocab_file=str(directory / VOCAB_PATH),
            vocoder_local_path=str(directory / "vocoder"),
            device=self.device,
        )
        # The random state is global: the seed and the sampling of one call go together
        self._lock = threading.Lock()
        logger.info("F5-TTS model ready: model={} device={} voices={}", model, self.device, [voice.id for voice in self._voices])

    @classmethod
    def fetch(cls, settings: Settings, model: str) -> Path:
        directory = weights_dir(settings, cls.name)
        for path in (CHECKPOINTS[model], VOCAB_PATH):
            fetch(huggingface_url(MODEL_REPO, path), directory / path)
        for name in VOCODER_FILES:
            fetch(huggingface_url(VOCODER_REPO, name), directory / "vocoder" / name)
        return directory

    def voices(self) -> List[Voice]:
        return list(self._voices)

    def synthesize(self, text: str, voice: Voice, speed: float) -> np.ndarray:
        """Blocking: call it from a worker thread."""
        reference = self._references[voice.id]
        # A long sentence goes in parts, so that the reference and a part fit one model call
        max_chars = int(len(reference.text.encode("utf-8")) / reference.seconds * (MAX_BATCH_SECONDS - reference.seconds) * speed)
        batches = chunk_text(text, max_chars=max_chars)

        with self._lock:
            seed_everything(SEED)
            audio, _, _ = next(infer_batch_process(
                (reference.audio, reference.sample_rate),
                reference.text,
                batches,
                self._model.ema_model,
                self._model.vocoder,
                mel_spec_type=self._model.mel_spec_type,
                progress=None,
                nfe_step=NFE_STEPS,
                speed=speed,
                device=self.device,
            ))
        if audio is None:
            logger.warning("Sentence the model split into nothing is not read: {} chars", len(text))
            return self.pause(voice.sample_rate)
        return np.asarray(audio, dtype=np.float32)

    def params(self) -> Dict[str, Any]:
        return {**super().params(), "nfe_steps": NFE_STEPS}

    def close(self) -> None:
        self._model = None

    @staticmethod
    def _load_voices(path: Path) -> Tuple[List[Voice], Dict[str, Reference]]:
        """Voices and their references from voices.yaml; the first voice is the default one."""
        if not path.is_file():
            raise BackendError(f"Backend 'f5tts' needs reference voices: '{path}' is not found")
        try:
            entries = yaml.safe_load(path.read_text(encoding="utf-8"))
            voices, references = [], {}
            for voice_id, entry in entries.items():
                audio = path.parent / entry["audio"]
                if not audio.is_file():
                    raise BackendError(f"Reference recording '{audio}' of voice '{voice_id}' is not found")
                if entry["gender"] not in ("f", "m"):
                    raise BackendError(f"Gender of voice '{voice_id}' is not 'f' or 'm'")
                voices.append(Voice(str(voice_id), entry["gender"], SAMPLE_RATE))
                references[str(voice_id)] = _read_reference(audio, str(entry["text"]))
        except (yaml.YAMLError, AttributeError, KeyError, TypeError, ValueError) as e:
            raise BackendError(f"Reference voices in '{path}' are not of the form id: gender, audio, text: {e!r}") from e
        if not voices:
            raise BackendError(f"Reference voices in '{path}' are empty")
        return voices, references


def _read_reference(path: Path, text: str) -> Reference:
    """The reference recording as the package prepares it: mono float with a short pause after it."""
    sample_rate, samples = wavfile.read(path)
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    tail = np.zeros(int(sample_rate * REFERENCE_TAIL_MS / 1000), dtype=np.float32)
    audio = torch.from_numpy(np.concatenate([samples, tail])).unsqueeze(0)

    # The text has to end with a sentence end and a space: the model reads the generated text right after it
    text = text.strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return Reference(audio, sample_rate, text + " ")
