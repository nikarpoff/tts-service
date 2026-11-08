"""
Domain entities of the speech synthesis
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

# Sample rates a client may ask for (TTS-1)
SampleRate = Literal[8000, 16000, 22050, 24000, 48000]


class AudioFormat(StrEnum):
    """How the speech is sent: raw PCM streamed by sentences or a whole WAV file"""
    PCM_S16LE = "pcm_s16le"
    WAV = "wav"


@dataclass(frozen=True)
class Voice:
    """A voice of the backend; the backend speaks it at its own sample rate"""
    id: str
    gender: Literal["f", "m"]
    sample_rate: int
