from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import SYNTHESIS_SETTINGS as CFG
from app.core.entities import AudioFormat, SampleRate


class InfoResponse(BaseModel):
    """Service configuration; the gateway puts it into its config_id"""
    model_config = ConfigDict(extra="forbid")

    service: str = Field(description="Service name")
    version: str = Field(description="Service version, the same as in pyproject.toml")
    git_sha: str = Field(description="Commit the image is built from")
    backend: str = Field(description="Model backend, MODEL_BACKEND")
    model: str = Field(description="Model of the backend, MODEL_NAME")
    device: Literal["cpu", "cuda"] = Field(description="Inference device, TARGET_PLATFORM")
    warmed_up: bool = Field(description="The model has passed its warmup calls")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters that affect the answers")


class SynthesizeRequest(BaseModel):
    """Text to speak and the audio wanted (TTS-1)"""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=CFG.MAX_TEXT_CHARS, description="Text to speak; `+` before a vowel marks the stress")
    voice: Optional[str] = Field(None, description="Voice id from /v1/voices; the default voice when unset")
    sample_rate: SampleRate = Field(CFG.DEFAULT_SAMPLE_RATE, description="Sample rate of the answer, Hz")
    format: AudioFormat = Field(AudioFormat.PCM_S16LE, description="pcm_s16le is streamed by sentences, wav comes as a whole file")
    speed: float = Field(1.0, ge=CFG.MIN_SPEED, le=CFG.MAX_SPEED, description="Speech rate, 1.0 is normal")


class VoiceInfo(BaseModel):
    """A voice of the backend (TTS-6)"""
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Voice id for /v1/synthesize")
    backend: str = Field(description="Backend that speaks it")
    sample_rate: int = Field(description="Native sample rate of the voice, Hz")
    gender: Literal["f", "m"] = Field(description="Voice gender")
    default: bool = Field(description="The voice of requests without `voice`")


class ErrorResponse(BaseModel):
    """Error of any endpoint"""
    model_config = ConfigDict(extra="forbid")

    detail: str = Field(description="Short English sentence about the reason")
