"""
Settings of this service on top of the shared ones.

The shared Settings are the same in the three AI services byte for byte; the
variables only the speech synthesis reads live here. They come from the same
environment and .env.
"""

from functools import lru_cache
from pathlib import Path
from typing import get_args

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config.config import PROJECT_ROOT
from app.core.entities import SampleRate


class ServiceSettings(BaseSettings):
    # 1. Cache
    CACHE_MAX_ITEMS: int = Field(256, ge=0, description="Synthesized phrases kept in memory, least recently used go first; 0 turns the cache off")
    PREWARM_PHRASES_PATH: Path = Field(Path("configs/prewarm_phrases.txt"), description="Phrases synthesized into the cache at startup, one per line; a relative path is taken from the service root")
    PREWARM_SAMPLE_RATE: int = Field(24000, description="Sample rate of the prewarmed phrases: the one the gateway asks for (APP_AUDIO_OUT_SAMPLE_RATE)")

    @field_validator("PREWARM_SAMPLE_RATE")
    @classmethod
    def check_sample_rate(cls, value: int) -> int:
        # A Literal of ints refuses the env string "24000": the set is checked here
        if value not in get_args(SampleRate):
            raise ValueError(f"expected one of {get_args(SampleRate)}")
        return value

    # An empty variable keeps the default
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    @property
    def prewarm_phrases_path(self) -> Path:
        return self.PREWARM_PHRASES_PATH if self.PREWARM_PHRASES_PATH.is_absolute() else PROJECT_ROOT / self.PREWARM_PHRASES_PATH


@lru_cache
def get_service_settings() -> ServiceSettings:
    return ServiceSettings()
