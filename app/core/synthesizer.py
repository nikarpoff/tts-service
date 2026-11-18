"""
Speech synthesis over the engine: cache, streaming by sentences, whole files.

A cache hit is sent at once and takes no place in the engine queue: it needs no model.
A miss takes one place for all its sentences. PCM goes out sentence by sentence: the
first one is synthesized before the response starts, so a full queue still gets its
503 and the client hears the first sentence while the next ones are synthesized.
"""

import time

from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

from loguru import logger

from app.core.audio import pcm_duration_ms, to_wav
from app.core.cache import AudioCache, CacheKey, load_phrases
from app.core.config import ServiceSettings
from app.core.engine import InferenceEngine
from app.core.entities import AudioFormat, Voice
from app.core.pipeline import SpeechPipeline
from app.utils.middleware import current_timings

PREWARM_SPEED = 1.0                 # the gateway does not change the speed


@dataclass
class Speech:
    """Synthesized speech ready for a response: a whole body or chunks by sentences"""
    audio_format: AudioFormat
    sample_rate: int
    cache_hit: bool
    body: Optional[bytes] = None
    chunks: Optional[AsyncIterator[bytes]] = field(default=None, repr=False)

    @property
    def media_type(self) -> str:
        if self.audio_format == AudioFormat.WAV:
            return "audio/wav"
        return f"audio/L16; rate={self.sample_rate}; channels=1"

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "X-Sample-Rate": str(self.sample_rate),
            "X-Audio-Format": self.audio_format.value,
            "X-Cache": "hit" if self.cache_hit else "miss",
        }


class SpeechSynthesizer:
    """Synthesis requests on the engine with the phrase cache"""

    def __init__(self, engine: InferenceEngine, settings: ServiceSettings) -> None:
        self.engine = engine
        self.settings = settings
        self.pipeline = SpeechPipeline(engine.backend)
        self.cache = AudioCache(settings.CACHE_MAX_ITEMS)

    async def startup(self) -> None:
        """Synthesizes the prewarm phrases into the cache."""
        if self.cache.max_items <= 0:
            logger.info("Cache is off: no phrases prewarmed")
            return

        path = self.settings.prewarm_phrases_path
        phrases = load_phrases(path)
        if len(phrases) > self.cache.max_items:
            logger.warning("Prewarm phrases {} exceed CACHE_MAX_ITEMS {}: the first ones are evicted", len(phrases), self.cache.max_items)

        started = time.perf_counter()
        voice = self.pipeline.voice(None)
        for phrase in phrases:
            key = self._key(self.pipeline.normalize(phrase), voice, self.settings.PREWARM_SAMPLE_RATE, PREWARM_SPEED)
            if key in self.cache:
                continue
            pcm = await self.engine.run(
                self.pipeline.render_all, self.pipeline.sentences(key.text), voice, key.sample_rate, key.speed
            )
            self.cache.put(key, pcm)

        logger.info(
            "Cache prewarmed: {} phrases from '{}' at {} Hz in {:.0f} ms",
            len(phrases), path, self.settings.PREWARM_SAMPLE_RATE, (time.perf_counter() - started) * 1000,
        )

    def voices(self) -> List[Voice]:
        return self.pipeline.voices()

    @property
    def default_voice(self) -> Voice:
        return self.pipeline.voice(None)

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str],
        sample_rate: int,
        speed: float,
        audio_format: AudioFormat,
    ) -> Speech:
        """Speech of the text; InvalidParamsError for an unknown voice or nothing to speak, ServiceBusyError on a full queue."""
        voice = self.pipeline.voice(voice_id)
        with current_timings().stage("normalize"):
            normalized = self.pipeline.normalize(text)
        key = self._key(normalized, voice, sample_rate, speed)

        cached = self.cache.get(key)
        logger.info(
            "Synthesis: chars={} voice={} rate={} speed={} format={} cache={}",
            len(text), voice.id, sample_rate, speed, audio_format.value, "hit" if cached is not None else "miss",
        )
        if cached is not None:
            return self._speech(cached, audio_format, sample_rate, cache_hit=True)

        sentences = self.pipeline.sentences(normalized)
        if audio_format == AudioFormat.WAV:
            with self.engine.admit():
                pcm = await self.engine.run(self.pipeline.render_all, sentences, voice, sample_rate, speed)
            self.cache.put(key, pcm)
            self._log_done(len(sentences), pcm, sample_rate)
            return self._speech(pcm, audio_format, sample_rate, cache_hit=False)

        chunks = self._stream(key, voice, sentences)
        # The first sentence before the response: a full queue still gets its 503, and
        # a started generator is closed on garbage collection, giving its place back
        first = await anext(chunks)
        return Speech(audio_format, sample_rate, cache_hit=False, chunks=_prepend(first, chunks))

    async def _stream(self, key: CacheKey, voice: Voice, sentences: List[str]) -> AsyncIterator[bytes]:
        parts = []
        with self.engine.admit():
            for number, sentence in enumerate(sentences, start=1):
                try:
                    chunk = await self.engine.run(self.pipeline.render, sentence, voice, key.sample_rate, key.speed)
                except Exception as e:
                    # The response has started: the client sees a cut stream, the log keeps the reason
                    logger.opt(exception=e).error("Synthesis failed at sentence {} of {}", number, len(sentences))
                    raise
                parts.append(chunk)
                yield chunk

        # Only a complete answer goes to the cache: a client gone midway leaves no cut audio
        pcm = b"".join(parts)
        self.cache.put(key, pcm)
        self._log_done(len(sentences), pcm, key.sample_rate)

    def _key(self, normalized: str, voice: Voice, sample_rate: int, speed: float) -> CacheKey:
        return CacheKey(self.pipeline.backend.name, voice.id, sample_rate, float(speed), normalized)

    @staticmethod
    def _speech(pcm: bytes, audio_format: AudioFormat, sample_rate: int, cache_hit: bool) -> Speech:
        if audio_format == AudioFormat.WAV:
            return Speech(audio_format, sample_rate, cache_hit, body=to_wav(pcm, sample_rate))
        return Speech(audio_format, sample_rate, cache_hit, chunks=_prepend(pcm, _empty()))

    @staticmethod
    def _log_done(sentences: int, pcm: bytes, sample_rate: int) -> None:
        logger.info(
            "Synthesis done: sentences={} audio_ms={:.0f} time_ms={:.1f}",
            sentences, pcm_duration_ms(pcm, sample_rate), current_timings().elapsed_ms(),
        )


async def _prepend(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    yield first
    async for chunk in rest:
        yield chunk


async def _empty() -> AsyncIterator[bytes]:
    return
    yield
