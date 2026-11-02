"""
Blocking synthesis steps over the backend: text to sentences, a sentence to PCM.

The service runs them on the engine pool; bench/latency.py and scripts/render_phrases.py
call them directly, so the measured and the rendered speech is the one the API sends.
"""

from typing import List, Optional

from app.core.audio import process
from app.core.backends import Backend
from app.core.entities import Voice
from app.core.exceptions import InvalidParamsError
from app.core.text import normalize, speakable, split_sentences
from app.utils.middleware import current_timings


class SpeechPipeline:
    """Normalization, sentences and the sound processing around the backend"""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self._voices = {voice.id: voice for voice in backend.voices()}

    def voices(self) -> List[Voice]:
        return list(self._voices.values())

    def voice(self, voice_id: Optional[str]) -> Voice:
        """The voice of the request, the default one when unset; 422 for an unknown one."""
        if voice_id is None:
            return self.backend.default_voice
        voice = self._voices.get(voice_id)
        if voice is None:
            raise InvalidParamsError("Unknown voice")
        return voice

    def normalize(self, text: str) -> str:
        """The text as the backend reads it; 422 when there is nothing to read."""
        normalized = normalize(text, keep_stress=self.backend.supports_stress)
        if not speakable(normalized):
            raise InvalidParamsError("Text has no words to speak")
        return normalized

    @staticmethod
    def sentences(normalized: str) -> List[str]:
        """Sentences to synthesize one by one; marks without words are not sent to the model."""
        return [sentence.strip() for sentence in split_sentences(normalized) if speakable(sentence)]

    def render(self, sentence: str, voice: Voice, sample_rate: int, speed: float) -> bytes:
        """PCM of one sentence at `sample_rate`. Blocking: call it from a worker thread."""
        timings = current_timings()
        with timings.stage("synthesize"):
            audio = self.backend.synthesize(sentence, voice, speed)
        with timings.stage("audio"):
            return process(audio, voice.sample_rate, sample_rate)

    def render_all(self, sentences: List[str], voice: Voice, sample_rate: int, speed: float) -> bytes:
        """PCM of all the sentences in a row. Blocking: call it from a worker thread."""
        return b"".join(self.render(sentence, voice, sample_rate, speed) for sentence in sentences)
