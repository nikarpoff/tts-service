"""
WAV container for the whole-file answers and the app phrases
"""

import io
import wave

from app.core.audio.process import BYTES_PER_SAMPLE


def to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Mono 16-bit PCM as a WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(BYTES_PER_SAMPLE)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buffer.getvalue()
