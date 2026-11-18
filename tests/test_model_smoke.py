"""
Smoke of a real model: a CPU model synthesizes non-empty speech of a plausible length (TTS-3).

The model is Piper with the default Russian voice: the library is in the dev group, so the
test needs no extra, only the network or a cache with the weights (~63 MB). Silero is
checked the same way where its extra is installed. The text goes the way of a request:
normalization, sentences, the model, the sound processing.
"""

from importlib.util import find_spec

import numpy as np
import pytest

from app.core.backends import build_backend
from app.core.config import Settings
from app.core.pipeline import SpeechPipeline
from conftest import leading_silence_ms, pcm_samples, peak_dbfs

pytestmark = pytest.mark.model

TEXT = "Впереди стол и два стула, слева дверь. Осторожно, ступени!"
SAMPLE_RATE = 24000
MS_PER_CHAR = (40.0, 150.0)         # Russian speech at the usual speed is 60-80 ms per character
MIN_RMS_DB = -30.0                  # below the peak: speech, not a click in silence

CPU_BACKENDS = [
    "piper",
    pytest.param("silero", marks=pytest.mark.skipif(find_spec("torch") is None, reason="the silero extra is not installed")),
]


@pytest.fixture(scope="module", params=CPU_BACKENDS)
def pipeline(request):
    # No .env and no fake of the fast tests: the settings of the model are given here
    settings = Settings(_env_file=None, MODEL_BACKEND=request.param, MODEL_NAME=None, TARGET_PLATFORM="cpu", ARTIFACTS_PATH=None)
    backend = build_backend(settings)
    yield SpeechPipeline(backend)
    backend.close()


def test_cpu_model_speaks_plausible_length(pipeline):
    assert pipeline.backend.device == "cpu"
    pipeline.backend.warmup()

    voice = pipeline.voice(None)
    sentences = pipeline.sentences(pipeline.normalize(TEXT))
    samples = pcm_samples(pipeline.render_all(sentences, voice, SAMPLE_RATE, 1.0))

    duration_ms = samples.size / SAMPLE_RATE * 1000
    per_char = duration_ms / len(TEXT)
    assert MS_PER_CHAR[0] <= per_char <= MS_PER_CHAR[1], f"{duration_ms:.0f} ms for {len(TEXT)} chars"

    # Speech: loud enough on average, normalized and starting at once (TTS-4)
    rms_db = 20 * np.log10(np.sqrt(np.mean(samples ** 2)))
    assert rms_db - peak_dbfs(samples) >= MIN_RMS_DB
    assert -1.5 <= peak_dbfs(samples) <= -0.5
    assert leading_silence_ms(samples, SAMPLE_RATE) <= 50


def test_cpu_model_speed_shortens_speech(pipeline):
    voice = pipeline.voice(None)
    sentences = pipeline.sentences(pipeline.normalize(TEXT))

    usual = len(pipeline.render_all(sentences, voice, SAMPLE_RATE, 1.0))
    fast = len(pipeline.render_all(sentences, voice, SAMPLE_RATE, 1.5))
    assert fast < 0.85 * usual


def test_cpu_model_reads_unreadable_as_pause(pipeline):
    # A sentence without Russian letters must not stop the answer: a short pause instead
    voice = pipeline.voice(None)
    pcm = pipeline.render("…", voice, SAMPLE_RATE, 1.0)
    assert 0 < len(pcm) // 2 <= SAMPLE_RATE


def test_piper_reads_stress_mark():
    # Piper reads the `+` of the request as the stress, not as a word (TTS-2)
    backend = build_backend(Settings(_env_file=None, MODEL_BACKEND="piper", MODEL_NAME=None, TARGET_PLATFORM="cpu", ARTIFACTS_PATH=None))
    try:
        phonemes = {text: "".join(backend._piper.phonemize(backend.espeak_text(text))[0]) for text in ("з+амок.", "зам+ок.")}
    finally:
        backend.close()

    assert phonemes["з+амок."] != phonemes["зам+ок."]
    # One word with one stress: the mark is not read aloud as "плюс"
    assert all(" " not in item and item.count("ˈ") == 1 for item in phonemes.values()), phonemes
