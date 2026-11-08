"""
Phrase cache (TTS-5): a repeat is a hit, the prewarm fills the cache at startup, LRU eviction.
"""

import asyncio

from contextlib import ExitStack
from pathlib import Path

from app.core.cache import AudioCache, CacheKey, load_phrases
from app.core.config import ServiceSettings
from app.core.synthesizer import SpeechSynthesizer

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PHRASES = ROOT / "tests" / "contract" / "prewarm_phrases.txt"


def key(text: str, voice: str = "fake_f") -> CacheKey:
    return CacheKey("fake", voice, 24000, 1.0, text)


def test_repeat_is_hit(client, auth):
    for audio_format in ("pcm_s16le", "wav"):
        request = {"text": f"Повтор фразы, {audio_format}. Второе предложение.", "format": audio_format}
        first = client.post("/v1/synthesize", json=request, headers=auth)
        second = client.post("/v1/synthesize", json=request, headers=auth)

        assert first.headers["X-Cache"] == "miss", audio_format
        assert second.headers["X-Cache"] == "hit", audio_format
        assert second.content == first.content, audio_format


def test_key_parts(client, auth):
    text = "Ключ кэша."
    assert client.post("/v1/synthesize", json={"text": text}, headers=auth).headers["X-Cache"] == "miss"

    # Another voice, rate or speed is another audio; the format is not: WAV wraps the same PCM
    for change in ({"voice": "fake_m"}, {"sample_rate": 16000}, {"speed": 1.5}):
        response = client.post("/v1/synthesize", json={"text": text, **change}, headers=auth)
        assert response.headers["X-Cache"] == "miss", change
    assert client.post("/v1/synthesize", json={"text": text, "format": "wav"}, headers=auth).headers["X-Cache"] == "hit"

    # The key is the normalized text: the explicit default voice and the missing final period hit too
    for same in ({"text": "Ключ кэша"}, {"text": "  Ключ   кэша. "}, {"text": text, "voice": "fake_f"}):
        assert client.post("/v1/synthesize", json=same, headers=auth).headers["X-Cache"] == "hit", same


def test_prewarm_at_startup(client, auth):
    synthesizer = client.app.state.synthesizer
    phrases = load_phrases(CONTRACT_PHRASES)
    assert len(synthesizer.cache) >= len(phrases)

    # The warnings of the passive mode are hits from the first request, as the gateway sends them
    for phrase in phrases:
        response = client.post(
            "/v1/synthesize", json={"text": phrase, "sample_rate": 24000, "format": "pcm_s16le", "speed": 1.0}, headers=auth
        )
        assert response.status_code == 200, phrase
        assert response.headers["X-Cache"] == "hit", phrase
        assert len(response.content) > 0, phrase


def test_hit_served_when_busy(client, auth):
    # A hit needs no model: a full queue does not stop it
    engine = client.app.state.engine
    with ExitStack() as stack:
        for _ in range(engine.settings.WORKERS + engine.settings.QUEUE_MAX):
            stack.enter_context(engine.admit())
        hit = client.post("/v1/synthesize", json={"text": "Впереди светофор"}, headers=auth)
        miss = client.post("/v1/synthesize", json={"text": "Новая фраза при занятой очереди."}, headers=auth)

    assert (hit.status_code, hit.headers["X-Cache"]) == (200, "hit")
    assert miss.status_code == 503


def test_prewarm_fills_new_cache(client, tmp_path):
    phrases = tmp_path / "phrases.txt"
    phrases.write_text("Впереди стол\n\nСлева дверь.\nВпереди стол.\n", encoding="utf-8")
    engine = client.app.state.engine

    synthesizer = SpeechSynthesizer(engine, ServiceSettings(_env_file=None, PREWARM_PHRASES_PATH=phrases, PREWARM_SAMPLE_RATE=16000))
    asyncio.run(synthesizer.startup())

    # «Впереди стол» and «Впереди стол.» are one phrase
    voice = engine.backend.default_voice.id
    assert len(synthesizer.cache) == 2
    assert CacheKey("fake", voice, 16000, 1.0, "Впереди стол.") in synthesizer.cache
    assert CacheKey("fake", voice, 16000, 1.0, "Слева дверь.") in synthesizer.cache

    # A cache turned off prewarms nothing
    disabled = SpeechSynthesizer(engine, ServiceSettings(_env_file=None, PREWARM_PHRASES_PATH=phrases, CACHE_MAX_ITEMS=0))
    asyncio.run(disabled.startup())
    assert len(disabled.cache) == 0


def test_lru_eviction():
    cache = AudioCache(2)
    cache.put(key("а"), b"a")
    cache.put(key("б"), b"b")
    assert cache.get(key("а")) == b"a"          # «а» is now the most recently used

    cache.put(key("в"), b"c")
    assert key("б") not in cache
    assert (cache.get(key("а")), cache.get(key("в"))) == (b"a", b"c")
    assert len(cache) == 2

    off = AudioCache(0)
    off.put(key("а"), b"a")
    assert off.get(key("а")) is None


def test_prewarm_matches_contract():
    # The gateway sends the warnings exactly as the contract lists them: any difference is a miss
    configured = ROOT / "configs" / "prewarm_phrases.txt"
    assert configured.read_bytes() == CONTRACT_PHRASES.read_bytes()
    assert ServiceSettings(_env_file=None).prewarm_phrases_path == configured

    phrases = load_phrases(configured)
    assert len(phrases) == 26
    assert len(set(phrases)) == len(phrases)
    assert "Осторожно, впереди ступени" in phrases
    assert all(not phrase.endswith(".") for phrase in phrases)
