"""
Shared service layer and the synthesis API: health, info, voices, synthesize, auth,
request id, Server-Timing and errors.
"""

import asyncio
import io
import json
import wave
import os
import re
import tomllib

from contextlib import ExitStack
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Tuple

import pytest

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from loguru import logger

from app.api_v1.errors import register_error_handlers
from app.api_v1.schemas import InfoResponse, VoiceInfo
from conftest import leading_silence_ms, pcm_samples, peak_dbfs
from app.utils.middleware import current_timings, RequestIdMiddleware, ServerTimingMiddleware

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests" / "contract"

GENERATED_ID = re.compile(r"[0-9a-f]{8}")
SERVER_TIMING = re.compile(r"(?:[a-z_]+;dur=\d+\.\d, )*total;dur=\d+\.\d")


@pytest.fixture(scope="module")
def probe(client: TestClient) -> Iterator[TestClient]:
    """The shared middlewares, error handlers and engine behind the routes the services add later."""
    engine = client.app.state.engine
    app = FastAPI()
    register_error_handlers(app)
    app.add_middleware(ServerTimingMiddleware)
    app.add_middleware(RequestIdMiddleware)

    @app.get("/probe/log")
    async def log_lines() -> dict:
        logger.info("probe: event loop")
        await engine.run(lambda: logger.info("probe: worker thread"))
        return {}

    @app.get("/probe/stream")
    async def stream() -> StreamingResponse:
        with current_timings().stage("prepare"):
            pass

        async def chunks() -> AsyncIterator[bytes]:
            yield b"first"
            yield b"second"

        return StreamingResponse(chunks(), media_type="audio/L16; rate=24000; channels=1")

    @app.get("/probe/timings")
    async def timings() -> dict:
        with current_timings().stage("decode"):
            pass
        return current_timings().as_dict()

    @app.get("/probe/busy")
    async def busy() -> dict:
        with engine.admit():
            return {}

    @app.get("/probe/params")
    async def params(n: int = Query(..., ge=1)) -> dict:
        return {"n": n}

    @app.get("/probe/error")
    async def error() -> dict:
        raise RuntimeError("probe failure")

    with TestClient(app, raise_server_exceptions=False) as probe_client:
        yield probe_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    # Not before the warmup
    engine = client.app.state.engine
    engine.warmed_up = False
    try:
        assert client.get("/health").status_code == 503
    finally:
        engine.warmed_up = True


def test_info(client, auth):
    response = client.get("/v1/info", headers=auth)
    assert response.status_code == 200
    info = response.json()

    # The shape of the reference example: the gateway builds its config_id of these keys
    example = json.loads((CONTRACT / "info.response.json").read_text())
    InfoResponse.model_validate(example)
    assert info.keys() == example.keys()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert info["service"] == project["name"] == example["service"]
    assert info["version"] == project["version"]
    assert info["git_sha"] == os.environ["GIT_SHA"]
    assert (info["backend"], info["model"], info["device"]) == ("fake", "fake", "cpu")
    assert info["warmed_up"] is True
    assert info["params"]["latency_ms"] == 0
    # The synthesis parameters of the reference example
    assert {"voice", "sample_rate", "cache_max_items"} <= info["params"].keys()
    assert info["params"]["cache_max_items"] == 256


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": "Basic dGVzdC10b2tlbg=="}],
    ids=["no_token", "wrong_token", "not_bearer"],
)
def test_info_unauthorized_401(client, headers):
    response = client.get("/v1/info", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_request_id_echoed(client, auth, probe):
    for path, headers in (("/health", {}), ("/v1/info", auth), ("/v1/info", {})):
        response = client.get(path, headers={**headers, "X-Request-ID": "req-0001"})
        assert response.headers["X-Request-ID"] == "req-0001", path

    # The streaming answer of the synthesis too
    response = client.post(
        "/v1/synthesize", json={"text": "Проверка идентификатора."}, headers={**auth, "X-Request-ID": "req-0003"}
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-0003"

    # Every log line of the request carries the id, the worker thread lines too
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        response = probe.get("/probe/log", headers={"X-Request-ID": "req-0002"})
    finally:
        logger.remove(sink)

    assert response.headers["X-Request-ID"] == "req-0002"
    lines = [record for record in records if record["message"].startswith("probe:")]
    assert len(lines) == 2
    assert all(record["extra"]["request_id"] == "req-0002" for record in lines)


def test_request_id_generated(client, auth):
    ids = [client.get("/v1/info", headers=auth).headers["X-Request-ID"] for _ in range(2)]
    assert all(GENERATED_ID.fullmatch(request_id) for request_id in ids)
    assert ids[0] != ids[1]

    # An id that would break a log line is replaced as well
    for bad_id in ("has spaces in it", "x" * 200):
        response = client.get("/health", headers={"X-Request-ID": bad_id})
        assert GENERATED_ID.fullmatch(response.headers["X-Request-ID"])


def test_server_timing_header(client, auth, probe):
    responses = {
        "health": client.get("/health"),
        "info": client.get("/v1/info", headers=auth),
        "unauthorized": client.get("/v1/info"),
        "not_found": client.get("/v1/missing", headers=auth),
        "timings": probe.get("/probe/timings"),
        "stream": probe.get("/probe/stream"),
        "internal_error": probe.get("/probe/error"),
        "synthesize_stream": client.post("/v1/synthesize", json={"text": "Заголовки потока. Второе предложение."}, headers=auth),
        "synthesize_wav": client.post("/v1/synthesize", json={"text": "Заголовки файла.", "format": "wav"}, headers=auth),
        "synthesize_422": client.post("/v1/synthesize", json={"text": ""}, headers=auth),
    }
    for name, response in responses.items():
        assert SERVER_TIMING.fullmatch(response.headers.get("Server-Timing", "")), name
        assert response.headers.get("X-Request-ID"), name

    # The same stages go to the `timings` of a body
    timings = responses["timings"]
    assert timings.headers["Server-Timing"].startswith("decode;dur=")
    assert timings.json().keys() == {"decode_ms", "total_ms"}

    # A streaming response gets the stages done before its body
    stream = responses["stream"]
    assert stream.headers["Server-Timing"].startswith("prepare;dur=")
    assert stream.content == b"firstsecond"

    # The synthesis stream: normalization and the first sentence are done before the body
    synthesis = responses["synthesize_stream"]
    assert synthesis.headers["Server-Timing"].startswith("normalize;dur=")
    assert "synthesize;dur=" in synthesis.headers["Server-Timing"]
    assert "audio;dur=" in responses["synthesize_wav"].headers["Server-Timing"]

    # The 500 comes from outside of the middlewares and keeps the headers
    error = responses["internal_error"]
    assert error.status_code == 500
    assert error.json() == {"detail": "Internal server error"}


def test_busy_503(client, probe):
    engine = client.app.state.engine
    places = engine.settings.WORKERS + engine.settings.QUEUE_MAX

    with ExitStack() as stack:
        for _ in range(places):
            stack.enter_context(engine.admit())
        response = probe.get("/probe/busy")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service is busy"}
    assert response.headers["Retry-After"] == "1"

    # The places are given back
    assert probe.get("/probe/busy").status_code == 200


def test_validation_error_422(probe):
    response = probe.get("/probe/params", params={"n": 0})
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request: Input should be greater than or equal to 1 at query.n"}


# =====================================================================
# Synthesis
# =====================================================================

def pcm_headers_ok(response, sample_rate: int, cache: str) -> None:
    assert response.headers["content-type"] == f"audio/L16; rate={sample_rate}; channels=1"
    assert response.headers["X-Sample-Rate"] == str(sample_rate)
    assert response.headers["X-Audio-Format"] == "pcm_s16le"
    assert response.headers["X-Cache"] == cache


def test_voices(client, auth):
    response = client.get("/v1/voices", headers=auth)
    assert response.status_code == 200
    voices = [VoiceInfo.model_validate(voice) for voice in response.json()]

    assert [voice.id for voice in voices] == ["fake_f", "fake_m"]
    assert all(voice.backend == "fake" and voice.sample_rate == 48000 for voice in voices)
    # Exactly one default voice, the one /v1/info shows
    assert [voice.id for voice in voices if voice.default] == ["fake_f"]
    assert client.get("/v1/info", headers=auth).json()["params"]["voice"] == "fake_f"

    assert client.get("/v1/voices").status_code == 401


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 24000, 48000])
def test_synthesize_pcm_ok(client, auth, sample_rate):
    text = f"Впереди стол. Справа 2 двери, частота {sample_rate}."
    response = client.post(
        "/v1/synthesize",
        json={"text": text, "sample_rate": sample_rate, "format": "pcm_s16le", "speed": 1.0},
        headers=auth,
    )
    assert response.status_code == 200
    pcm_headers_ok(response, sample_rate, "miss")
    # Streamed: no length known before the body, the server sends it chunked
    assert "content-length" not in response.headers

    samples = pcm_samples(response.content)
    # Two sentences of the fake tone: 60 ms per character of the normalized text, the silence cut
    assert len(response.content) % 2 == 0
    assert 1.0 < len(samples) / sample_rate < 6.0
    assert peak_dbfs(samples) <= -1.0 + 1e-3
    assert leading_silence_ms(samples, sample_rate) <= 50

    # Optional fields take their defaults: the default voice, 24 kHz, PCM
    response = client.post("/v1/synthesize", json={"text": f"Только текст {sample_rate}."}, headers=auth)
    assert response.status_code == 200
    pcm_headers_ok(response, 24000, "miss")


def test_synthesize_wav_ok(client, auth):
    response = client.post(
        "/v1/synthesize",
        json={"text": "Слева дверь, справа окно.", "voice": "fake_m", "sample_rate": 16000, "format": "wav", "speed": 2.0},
        headers=auth,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["X-Sample-Rate"] == "16000"
    assert response.headers["X-Audio-Format"] == "wav"
    assert response.headers["X-Cache"] == "miss"
    assert int(response.headers["content-length"]) == len(response.content)

    with wave.open(io.BytesIO(response.content)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        frames = wav.readframes(wav.getnframes())
    samples = pcm_samples(frames)
    assert peak_dbfs(samples) <= -1.0 + 1e-3

    # Twice the speed - half the duration of the same text
    slow = client.post(
        "/v1/synthesize",
        json={"text": "Слева дверь, справа окно.", "voice": "fake_m", "sample_rate": 16000, "format": "wav", "speed": 1.0},
        headers=auth,
    )
    assert len(slow.content) > 1.8 * len(frames)


def test_synthesize_empty_text_422(client, auth):
    for text in ("", "   ", "...", "?!"):
        response = client.post("/v1/synthesize", json={"text": text}, headers=auth)
        assert response.status_code == 422, text
        assert response.json()["detail"], text

    assert client.post("/v1/synthesize", json={}, headers=auth).status_code == 422
    response = client.post("/v1/synthesize", json={"text": "..."}, headers=auth)
    assert response.json() == {"detail": "Text has no words to speak"}


def test_synthesize_too_long_422(client, auth):
    assert client.post("/v1/synthesize", json={"text": "а" * 1000}, headers=auth).status_code == 200

    response = client.post("/v1/synthesize", json={"text": "а" * 1001}, headers=auth)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request: String should have at most 1000 characters at body.text"}


def test_synthesize_unknown_voice_422(client, auth):
    response = client.post("/v1/synthesize", json={"text": "Привет.", "voice": "nobody"}, headers=auth)
    assert response.status_code == 422
    # The reference error example of the contract
    assert response.json() == json.loads((CONTRACT / "error.response.json").read_text())


@pytest.mark.parametrize(
    "params",
    [{"sample_rate": 44100}, {"sample_rate": 0}, {"sample_rate": "fast"}],
    ids=["44100", "zero", "not_a_number"],
)
def test_synthesize_bad_sample_rate_422(client, auth, params):
    response = client.post("/v1/synthesize", json={"text": "Привет.", **params}, headers=auth)
    assert response.status_code == 422
    assert response.json()["detail"].endswith("at body.sample_rate")


@pytest.mark.parametrize(
    "params",
    [{"speed": 0.4}, {"speed": 2.1}, {"format": "mp3"}, {"extra": 1}],
    ids=["slow", "fast", "format", "unknown_field"],
)
def test_synthesize_bad_params_422(client, auth, params):
    response = client.post("/v1/synthesize", json={"text": "Привет.", **params}, headers=auth)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}],
    ids=["no_token", "wrong_token"],
)
def test_synthesize_unauthorized_401(client, headers):
    response = client.post("/v1/synthesize", json={"text": "Привет."}, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}


def test_synthesize_busy_503(client, auth):
    engine = client.app.state.engine
    places = engine.settings.WORKERS + engine.settings.QUEUE_MAX

    for audio_format in ("pcm_s16le", "wav"):
        with ExitStack() as stack:
            for _ in range(places):
                stack.enter_context(engine.admit())
            response = client.post(
                "/v1/synthesize", json={"text": f"Очередь занята, {audio_format}.", "format": audio_format}, headers=auth
            )

        assert response.status_code == 503, audio_format
        assert response.json() == {"detail": "Service is busy"}
        assert response.headers["Retry-After"] == "1"

    # The places are given back, the streams included
    assert engine._admitted == 0
    assert client.post("/v1/synthesize", json={"text": "Очередь свободна."}, headers=auth).status_code == 200
    assert engine._admitted == 0


async def call_asgi(app, path: str, body: bytes, headers: Dict[str, str], events: List[Tuple[str, Any]]) -> None:
    """Runs one request through the ASGI app, logging every body message as it is sent.

    The test client gathers the whole body before it returns: only the ASGI messages
    show when each chunk leaves the service.
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(name.lower().encode(), value.encode()) for name, value in headers.items()],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    request_sent = False

    async def receive() -> dict:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            events.append(("start", message["status"]))
        elif message["type"] == "http.response.body" and message.get("body"):
            events.append(("chunk", len(message["body"])))

    await app(scope, receive, send)


def test_synthesize_streams_first_sentence_early(client, auth, monkeypatch):
    backend = client.app.state.engine.backend
    sentences = ["Первое предложение потока.", "Второе предложение потока.", "Третье предложение потока."]
    events: List[Tuple[str, Any]] = []

    # Every sentence takes a while, and the backend marks when it is done with one
    synthesize = backend.synthesize
    monkeypatch.setattr(backend, "latency_ms", 100)

    def synthesize_logged(text, voice, speed):
        audio = synthesize(text, voice, speed)
        events.append(("synthesized", text))
        return audio

    monkeypatch.setattr(backend, "synthesize", synthesize_logged)

    body = json.dumps({"text": " ".join(sentences)}).encode()
    headers = {**auth, "Content-Type": "application/json", "Content-Length": str(len(body))}
    asyncio.run(call_asgi(client.app, "/v1/synthesize", body, headers, events))

    assert events[0] == ("synthesized", sentences[0])
    assert events[1] == ("start", 200)
    # A chunk per sentence, each sent before the next sentence is synthesized
    assert [kind for kind, _ in events] == ["synthesized", "start", "chunk", "synthesized", "chunk", "synthesized", "chunk"]
    assert events.index(("synthesized", sentences[1])) > 2
    assert events.index(("synthesized", sentences[2])) > events.index(("synthesized", sentences[1]))
