import os

from typing import Dict, Iterator

import numpy as np
import pytest

from fastapi.testclient import TestClient

TOKEN = "test-token"

# The settings are read once, at the app import: the environment is set before it.
# Variables win over a local .env, so the tests run the same with it and without it.
os.environ.update({
    "MODEL_BACKEND": "fake",
    "MODEL_NAME": "fake",
    "TARGET_PLATFORM": "cpu",
    "FAKE_LATENCY_MS": "0",
    "AUTH_TOKEN": TOKEN,
    "WORKERS": "1",
    "QUEUE_MAX": "4",
    "GIT_SHA": "abc1234",
    "CACHE_MAX_ITEMS": "256",
    "PREWARM_PHRASES_PATH": "configs/prewarm_phrases.txt",
    "PREWARM_SAMPLE_RATE": "24000",
})


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    from app.main import app

    # `with` runs the lifespan: the engine is built and warmed up
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth() -> Dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def pcm_samples(pcm: bytes) -> np.ndarray:
    """Signed 16-bit little-endian PCM as float samples in [-1, 1]."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32767


def peak_dbfs(samples: np.ndarray) -> float:
    return 20 * np.log10(np.max(np.abs(samples)))


def leading_silence_ms(samples: np.ndarray, sample_rate: int, threshold_db: float = -40.0) -> float:
    """Time before the first sample louder than `threshold_db` below the peak."""
    threshold = np.max(np.abs(samples)) * 10 ** (threshold_db / 20)
    return int(np.argmax(np.abs(samples) >= threshold)) / sample_rate * 1000
