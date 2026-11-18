"""
Configuration: one version everywhere, the defaults written down in .env.example.
"""

import sys
import tomllib

from pathlib import Path

from app.core.config import ServiceSettings, Settings

ROOT = Path(__file__).resolve().parents[1]


def clear_settings_env(monkeypatch) -> None:
    for name in (*Settings.model_fields, *ServiceSettings.model_fields):
        monkeypatch.delenv(name, raising=False)


def test_versions_match(client, auth):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

    assert client.app.version == version
    assert client.get("/v1/info", headers=auth).json()["version"] == version


def test_settings_defaults(monkeypatch):
    clear_settings_env(monkeypatch)
    settings = Settings(_env_file=None)

    # The fake backend by default: the container starts without weights and GPU
    assert (settings.MODEL_BACKEND, settings.MODEL_NAME, settings.TARGET_PLATFORM) == ("fake", None, "cpu")
    assert (settings.WORKERS, settings.QUEUE_MAX, settings.FAKE_LATENCY_MS) == (1, 4, 0)
    assert settings.AUTH_TOKEN is None


def test_env_example_matches_defaults(monkeypatch):
    clear_settings_env(monkeypatch)
    example = ROOT / ".env.example"

    assert Settings(_env_file=example).model_dump() == Settings(_env_file=None).model_dump()
    assert [name for name in Settings.model_fields if name not in example.read_text()] == []


def test_service_settings_defaults(monkeypatch):
    clear_settings_env(monkeypatch)
    settings = ServiceSettings(_env_file=None)

    assert (settings.CACHE_MAX_ITEMS, settings.PREWARM_SAMPLE_RATE) == (256, 24000)
    assert settings.prewarm_phrases_path == ROOT / "configs" / "prewarm_phrases.txt"


def test_service_env_example_matches_defaults(monkeypatch):
    clear_settings_env(monkeypatch)
    example = ROOT / ".env.example"

    assert ServiceSettings(_env_file=example).model_dump() == ServiceSettings(_env_file=None).model_dump()
    assert [name for name in ServiceSettings.model_fields if name not in example.read_text()] == []


def test_prewarm_sample_rate_checked(monkeypatch):
    monkeypatch.setenv("PREWARM_SAMPLE_RATE", "16000")
    assert ServiceSettings(_env_file=None).PREWARM_SAMPLE_RATE == 16000

    monkeypatch.setenv("PREWARM_SAMPLE_RATE", "44100")
    try:
        ServiceSettings(_env_file=None)
    except ValueError as e:
        assert "PREWARM_SAMPLE_RATE" in str(e)
    else:
        raise AssertionError("44100 Hz must be refused")


def test_python_312():
    # uv picks the interpreter by .python-version: the lock and the image are on 3.12
    assert sys.version_info[:2] == (3, 12)
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
