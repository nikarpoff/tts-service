"""
Backend registry: MODEL_BACKEND picks the backend, a wrong one stops the start with a clear message.
"""

import re
import subprocess
import sys

from pathlib import Path

import pytest

from loguru import logger

from app.core.backends import registry
from app.core.backends.fake import FakeBackend
from app.core.backends.registry import BACKENDS, BackendSpec, build_backend
from app.core.backends.weights import fetch, weights_dir
from app.core.config import Settings
from app.core.exceptions import BackendError
from scripts.fetch_weights import main as fetch_weights_main

ROOT = Path(__file__).resolve().parents[1]

EXTRA_BACKENDS = sorted(name for name, spec in BACKENDS.items() if spec.extra)


def make_settings(**values) -> Settings:
    # No .env: the test sees the defaults and its own values only
    return Settings(_env_file=None, **values)


def test_registry_selects_fake():
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        backend = build_backend(make_settings(MODEL_BACKEND="fake", MODEL_NAME=None))
    finally:
        logger.remove(sink)

    assert isinstance(backend, FakeBackend)
    assert (backend.name, backend.model, backend.device) == ("fake", "fake", "cpu")
    # Synthetic answers must not pass for a measurement unnoticed
    assert any(record["level"].name == "WARNING" for record in records)


def test_registry_unknown_backend_fails():
    message = f"Unknown backend 'yolo9000', expected one of: {', '.join(sorted(BACKENDS))}"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_BACKEND="yolo9000"))

    message = "Unknown model 'large' of backend 'fake', expected one of: fake"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_BACKEND="fake", MODEL_NAME="large"))


@pytest.mark.parametrize("name", EXTRA_BACKENDS)
def test_registry_missing_extra_fails(name, monkeypatch):
    # None of the extra libraries is installed, as in the fast environment
    monkeypatch.setattr(registry, "find_spec", lambda package: None)

    message = f"Backend '{name}' requires the '{BACKENDS[name].extra}' extra"
    with pytest.raises(BackendError, match=re.escape(message)):
        build_backend(make_settings(MODEL_BACKEND=name))


def test_registry_missing_library_on_import_fails(tmp_path, monkeypatch):
    # The backend module imports a library of its extra, and the library is not installed
    (tmp_path / "alv_probe_backend.py").write_text("import alv_missing_library\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(BACKENDS, "probe", BackendSpec("alv_probe_backend:ProbeBackend", "probe"))

    with pytest.raises(BackendError, match=re.escape("Backend 'probe' requires the 'probe' extra")):
        build_backend(make_settings(MODEL_BACKEND="probe"))


@pytest.mark.parametrize("name", EXTRA_BACKENDS)
def test_registry_real_backend_declares_models(name):
    # The registry entry points at the module of the backend, its extra and the packages checked before the import
    spec = BACKENDS[name]
    module_name, _ = spec.path.split(":")
    assert module_name == f"app.core.backends.{name}"
    assert spec.extra == name and spec.packages


def test_registry_imports_no_model_library():
    # The fast environment builds fake without torch or piper: nothing of the real backends is imported
    code = (
        "import sys; from app.core.backends import build_backend; from app.core.config import Settings;"
        "build_backend(Settings(_env_file=None, MODEL_BACKEND='fake'));"
        "print(sorted(m for m in ('torch', 'piper', 'f5_tts', 'onnxruntime') if m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_weights_dir_under_artifacts_path(tmp_path, monkeypatch):
    assert weights_dir(make_settings(ARTIFACTS_PATH=tmp_path), "piper") == tmp_path / "piper"

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert weights_dir(make_settings(ARTIFACTS_PATH=None), "silero") == tmp_path / "cache" / "tts-service" / "silero"


def test_weights_fetched_once(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    target = tmp_path / "weights" / "model.bin"

    assert fetch(source.as_uri(), target) == target
    assert target.read_bytes() == b"weights"

    # The file in place is not downloaded again: a start needs no network
    source.unlink()
    assert fetch(source.as_uri(), target) == target


def test_weights_unreachable_fail(tmp_path):
    target = tmp_path / "model.bin"
    with pytest.raises(BackendError, match="cannot be downloaded"):
        fetch((tmp_path / "missing.bin").as_uri(), target)
    # No partial file passes for the weights
    assert list(tmp_path.iterdir()) == []


def test_fetch_weights_script(capsys):
    assert fetch_weights_main(["--backend", "fake"]) == 0
    assert capsys.readouterr().out == "fake/fake\tвесов нет\n"

    assert fetch_weights_main(["--backend", "yolo9000"]) == 1
    assert "неизвестный бэкенд 'yolo9000'" in capsys.readouterr().err
    assert fetch_weights_main(["--backend", "piper", "--model", "ru_RU-nobody-low"]) == 1
    assert "неизвестная модель 'ru_RU-nobody-low'" in capsys.readouterr().err


def test_fetch_weights_script_missing_extra(monkeypatch, capsys):
    monkeypatch.setattr(registry, "find_spec", lambda package: None)
    assert fetch_weights_main(["--backend", "silero"]) == 1
    assert "Backend 'silero' requires the 'silero' extra" in capsys.readouterr().err
