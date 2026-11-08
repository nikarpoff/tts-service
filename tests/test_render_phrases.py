"""
App phrases (TTS-8): YAML id -> text becomes phrase_<id>.wav in the target directory.
"""

import io
import subprocess
import sys
import wave

from pathlib import Path
from typing import Tuple

import pytest

from conftest import leading_silence_ms, pcm_samples, peak_dbfs
from scripts.render_phrases import main

ROOT = Path(__file__).resolve().parents[1]

PHRASES = """\
no_connection: Нет связи с сервером
passive_on: Пассивный режим включён
asr_empty: Не расслышал вопрос. Повторите, пожалуйста.
"""


@pytest.fixture
def phrases(tmp_path) -> Path:
    path = tmp_path / "phrases.yaml"
    path.write_text(PHRASES, encoding="utf-8")
    return path


def read_wav(path: Path) -> Tuple[int, bytes]:
    with wave.open(io.BytesIO(path.read_bytes())) as wav:
        assert (wav.getnchannels(), wav.getsampwidth()) == (1, 2)
        return wav.getframerate(), wav.readframes(wav.getnframes())


def test_renders_every_phrase(phrases, tmp_path):
    out = tmp_path / "res" / "raw"
    assert main(["--phrases", str(phrases), "--out", str(out), "--backend", "fake"]) == 0

    assert sorted(path.name for path in out.iterdir()) == [
        "phrase_asr_empty.wav", "phrase_no_connection.wav", "phrase_passive_on.wav",
    ]
    for path in out.iterdir():
        rate, pcm = read_wav(path)
        samples = pcm_samples(pcm)
        assert rate == 24000
        assert len(samples) / rate > 0.5
        assert peak_dbfs(samples) <= -1.0 + 1e-3
        assert leading_silence_ms(samples, rate) <= 50

    # Two sentences sound longer than one short phrase
    assert (out / "phrase_asr_empty.wav").stat().st_size > (out / "phrase_no_connection.wav").stat().st_size


def test_options(phrases, tmp_path):
    out = tmp_path / "raw"
    assert main(["--phrases", str(phrases), "--out", str(out), "--backend", "fake", "--voice", "fake_m", "--sample-rate", "16000"]) == 0
    assert read_wav(out / "phrase_passive_on.wav")[0] == 16000

    # The same input renders the same files: the app resources change only with the phrases
    first = (out / "phrase_passive_on.wav").read_bytes()
    assert main(["--phrases", str(phrases), "--out", str(out), "--backend", "fake", "--voice", "fake_m", "--sample-rate", "16000"]) == 0
    assert (out / "phrase_passive_on.wav").read_bytes() == first


@pytest.mark.parametrize(
    "content",
    ["Bad-Id: Текст\n", "1st: Текст\n", "empty: ''\n", "- список\n- фраз\n", "{}\n", "a: [не, строка]\n", "a: : :\n"],
    ids=["dash_and_capital", "digit_first", "empty_text", "list", "empty", "not_text", "broken_yaml"],
)
def test_bad_phrases_fail(tmp_path, content, capsys):
    path = tmp_path / "phrases.yaml"
    path.write_text(content, encoding="utf-8")
    out = tmp_path / "raw"

    assert main(["--phrases", str(path), "--out", str(out), "--backend", "fake"]) == 1
    assert "Ошибка" in capsys.readouterr().err
    assert not out.exists()


def test_missing_file_and_unknown_voice_fail(phrases, tmp_path):
    out = tmp_path / "raw"
    assert main(["--phrases", str(tmp_path / "missing.yaml"), "--out", str(out), "--backend", "fake"]) == 1
    assert main(["--phrases", str(phrases), "--out", str(out), "--backend", "fake", "--voice", "nobody"]) == 1
    assert main(["--phrases", str(phrases), "--out", str(out), "--backend", "yolo9000"]) == 1
    assert not out.exists()


def test_runs_as_script(phrases, tmp_path):
    # The way the app task runs it: a file path from the service root, any target directory
    out = tmp_path / "app" / "src" / "main" / "res" / "raw"
    result = subprocess.run(
        [sys.executable, "scripts/render_phrases.py", "--phrases", str(phrases), "--out", str(out), "--backend", "fake"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "phrase_no_connection.wav").exists()
    assert "phrase_no_connection.wav" in result.stdout
