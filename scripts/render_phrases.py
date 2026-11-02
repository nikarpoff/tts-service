"""
Системные фразы приложения в WAV (TTS-8, MOB-11).

Читает YAML вида `id: текст` и пишет в каталог назначения `phrase_<id>.wav`
(моно, 16 бит) тем же путём, что и API: нормализация, предложения, бэкенд, обработка
звука. Так системные сообщения звучат одним голосом с ответами и без сети. Каталог
назначения - ресурсы приложения; скрипт только пишет файлы, коммитит их репозиторий
приложения.

    uv run scripts/render_phrases.py \\
        --phrases ../assistant-low-vision-mobile/phrases/phrases.yaml \\
        --out ../assistant-low-vision-mobile/app/src/main/res/raw --backend fake
"""

import argparse
import re
import sys

from pathlib import Path

import yaml

# Скрипт запускается файлом из корня сервиса: пакет app лежит уровнем выше
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.audio import pcm_duration_ms, to_wav  # noqa: E402
from app.core.backends import build_backend  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.exceptions import AppError  # noqa: E402
from app.core.pipeline import SpeechPipeline  # noqa: E402
from app.utils.logging import configure_logging  # noqa: E402

# Имя ресурса Android: строчные латинские буквы, цифры и подчёркивание, первой - буква
PHRASE_ID = re.compile(r"[a-z][a-z0-9_]*")
SAMPLE_RATES = [8000, 16000, 22050, 24000, 48000]


class PhrasesError(Exception):
    """Файл фраз не читается или не того вида."""


def load_phrases(path: Path) -> dict[str, str]:
    """Фразы файла: id → текст; ошибка называет первую неверную запись."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise PhrasesError(f"не читается файл фраз '{path}': {e}") from e

    if not isinstance(data, dict) or not data:
        raise PhrasesError(f"в '{path}' ожидается непустой словарь id: текст")
    for phrase_id, text in data.items():
        if not isinstance(phrase_id, str) or not PHRASE_ID.fullmatch(phrase_id):
            raise PhrasesError(f"id '{phrase_id}' не подходит для ресурса Android: нужны [a-z][a-z0-9_]*")
        if not isinstance(text, str) or not text.strip():
            raise PhrasesError(f"у фразы '{phrase_id}' пустой текст")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phrases", type=Path, required=True, help="YAML с фразами id: текст")
    parser.add_argument("--out", type=Path, required=True, help="каталог для phrase_<id>.wav; создаётся, если его нет")
    parser.add_argument("--backend", default=None, help="MODEL_BACKEND; по умолчанию из окружения")
    parser.add_argument("--model", default=None, help="MODEL_NAME; по умолчанию модель бэкенда")
    parser.add_argument("--voice", default=None, help="голос; по умолчанию голос бэкенда по умолчанию")
    parser.add_argument("--sample-rate", type=int, default=24000, choices=SAMPLE_RATES,
                        help="частота WAV, Гц (по умолчанию %(default)s - как у ответов)")
    args = parser.parse_args(argv)

    overrides = {"MODEL_BACKEND": args.backend, "MODEL_NAME": args.model}
    settings = Settings(**{name: value for name, value in overrides.items() if value is not None})
    configure_logging(settings.LOG_LEVEL, settings.LOG_DIAGNOSE)

    try:
        phrases = load_phrases(args.phrases)
        pipeline = SpeechPipeline(build_backend(settings))
        voice = pipeline.voice(args.voice)
        # Сначала синтезируются все фразы: ошибка в одной не оставляет каталог наполовину обновлённым
        rendered = {}
        for phrase_id, text in phrases.items():
            sentences = pipeline.sentences(pipeline.normalize(text))
            rendered[phrase_id] = pipeline.render_all(sentences, voice, args.sample_rate, 1.0)
    except (PhrasesError, AppError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for phrase_id, pcm in rendered.items():
        target = args.out / f"phrase_{phrase_id}.wav"
        target.write_bytes(to_wav(pcm, args.sample_rate))
        print(f"{target}\t{pcm_duration_ms(pcm, args.sample_rate) / 1000:.2f} с")

    pipeline.backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
