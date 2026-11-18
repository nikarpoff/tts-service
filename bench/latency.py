"""
Замер синтеза в процессе, без HTTP (SVC-10, TTS-9).

Прогрев, затем N повторов синтеза фиксированного текста тем же путём, что и в API:
нормализация, предложения, бэкенд, обработка звука. Кэш не участвует: замеряется сам
синтез. Задержка - время до первого фрагмента (первого готового предложения), к ней
добавлены RTF (время синтеза / длительность звука) и полное время. Результат - JSON
общей схемы EXP-2 в stdout, логи - в stderr.

    uv run python -m bench.latency --backend fake --n 5 --warmup 1
    uv run python -m bench.latency --backend silero --model v4_ru --n 50 --warmup 5 --device cpu
"""

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.core.audio import pcm_duration_ms
from app.core.backends import build_backend
from app.core.config import get_project, Settings
from app.core.exceptions import AppError
from app.core.pipeline import SpeechPipeline
from app.utils.logging import configure_logging

ROOT = Path(__file__).resolve().parents[1]
# Одно предложение до 80 символов: на нём задан порог TTS-7
DEFAULT_TEXT = "Впереди стол и два стула, слева дверь, справа окно."


class Hardware(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu: str
    gpu: str | None


class BenchResult(BaseModel):
    """Общая схема результата замера сервиса (EXP-2)."""
    model_config = ConfigDict(extra="forbid")

    service: str
    backend: str
    model: str
    device: str
    git_sha: str
    n: int
    warmup: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    hardware: Hardware
    started_at: str


class SynthesisBenchResult(BenchResult):
    """EXP-2 с полями синтеза: задержка в общих полях - время до первого фрагмента."""
    total_p50_ms: float = Field(description="Полное время синтеза текста, p50")
    total_p95_ms: float = Field(description="Полное время синтеза текста, p95")
    rtf_mean: float = Field(description="Время синтеза / длительность звука, среднее")
    rtf_p95: float = Field(description="То же, p95")
    sample_rate: int
    text_chars: int


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank: x[ceil(q·n)-1] по отсортированной выборке - одно правило для всех отчётов."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def std(values: list[float]) -> float:
    """Выборочное стандартное отклонение (n-1); для одного замера - 0."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def git_sha(settings: Settings) -> str:
    """Коммит из GIT_SHA образа, иначе из рабочей копии."""
    if settings.GIT_SHA != "unknown":
        return settings.GIT_SHA
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def hardware(device: str) -> Hardware:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    gpu = None
    if device == "cuda":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
            )
            gpu = result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
        except (OSError, subprocess.SubprocessError):
            gpu = None
    return Hardware(cpu=cpu, gpu=gpu)


def measure(pipeline: SpeechPipeline, text: str, sample_rate: int) -> tuple[float, float, float]:
    """Один синтез: (до первого фрагмента, полное время, длительность звука) в мс."""
    voice = pipeline.voice(None)
    started = time.perf_counter()
    first_ms = None
    audio_ms = 0.0
    for sentence in pipeline.sentences(pipeline.normalize(text)):
        pcm = pipeline.render(sentence, voice, sample_rate, 1.0)
        if first_ms is None:
            first_ms = (time.perf_counter() - started) * 1000
        audio_ms += pcm_duration_ms(pcm, sample_rate)
    return first_ms or 0.0, (time.perf_counter() - started) * 1000, audio_ms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=None, help="MODEL_BACKEND; по умолчанию из окружения")
    parser.add_argument("--model", default=None, help="MODEL_NAME; по умолчанию модель бэкенда")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="TARGET_PLATFORM; по умолчанию из окружения")
    parser.add_argument("--n", type=int, default=50, help="число замеров (по умолчанию %(default)s)")
    parser.add_argument("--warmup", type=int, default=5, help="холостых синтезов до замеров (по умолчанию %(default)s)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="текст синтеза (по умолчанию «%(default)s»)")
    parser.add_argument("--sample-rate", type=int, default=24000, choices=[8000, 16000, 22050, 24000, 48000],
                        help="частота ответа, Гц (по умолчанию %(default)s)")
    args = parser.parse_args(argv)
    if args.n < 1 or args.warmup < 0:
        parser.error("--n должен быть не меньше 1, --warmup - не меньше 0")

    # Явные флаги важнее окружения, остальное (ARTIFACTS_PATH, FAKE_LATENCY_MS) - из него
    overrides = {"MODEL_BACKEND": args.backend, "MODEL_NAME": args.model, "TARGET_PLATFORM": args.device}
    settings = Settings(**{name: value for name, value in overrides.items() if value is not None})
    configure_logging(settings.LOG_LEVEL, settings.LOG_DIAGNOSE)

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        backend = build_backend(settings)
        pipeline = SpeechPipeline(backend)
        for _ in range(args.warmup):
            measure(pipeline, args.text, args.sample_rate)
        runs = [measure(pipeline, args.text, args.sample_rate) for _ in range(args.n)]
    except AppError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    first = [run[0] for run in runs]
    total = [run[1] for run in runs]
    rtf = [run[1] / run[2] if run[2] else 0.0 for run in runs]

    result = SynthesisBenchResult(
        service=get_project().name,
        backend=backend.name,
        model=backend.model,
        device=backend.device,
        git_sha=git_sha(settings),
        n=args.n,
        warmup=args.warmup,
        p50_ms=round(percentile(first, 0.5), 1),
        p95_ms=round(percentile(first, 0.95), 1),
        mean_ms=round(statistics.fmean(first), 1),
        std_ms=round(std(first), 1),
        hardware=hardware(backend.device),
        started_at=started_at,
        total_p50_ms=round(percentile(total, 0.5), 1),
        total_p95_ms=round(percentile(total, 0.95), 1),
        rtf_mean=round(statistics.fmean(rtf), 4),
        rtf_p95=round(percentile(rtf, 0.95), 4),
        sample_rate=args.sample_rate,
        text_chars=len(args.text),
    )
    print(json.dumps(result.model_dump(), ensure_ascii=False))
    backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
