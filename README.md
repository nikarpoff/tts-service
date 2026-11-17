# tts-service

Сервис синтеза речи голосового ассистента для людей с нарушениями зрения. Принимает текст ответа и возвращает русскую речь: потоком PCM по предложениям, чтобы первое предложение зазвучало раньше, чем синтезированы остальные, или целым WAV. Частые фразы, например предупреждения пассивного режима, отдаются из кэша. Сервис не сочиняет текст: его строит Шлюз (`assistant-low-vision-platform`), он же - единственный потребитель API. Модели - Silero и Piper на CPU, F5-TTS на видеокарте; бэкенд `fake` (тон вместо речи) отвечает без весов и включён по умолчанию. Требования - в [docs/spec.md](docs/spec.md).

## Назначение

Детектор, ASR и TTS устроены одинаково (раздел SVC спецификации), поэтому Шлюз работает с любым бэкендом любого из них одинаково:

- модель выбирается настройками `MODEL_BACKEND` и `MODEL_NAME`, загружается и прогревается при старте (3 холостых вызова), `/health` отвечает только после прогрева;
- инференс идёт в пуле из `WORKERS` потоков, ещё `QUEUE_MAX` запросов могут ждать; следующий запрос сразу получает 503: устаревшее описание сцены пользователю бесполезно;
- вход читается в память, временных файлов нет;
- у каждого запроса есть идентификатор `X-Request-ID`, он попадает в каждую строку лога о запросе;
- каждый ответ, включая потоковый, несёт заголовок `Server-Timing` с временем этапов;
- `/v1/info` описывает конфигурацию (сервис, версия, коммит, бэкенд, модель, устройство), Шлюз включает её в свой `config_id`.

### Как работает синтез

1. Текст нормализуется (TTS-2): числа - словами с согласованием рода («2 двери» → «две двери», «21 стул» → «двадцать один стул»), «%», «°», «№» и сокращения («ул.», «д.», «т.е.», «т.д.») - словами. Ударение `+` перед гласной передаётся бэкенду, если он его понимает, иначе убирается. Без знака в конце текст получает точку.
2. Ключ кэша - бэкенд, голос, частота, скорость и нормализованный текст. Попадание (`X-Cache: hit`) отдаётся сразу, без модели и без места в очереди: на занятом сервисе предупреждения тоже звучат.
3. Промах занимает одно место в очереди на весь запрос. Текст режется на предложения по `.`, `!`, `?`, `…` с пробелом после; точка сокращения или инициала предложение не заканчивает.
4. Каждое предложение синтезируется и обрабатывается (TTS-4): пересэмплирование к запрошенной частоте, тишина в начале урезается до 10 мс (предел - 50 мс), пик нормализуется к -1 dBFS, края сглаживаются по 5 мс.
5. `pcm_s16le` уходит потоком: первое предложение синтезируется до начала ответа (поэтому на заполненной очереди приходит 503, а не оборванный поток), следующие - пока клиент играет предыдущие. `wav` отдаётся целым файлом.
6. Полный ответ кладётся в кэш (LRU, `CACHE_MAX_ITEMS`). При старте в кэш синтезируются фразы `PREWARM_PHRASES_PATH` - предупреждения пассивного режима Шлюза.

Фразы прогрева `configs/prewarm_phrases.txt` совпадают с контрактом `tests/contract/prewarm_phrases.txt` байт в байт (тест `test_prewarm_matches_contract`): Шлюз строит предупреждения из тех же строк, и любое расхождение дало бы промах кэша. Фраза с точкой в конце и без неё - один ключ: нормализация ставит точку, так что попадание звучит ровно как новый синтез.

## API

| Метод | Путь | |
| --- | --- | --- |
| `GET` | `/health` | `{"status": "ok"}`, когда модель загружена и прогрета |
| `GET` | `/v1/info` | конфигурация сервиса для `config_id` Шлюза |
| `GET` | `/v1/voices` | голоса загруженной модели |
| `POST` | `/v1/synthesize` | речь текста: PCM потоком по предложениям или WAV |
| `GET` | `/docs` | Swagger UI |

Запросы к `/v1/*` требуют `Authorization: Bearer <AUTH_TOKEN>`; если `AUTH_TOKEN` не задан, аутентификация выключена.

Заголовок `X-Request-ID` запроса возвращается в ответе; без него сервис создаёт свой (8 hex-символов). Значение с пробелами, управляющими символами или длиннее 128 символов заменяется новым: оно пишется в каждую строку лога.

Заголовок `Server-Timing` перечисляет этапы, известные до начала тела ответа, и `total` - время до ответа: `decode;dur=2.1, inference;dur=11.6, total;dur=15.0`.

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "X-Request-ID: 3f2c1ab0" \
  "http://localhost:8103/v1/info"
```

```json
{
  "service": "tts-service",
  "version": "0.1.0",
  "git_sha": "3f2c1ab",
  "backend": "fake",
  "model": "fake",
  "device": "cpu",
  "warmed_up": true,
  "params": {"voice": "fake_f", "sample_rate": 48000, "latency_ms": 0, "cache_max_items": 256}
}
```

| Поле | |
| --- | --- |
| `service`, `version` | имя и версия из `pyproject.toml` |
| `git_sha` | коммит, из которого собран образ (аргумент сборки `GIT_SHA`) |
| `backend`, `model` | `MODEL_BACKEND` и модель бэкенда (`MODEL_NAME` или модель по умолчанию) |
| `device` | `TARGET_PLATFORM`: `cpu` или `cuda` |
| `warmed_up` | модель прошла холостые вызовы |
| `params` | параметры, от которых зависит ответ: голос по умолчанию `voice` и его частота `sample_rate`, размер кэша `cache_max_items`; у `fake` - ещё `latency_ms` |

### Голоса

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" "http://localhost:8103/v1/voices"
```

```json
[
  {"id": "fake_f", "backend": "fake", "sample_rate": 48000, "gender": "f", "default": true},
  {"id": "fake_m", "backend": "fake", "sample_rate": 48000, "gender": "m", "default": false}
]
```

`sample_rate` - родная частота голоса; ответ пересэмплируется к запрошенной.

### Синтез

```bash
curl -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "Осторожно, впереди ступени.", "sample_rate": 24000, "format": "pcm_s16le", "speed": 1.0}' \
  --output answer.pcm "http://localhost:8103/v1/synthesize"

curl -H "Authorization: Bearer $AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"text": "Слева дверь, справа 2 стула.", "format": "wav"}' \
  --output answer.wav "http://localhost:8103/v1/synthesize"
```

| Поле | По умолчанию | |
| --- | --- | --- |
| `text` | обязательное | 1-1000 символов; `+` перед гласной - ударение |
| `voice` | голос по умолчанию | `id` из `/v1/voices` |
| `sample_rate` | `24000` | 8000, 16000, 22050, 24000 или 48000 |
| `format` | `pcm_s16le` | `pcm_s16le` - потоком по предложениям, `wav` - целым файлом |
| `speed` | `1.0` | 0.5-2.0 |

Ответ `pcm_s16le`: `Content-Type: audio/L16; rate=<частота>; channels=1`, тело chunked - моно, 16 бит со знаком, little-endian, фрагмент на предложение. Ответ `wav`: `Content-Type: audio/wav` с длиной тела. Заголовки обоих:

| Заголовок | |
| --- | --- |
| `X-Sample-Rate` | частота звука, Гц |
| `X-Audio-Format` | `pcm_s16le` или `wav` |
| `X-Cache` | `hit` - из кэша без синтеза, `miss` - синтезировано |
| `Server-Timing` | `normalize`, `synthesize` и `audio` (для потока - первого предложения) и `total` |

## Коды ответов

Ошибки приходят одним предложением на английском: `{"detail": "<причина>"}`.

| Код | Когда |
| --- | --- |
| `200` | успех |
| `400` | вход пустой или битый |
| `401` | нет токена или неверный токен: `{"detail": "Invalid token"}` |
| `413` | вход больше лимита |
| `415` | неподдерживаемый `Content-Type` |
| `422` | параметры не прошли проверку: `{"detail": "Invalid request: <что не так> at <где>"}` |
| `500` | внутренняя ошибка: `{"detail": "Internal server error"}`, подробности только в логе |
| `503` | очередь заполнена: `{"detail": "Service is busy"}` и `Retry-After: 1`; у `/health` - модель ещё не прогрета |

Коды `/v1/synthesize`:

| Код | Когда |
| --- | --- |
| `200` | речь: поток PCM или WAV |
| `401` | нет токена или неверный токен |
| `422` | пустой текст или текст без слов (`{"detail": "Text has no words to speak"}`), длиннее 1000 символов, неизвестный голос (`{"detail": "Unknown voice"}`), неверная частота, скорость, формат или лишнее поле |
| `503` | очередь заполнена, `Retry-After: 1`; попадание в кэш отдаётся и при заполненной очереди |

Ошибка модели посреди потока (после первого предложения) обрывает поток: заголовки уже отправлены, причина - в логе.

## Конфигурация

Все переменные необязательные, значения по умолчанию - в таблице и в [.env.example](.env.example).

| Переменная | По умолчанию | |
| --- | --- | --- |
| `MODEL_BACKEND` | `fake` | бэкенд из реестра (раздел «Бэкенды и модели») |
| `MODEL_NAME` | - | модель бэкенда; не задана - модель бэкенда по умолчанию |
| `TARGET_PLATFORM` | `cpu` | `cpu` или `cuda` |
| `ARTIFACTS_PATH` | - | каталог весов, в нём - подкаталог на бэкенд; не задан - `$XDG_CACHE_HOME/tts-service/` (раздел «Веса») |
| `FAKE_LATENCY_MS` | `0` | время инференса, которое изображает `fake`, мс на предложение |
| `AUTH_TOKEN` | - | Bearer-токен; не задан - аутентификация выключена |
| `WORKERS` | `1` | потоков инференса |
| `QUEUE_MAX` | `4` | запросов в ожидании потока; сверх этого - 503 |
| `CACHE_MAX_ITEMS` | `256` | фраз в кэше синтеза (LRU в памяти); `0` - кэш выключен |
| `PREWARM_PHRASES_PATH` | `configs/prewarm_phrases.txt` | фразы, синтезируемые в кэш при старте, по одной на строку; относительный путь - от корня сервиса |
| `PREWARM_SAMPLE_RATE` | `24000` | частота прогретых фраз; должна совпадать с `APP_AUDIO_OUT_SAMPLE_RATE` Шлюза, иначе предупреждения промахнутся мимо кэша |
| `GIT_SHA` | `unknown` | коммит для `/v1/info`; задаёт аргумент сборки образа, в `.env` не пишется |
| `LOG_LEVEL` | `INFO` | `TRACE`, `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_DIAGNOSE` | `False` | значения переменных в трейсбеках; только для отладки - в логи попадут данные запросов и токены |

Пороги моделей и эвристик - не переменные окружения, а константы в `app/core/config/constants.py`: предел текста, порог тишины, длина краёв, целевой пик. Кэш на 256 фраз по 3 с при 24 кГц занимает около 40 МБ.

## Бэкенды и модели

Бэкенд выбирается `MODEL_BACKEND`, модель - `MODEL_NAME`. Реестр (`app/core/backends/registry.py`) импортирует модуль только выбранного бэкенда, поэтому библиотеки настоящих моделей - необязательные extras: без них сервис устанавливается и проходит быстрые тесты. Неизвестный бэкенд, неизвестная модель или бэкенд без своего extra останавливают старт с понятным сообщением, например `Backend 'silero' requires the 'silero' extra`.

| `MODEL_BACKEND` | Extra | `MODEL_NAME` (по умолчанию - первая) | Голоса (`/v1/voices`) | Частота | Устройство |
| --- | --- | --- | --- | --- | --- |
| `silero` | `silero` | `v5_ru`; `v5_1_ru` - `v5_6_ru`, `v4_ru` | `xenia` (по умолчанию), `kseniya`, `baya` - женские; `aidar`, `eugene` - мужские | 48000 | CPU, CUDA |
| `piper` | `piper` | `ru_RU-denis-medium`; `ru_RU-dmitri-medium`, `ru_RU-irina-medium`, `ru_RU-ruslan-medium` | один голос модели: `denis`, `dmitri`, `ruslan` - мужские, `irina` - женский | 22050 | CPU; CUDA при `onnxruntime-gpu` |
| `f5tts` | `f5tts` | `F5TTS_v1_Base_v2`; `F5TTS_v1_Base`, `F5TTS_v1_Base_accent_tune` | голоса из эталонных записей `voices.yaml` (ниже) | 24000 | CUDA (на CPU секунды на предложение) |
| `fake` | - | `fake` | `fake_f` (по умолчанию), `fake_m` | 48000 | без весов |

- **Silero** - Silero TTS для русского языка (`snakers4/silero-models`): файл `torch.package` с акустической моделью, вокодером и расстановщиком ударений, ему нужен только torch. Ударения и «ё» модель ставит сама (TTS-2), явное `+` перед гласной она понимает и предпочитает своему. Скорость передаётся через SSML `<prosody rate="…%">`. Предложение без русских букв (например, «Hello») модель не читает: вместо него - пауза 100 мс, ответ продолжается.
- **Piper** - голоса VITS в ONNX (`rhasspy/piper-voices`), библиотека `piper-tts`. Текст переводится в фонемы встроенным espeak-ng, он же ставит ударения; `+` перед гласной превращается в знак ударения espeak-ng (U+0301), а не читается словом «плюс». Скорость - `length_scale` модели, делённый на `speed`.
- **F5-TTS** - модель flow matching, дообученная на русской речи (`Misha24-10/F5-TTS_RUSSIAN`), с вокодером Vocos (`charactr/vocos-mel-24khz`). Голос клонируется с эталонной записи, поэтому до старта в каталог весов кладётся `f5tts/voices/voices.yaml` (первый голос - по умолчанию) с записями рядом:

  ```yaml
  anna:
    gender: f                 # f или m
    audio: anna.wav           # WAV 5-12 с чистой речи, моно
    text: "Текст записи с +ударениями."
  ```

  Записи читаются один раз при старте. Модель обучена на тексте с ударениями `+` и без них читает хуже; сервис передаёт ударения запроса как есть и своих не ставит. 16 шагов решателя, фиксированное зерно: одна и та же фраза звучит одинаково. Проверяется на машине моделей с видеокартой.

На `fake` сервис пишет предупреждение при старте, а `/v1/info` показывает `backend: fake`, поэтому подмена модели в замерах сразу видна.

### Веса

Веса не входят в образ и в репозиторий. Они лежат в `ARTIFACTS_PATH/<бэкенд>/` (в развёртывании - именованный том), а без `ARTIFACTS_PATH` - в `$XDG_CACHE_HOME/tts-service/<бэкенд>/` (по умолчанию `~/.cache/tts-service/`). Недостающий файл скачивается при старте один раз; при скачанных файлах сеть не нужна. Недокачанный файл не принимается за веса.

| Бэкенд | Файлы | Размер |
| --- | --- | --- |
| `silero` | `<модель>.pt` с `models.silero.ai` | 145 МБ (`v5_ru`), 40 МБ (`v4_ru`) |
| `piper` | `<модель>.onnx` и `<модель>.onnx.json` с Hugging Face `rhasspy/piper-voices` | 63 МБ |
| `f5tts` | чекпойнт и `vocab.txt` из `Misha24-10/F5-TTS_RUSSIAN`, `vocoder/` из `charactr/vocos-mel-24khz` | 1,4 ГБ и 54 МБ |

Скачать веса заранее, для старта без сети (DEP-6; в развёртывании его вызывает `deployments/fetch-weights.sh` в образе сервиса):

```bash
MODEL_BACKEND=piper ARTIFACTS_PATH=./model uv run python -m scripts.fetch_weights
python -m scripts.fetch_weights --backend silero --model v5_ru   # в контейнере
```

Скрипт печатает `<бэкенд>/<модель>\t<каталог весов>` и модель в память не загружает.

Память процесса на CPU: Silero `v5_ru` - около 730 МБ, Piper - около 290 МБ.

### Лицензии

Проверено по первоисточникам 2026-11-17: файлы `LICENSE` репозиториев, карточки моделей на Hugging Face, метаданные PyPI.

| Модель | Лицензия | Коммерческое использование |
| --- | --- | --- |
| `fake` (код сервиса, без весов) | Apache-2.0 | да |
| Silero TTS `v4_ru`, `v5_ru` - `v5_6_ru` (веса, `snakers4/silero-models`) | CC BY-NC-SA 4.0 | нет |
| torch (библиотека бэкенда `silero`) | BSD-3-Clause | да |
| `piper-tts` 1.8 (библиотека) и встроенный espeak-ng | GPL-3.0-or-later | да, но образ с extra `piper` распространяется на условиях GPL |
| onnxruntime (библиотека бэкенда `piper`) | MIT | да |
| Piper `ru_RU-denis-medium` (репозиторий `rhasspy/piper-voices` - MIT; данные - CC0, базовая модель `rdh` - CC0) | MIT | да |
| Piper `ru_RU-dmitri-medium` (данные - CC0, базовая модель - Hi-Fi TTS, CC BY 4.0) | MIT | да, с указанием Hi-Fi TTS |
| Piper `ru_RU-irina-medium` (данные RHVoice, лицензия в карточке не указана) | не определена | нет, пока не выяснена |
| Piper `ru_RU-ruslan-medium` (корпус RUSLAN) | CC BY-NC-SA 4.0 | нет |
| `f5-tts` (библиотека) | MIT | да |
| F5-TTS_RUSSIAN `F5TTS_v1_Base*` (веса) | CC BY-NC 4.0 | нет |
| Vocos `charactr/vocos-mel-24khz` (вокодер F5-TTS) | MIT | да |

Модели с некоммерческой лицензией и GPL-библиотека подключаются только своими extras (NFR-13). Синтезированный звук моделей с CC BY-NC и CC BY-NC-SA можно использовать в исследовании, но не распространять в составе приложения под Apache-2.0; поэтому системные фразы приложения синтезированы голосом Piper `ru_RU-denis-medium` (раздел «Фразы для приложения»). Эталонные записи голосов F5-TTS приносит развёртывание; на них нужно согласие диктора.

## Запуск

Порт на хосте - `127.0.0.1:8103`, сеть Docker `alv-net` compose создаёт сам. Каталог `./model` монтируется в `/opt/model` под веса.

На CPU:

```bash
cp .env.example .env
docker compose -f compose-cpu.yml up -d --build
curl http://localhost:8103/health
```

На GPU (нужен NVIDIA Container Toolkit; в `.env` - `TARGET_PLATFORM=cuda`):

```bash
docker compose up -d --build
```

Аргументы сборки берутся из окружения: `GIT_SHA=$(git rev-parse --short HEAD)` попадает в `/v1/info`, `EXTRAS` задаёт extras образа: по умолчанию в `Dockerfile.cpu` - `cpu silero piper`, в `Dockerfile` - `gpu silero piper f5tts`; `EXTRAS=""` - лёгкий образ только с `fake`. Веса в образ не входят: каталог `./model` монтируется в `/opt/model`, для него в `.env` задаётся `ARTIFACTS_PATH=/opt/model`.

Локально:

```bash
uv sync                                          # fake; с моделями: uv sync --extra cpu --extra silero --extra piper
uv run fastapi dev app/main.py
uv run pytest -q -m "not model"
uv run pytest tests/test_model_smoke.py -q -m model
```

Тесты с маркером `model` скачивают веса и в быстрый прогон не входят. `tests/test_model_smoke.py` синтезирует текст голосом Piper `ru_RU-denis-medium` тем же путём, что и API, и проверяет длительность от 40 до 150 мс на символ, громкость, пик и тишину в начале, ускорение от `speed` и чтение ударения `+`; `piper-tts` для него стоит в группе `dev`, поэтому extras не нужны (в образы группа `dev` не ставится). Если установлен extra `silero`, те же проверки проходит и Silero. CI (GitHub Actions) прогоняет быстрые тесты на каждый пуш и PR, а на пуш в `dev`, `main` и тег `v*` публикует образы в `ghcr.io/nikarpoff/tts-service` с тегами `latest` (ветка `main`), `sha-<7>` и `v<semver>` и суффиксом варианта: `latest-cpu`, `v0.1.0-cuda`. На `dev` собирается только `-cpu`.

### Фразы для приложения

Системные фразы приложения (MOB-11) - WAV из файла фраз вида `id: текст`. Скрипт пишет `phrase_<id>.wav` (моно, 16 бит) в каталог назначения тем же путём, что и API, и ничего не коммитит: файлы забирает репозиторий приложения.

```bash
uv run scripts/render_phrases.py \
  --phrases ../assistant-low-vision-mobile/phrases/phrases.yaml \
  --out ../assistant-low-vision-mobile/app/src/main/res/raw \
  --backend fake [--model <модель>] [--voice <голос>] [--sample-rate 24000]
```

`id` - имя ресурса Android: `[a-z][a-z0-9_]*`. Неверный файл, `id`, голос или бэкенд - код выхода 1 без записи файлов.

### Устройство

```
app/
├── main.py                 # настройки -> логирование -> FastAPI(lifespan) -> /health -> middleware -> роутеры
├── api_v1/
│   ├── views.py            # /v1/info, /v1/voices, /v1/synthesize; lifespan роутера прогревает кэш
│   ├── schemas.py          # контракт API (pydantic)
│   └── errors.py           # AppError, 422 и 500 -> {"detail": ...}
├── core/
│   ├── config/             # Settings (общие), ServiceSettings (синтез), версия из pyproject.toml, константы
│   ├── auth/               # Bearer-токен
│   ├── backends/           # реестр, интерфейс синтеза (voices, synthesize, fetch), веса; fake, silero, piper, f5tts
│   ├── text/               # нормализация (TTS-2) и предложения
│   ├── audio/              # пересэмплирование, тишина, пик, края (TTS-4); WAV
│   ├── cache.py            # LRU синтезированных фраз (TTS-5)
│   ├── pipeline.py         # блокирующие шаги: текст -> предложения -> PCM
│   ├── synthesizer.py      # кэш, поток по предложениям, WAV поверх движка
│   ├── engine.py           # бэкенд в пуле WORKERS, очередь QUEUE_MAX, прогрев
│   ├── entities.py         # Voice, AudioFormat, SampleRate
│   └── exceptions.py       # доменные ошибки со status_code
└── utils/                  # lifespan, логирование, middleware X-Request-ID и Server-Timing
configs/prewarm_phrases.txt # фразы прогрева кэша = контракт prewarm_phrases.txt
bench/latency.py            # замер синтеза в процессе
scripts/render_phrases.py   # фразы приложения в WAV
scripts/fetch_weights.py    # веса текущей модели заранее
tests/
├── contract/               # эталонные примеры из Шлюза; правятся только синхронизацией
└── test_*.py
```

## Замеры

Время этапов, известных до начала звука, - в заголовке `Server-Timing` каждого ответа, включая потоковый.

Замер синтеза в процессе, без HTTP и без кэша (SVC-10, TTS-9) - тем же путём, что и API:

```bash
uv run python -m bench.latency --backend fake --n 5 --warmup 1
uv run python -m bench.latency --backend silero --model v4_ru --device cpu --n 50 --warmup 5 [--text "..."] [--sample-rate 24000]
```

Скрипт печатает в stdout JSON общей схемы EXP-2; `p50_ms`, `p95_ms`, `mean_ms`, `std_ms` - время до первого фрагмента (первого готового предложения), перцентили - nearest-rank, отклонение - выборочное. Дополнительно: `total_p50_ms`, `total_p95_ms` - полное время синтеза текста, `rtf_mean`, `rtf_p95` - время синтеза к длительности звука, `sample_rate`, `text_chars`. Текст по умолчанию - одно предложение до 80 символов, на нём задан порог TTS-7.

```json
{"service": "tts-service", "backend": "fake", "model": "fake", "device": "cpu", "git_sha": "e2319f1", "n": 5, "warmup": 1, "p50_ms": 32.7, "p95_ms": 45.7, "mean_ms": 35.2, "std_ms": 9.0, "hardware": {"cpu": "Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz", "gpu": null}, "started_at": "2026-11-02T22:49:55Z", "total_p50_ms": 32.7, "total_p95_ms": 45.7, "rtf_mean": 0.0115, "rtf_p95": 0.0149, "sample_rate": 24000, "text_chars": 51}
```

Сами замеры, сравнение моделей и тест слушателей - задача `002-hypothesis-experiment`.
