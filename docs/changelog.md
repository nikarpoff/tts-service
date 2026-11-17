# Changelog

### v0.1.0
- Добавлены копии эталонных контрактов из репозитория Шлюза в `tests/contract/`.
- Добавлена выдержка из общей спецификации в `docs/spec.md`.
- Добавлен каркас сервиса по `ainergy-python-style` (`--ml`): FastAPI, uv, Python 3.12 (`requires-python = ">=3.12,<3.13"`, `.python-version`), профили `cpu` и `gpu` для torch.
- Добавлен общий слой сервисов ИИ (SVC): Bearer-аутентификация, ошибки `{"detail": ...}`, заголовки `X-Request-ID` и `Server-Timing` на чистых ASGI-middleware, `request_id` в каждой строке лога, движок с пулом `WORKERS` и очередью `QUEUE_MAX` (сверх неё - 503 с `Retry-After: 1`), реестр бэкендов с ленивым импортом и бэкендом `fake` (`FAKE_LATENCY_MS`), прогрев тремя холостыми вызовами, `/health` и `/v1/info`.
- Добавлены образы `Dockerfile` (CUDA) и `Dockerfile.cpu` с аргументами `EXTRAS` и `GIT_SHA`, `compose.yml` и `compose-cpu.yml` на `127.0.0.1:8103` в сети `alv-net`, `.env.example`.
- Добавлен CI в GitHub Actions: быстрые тесты и публикация образов в GHCR.
- README переписан на русском по разделам SVC-11.
- Добавлен синтез `POST /v1/synthesize` (TTS-1): `pcm_s16le` потоком по предложениям (первое синтезируется до начала ответа), `wav` целым файлом, заголовки `X-Sample-Rate`, `X-Audio-Format`, `X-Cache`; 422 на пустой, слишком длинный текст, неизвестный голос и неверные параметры; 503 на заполненной очереди.
- Добавлен `GET /v1/voices` (TTS-6); в `params` `/v1/info` - голос по умолчанию, его частота и `cache_max_items`.
- Добавлена нормализация текста (TTS-2): числа словами с согласованием рода, «%», «°», «№», сокращения, ударение `+` для бэкендов с его поддержкой, точка в конце текста без знака.
- Добавлена обработка звука (TTS-4): пересэмплирование, тишина в начале до 10 мс, пик -1 dBFS, края по 5 мс.
- Добавлен LRU-кэш фраз (`CACHE_MAX_ITEMS`) с прогревом при старте из `PREWARM_PHRASES_PATH` на частоте `PREWARM_SAMPLE_RATE`; `configs/prewarm_phrases.txt` совпадает с контрактом. Попадание отдаётся и при заполненной очереди.
- Бэкенд `fake` синтезирует тон 60 мс на символ с задержкой `FAKE_LATENCY_MS` на предложение, голоса `fake_f` и `fake_m`.
- Добавлены `scripts/render_phrases.py` (WAV системных фраз приложения, TTS-8) и `bench/latency.py` (время до первого фрагмента и RTF в JSON EXP-2).
- Добавлены зависимости `scipy` и `pyyaml`.
- Добавлены бэкенды `silero` (Silero TTS `v5_ru` и другие русские модели, голоса `xenia`, `kseniya`, `baya`, `aidar`, `eugene`, автоударение, скорость через SSML), `piper` (голоса `ru_RU-denis-medium`, `ru_RU-dmitri-medium`, `ru_RU-irina-medium`, `ru_RU-ruslan-medium`, фонемы и ударения espeak-ng, `+` - знак ударения) и `f5tts` (F5-TTS_RUSSIAN с вокодером Vocos, голоса из эталонных записей `voices.yaml`, для видеокарты) - каждый своим extra с ленивым импортом. Предложение, которое модель не может прочитать, становится паузой 100 мс.
- Веса моделей скачиваются при первом старте в `ARTIFACTS_PATH/<бэкенд>/` или в `~/.cache/tts-service/<бэкенд>/` и больше не качаются; `scripts/fetch_weights.py` скачивает веса текущей модели заранее, не загружая её (DEP-6).
- Образы ставят бэкенды по умолчанию: `Dockerfile.cpu` - `cpu silero piper`, `Dockerfile` - `gpu silero piper f5tts`. torchaudio (зависимость F5-TTS) берётся из индекса PyTorch своего профиля.
- Добавлен модельный тест `tests/test_model_smoke.py` (маркер `model`): Piper, а при extra `silero` и Silero, синтезирует речь от 40 до 150 мс на символ с нормализованным пиком, ускоряется от `speed` и читает ударение `+`; `piper-tts` для него - в группе `dev`.
- В README - таблицы бэкендов, весов и лицензий моделей и библиотек; системные фразы приложения перезаписаны голосом Piper `ru_RU-denis-medium` (MIT, данные CC0), так как веса Silero и F5-TTS_RUSSIAN некоммерческие.
- Зависимость `huggingface-hub` понижена до `>=1.0.0`: её требует `f5-tts` через `cached-path`.
- В README уточнено, где ставится GPL-библиотека `piper-tts`: в образ - только через extra `piper`, в группе `dev` - для теста модели, в образы группа `dev` не попадает.
