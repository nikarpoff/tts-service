# Model and heuristic thresholds live here, not in env: they are tuned together with the code


class ENGINE_SETTINGS:
    """Inference engine shared by the services."""
    WARMUP_CALLS = 3                    # idle calls at startup: kernels, allocators and caches are ready before the first request


class SYNTHESIS_SETTINGS:
    """Limits of /v1/synthesize (TTS-1)."""
    MAX_TEXT_CHARS = 1000               # a longer text gets 422
    MIN_SPEED = 0.5
    MAX_SPEED = 2.0
    DEFAULT_SAMPLE_RATE = 24000         # the gateway default, APP_AUDIO_OUT_SAMPLE_RATE
    WARMUP_TEXT = "Впереди стол и два стула, слева дверь."


class AUDIO_SETTINGS:
    """Sound of the answer (TTS-4)."""
    PEAK_DBFS = -1.0                    # peak normalization target: no clipping after the client resamples
    FADE_MS = 5.0                       # fade in and out: no clicks at the sentence joints
    KEEP_LEADING_SILENCE_MS = 10.0      # lead kept before the first sound; the limit is 50 ms, it adds to the latency directly
    SILENCE_DB = -40.0                  # quieter than the peak by this much counts as silence
    MIN_PEAK = 1e-4                     # a quieter signal is silence and is not amplified
