from contextlib import asynccontextmanager
from typing import AsyncIterator, List

from fastapi import (
    APIRouter,
    FastAPI,
    Request,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from app.core.auth import verify_token
from app.core.config import get_project, get_service_settings
from app.core.engine import InferenceEngine
from app.core.synthesizer import SpeechSynthesizer
from app.api_v1.schemas import ErrorResponse, InfoResponse, SynthesizeRequest, VoiceInfo


@asynccontextmanager
async def synthesis_lifespan(app: FastAPI) -> AsyncIterator[None]:
    # FastAPI runs a router lifespan inside the app one: the engine is warmed up by now,
    # and the service takes connections only after the cache is prewarmed too
    synthesizer = SpeechSynthesizer(app.state.engine, get_service_settings())
    try:
        await synthesizer.startup()
    except OSError as e:
        logger.error("Bad configuration: cannot read the prewarm phrases: {}", e)
        raise SystemExit(1) from None
    app.state.synthesizer = synthesizer
    yield


router = APIRouter(prefix="/v1", dependencies=[Depends(verify_token)], lifespan=synthesis_lifespan)

UNAUTHORIZED = {status.HTTP_401_UNAUTHORIZED: {"model": ErrorResponse, "description": "No token or a wrong one"}}


def get_engine(request: Request) -> InferenceEngine:
    # Prefer app.state singleton created on startup
    svc = getattr(request.app.state, "engine", None)
    if isinstance(svc, InferenceEngine):
        return svc
    raise HTTPException(500, "service is not initialized")


def get_synthesizer(request: Request) -> SpeechSynthesizer:
    svc = getattr(request.app.state, "synthesizer", None)
    if isinstance(svc, SpeechSynthesizer):
        return svc
    raise HTTPException(500, "service is not initialized")


@router.get("/info", responses=UNAUTHORIZED)
async def info(
    engine: InferenceEngine = Depends(get_engine),
    synthesizer: SpeechSynthesizer = Depends(get_synthesizer),
) -> InfoResponse:
    """Service configuration for the gateway config_id"""
    project = get_project()
    backend = engine.backend
    return InfoResponse(
        service=project.name,
        version=project.version,
        git_sha=engine.settings.GIT_SHA,
        backend=backend.name,
        model=backend.model,
        device=backend.device,
        warmed_up=engine.warmed_up,
        params={**backend.params(), "cache_max_items": synthesizer.cache.max_items},
    )


@router.get("/voices", responses=UNAUTHORIZED)
async def voices(synthesizer: SpeechSynthesizer = Depends(get_synthesizer)) -> List[VoiceInfo]:
    """Voices of the loaded model"""
    default = synthesizer.default_voice
    return [
        VoiceInfo(
            id=voice.id,
            backend=synthesizer.pipeline.backend.name,
            sample_rate=voice.sample_rate,
            gender=voice.gender,
            default=voice == default,
        )
        for voice in synthesizer.voices()
    ]


@router.post("/synthesize",
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "content": {"audio/L16": {}, "audio/wav": {}},
            "description": "PCM streamed by sentences or a whole WAV file; X-Sample-Rate, X-Audio-Format and X-Cache headers",
        },
        **UNAUTHORIZED,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"model": ErrorResponse, "description": "Empty or too long text, unknown voice, wrong parameters"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "The queue is full; Retry-After: 1"},
    },
)
async def synthesize(
    request: SynthesizeRequest,
    synthesizer: SpeechSynthesizer = Depends(get_synthesizer),
) -> Response:
    """Speech of the text"""
    speech = await synthesizer.synthesize(
        request.text, request.voice, request.sample_rate, request.speed, request.format
    )
    if speech.chunks is not None:
        return StreamingResponse(speech.chunks, media_type=speech.media_type, headers=speech.headers)
    return Response(speech.body, media_type=speech.media_type, headers=speech.headers)
