"""
Reference examples of the contract (SVC-12) pass the API schemas; the gateway holds the same files.
"""

import json

from pathlib import Path
from typing import List

import pytest

from pydantic import TypeAdapter

from app.api_v1.schemas import ErrorResponse, InfoResponse, SynthesizeRequest, VoiceInfo
from bench.latency import BenchResult, main as bench_main, SynthesisBenchResult

CONTRACT = Path(__file__).resolve().parent / "contract"

SCHEMAS = {
    "synthesize.request.json": TypeAdapter(SynthesizeRequest),
    "voices.response.json": TypeAdapter(List[VoiceInfo]),
    "info.response.json": TypeAdapter(InfoResponse),
    "error.response.json": TypeAdapter(ErrorResponse),
    "bench.result.json": TypeAdapter(BenchResult),
}


def test_every_example_has_schema():
    examples = {path.name for path in CONTRACT.glob("*.json")}
    assert examples == SCHEMAS.keys()


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_example_matches_schema(name):
    example = json.loads((CONTRACT / name).read_text(encoding="utf-8"))
    validated = SCHEMAS[name].validate_python(example)

    # Nothing of the example is lost or renamed on the way through the schema
    assert SCHEMAS[name].dump_python(validated, mode="json") == example


def test_schemas_forbid_unknown_fields():
    example = json.loads((CONTRACT / "synthesize.request.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError):
        SynthesizeRequest.model_validate({**example, "pitch": 1.2})


def test_bench_output_matches_schema(capsys):
    assert bench_main(["--backend", "fake", "--n", "3", "--warmup", "1"]) == 0
    result = SynthesisBenchResult.model_validate_json(capsys.readouterr().out)

    # The common EXP-2 keys of the example, then the synthesis ones
    example = json.loads((CONTRACT / "bench.result.json").read_text(encoding="utf-8"))
    assert example.keys() <= result.model_dump().keys()
    assert (result.service, result.backend, result.n, result.warmup) == ("tts-service", "fake", 3, 1)
    assert result.p50_ms <= result.p95_ms
    assert 0 < result.rtf_mean < 1
