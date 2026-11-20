# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import json
from email import policy
from email.parser import BytesParser

import httpx
import pytest

from openviking.models.embedder.http_job_embedders import HttpJobDenseEmbedder
from openviking.models.http_job import HttpJobError
from openviking_cli.utils.config.embedding_config import EmbeddingConfig, HttpJobEmbeddingConfig

ORIGIN = "https://jobs.example.test"


def test_convertor_multipart_poll_download():
    calls = []

    async def handle(request):
        calls.append((request.method, request.url.path))
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.method == "POST":
            body = await request.aread()
            parts = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + body
            )
            fields = {part.get_param("name", header="content-disposition"): part for part in parts.iter_parts()}
            assert set(fields) == {"input", "source_format", "to_format", "options[model]", "file"}
            assert fields["input"].get_payload() == "file"
            assert fields["source_format"].get_payload() == "json"
            assert fields["to_format"].get_payload() == "embd"
            assert fields["options[model]"].get_payload() == "model-key"
            assert fields["file"].get_filename() == "input.json"
            file_content = fields["file"].get_payload(decode=True)
            assert isinstance(file_content, bytes)
            assert json.loads(file_content) == {"input": ["first", "second"]}
            return httpx.Response(202, json={"conversion_id": 7})
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"status": ["pending", "processing", "completed"][len(calls) - 2]})
        return httpx.Response(200, json={"dimensions": 2, "data": [
            {"index": 1, "embedding": [3, 4]}, {"index": 0, "embedding": [1, 2]},
        ]})

    embedder = HttpJobDenseEmbedder(
        "model-key", dimension=2, convertor=True, bearer_token="test-token", poll_interval=0.001,
        mapping=HttpJobEmbeddingConfig(preset="convertor").make_mapping(ORIGIN),
        transport_factory=lambda: httpx.MockTransport(handle),
    )
    assert embedder.embed_batch(["first", "second"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == [
        ("POST", "/api/v1/convert"), *[("GET", "/api/v1/convert/7/status")] * 3,
        ("GET", "/api/v1/convert/7/download"),
    ]


def test_json_config_factory_async(monkeypatch):
    calls = []

    async def handle(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            assert json.loads(await request.aread()) == {"texts": ["text"], "engine": "model-key", "flag": True}
            return httpx.Response(202, json={"job": {"key": "abc"}})
        return httpx.Response(200, json={"state": "ready", "out": {"vectors": [
            {"index": 0, "embedding": [1, 2]},
        ]}})

    from openviking.models.embedder import http_job_embedders

    transport = http_job_embedders.HttpJobTransport
    monkeypatch.setattr(
        http_job_embedders, "HttpJobTransport",
        lambda *args, **kwargs: transport(
            *args, **{**kwargs, "transport_factory": lambda: httpx.MockTransport(handle)}
        ),
    )
    config = EmbeddingConfig.model_validate({"dense": {
        "provider": "http_job", "model": "model-key", "dimension": 2, "api_base": ORIGIN,
        "extra_body": {"flag": True, "texts": ["ignored"]},
        "http_job": {
            "submit_route": "/jobs", "poll_route": "/jobs/{id}", "id_path": ["job", "key"],
            "status_path": ["state"], "success_statuses": ["ready"],
            "result_path": ["out", "vectors"], "input_key": "texts", "model_key": "engine",
        },
    }})
    embedder = config.get_embedder()
    assert isinstance(embedder, HttpJobDenseEmbedder)
    assert embedder.get_dimension() == config.get_dimension() == 2
    assert asyncio.run(embedder.embed_async("text")).dense_vector == [1.0, 2.0]
    assert calls == [("POST", "/jobs"), ("GET", "/jobs/abc")]


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"dimensions": 3, "data": []}, "dimension"),
        ({"dimensions": 2, "data": []}, "count"),
        ({"dimensions": 2, "data": [{"index": True, "embedding": [1, 2]}]}, "index"),
        ({"dimensions": 2, "data": [{"index": 1, "embedding": [1, 2]}]}, "index"),
        ({"dimensions": 2, "data": [{"index": 0, "embedding": [1]}]}, "dimension"),
        ({"dimensions": 2, "data": [{"index": 0, "embedding": [1, True]}]}, "value"),
        ({"dimensions": 2, "data": [{"index": 0, "embedding": [1, float("nan")]}]}, "value"),
    ],
)
def test_convertor_rejects_invalid_result(result, message):
    embedder = HttpJobDenseEmbedder(
        "model-key", dimension=2, convertor=True,
        mapping=HttpJobEmbeddingConfig(preset="convertor").make_mapping(ORIGIN),
    )
    with pytest.raises(HttpJobError, match=message):
        embedder._vectors(result, 1)


@pytest.mark.parametrize("status", ["failed", "expired"])
def test_convertor_terminal_failure_does_not_download(status):
    def handle(request):
        if request.url.path == "/api/v1/convert":
            return httpx.Response(202, json={"conversion_id": 7})
        assert request.url.path == "/api/v1/convert/7/status"
        return httpx.Response(200, json={"status": status})

    embedder = HttpJobDenseEmbedder(
        "model-key", dimension=2, convertor=True,
        mapping=HttpJobEmbeddingConfig(preset="convertor").make_mapping(ORIGIN),
        transport_factory=lambda: httpx.MockTransport(handle),
    )
    with pytest.raises(HttpJobError, match="Remote HTTP job failed"):
        embedder.embed("text")
