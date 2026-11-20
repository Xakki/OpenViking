# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import json
from email import policy
from email.parser import BytesParser

import httpx
import pytest

from openviking.models.http_job import HttpJobError
from openviking.models.vlm.backends import http_job_vlm
from openviking.models.vlm.backends.openai_vlm import OpenAIVLM
from openviking.models.vlm.base import UnsupportedMediaInputError
from openviking.models.vlm.retry_vlm import RetryVLM
from openviking.server.gateway_vlm import vlm_upstreams
from openviking_cli.utils.config.vlm_config import VLMConfig

ORIGIN = "https://jobs.example.test"


def mock_transport(monkeypatch, handler):
    transport = http_job_vlm.HttpJobTransport
    monkeypatch.setattr(
        http_job_vlm,
        "HttpJobTransport",
        lambda *args, **kwargs: transport(
            *args, **{**kwargs, "transport_factory": lambda: httpx.MockTransport(handler)}
        ),
    )


def test_generic_job_vlm_without_credentials(monkeypatch):
    async def handle(request):
        assert "authorization" not in request.headers
        if request.method == "POST":
            assert request.url.path == "/jobs"
            assert json.loads(await request.aread()) == {
                "model": "example", "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.0, "max_tokens": 9, "thinking": False,
            }
            return httpx.Response(202, json={"id": "a"})
        assert request.url.path == "/jobs/a"
        return httpx.Response(200, json={"status": "completed", "data": "done"})

    mock_transport(monkeypatch, handle)
    cfg = VLMConfig.model_validate({
        "provider": "http_job", "api_base": ORIGIN, "model": "example", "temperature": 0.0,
        "http_job": {"preset": "generic", "submit_route": "/jobs", "poll_route": "/jobs/{id}"},
    })
    assert cfg.is_available()
    assert asyncio.run(cfg.get_completion_async("hello", max_tokens=9)) == "done"
    assert vlm_upstreams(cfg)["skipped"] == [{"id": "vlm-default", "reason": "http_job"}]


def test_convertor_job_vlm_sends_multipart_then_downloads(monkeypatch):
    calls = []

    async def handle(request):
        calls.append((request.method, request.url.path))
        assert request.headers["authorization"] == "Bearer test-token"
        if request.method == "POST":
            body = await request.aread()
            parts = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + body
            )
            fields = {part.get_param("name", header="content-disposition"): part for part in parts.iter_parts()}
            assert {key: part.get_payload() for key, part in fields.items() if key != "file"} == {
                "input": "file", "source_format": "json", "to_format": "json_ai", "options[model]": "gonka/fast",
            }
            content = fields["file"].get_payload(decode=True)
            assert isinstance(content, bytes)
            assert json.loads(content) == {
                "messages": [{"role": "user", "content": "hello"}], "params": {"temperature": 0.0},
            }
            return httpx.Response(202, json={"conversion_id": 7})
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    mock_transport(monkeypatch, handle)
    cfg = VLMConfig(provider="http_job", model="gonka/fast", api_key="test-token", temperature=0.0,
                    http_job={"preset": "convertor"})
    assert cfg.get_vlm_instance().api_base == "https://convertor.xakki.pro"
    assert cfg.get_completion("hello") == "done"
    assert calls == [("POST", "/api/v1/convert"), ("GET", "/api/v1/convert/7/status"),
                     ("GET", "/api/v1/convert/7/download")]


def test_job_vlm_never_retries_submit_or_sends_unsupported_inputs(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(503)

    mock_transport(monkeypatch, handle)
    cfg = VLMConfig(provider="http_job", model="gonka/fast", api_key="test-token",
                    http_job={"preset": "convertor"}, max_retries=3)
    backend = cfg.get_vlm_instance()
    with pytest.raises(ValueError, match="does not support tools"):
        backend.get_completion("hello", tools=[{"type": "function"}])
    with pytest.raises(UnsupportedMediaInputError):
        backend.get_vision_completion()
    assert calls == []
    with pytest.raises(HttpJobError):
        backend.get_completion("hello")
    assert calls == ["POST"]


def test_job_vlm_credential_inherits_top_level_settings():
    cfg = VLMConfig.model_validate({"model": "gonka/fast", "http_job": {"preset": "convertor"},
                                    "api_key": "test-token", "credentials": [{"provider": "http_job"}]})
    assert isinstance(cfg.get_vlm_instance(), http_job_vlm.HttpJobVLM)


def test_default_vlm_backend_still_uses_openai_retry_wrapper():
    cfg = VLMConfig(model="example", api_key="test-token", api_base=ORIGIN)
    backend = cfg.get_vlm_instance()
    assert isinstance(backend, RetryVLM)
    assert isinstance(backend.inner, OpenAIVLM)
