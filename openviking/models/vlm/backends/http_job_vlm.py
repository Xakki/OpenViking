# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Text completions backed by generic JSON jobs or Convertor JSON AI jobs."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from openviking.models.http_job import HttpJobError, HttpJobMapping, HttpJobTransport, HttpJobUpload
from openviking.models.vlm.base import UnsupportedMediaInputError, VLMBase
from openviking.models.vlm.registry import DEFAULT_CONVERTOR_ORIGIN

_CONVERTOR_PARAMS = frozenset({"temperature", "top_p", "max_tokens", "stop", "response_format"})
_UNSUPPORTED_FIELDS = frozenset({"stream", "stream_options", "tools", "tool_choice", "functions", "function_call", "parallel_tool_calls", "modalities", "audio"})


def _mapping(config: dict[str, Any]) -> tuple[HttpJobMapping, bool, dict[str, Any]]:
    options = dict(config.get("http_job") or {})
    preset = options.pop("preset", None)
    timing = {
        key: options.pop(key)
        for key in ("poll_interval", "get_retries", "retry_interval", "max_response_bytes")
        if key in options
    }
    if preset == "convertor":
        if options:
            raise ValueError("Convertor preset does not accept route or result mapping overrides")
        return HttpJobMapping(
            origin=config.get("api_base") or DEFAULT_CONVERTOR_ORIGIN,
            submit_route="/api/v1/convert",
            poll_route="/api/v1/convert/{id}/status",
            result_route="/api/v1/convert/{id}/download",
            id_path=("conversion_id",),
            status_path=("status",),
            pending_statuses=frozenset({"pending", "processing"}),
            success_statuses=frozenset({"completed"}),
            failure_statuses=frozenset({"failed", "expired"}),
            multipart_field="file",
        ), True, timing
    if preset not in (None, "generic"):
        raise ValueError("Unknown VLM HTTP job preset")
    for key in ("id_path", "status_path", "result_path", "pending_statuses", "success_statuses", "failure_statuses"):
        if key in options:
            values = options[key]
            if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"HTTP job mapping {key} must be an array of strings")
            options[key] = tuple(values) if key.endswith("_path") else frozenset(values)
    if "multipart_field" in options or "origin" in options:
        raise ValueError("Generic VLM HTTP jobs use api_base and JSON submission")
    options.setdefault("id_path", ("id",))
    options.setdefault("status_path", ("status",))
    options.setdefault("result_path", ("data",))
    options.setdefault("pending_statuses", frozenset({"pending", "processing"}))
    options.setdefault("success_statuses", frozenset({"completed"}))
    options.setdefault("failure_statuses", frozenset({"failed", "expired"}))
    try:
        return HttpJobMapping(origin=config.get("api_base"), **options), False, timing
    except TypeError:
        raise ValueError("Invalid or incomplete generic VLM HTTP job mapping") from None


class HttpJobVLM(VLMBase):
    """Explicitly selected, text-only submit/poll/download VLM."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        super().__init__(config)
        self.provider = "http_job"
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("HTTP job VLM requires a model")
        if self.extra_headers:
            raise ValueError("HTTP job VLM only supports Bearer authentication via api_key")
        if "stream" in config or _UNSUPPORTED_FIELDS.intersection(self.extra_request_body):
            raise ValueError("HTTP job VLM does not support tools, media or native streaming")
        if {"messages", "model", "params"}.intersection(self.extra_request_body):
            raise ValueError("HTTP job VLM extra_request_body cannot override the request envelope")
        self._mapping, self._convertor, options = _mapping(config)
        if self._convertor:
            if not self.api_key:
                raise ValueError("Convertor VLM requires Bearer authentication via api_key")
            if self.extra_request_body.keys() - _CONVERTOR_PARAMS:
                raise ValueError("Unsupported Convertor VLM generation parameter")
        self.api_base = self._mapping.origin
        self._transport_options = dict(
            bearer_token=self.api_key,
            timeout=self.timeout,
            transport_factory=transport_factory,
            **options,
        )
        self._transport = self._new_transport()

    def _new_transport(self) -> HttpJobTransport:
        return HttpJobTransport(self._mapping, **self._transport_options)

    def _request(
        self, prompt, thinking, tools, tool_choice, messages, max_tokens=None, reasoning_effort=None
    ) -> tuple[dict[str, Any], HttpJobUpload | None]:
        if tools or tool_choice is not None:
            raise ValueError("HTTP job VLM does not support tools")
        normalized = []
        for message in messages if messages is not None else [{"role": "user", "content": prompt}]:
            if not isinstance(message, dict) or _UNSUPPORTED_FIELDS.intersection(message) or "tool_calls" in message:
                raise ValueError("HTTP job VLM only supports plain text messages, without tools")
            role, content = message.get("role"), message.get("content")
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant") or not isinstance(content, str):
                raise ValueError("HTTP job VLM only supports system, user and assistant text messages")
            normalized.append({"role": role, "content": content})
        if not any(message["role"] == "user" for message in normalized):
            raise ValueError("HTTP job VLM requires at least one user message")
        params = dict(self.extra_request_body)
        if self.temperature is not None:
            params.setdefault("temperature", self.temperature)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            params.setdefault("max_tokens", self.max_tokens)
        effort = reasoning_effort or self.config.get("reasoning_effort")
        if self._convertor:
            if (self.thinking if thinking is None else thinking) or effort:
                raise ValueError("Convertor VLM does not support thinking or reasoning effort controls")
            body: dict[str, Any] = {"messages": normalized}
            if params:
                body["params"] = params
            try:
                content = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
            except (ValueError, TypeError, RecursionError):
                raise ValueError("Invalid Convertor VLM request payload") from None
            return {
                "input": "file",
                "source_format": "json",
                "to_format": "json_ai",
                "options[model]": self.model,
            }, HttpJobUpload("messages.json", content, "application/json")
        if thinking is not None or self.thinking:
            params.setdefault("thinking", self.thinking if thinking is None else thinking)
        if effort:
            params["reasoning_effort"] = effort
        return {"model": self.model, "messages": normalized, **params}, None

    def _completion(self, result: Any) -> str:
        if self._convertor:
            try:
                message = result["choices"][0]["message"]
                if message.get("tool_calls") or message.get("function_call"):
                    raise HttpJobError("HTTP job result contains unsupported tool calls")
                result = message["content"]
            except (AttributeError, KeyError, IndexError, TypeError):
                raise HttpJobError("Convertor VLM result is missing completion content") from None
        if not isinstance(result, str):
            raise HttpJobError("HTTP job completion content is not text")
        return self._clean_response(result)

    def get_completion(self, prompt="", thinking=None, tools=None, tool_choice=None, messages=None) -> str:
        payload, upload = self._request(prompt, thinking, tools, tool_choice, messages)
        return self._completion(self._transport.run(payload, upload=upload))

    async def get_completion_async(
        self, prompt="", thinking=None, tools=None, tool_choice=None, messages=None, max_tokens=None, reasoning_effort=None
    ) -> str:
        payload, upload = self._request(prompt, thinking, tools, tool_choice, messages, max_tokens, reasoning_effort)
        # Each call owns its client; the VLM close interface is synchronous and
        # cannot safely close clients on arbitrary owner loops.
        transport = self._new_transport()
        try:
            return self._completion(await transport.arun(payload, upload=upload))
        finally:
            await transport.aclose()

    def get_vision_completion(self, *args, **kwargs) -> str:
        raise UnsupportedMediaInputError("HTTP job VLM does not support vision inputs")

    async def get_vision_completion_async(self, *args, **kwargs) -> str:
        return self.get_vision_completion(*args, **kwargs)
