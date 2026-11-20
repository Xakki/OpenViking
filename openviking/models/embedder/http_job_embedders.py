# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Text embedding through a configured HTTP job service."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from openviking.models.embedder.base import DenseEmbedderBase, EmbeddingInput, EmbedResult
from openviking.models.http_job import HttpJobError, HttpJobMapping, HttpJobTransport, HttpJobUpload


class HttpJobDenseEmbedder(DenseEmbedderBase):
    def __init__(
        self,
        model_name: str,
        *,
        mapping: HttpJobMapping,
        dimension: int,
        convertor: bool = False,
        input_key: str = "input",
        model_key: str = "model",
        extra_body: dict[str, Any] | None = None,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        poll_interval: float = 0.5,
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        super().__init__(model_name, {"provider": "http_job"})
        self._dimension = dimension
        self._convertor = convertor
        self._input_key = input_key
        self._model_key = model_key
        self._extra_body = dict(extra_body or {})
        self._new_transport = lambda: HttpJobTransport(
            mapping,
            bearer_token=bearer_token,
            timeout=timeout,
            poll_interval=poll_interval,
            transport_factory=transport_factory,
        )
        self._sync_transport = self._new_transport()

    def get_dimension(self) -> int:
        return self._dimension

    def _submission(self, texts: list[str]) -> tuple[dict[str, Any], HttpJobUpload | None]:
        if self._convertor:
            return {
                "input": "file",
                "source_format": "json",
                "to_format": "embd",
                "options[model]": self.model_name,
            }, HttpJobUpload(
                "input.json", json.dumps({"input": texts}).encode("utf-8"), "application/json"
            )
        return {**self._extra_body, self._input_key: texts, self._model_key: self.model_name}, None

    def _vectors(self, result: Any, count: int) -> list[list[float]]:
        if self._convertor:
            dimensions = result.get("dimensions") if isinstance(result, dict) else None
            if type(dimensions) is not int or dimensions != self._dimension:
                raise HttpJobError("HTTP job returned an unexpected embedding dimension")
            result = result.get("data")
        if not isinstance(result, list) or len(result) != count:
            raise HttpJobError("HTTP job returned an unexpected embedding count")
        vectors: dict[int, list[float]] = {}
        for row in result:
            index = row.get("index") if isinstance(row, dict) else None
            if type(index) is not int or not 0 <= index < count or index in vectors:
                raise HttpJobError("HTTP job returned an invalid embedding index")
            vector = row.get("embedding")
            if not isinstance(vector, list) or len(vector) != self._dimension:
                raise HttpJobError("HTTP job returned an unexpected embedding dimension")
            if any(type(value) not in (int, float) for value in vector):
                raise HttpJobError("HTTP job returned an invalid embedding value")
            try:
                values = [float(value) for value in vector]
            except OverflowError:
                raise HttpJobError("HTTP job returned an invalid embedding value") from None
            if not all(math.isfinite(value) for value in values):
                raise HttpJobError("HTTP job returned an invalid embedding value")
            vectors[index] = values
        return [vectors[index] for index in range(count)]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, str) or any(not isinstance(text, str) for text in texts):
            raise ValueError("HTTP job embedding requires text inputs")
        if not texts:
            return []
        payload, upload = self._submission(list(texts))
        return self._vectors(self._sync_transport.run(payload, upload=upload), len(texts))

    def _prepare_text(self, content: EmbeddingInput) -> str:
        text = self.prepare_embedding_input(content)
        if not isinstance(text, str):
            raise ValueError("HTTP job embedding requires text input")
        return text

    def embed(self, content: EmbeddingInput, is_query: bool = False) -> EmbedResult:
        return EmbedResult(dense_vector=self.embed_batch([self._prepare_text(content)])[0])

    async def embed_async(self, content: EmbeddingInput, is_query: bool = False) -> EmbedResult:
        payload, upload = self._submission([self._prepare_text(content)])

        async def request() -> EmbedResult:
            # Keep client ownership with this operation, including on cancellation.
            transport = self._new_transport()
            try:
                result = await transport.arun(payload, upload=upload)
                return EmbedResult(dense_vector=self._vectors(result, 1)[0])
            finally:
                await transport.aclose()

        return await self._run_once_async(request, is_query=is_query, operation_name="embedding")
