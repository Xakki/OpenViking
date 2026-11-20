# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Configured, same-origin HTTP job submission and polling."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from openviking.utils.async_client_cache import LoopScopedAsyncClientCache

__all__ = [
    "HttpJobError",
    "HttpJobMapping",
    "HttpJobTimeout",
    "HttpJobTransport",
    "HttpJobUpload",
]

JsonPath = tuple[str, ...]
_ROUTE = re.compile(r"/[A-Za-z0-9_./~-]*(?:\{id\}[A-Za-z0-9_./~-]*)?")
_TOKEN = re.compile(r"[A-Za-z0-9._~-]+")
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_RETRY_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class HttpJobError(RuntimeError):
    """A sanitized transport, response contract, or remote job failure."""


class HttpJobTimeout(HttpJobError):
    """The total job deadline elapsed; remote work may still be running."""


def _validate_route(route: str, *, job_id: bool) -> None:
    if (
        not isinstance(route, str)
        or not _ROUTE.fullmatch(route)
        or "//" in route
        or any(part in {".", ".."} for part in route.split("/"))
        or ("{id}" in route) != job_id
    ):
        raise ValueError("Invalid HTTP job route")


def _origin(value: str) -> httpx.URL:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"https?://[^/\\?#@\s]+/?", value)
        or any(ord(char) < 33 or ord(char) >= 127 for char in value)
        or "%" in value
    ):
        raise ValueError("Expected an HTTP(S) origin without credentials, path or query")
    try:
        url = httpx.URL(value)
    except (httpx.InvalidURL, ValueError):
        raise ValueError("Invalid HTTP job origin") from None
    if not url.host or url.port == 0 or (url.port is not None and url.port > 65535):
        raise ValueError("Invalid HTTP job origin")
    return url


@dataclass(frozen=True)
class HttpJobMapping:
    """Routes and JSON object paths; an empty result path selects the entire JSON body.

    Omit result_route to extract from the successful poll response. Set multipart_field
    to upload one file with optional form fields; otherwise submit a JSON object.
    """

    origin: str
    submit_route: str
    poll_route: str
    id_path: JsonPath
    status_path: JsonPath
    pending_statuses: frozenset[str]
    success_statuses: frozenset[str]
    failure_statuses: frozenset[str]
    result_path: JsonPath = ()
    result_route: str | None = None
    multipart_field: str | None = None

    def __post_init__(self) -> None:
        _origin(self.origin)
        _validate_route(self.submit_route, job_id=False)
        _validate_route(self.poll_route, job_id=True)
        if self.result_route is not None:
            _validate_route(self.result_route, job_id=True)
        for path in (self.id_path, self.status_path, self.result_path):
            if not isinstance(path, tuple) or any(
                not isinstance(key, str) or not key for key in path
            ):
                raise ValueError("Expected a tuple of nonempty JSON keys")
        if not self.id_path or not self.status_path:
            raise ValueError("Job ID and status paths cannot be empty")
        seen: set[str] = set()
        for statuses in (self.pending_statuses, self.success_statuses, self.failure_statuses):
            if (
                not isinstance(statuses, frozenset)
                or any(not isinstance(status, str) or not status.strip() for status in statuses)
                or seen.intersection(statuses)
            ):
                raise ValueError("Expected disjoint frozen sets of nonempty statuses")
            seen.update(statuses)
        if not self.success_statuses or not self.failure_statuses:
            raise ValueError("Success and failure status sets cannot be empty")
        if self.multipart_field is not None and (
            not isinstance(self.multipart_field, str) or not _TOKEN.fullmatch(self.multipart_field)
        ):
            raise ValueError("Invalid multipart field")


@dataclass(frozen=True)
class HttpJobUpload:
    """One in-memory multipart upload; no files are opened by the transport."""

    filename: str
    content: bytes = field(repr=False)
    content_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.filename, str)
            or not self.filename
            or any(ord(char) < 32 or char in '\\"/\x7f' for char in self.filename)
            or not isinstance(self.content, bytes)
            or not isinstance(self.content_type, str)
            or not re.fullmatch(r"[\w.+-]+/[\w.+-]+", self.content_type, re.ASCII)
        ):
            raise ValueError("Invalid multipart upload")


def _extract(value: Any, path: JsonPath) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise HttpJobError("HTTP job response is missing a mapped field")
        value = value[key]
    return value


def _check_deadline(deadline: float) -> None:
    if asyncio.get_running_loop().time() >= deadline:
        raise HttpJobTimeout("HTTP job deadline exceeded")


class _LoopClient:
    def __init__(self) -> None:
        self.client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            await client.aclose()


class HttpJobTransport:
    """Execute one job per call. Cancelling locally does not cancel the remote job.

    Close async clients with aclose() on each owning loop before its shutdown.
    transport_factory is a trusted hook for custom async HTTP transports (e.g. tests).
    """

    def __init__(
        self,
        mapping: HttpJobMapping,
        *,
        bearer_token: str | None = None,
        timeout: float = 60.0,
        poll_interval: float = 0.5,
        get_retries: int = 2,
        retry_interval: float = 0.2,
        max_response_bytes: int = 8 * 1024 * 1024,
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        self._mapping = mapping
        self._origin = _origin(mapping.origin)
        if bearer_token is not None:
            if self._origin.scheme != "https":
                raise ValueError("HTTP job bearer authentication requires HTTPS")
            if not isinstance(bearer_token, str) or not re.fullmatch(
                r"[A-Za-z0-9._~+/-]+=*", bearer_token
            ):
                raise ValueError("Invalid HTTP job bearer token")
        self._auth = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
        for value in (timeout, poll_interval, retry_interval):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("HTTP job timing must be finite and positive")
        if type(get_retries) is not int or not 0 <= get_retries <= 10:
            raise ValueError("HTTP job GET retries must be between 0 and 10")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("HTTP job response size limit must be positive")
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._get_retries = get_retries
        self._retry_interval = retry_interval
        self._max_response_bytes = max_response_bytes
        self._transport_factory = transport_factory
        self._clients = LoopScopedAsyncClientCache()

    def _new_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=self._transport_factory() if self._transport_factory else None,
            follow_redirects=False,
            trust_env=False,
            timeout=None,
        )

    def run(self, payload: Mapping[str, Any], *, upload: HttpJobUpload | None = None) -> Any:
        """Run synchronously, outside an active asyncio event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use arun() inside an active event loop")

        async def run_private() -> Any:
            async with self._new_client() as client:
                return await self._run(client, payload, upload)

        return asyncio.run(run_private())

    async def arun(self, payload: Mapping[str, Any], *, upload: HttpJobUpload | None = None) -> Any:
        """Run asynchronously with a client cached on the calling loop."""
        holder = self._clients.get(_LoopClient)
        if holder.client is None:
            holder.client = self._new_client()
        return await self._run(holder.client, payload, upload)

    async def aclose(self) -> None:
        """Close this transport's client on the current loop after its jobs end."""
        await self._clients.get(_LoopClient).aclose()

    async def _run(
        self, client: httpx.AsyncClient, payload: Mapping[str, Any], upload: HttpJobUpload | None
    ) -> Any:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout
        task = loop.create_task(self._execute(client, payload, upload, deadline))
        try:
            # Keep caller cancellation distinct from deadline expiry on Python 3.10.
            done, _ = await asyncio.wait({task}, timeout=max(0, deadline - loop.time()))
            if not done:
                raise HttpJobTimeout("HTTP job deadline exceeded")
            _check_deadline(deadline)
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _execute(
        self,
        client: httpx.AsyncClient,
        payload: Mapping[str, Any],
        upload: HttpJobUpload | None,
        deadline: float,
    ) -> Any:
        mapping = self._mapping
        if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
            raise ValueError("HTTP job payload must be a mapping with string keys")
        if mapping.multipart_field is None:
            if upload is not None:
                raise ValueError("JSON submission does not accept an upload")
            body = {"json": dict(payload)}
        else:
            if (
                not isinstance(upload, HttpJobUpload)
                or mapping.multipart_field in payload
                or any(
                    (key != "options[model]" and not _TOKEN.fullmatch(key))
                    or not isinstance(value, str)
                    for key, value in payload.items()
                )
            ):
                raise ValueError("Multipart submission requires an upload and string form fields")
            body = {
                "data": dict(payload),
                "files": {
                    mapping.multipart_field: (upload.filename, upload.content, upload.content_type)
                },
            }
        submitted = await self._request(client, "POST", mapping.submit_route, deadline, **body)
        job_id = _extract(submitted, mapping.id_path)
        if type(job_id) is int and job_id.bit_length() <= 256:
            job_id = str(job_id)
        if (
            not isinstance(job_id, str)
            or len(job_id) > 256
            or not _TOKEN.fullmatch(job_id)
            or job_id in {".", ".."}
        ):
            raise HttpJobError("HTTP job response contains an invalid job ID")
        poll_route = mapping.poll_route.replace("{id}", job_id)
        while True:
            response = await self._request(client, "GET", poll_route, deadline)
            status = _extract(response, mapping.status_path)
            if not isinstance(status, str):
                raise HttpJobError("HTTP job response contains an invalid status")
            if status in mapping.success_statuses:
                if mapping.result_route is not None:
                    response = await self._request(
                        client, "GET", mapping.result_route.replace("{id}", job_id), deadline
                    )
                return _extract(response, mapping.result_path)
            if status in mapping.failure_statuses:
                raise HttpJobError("Remote HTTP job failed")
            if status not in mapping.pending_statuses:
                raise HttpJobError("HTTP job response contains an unmapped status")
            await asyncio.sleep(self._poll_interval)

    async def _request(
        self, client: httpx.AsyncClient, method: str, route: str, deadline: float, **body: Any
    ) -> Any:
        try:
            # Building the request directly keeps the client's cookie jar out of job requests.
            request = httpx.Request(
                method, self._origin.copy_with(path=route), headers=self._auth, **body
            )
        except (ValueError, TypeError, RecursionError):
            raise ValueError("Invalid HTTP job request payload") from None
        retries_left = self._get_retries if method == "GET" else 0
        while True:
            _check_deadline(deadline)
            try:
                response = await client.send(request, follow_redirects=False, stream=True)
                try:
                    if response.is_success:
                        content = bytearray()
                        async for chunk in response.aiter_bytes():
                            if len(chunk) > self._max_response_bytes - len(content):
                                raise HttpJobError("HTTP job response exceeded the size limit")
                            content.extend(chunk)
                        try:
                            return json.loads(content)
                        except (ValueError, RecursionError):
                            raise HttpJobError("HTTP job response is not valid JSON") from None
                    if not retries_left or response.status_code not in _RETRY_STATUS:
                        raise HttpJobError(
                            f"HTTP job {method} returned HTTP {response.status_code}"
                        )
                finally:
                    await response.aclose()
            except _RETRY_ERRORS:
                if not retries_left:
                    raise HttpJobError(f"HTTP job {method} transport failed") from None
            except httpx.HTTPError:
                raise HttpJobError(f"HTTP job {method} transport failed") from None
            retries_left -= 1
            await asyncio.sleep(self._retry_interval)
