# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import json
import unittest
from dataclasses import replace

import httpx

from openviking.models.http_job import (
    HttpJobError,
    HttpJobMapping,
    HttpJobTimeout,
    HttpJobTransport,
    HttpJobUpload,
)

MAPPING = HttpJobMapping(
    origin="https://jobs.invalid",
    submit_route="/queue",
    poll_route="/queue/{id}",
    id_path=("task", "id"),
    status_path=("state",),
    pending_statuses=frozenset({"waiting"}),
    success_statuses=frozenset({"ready"}),
    failure_statuses=frozenset({"failed"}),
    result_path=("payload", "items"),
)


def transport(handler, mapping=MAPPING, **options):
    return HttpJobTransport(
        mapping,
        poll_interval=0.001,
        retry_interval=0.001,
        transport_factory=lambda: httpx.MockTransport(handler),
        **options,
    )


class HttpJobTests(unittest.TestCase):
    def test_json_body_nested_inline_result_and_get_retry(self):
        calls = []

        def handler(request):
            calls.append((request.method, request.url.path))
            self.assertEqual(request.url.host, "jobs.invalid")
            self.assertEqual(request.headers["authorization"], "Bearer fake-token")
            self.assertNotIn("cookie", request.headers)
            if request.method == "POST":
                self.assertEqual(json.loads(request.content), {"input": ["first", "second"]})
                return httpx.Response(
                    202, json={"task": {"id": 42}}, headers={"Set-Cookie": "other=ignore"}
                )
            if len(calls) == 2:
                raise httpx.ReadTimeout("private", request=request)
            if len(calls) == 3:
                return httpx.Response(503)
            if len(calls) == 4:
                return httpx.Response(200, json={"state": "waiting"})
            return httpx.Response(200, json={"state": "ready", "payload": {"items": [2, 1]}})

        result = transport(handler, bearer_token="fake-token").run({"input": ["first", "second"]})
        self.assertEqual(result, [2, 1])
        self.assertEqual(calls, [("POST", "/queue"), *(("GET", "/queue/42"),) * 4])

    def test_second_contract_multipart_and_separate_result(self):
        mapping = replace(
            MAPPING,
            submit_route="/uploads",
            poll_route="/status/{id}.json",
            result_route="/output/{id}",
            multipart_field="asset",
            id_path=("reference",),
            status_path=("meta", "phase"),
            success_statuses=frozenset({"finished"}),
            result_path=(),
        )
        calls = []

        def handler(request):
            calls.append(request.url.path)
            if request.url.path == "/uploads":
                self.assertEqual(request.method, "POST")
                self.assertTrue(request.headers["content-type"].startswith("multipart/form-data;"))
                self.assertIn(b'name="asset"; filename="note.txt"', request.content)
                self.assertIn(b"Content-Type: text/plain\r\n\r\nhello", request.content)
                self.assertIn(b'name="mode"\r\n\r\nfast', request.content)
                return httpx.Response(202, json={"reference": "a-b"})
            if request.url.path == "/status/a-b.json":
                return httpx.Response(200, json={"meta": {"phase": "finished"}})
            return httpx.Response(200, json=[{"value": 5}])

        self.assertEqual(
            transport(handler, mapping).run(
                {"mode": "fast"}, upload=HttpJobUpload("note.txt", b"hello", "text/plain")
            ),
            [{"value": 5}],
        )
        self.assertEqual(calls, ["/uploads", "/status/a-b.json", "/output/a-b"])

    def test_post_never_retried_or_redirected(self):
        for kind in ("timeout", 503, 307):
            with self.subTest(kind=kind):
                calls = []

                def handler(request, calls=calls, kind=kind):
                    calls.append(str(request.url))
                    if not isinstance(kind, int):
                        raise httpx.ReadTimeout("private", request=request)
                    return httpx.Response(kind, headers={"Location": "https://else.invalid/secret"})

                with self.assertRaises(HttpJobError) as caught:
                    transport(handler).run({})
                self.assertEqual(calls, ["https://jobs.invalid/queue"])
                self.assertNotRegex(str(caught.exception), "private|else.invalid|secret")

    def test_get_retry_cap_nonretryable_status_and_redirect(self):
        for status, expected_poll_count in ((503, 3), (401, 1), (302, 1)):
            with self.subTest(status=status):
                calls = []

                def handler(request, calls=calls, status=status):
                    calls.append(request.url.path)
                    if request.method == "POST":
                        return httpx.Response(202, json={"task": {"id": 42}})
                    return httpx.Response(status, headers={"Location": "/elsewhere"})

                with self.assertRaisesRegex(HttpJobError, str(status)):
                    transport(handler).run({})
                self.assertEqual(calls, ["/queue", *["/queue/42"] * expected_poll_count])

    def test_rejects_server_supplied_route_fragments(self):
        for job_id in ("../result", "..", "//host", "%2F", "a?b", True):
            with self.subTest(job_id=job_id):

                def handler(request, job_id=job_id):
                    self.assertEqual(request.method, "POST")
                    return httpx.Response(202, json={"task": {"id": job_id}})

                with self.assertRaisesRegex(HttpJobError, "invalid job ID"):
                    transport(handler).run({})

    def test_rejects_bad_mapping_and_auth_before_io(self):
        for fields in (
            {"origin": "https://user:pass@jobs.invalid"},
            {"origin": "https://jobs.invalid/prefix"},
            {"submit_route": "//else.invalid/a"},
            {"poll_route": "/../{id}"},
            {"poll_route": "/status/{id}?where=else"},
            {"poll_route": "/fixed"},
            {"id_path": ()},
            {"pending_statuses": frozenset({"ready"})},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(MAPPING, **fields)
        with self.assertRaisesRegex(ValueError, "requires HTTPS"):
            HttpJobTransport(replace(MAPPING, origin="http://jobs.invalid"), bearer_token="test")
        with self.assertRaisesRegex(ValueError, "Invalid HTTP job bearer token"):
            HttpJobTransport(MAPPING, bearer_token="bad\r\nX-Injection: value")

    def test_bad_json_limit_missing_field_unknown_status_and_failed_job(self):
        for response, error in (
            (httpx.Response(200, content=b"not JSON"), "not valid JSON"),
            (httpx.Response(200, content=b"x" * 64), "size limit"),
            (httpx.Response(200, json={"state": "ready"}), "missing a mapped field"),
            (httpx.Response(200, json={"state": "unexpected"}), "unmapped status"),
            (
                httpx.Response(200, json={"state": "failed", "reason": "private"}),
                "Remote HTTP job failed",
            ),
        ):
            with self.subTest(error=error):

                def handler(request, response=response):
                    if request.method == "POST":
                        return httpx.Response(202, json={"task": {"id": 42}})
                    return response

                with self.assertRaisesRegex(HttpJobError, error) as caught:
                    transport(handler, max_response_bytes=48).run({})
                self.assertNotIn("private", str(caught.exception))

    def test_client_is_scoped_to_the_owning_loop(self):
        created = []
        closed = []

        def factory():
            loop = asyncio.get_running_loop()
            created.append(loop)

            def handler(request):
                self.assertIs(loop, asyncio.get_running_loop())
                if request.method == "POST":
                    return httpx.Response(202, json={"task": {"id": 42}})
                return httpx.Response(200, json={"state": "ready", "payload": {"items": []}})

            class Transport(httpx.MockTransport):
                async def aclose(self):
                    closed.append(asyncio.get_running_loop())

            return Transport(handler)

        client = HttpJobTransport(MAPPING, transport_factory=factory)

        async def run_twice():
            await client.arun({})
            await client.arun({})
            await client.aclose()

        asyncio.run(run_twice())
        asyncio.run(run_twice())
        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        self.assertEqual(created, closed)


class AsyncHttpJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_includes_hanging_response_and_cleans_up(self):
        cleaned = asyncio.Event()

        async def handler(request):
            self.assertEqual(request.method, "POST")
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        client = transport(handler, timeout=0.03)
        self.addAsyncCleanup(client.aclose)
        with self.assertRaises(HttpJobTimeout):
            await asyncio.wait_for(client.arun({}), timeout=2)
        self.assertTrue(cleaned.is_set())

    async def test_cancellation_is_not_a_retry(self):
        started = asyncio.Event()
        cleaned = asyncio.Event()
        calls = []

        async def handler(request):
            calls.append(request.method)
            if request.method == "POST":
                return httpx.Response(202, json={"task": {"id": 42}})
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned.set()

        client = transport(handler)
        self.addAsyncCleanup(client.aclose)
        task = asyncio.create_task(client.arun({}))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cleaned.is_set())
        self.assertEqual(calls, ["POST", "GET"])

    async def test_deadline_covers_poll_and_retry_waits(self):
        calls = []

        def handler(request):
            calls.append(request.method)
            if request.method == "POST":
                return httpx.Response(202, json={"task": {"id": 42}})
            if len(calls) == 2:
                return httpx.Response(200, json={"state": "waiting"})
            return httpx.Response(503)

        client = HttpJobTransport(
            MAPPING,
            timeout=0.08,
            poll_interval=0.001,
            retry_interval=5,
            transport_factory=lambda: httpx.MockTransport(handler),
        )
        self.addAsyncCleanup(client.aclose)
        with self.assertRaises(HttpJobTimeout):
            await asyncio.wait_for(client.arun({}), timeout=2)
        self.assertEqual(calls, ["POST", "GET", "GET"])
