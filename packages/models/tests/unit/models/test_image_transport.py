"""ImageServerTransport against a scripted server (httpx.MockTransport): the byte cap, the one
origin, redirects, and the log of each attempt."""

from __future__ import annotations

import gzip
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from threetears.models.providers.image.transport import ImageServerTransport, ResponseTooLargeError, RetryPolicy

BASE = "http://gpu-box.lan:8188"
_LOGGER = "threetears.models.providers.image.transport"


async def _no_sleep(_seconds: float) -> None:
    return None


def _transport(handler: Callable[[httpx.Request], httpx.Response], **kw: object) -> ImageServerTransport:
    return ImageServerTransport(
        BASE,
        name="TestClient",
        timeout=5.0,
        retry=RetryPolicy(attempts=2, sleep=_no_sleep),
        transport=httpx.MockTransport(handler),
        **kw,  # type: ignore[arg-type]
    )


async def test_a_body_under_the_cap_is_read() -> None:
    async with _transport(lambda _r: httpx.Response(200, content=b"x" * 10), max_response_bytes=10) as server:
        response = await server.request("GET", "/view", repeatable=True)
    assert response.content == b"x" * 10


async def test_a_declared_body_over_the_cap_is_refused_before_reading() -> None:
    answer = httpx.Response(200, headers={"content-length": "11"}, content=b"x" * 11)
    async with _transport(lambda _r: answer, max_response_bytes=10) as server:
        with pytest.raises(ResponseTooLargeError) as err:
            await server.request("GET", "/view", repeatable=True)
    assert err.value.limit == 10 and err.value.size == 11


async def test_an_undeclared_body_over_the_cap_is_refused_while_reading() -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"x" * 6
        yield b"x" * 6

    seen: list[int] = []

    def counting(_r: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(200, content=chunks())  # streamed: no content-length

    async with _transport(counting, max_response_bytes=10) as server:
        with pytest.raises(ResponseTooLargeError):
            await server.request("GET", "/view", repeatable=True)
    assert len(seen) == 1  # an answer too large is not asked for again


async def test_a_gzipped_body_is_read_once_decoded() -> None:
    answer = httpx.Response(200, headers={"content-encoding": "gzip"}, content=gzip.compress(b'{"a": 1}'))
    async with _transport(lambda _r: answer) as server:
        response = await server.request("GET", "/queue", repeatable=True)
    assert response.json() == {"a": 1}


@pytest.mark.parametrize("path", ["http://elsewhere:80/x", "//elsewhere/x", "history", "/a\\@elsewhere"])
async def test_only_a_path_under_the_base_is_taken(path: str) -> None:
    async with _transport(lambda _r: httpx.Response(200)) as server:
        with pytest.raises(ValueError):
            await server.request("GET", path, repeatable=True)


async def test_a_redirect_is_not_followed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://elsewhere/"})

    async with _transport(handler) as server:
        response = await server.request("GET", "/queue", repeatable=True)
    assert response.status_code == 302


def test_the_base_must_be_an_http_address() -> None:
    with pytest.raises(ValueError):
        ImageServerTransport("file:///etc", name="TestClient", timeout=1.0)


async def test_each_attempt_is_logged_and_running_out_is_an_error(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG", logger=_LOGGER):
        async with _transport(lambda _r: httpx.Response(503)) as server:
            response = await server.request("GET", "/queue", repeatable=True)
    assert response.status_code == 503
    attempts = [r.extra_data for r in caplog.records if r.getMessage() == "image server request attempt"]  # type: ignore[attr-defined]
    assert [(a["method"], a["path"], a["attempt"], a["outcome"]) for a in attempts] == [
        ("GET", "/queue", 1, "503"),
        ("GET", "/queue", 2, "503"),
    ]
    gave_up = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(gave_up) == 1 and gave_up[0].extra_data["attempts"] == 2  # type: ignore[attr-defined]


async def test_a_request_that_starts_work_is_not_sent_again_once_it_left(caplog: pytest.LogCaptureFixture) -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with caplog.at_level("DEBUG", logger=_LOGGER):
        async with _transport(handler) as server:
            with pytest.raises(httpx.ReadTimeout):
                await server.request("POST", "/prompt", repeatable=False, json={})
    attempts = [r for r in caplog.records if r.getMessage() == "image server request attempt"]
    assert len(attempts) == 1
    assert [r.getMessage() for r in caplog.records if r.levelname == "ERROR"] == ["image server request failed"]
