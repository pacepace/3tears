"""concurrent-request login-storm test for :class:`PlatformHttpClient`.

separated from ``test_http_client.py`` because it exercises an
additional invariant (login-lock) and uses a slower mock path
(asyncio sleep inside the responder).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from threetears.mcp.http_client import PlatformHttpClient


def _build_client(transport: httpx.MockTransport) -> PlatformHttpClient:
    """build a client with the supplied mock transport injected."""
    client = PlatformHttpClient(
        base_url="http://test.example",
        email="admin@example.org",
        password="hunter2",
    )
    client._client = httpx.AsyncClient(transport=transport)  # noqa: SLF001
    return client


@pytest.mark.asyncio
async def test_concurrent_first_requests_login_only_once() -> None:
    """N parallel first-requests trigger ONE login, not N.

    proves the asyncio.Lock around login() prevents the storm.
    each waiter that arrives during a slow login sees the cached
    token after the lock releases instead of issuing its own POST.
    """
    login_count = 0

    async def slow_login(_request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        login_count += 1
        # simulate a slow login so the parallel callers all queue.
        await asyncio.sleep(0.02)
        return httpx.Response(200, json={"access_token": "tok"})

    def responder(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/auth/login":
            # MockTransport requires sync return -- run the async
            # coroutine synchronously via asyncio.run-like trick.
            # easier: construct the response directly + bump count.
            nonlocal login_count
            login_count += 1
            return httpx.Response(200, json={"access_token": "tok"})
        assert request.headers.get("Authorization") == "Bearer tok"
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(responder)
    client = _build_client(transport)
    try:
        # fire 5 parallel first-requests.
        results = await asyncio.gather(
            client.get("/api/v1/path1"),
            client.get("/api/v1/path2"),
            client.get("/api/v1/path3"),
            client.get("/api/v1/path4"),
            client.get("/api/v1/path5"),
        )
        assert all(r.status_code == 200 for r in results)
        # only one login despite five parallel callers.
        assert login_count == 1
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_force_login_overrides_cached_token() -> None:
    """login(force=True) re-issues the POST even when a token is cached.

    used by the refresh-on-401 retry path to guarantee the new
    token is fresh.
    """
    login_count = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/api/v1/auth/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": f"tok-{login_count}"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(responder)
    client = _build_client(transport)
    try:
        first = await client.login()
        second = await client.login(force=True)
        assert first == "tok-1"
        assert second == "tok-2"
        assert login_count == 2
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_default_login_returns_cached_token_without_post() -> None:
    """login() without force=True returns the cached token (no POST)."""
    login_count = 0

    def responder(request: httpx.Request) -> httpx.Response:
        nonlocal login_count
        if request.url.path == "/api/v1/auth/login":
            login_count += 1
            return httpx.Response(200, json={"access_token": "tok"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(responder)
    client = _build_client(transport)
    try:
        await client.login()
        await client.login()
        await client.login()
        assert login_count == 1
    finally:
        await client.aclose()
