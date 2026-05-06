"""unit tests for :class:`threetears.mcp.http_client.PlatformHttpClient`.

every test runs against ``httpx.MockTransport`` so no real network is
involved. covers JWT login envelope, refresh-on-401, error mapping,
and the env-var convenience constructor.
"""

from __future__ import annotations

import json

import httpx
import pytest
from threetears.mcp.http_client import PlatformHttpClient, PlatformHttpError


class _FakeTransport:
    """httpx.MockTransport-ish, with a recorded request log."""

    def __init__(self, responder: object) -> None:
        """capture responder callable shaped like httpx.MockTransport.

        :param responder: callable taking httpx.Request returning httpx.Response
        :ptype responder: object
        """
        self.requests: list[httpx.Request] = []
        self._responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """record + dispatch."""
        self.requests.append(request)
        return self._responder(request)  # type: ignore[operator]


def _build_client(transport: httpx.MockTransport) -> PlatformHttpClient:
    """build a client with the supplied mock transport injected."""
    client = PlatformHttpClient(
        base_url="http://test.example",
        email="admin@example.org",
        password="hunter2",
    )
    client._client = httpx.AsyncClient(transport=transport)  # noqa: SLF001
    return client


class TestLogin:
    """``POST login_path`` happy path + error mapping."""

    @pytest.mark.asyncio
    async def test_login_caches_token_from_default_field(self) -> None:
        """successful login captures access_token from the response."""
        def responder(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/v1/auth/login"
            body = json.loads(request.content)
            assert body == {"email": "admin@example.org", "password": "hunter2"}
            return httpx.Response(200, json={"access_token": "deadbeef"})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            token = await client.login()
            assert token == "deadbeef"
            assert client.token == "deadbeef"
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_login_failure_raises_PlatformHttpError(self) -> None:
        """401 from login surfaces as PlatformHttpError with status."""
        def responder(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "bad creds"})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            with pytest.raises(PlatformHttpError) as exc_info:
                await client.login()
            assert exc_info.value.status_code == 401
            assert b"bad creds" in exc_info.value.body
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_login_missing_token_field_raises(self) -> None:
        """200 response without access_token surfaces as PlatformHttpError."""
        def responder(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"refresh_token": "x"})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            with pytest.raises(PlatformHttpError, match="missing"):
                await client.login()
        finally:
            await client.aclose()


class TestRequestWithRefreshOn401:
    """authenticated request flow + refresh-on-401 retry semantics."""

    @pytest.mark.asyncio
    async def test_first_request_logs_in_then_proceeds(self) -> None:
        """no token cached -> client logs in, then sends Authorization header."""
        login_calls = 0

        def responder(request: httpx.Request) -> httpx.Response:
            nonlocal login_calls
            if request.url.path == "/api/v1/auth/login":
                login_calls += 1
                return httpx.Response(200, json={"access_token": "tok"})
            assert request.headers.get("Authorization") == "Bearer tok"
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            response = await client.get("/api/v1/admin/conversations")
            assert response.status_code == 200
            assert login_calls == 1
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_refresh_on_401_retries_once_with_new_token(self) -> None:
        """first request returns 401 -> client re-logins, retries with fresh token."""
        seen_tokens: list[str] = []
        login_count = 0

        def responder(request: httpx.Request) -> httpx.Response:
            nonlocal login_count
            if request.url.path == "/api/v1/auth/login":
                login_count += 1
                return httpx.Response(200, json={"access_token": f"tok-{login_count}"})
            seen_tokens.append(request.headers.get("Authorization", ""))
            # first call (with tok-1) returns 401; second (with tok-2) succeeds.
            if login_count == 1:
                return httpx.Response(401, json={"detail": "expired"})
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            response = await client.get("/api/v1/admin/sessions")
            assert response.status_code == 200
            assert login_count == 2
            assert seen_tokens == ["Bearer tok-1", "Bearer tok-2"]
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_persistent_401_after_refresh_raises(self) -> None:
        """second 401 after refresh -> PlatformHttpError(status=401)."""
        def responder(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json={"access_token": "x"})
            return httpx.Response(401, json={"detail": "deactivated"})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            with pytest.raises(PlatformHttpError) as exc_info:
                await client.get("/api/v1/me")
            assert exc_info.value.status_code == 401
            assert b"deactivated" in exc_info.value.body
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_non_401_status_passed_through_unraised(self) -> None:
        """500 etc. is returned to the caller, NOT raised by .request."""
        def responder(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json={"access_token": "x"})
            return httpx.Response(500, text="boom")

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            response = await client.get("/api/v1/admin/health")
            assert response.status_code == 500
        finally:
            await client.aclose()


class TestVerbHelpers:
    """get / post / patch / delete forward to .request with the right method."""

    @pytest.mark.asyncio
    async def test_post_sends_json_body(self) -> None:
        """post(path, json=...) puts the body on the wire."""
        def responder(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json={"access_token": "x"})
            assert request.method == "POST"
            assert json.loads(request.content) == {"key": "value"}
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            response = await client.post("/api/v1/admin/things", json={"key": "value"})
            assert response.status_code == 200
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_get_sends_query_params(self) -> None:
        """get(path, params=...) puts the params on the wire."""
        def responder(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/v1/auth/login":
                return httpx.Response(200, json={"access_token": "x"})
            assert request.method == "GET"
            assert request.url.params.get("limit") == "10"
            return httpx.Response(200, json={"ok": True})

        transport = httpx.MockTransport(responder)
        client = _build_client(transport)
        try:
            response = await client.get("/api/v1/admin/conversations", params={"limit": 10})
            assert response.status_code == 200
        finally:
            await client.aclose()


class TestFromEnv:
    """env-var constructor reads conventional METALLM_* vars."""

    @pytest.mark.asyncio
    async def test_from_env_picks_up_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env-var values flow through to the client config."""
        monkeypatch.setenv("METALLM_E2E_URL", "http://override.example")
        monkeypatch.setenv("METALLM_ADMIN_EMAIL", "ops@example.org")
        monkeypatch.setenv("METALLM_ADMIN_PASSWORD", "topsecret")
        client = PlatformHttpClient.from_env()
        try:
            assert client._base_url == "http://override.example"  # noqa: SLF001
            assert client._email == "ops@example.org"  # noqa: SLF001
            assert client._password == "topsecret"  # noqa: SLF001
        finally:
            await client.aclose()
