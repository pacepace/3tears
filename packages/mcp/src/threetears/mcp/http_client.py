"""typed httpx client used by both MCP servers and CLI scripts.

one HTTP-client implementation, two transports:

- MCP server tool handlers consume this client to call their backend
  REST API (e.g. metallm-mcp-server.py's ``list_conversations`` tool
  delegates to ``client.get("/api/v1/admin/conversations")``).
- CLI scripts (``debug-api.py``, ``debug-token.py``,
  ``settings-transfer.py``) consume the same client; the shell
  wrappers (``debug-api.sh`` etc.) delegate to thin Python entries
  that import :class:`PlatformHttpClient`.

JWT login + automatic refresh-on-401 lives here exactly once. before
this consolidation each consumer reimplemented its own auth logic
(metallm-mcp-server's ``_login`` + ``_api``, debug-api's bespoke
auth, settings-transfer's standalone httpx). the consolidation
removes a real source of drift: a JWT-shape change today requires
updating N call sites; with this client it's one update.

named for what it is, not for what consumes it: ``PlatformHttpClient``
is a generic typed REST client with platform conventions (admin
login envelope, JWT bearer, refresh-on-401). the ``mcp`` package
ownership is incidental -- v1 cohesion. promote to a less
mcp-specific home if a third consumer ever appears.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin

import httpx
from threetears.observe import get_logger

__all__ = [
    "PlatformHttpClient",
    "PlatformHttpError",
]

log = get_logger(__name__)


class PlatformHttpError(RuntimeError):
    """raised when a platform HTTP call fails after all retries.

    carries the upstream HTTP status + response body so callers
    can pattern-match on error shape without re-parsing.

    :ivar status_code: HTTP status code returned by the upstream
    :ivar body: upstream response body (bytes); decode as needed
    """

    def __init__(self, message: str, *, status_code: int, body: bytes) -> None:
        """capture status + body alongside the message.

        :param message: human-readable error description
        :ptype message: str
        :param status_code: upstream HTTP status code
        :ptype status_code: int
        :param body: upstream response body
        :ptype body: bytes
        :return: nothing
        :rtype: None
        """
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class PlatformHttpClient:
    """typed REST client with JWT login + refresh-on-401.

    constructors:

    - :meth:`__init__` -- explicit base_url + creds (preferred for
      tests and for callers that already hold their config).
    - :meth:`from_env` -- read base_url + admin creds from env vars
      (CLI scripts and CLI-style MCP servers use this).

    :param base_url: API root URL (e.g. ``http://localhost:8002``)
    :ptype base_url: str
    :param email: admin email used for ``POST /api/v1/auth/login``
    :ptype email: str
    :param password: admin password used for the login call
    :ptype password: str
    :param login_path: path the login POST hits;
        ``"/api/v1/auth/login"`` is the platform default
    :ptype login_path: str
    :param token_field: JSON field on the login response that
        carries the JWT; ``"access_token"`` is the platform default
    :ptype token_field: str
    :param timeout: per-request timeout in seconds
    :ptype timeout: float
    """

    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        password: str,
        login_path: str = "/api/v1/auth/login",
        token_field: str = "access_token",
        timeout: float = 30.0,
    ) -> None:
        """capture config; no I/O until :meth:`login` or first request.

        :param base_url: API root URL
        :ptype base_url: str
        :param email: admin email
        :ptype email: str
        :param password: admin password
        :ptype password: str
        :param login_path: path for ``POST`` login
        :ptype login_path: str
        :param token_field: response JSON field carrying the JWT
        :ptype token_field: str
        :param timeout: per-request timeout
        :ptype timeout: float
        :return: nothing
        :rtype: None
        :raises ValueError: when ``base_url`` is empty
        """
        if not base_url:
            raise ValueError("base_url must be non-empty")
        self._base_url = base_url.rstrip("/")
        self._email = email
        self._password = password
        self._login_path = login_path
        self._token_field = token_field
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._token: str | None = None
        # serializes login() so concurrent first-requests + concurrent
        # 401-refreshes don't trigger a login storm. each waiter that
        # arrives while a login is in flight observes the freshly-cached
        # token after the lock releases instead of issuing its own.
        self._login_lock = asyncio.Lock()

    @classmethod
    def from_env(
        cls,
        *,
        url_env_var: str = "METALLM_E2E_URL",
        email_env_var: str = "METALLM_ADMIN_EMAIL",
        password_env_var: str = "METALLM_ADMIN_PASSWORD",
        url_default: str = "http://localhost:8002",
        email_default: str = "admin@example.org",
        password_default: str = "password",
    ) -> PlatformHttpClient:
        """build a client from env vars (CLI ergonomics).

        defaults match the existing ``debug-api.sh`` / ``debug-token.sh``
        defaults so a drop-in rewrite of those scripts keeps the same
        out-of-the-box behaviour.

        :param url_env_var: env var holding the API root URL
        :ptype url_env_var: str
        :param email_env_var: env var holding the admin email
        :ptype email_env_var: str
        :param password_env_var: env var holding the admin password
        :ptype password_env_var: str
        :param url_default: fallback URL when env var unset
        :ptype url_default: str
        :param email_default: fallback email when env var unset
        :ptype email_default: str
        :param password_default: fallback password when env var unset
        :ptype password_default: str
        :return: configured client (not logged in yet)
        :rtype: PlatformHttpClient
        """
        return cls(
            base_url=os.environ.get(url_env_var, url_default),
            email=os.environ.get(email_env_var, email_default),
            password=os.environ.get(password_env_var, password_default),
        )

    async def __aenter__(self) -> PlatformHttpClient:
        """return self for ``async with`` ergonomics.

        :return: self
        :rtype: PlatformHttpClient
        """
        return self

    async def __aexit__(self, *_args: Any) -> None:
        """close the underlying httpx client on context exit.

        :return: nothing
        :rtype: None
        """
        await self.aclose()

    async def aclose(self) -> None:
        """close the underlying httpx client.

        :return: nothing
        :rtype: None
        """
        await self._client.aclose()

    async def login(self, *, force: bool = False) -> str:
        """POST ``login_path`` with email/password; cache the token.

        serialized via :attr:`_login_lock` so concurrent callers
        don't issue parallel login requests. when ``force=False``
        (the default) waiters that arrive after another login
        completed observe the freshly-cached token without issuing
        their own POST. ``force=True`` forces a re-login (the
        refresh-on-401 retry path uses this so the new token is
        guaranteed fresh).

        :param force: re-login even if a token is already cached
        :ptype force: bool
        :return: the bearer token
        :rtype: str
        :raises PlatformHttpError: when login returns non-2xx or
            the response body lacks ``token_field``
        """
        async with self._login_lock:
            if not force and self._token is not None:
                return self._token
            url = self._url(self._login_path)
            response = await self._client.post(
                url,
                json={"email": self._email, "password": self._password},
            )
            if response.status_code >= 400:
                raise PlatformHttpError(
                    f"login failed: {response.status_code}",
                    status_code=response.status_code,
                    body=response.content,
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise PlatformHttpError(
                    f"login response not JSON: {response.text[:200]}",
                    status_code=response.status_code,
                    body=response.content,
                ) from exc
            token = payload.get(self._token_field)
            if not token:
                raise PlatformHttpError(
                    f"login response missing {self._token_field!r}",
                    status_code=response.status_code,
                    body=response.content,
                )
            self._token = token
            return token

    @property
    def token(self) -> str | None:
        """current cached token (None until first :meth:`login`).

        :return: bearer token or None
        :rtype: str | None
        """
        return self._token

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """authenticated request with refresh-on-401.

        flow: ensure a token (login if not yet), send the request
        with ``Authorization: Bearer <token>``, on 401 re-login once
        and retry. on a second 401 the request fails -- credentials
        are wrong or the user was deactivated.

        :param method: HTTP verb (GET / POST / PATCH / DELETE / etc.)
        :ptype method: str
        :param path: API path (joined with ``base_url``); leading
            slash optional
        :ptype path: str
        :param json: optional JSON body
        :ptype json: Any
        :param params: optional query-string parameters
        :ptype params: Mapping[str, Any] | None
        :param headers: optional extra request headers
        :ptype headers: Mapping[str, str] | None
        :return: full httpx response (caller decides what to do
            with non-2xx; this method does not raise on 4xx/5xx
            other than the 401 retry)
        :rtype: httpx.Response
        :raises PlatformHttpError: when the second attempt also
            returns 401 (auth genuinely failed, not a stale token)
        """
        if self._token is None:
            await self.login()

        response = await self._send(method, path, json=json, params=params, headers=headers)
        if response.status_code == 401:
            log.info(
                "401 from upstream; refreshing token and retrying once",
                extra={"extra_data": {"method": method, "path": path}},
            )
            await self.login(force=True)
            response = await self._send(method, path, json=json, params=params, headers=headers)
            if response.status_code == 401:
                raise PlatformHttpError(
                    "auth failed after refresh-on-401 retry",
                    status_code=401,
                    body=response.content,
                )
        return response

    async def _send(
        self,
        method: str,
        path: str,
        *,
        json: Any,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
    ) -> httpx.Response:
        """execute one request with the current token; no retry logic.

        :param method: HTTP verb
        :ptype method: str
        :param path: API path
        :ptype path: str
        :param json: optional JSON body
        :ptype json: Any
        :param params: optional query string
        :ptype params: Mapping[str, Any] | None
        :param headers: optional extra headers
        :ptype headers: Mapping[str, str] | None
        :return: httpx response
        :rtype: httpx.Response
        """
        merged_headers: dict[str, str] = {}
        if self._token is not None:
            merged_headers["Authorization"] = f"Bearer {self._token}"
        if headers:
            merged_headers.update(headers)
        url = self._url(path)
        return await self._client.request(
            method,
            url,
            json=json,
            params=dict(params) if params else None,
            headers=merged_headers,
        )

    async def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> httpx.Response:
        """GET ``path`` with refresh-on-401.

        :param path: API path
        :ptype path: str
        :param params: optional query string
        :ptype params: Mapping[str, Any] | None
        :return: response
        :rtype: httpx.Response
        """
        return await self.request("GET", path, params=params)

    async def post(self, path: str, *, json: Any = None) -> httpx.Response:
        """POST ``path`` with optional JSON body.

        :param path: API path
        :ptype path: str
        :param json: optional JSON body
        :ptype json: Any
        :return: response
        :rtype: httpx.Response
        """
        return await self.request("POST", path, json=json)

    async def patch(self, path: str, *, json: Any = None) -> httpx.Response:
        """PATCH ``path`` with optional JSON body.

        :param path: API path
        :ptype path: str
        :param json: optional JSON body
        :ptype json: Any
        :return: response
        :rtype: httpx.Response
        """
        return await self.request("PATCH", path, json=json)

    async def delete(self, path: str) -> httpx.Response:
        """DELETE ``path``.

        :param path: API path
        :ptype path: str
        :return: response
        :rtype: httpx.Response
        """
        return await self.request("DELETE", path)

    async def upload(
        self,
        path: str,
        *,
        file_field: str,
        file_data: bytes,
        filename: str,
        content_type: str,
        extra_form: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """POST ``path`` with multipart/form-data and refresh-on-401.

        ``post()`` is JSON-only by design; multipart requests use
        ``files=`` / ``data=`` on the underlying httpx call instead.
        Rather than expand ``post()``'s contract, this is its own
        method so the JSON path stays simple and callers see an
        explicit upload site.

        Mirrors :meth:`request` for the auth + 401-retry semantics:
        ensure a token, send the upload, on 401 re-login once and
        retry once. A second 401 raises :class:`PlatformHttpError`.

        :param path: API path; leading slash optional
        :ptype path: str
        :param file_field: name of the multipart field to attach the
            file under (e.g. ``"file"`` for a FastAPI
            ``UploadFile = File(...)`` parameter named ``file``)
        :ptype file_field: str
        :param file_data: raw bytes to upload
        :ptype file_data: bytes
        :param filename: original filename to send (servers commonly
            use this for content-type fallback / extension sniffing)
        :ptype filename: str
        :param content_type: MIME type of the upload
        :ptype content_type: str
        :param extra_form: additional form fields to include alongside
            the file
        :ptype extra_form: Mapping[str, str] | None
        :return: full httpx response
        :rtype: httpx.Response
        :raises PlatformHttpError: when the second attempt also
            returns 401 (auth genuinely failed, not a stale token)
        """
        if self._token is None:
            await self.login()

        response = await self._send_upload(
            path,
            file_field=file_field,
            file_data=file_data,
            filename=filename,
            content_type=content_type,
            extra_form=extra_form,
        )
        if response.status_code == 401:
            log.info(
                "401 on upload; refreshing token and retrying once",
                extra={"extra_data": {"path": path, "filename": filename}},
            )
            await self.login(force=True)
            response = await self._send_upload(
                path,
                file_field=file_field,
                file_data=file_data,
                filename=filename,
                content_type=content_type,
                extra_form=extra_form,
            )
            if response.status_code == 401:
                raise PlatformHttpError(
                    "auth failed after refresh-on-401 retry on upload",
                    status_code=401,
                    body=response.content,
                )
        return response

    async def _send_upload(
        self,
        path: str,
        *,
        file_field: str,
        file_data: bytes,
        filename: str,
        content_type: str,
        extra_form: Mapping[str, str] | None,
    ) -> httpx.Response:
        """execute one multipart POST with the current token; no retry."""
        merged_headers: dict[str, str] = {}
        if self._token is not None:
            merged_headers["Authorization"] = f"Bearer {self._token}"
        url = self._url(path)
        files = {file_field: (filename, file_data, content_type)}
        data = dict(extra_form) if extra_form else None
        return await self._client.post(
            url,
            files=files,
            data=data,
            headers=merged_headers,
        )

    def _url(self, path: str) -> str:
        """build the full URL for ``path`` (joins to ``base_url``).

        :param path: API path; leading slash optional
        :ptype path: str
        :return: full URL
        :rtype: str
        """
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return urljoin(self._base_url + "/", path.lstrip("/"))
