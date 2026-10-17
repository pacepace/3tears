"""SwarmUI: a client for its HTTP API.

SwarmUI is a front end that runs pictures on one or more ComfyUI backends (one per GPU, often on
other machines). :class:`SwarmUIClient` is its JSON API: a session, a picture from text
(``GenerateText2Image`` with any of its parameters), its backends and their state, switching a
backend off and on (SwarmUI keeps a dead connection to a ComfyUI that restarted), and a rescan of
its model folders. :meth:`SwarmUIClient.comfy` is the ComfyUI it proxies for raw graphs.

SwarmUI answers a refusal with HTTP 200 and an ``error`` in the body; every call here raises
:class:`SwarmUIError` for it instead of handing back a reply that looks like success. Two refusals
are handled where they happen, because neither means the request ran: an expired session (a new
one is taken and the call sent again), and, for a picture, a model or LoRA added since SwarmUI's
last scan (one rescan, then the call again) and a pinned backend that is still connecting (the
call again after a pause, a bounded number of times). Reads are sent again when the server can't
be reached or answers 5xx; a picture is sent again only when the request never left.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Any

import httpx

from threetears.models.providers.image import http_retry
from threetears.models.providers.image.comfyui import ComfyUIClient
from threetears.models.providers.image.http_retry import RetryPolicy
from threetears.observe import get_logger

__all__ = ["SwarmUIClient", "SwarmUIError"]

log = get_logger(__name__)

#: SwarmUI's ``error_id`` for a session it no longer knows
_INVALID_SESSION = "invalid_session_id"
#: SwarmUI's words for a model or LoRA it hasn't listed yet (added since its last scan)
_NOT_LISTED = ("option does not exist", "Invalid model value")
#: SwarmUI's words when no backend can take the job (a pinned one still connecting)
_NO_BACKEND = "No backends match"
#: the most of an answer's text kept on an error
_MAX_ERROR_TEXT = 2000


class SwarmUIError(RuntimeError):
    """SwarmUI refused a call: its ``error`` text, or the HTTP status when it answered with one.

    :param message: what went wrong, in SwarmUI's words when it gave any
    :ptype message: str
    :param error_id: SwarmUI's ``error_id``, when it gave one
    :ptype error_id: str | None
    :param status_code: the HTTP status, for a non-2xx answer
    :ptype status_code: int | None
    """

    def __init__(self, message: str, *, error_id: str | None = None, status_code: int | None = None) -> None:
        """record the message, the error id and the status.

        :param message: what went wrong
        :ptype message: str
        :param error_id: SwarmUI's ``error_id``, if any
        :ptype error_id: str | None
        :param status_code: the HTTP status, if any
        :ptype status_code: int | None
        :return: nothing
        :rtype: None
        """
        super().__init__(message)
        self.error_id = error_id
        self.status_code = status_code


class SwarmUIClient:
    """one SwarmUI server's API, open for the length of an ``async with`` block.

    :param base_url: SwarmUI's address (``http://host:7801``)
    :ptype base_url: str
    :param timeout: seconds an ordinary call may take
    :ptype timeout: float
    :param generate_timeout: seconds a picture may take (a cold model loads first)
    :ptype generate_timeout: float
    :param retry: sends per request and the pauses between them; ``None`` is three sends
    :ptype retry: RetryPolicy | None
    :param backend_waits: how many times a picture is asked for again while no backend can take it
    :ptype backend_waits: int
    :param backend_wait_seconds: the pause before each of those
    :ptype backend_wait_seconds: float
    :param transport: an httpx transport (a test's ``httpx.MockTransport``); ``None`` is the network
    :ptype transport: httpx.AsyncBaseTransport | None
    :param sleep: how the backend waits are taken; ``None`` is ``asyncio.sleep``
    :ptype sleep: Callable[[float], Awaitable[None]] | None
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        generate_timeout: float = 900.0,
        retry: RetryPolicy | None = None,
        backend_waits: int = 11,
        backend_wait_seconds: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """capture the settings; the connection opens on ``async with``.

        :param base_url: SwarmUI's address
        :ptype base_url: str
        :param timeout: seconds an ordinary call may take
        :ptype timeout: float
        :param generate_timeout: seconds a picture may take
        :ptype generate_timeout: float
        :param retry: sends per request and pauses, or ``None``
        :ptype retry: RetryPolicy | None
        :param backend_waits: asks again while no backend can take a picture
        :ptype backend_waits: int
        :param backend_wait_seconds: the pause before each
        :ptype backend_wait_seconds: float
        :param transport: an httpx transport, or ``None``
        :ptype transport: httpx.AsyncBaseTransport | None
        :param sleep: how a backend wait is taken, or ``None``
        :ptype sleep: Callable[[float], Awaitable[None]] | None
        :return: nothing
        :rtype: None
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generate_timeout = generate_timeout
        self._retry = retry or RetryPolicy()
        self._backend_waits = backend_waits
        self._backend_wait_seconds = backend_wait_seconds
        self._transport = transport
        self._sleep = sleep
        self._session: str | None = None
        self._stack: contextlib.AsyncExitStack | None = None
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> SwarmUIClient:
        """open the connection.

        :return: self
        :rtype: SwarmUIClient
        """
        stack = contextlib.AsyncExitStack()
        self._http = await stack.enter_async_context(httpx.AsyncClient(timeout=self.timeout, transport=self._transport))
        self._stack = stack
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """close the connection.

        :param exc_type: the exception's type, if the block raised
        :ptype exc_type: type[BaseException] | None
        :param exc: the exception, if the block raised
        :ptype exc: BaseException | None
        :param tb: its traceback
        :ptype tb: TracebackType | None
        :return: nothing
        :rtype: None
        """
        stack, self._stack, self._http = self._stack, None, None
        if stack is not None:
            await stack.aclose()

    # --- the API -------------------------------------------------------------------------------

    async def _post(
        self, name: str, payload: Mapping[str, Any], *, repeatable: bool, timeout: float | None = None
    ) -> dict[str, Any]:
        """one call to ``/API/<name>``: SwarmUI's reply, or SwarmUIError for its ``error``.

        :param name: the API call (``GetNewSession``, ``ListBackends``...)
        :ptype name: str
        :param payload: its JSON body
        :ptype payload: Mapping[str, Any]
        :param repeatable: True when sending it twice changes nothing
        :ptype repeatable: bool
        :param timeout: this call's timeout; ``None`` is the client's
        :ptype timeout: float | None
        :return: the reply
        :rtype: dict[str, Any]
        :raises RuntimeError: when used outside ``async with``
        :raises SwarmUIError: when SwarmUI refuses (an ``error`` in the reply, or a non-2xx status)
        """
        if self._http is None:
            raise RuntimeError("SwarmUIClient is used inside `async with` only")
        kwargs: dict[str, Any] = {"json": dict(payload)}
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = await http_retry.send(
            self._http, "POST", f"{self.base_url}/API/{name}", repeatable=repeatable, policy=self._retry, **kwargs
        )
        if response.status_code >= 400:
            raise SwarmUIError(
                f"SwarmUI {name} answered {response.status_code}: {response.text[:_MAX_ERROR_TEXT]}",
                status_code=response.status_code,
            )
        reply = response.json()
        if not isinstance(reply, dict):
            raise SwarmUIError(f"SwarmUI {name} answered something that isn't an object")
        if "error" in reply:
            raise SwarmUIError(f"SwarmUI: {reply['error']}", error_id=reply.get("error_id"))
        return reply

    async def session(self) -> str:
        """this client's session id (a new one on first use).

        :return: the session id
        :rtype: str
        """
        if self._session is None:
            self._session = str((await self._post("GetNewSession", {}, repeatable=True))["session_id"])
        return self._session

    async def call(
        self, name: str, payload: Mapping[str, Any] | None = None, *, repeatable: bool, timeout: float | None = None
    ) -> dict[str, Any]:
        """any SwarmUI API call that takes a session: the session is added, and taken again once if
        SwarmUI no longer knows it.

        :param name: the API call
        :ptype name: str
        :param payload: its body, without ``session_id``
        :ptype payload: Mapping[str, Any] | None
        :param repeatable: True when sending it twice changes nothing
        :ptype repeatable: bool
        :param timeout: this call's timeout; ``None`` is the client's
        :ptype timeout: float | None
        :return: the reply
        :rtype: dict[str, Any]
        :raises SwarmUIError: when SwarmUI refuses
        """
        body = dict(payload or {})
        for fresh in (False, True):
            if fresh:
                self._session = None
            try:
                return await self._post(
                    name, {**body, "session_id": await self.session()}, repeatable=repeatable, timeout=timeout
                )
            except SwarmUIError as exc:
                if exc.error_id != _INVALID_SESSION or fresh:
                    raise
        raise AssertionError("unreachable")  # the loop returns or raises

    async def refresh_models(self) -> None:
        """rescan the model folders, so a model or LoRA copied in since start-up can be used.

        :return: nothing
        :rtype: None
        """
        await self.call("TriggerRefresh", {"strong": True}, repeatable=True, timeout=self.generate_timeout)

    async def generate(self, params: Mapping[str, Any]) -> list[str]:
        """one ``GenerateText2Image`` call: the saved pictures' paths as SwarmUI names them.

        ``params`` are SwarmUI's own parameter ids (``prompt``, ``model``, ``seed``, ``width``,
        ``loras``, ``exactbackendid``, ...); ``images`` defaults to 1. A model or LoRA it hasn't
        listed yet gets one rescan and a second ask; while no backend can take the job, it is asked
        again after a pause, up to ``backend_waits`` times.

        :param params: the parameters
        :ptype params: Mapping[str, Any]
        :return: the pictures' paths (``View/...``/``Output/...``, as SwarmUI answers them)
        :rtype: list[str]
        :raises SwarmUIError: when SwarmUI refuses the picture
        """
        body = {"images": 1, **params}
        rescanned, waits = False, 0
        while True:
            try:
                reply = await self.call("GenerateText2Image", body, repeatable=False, timeout=self.generate_timeout)
            except SwarmUIError as exc:
                said = str(exc)
                if not rescanned and any(words in said for words in _NOT_LISTED):
                    rescanned = True
                    log.info("swarmui: rescanning its models", extra={"extra_data": {"error": said}})
                    await self.refresh_models()
                    continue
                if _NO_BACKEND in said and waits < self._backend_waits:
                    waits += 1
                    log.info(
                        "swarmui: no backend can take it yet; asking again",
                        extra={"extra_data": {"wait": waits, "of": self._backend_waits}},
                    )
                    await (self._sleep or asyncio.sleep)(self._backend_wait_seconds)
                    continue
                raise
            return [str(p) for p in reply.get("images") or []]

    async def list_backends(self) -> dict[str, dict[str, Any]]:
        """every backend by its id: ``status`` (running, idle, loading, errored, disabled...),
        ``title``, ``settings`` (its ``Address``...).

        :return: the backends
        :rtype: dict[str, dict[str, Any]]
        """
        reply = await self.call("ListBackends", repeatable=True)
        return {str(k): v for k, v in reply.items() if isinstance(v, dict)}

    async def toggle_backend(self, backend_id: int, enabled: bool, *, timeout: float | None = None) -> None:
        """switch a backend off or on. On makes SwarmUI connect to it again, and waits for that.

        :param backend_id: the backend's id
        :ptype backend_id: int
        :param enabled: on or off
        :ptype enabled: bool
        :param timeout: this call's timeout; connecting can take longer than an ordinary call
        :ptype timeout: float | None
        :return: nothing
        :rtype: None
        """
        await self.call(
            "ToggleBackend", {"backend_id": backend_id, "enabled": enabled}, repeatable=True, timeout=timeout
        )

    async def reconnect_backend(self, backend_id: int, *, timeout: float | None = None) -> None:
        """off and on: SwarmUI drops its connection to a backend and makes a new one.

        :param backend_id: the backend's id
        :ptype backend_id: int
        :param timeout: each call's timeout
        :ptype timeout: float | None
        :return: nothing
        :rtype: None
        """
        for enabled in (False, True):
            await self.toggle_backend(backend_id, enabled, timeout=timeout)

    def comfy(self, **kwargs: Any) -> ComfyUIClient:
        """the ComfyUI SwarmUI proxies for raw graphs (``/ComfyBackendDirect``), not yet open.

        :param kwargs: ComfyUIClient's settings (``timeout``, ``client_id``, ``transport``...)
        :ptype kwargs: Any
        :return: the client, for ``async with``
        :rtype: ComfyUIClient
        """
        kwargs.setdefault("transport", self._transport)
        return ComfyUIClient(f"{self.base_url}/ComfyBackendDirect", **kwargs)
