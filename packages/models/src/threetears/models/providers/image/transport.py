"""the one place the self-hosted image clients open a socket.

:class:`ComfyUIClient` and :class:`SwarmUIClient` talk to a server the operator configured (a GPU
box on the LAN) through :class:`ImageServerTransport`, which alone owns the ``httpx.AsyncClient``.
3tears-models may not depend on 3tears core, so core's traced HTTP client is out of reach; this
module carries the same obligations instead:

- **timeout:** every request has one (the transport's), and a call may set its own.
- **bounded retry with backoff:** a read (``repeatable=True``) is sent again when the server can't
  be reached, times out, drops the connection or answers 5xx; a request that starts work
  (``repeatable=False``: queue a graph, draw a picture, upload a file) is sent again only when it
  never left (:data:`NOT_SENT`), because a slow answer may mean the server took it and a second
  send would do it twice. Pauses double from ``first_delay`` up to ``max_delay``.
- **per-attempt accounting:** each attempt is logged with its method, path, attempt number,
  outcome and time taken; a request that fails for good is logged as an error with its attempts.
- **one origin:** a caller names a path, never a URL. The path is joined to the configured base
  URL, and a request whose URL left that origin (scheme, host, port) is refused. Redirects are
  not followed. Private and loopback addresses are allowed on purpose: these servers sit on the
  LAN by design, and the address comes from the operator's settings, never from request data.
- **bounded reads:** a body is read in chunks under a byte cap (a declared ``Content-Length`` over
  it is refused before reading), so an answer too large raises :class:`ResponseTooLargeError`
  instead of filling memory.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Awaitable, Callable, Mapping
from types import TracebackType
from typing import Any

import httpx

from threetears.observe import get_logger, retry_bounded

__all__ = [
    "DEFAULT_MAX_RESPONSE_BYTES",
    "NOT_SENT",
    "ImageServerTransport",
    "ResponseTooLargeError",
    "RetryPolicy",
]

log = get_logger(__name__)

#: failures that mean the request never reached the server: safe to send again whatever it does
NOT_SENT: tuple[type[Exception], ...] = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)

#: the largest body read by default: a big 16-bit PNG fits, a runaway answer does not
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024 * 1024

#: headers that describe the body as it came over the wire, not as it was read (decoded)
_WIRE_HEADERS = frozenset({"content-encoding", "content-length", "transfer-encoding"})


class RetryPolicy:
    """how many times a request is sent and how long to pause between sends.

    :param attempts: sends per request (at least 1)
    :ptype attempts: int
    :param first_delay: the first pause, doubling up to ``max_delay``
    :ptype first_delay: float
    :param max_delay: the longest pause
    :ptype max_delay: float
    :param sleep: how a pause is taken; ``None`` is ``asyncio.sleep`` (a test passes its own)
    :ptype sleep: Callable[[float], Awaitable[None]] | None
    """

    def __init__(
        self,
        *,
        attempts: int = 3,
        first_delay: float = 0.5,
        max_delay: float = 8.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """capture the policy.

        :param attempts: sends per request
        :ptype attempts: int
        :param first_delay: the first pause
        :ptype first_delay: float
        :param max_delay: the longest pause
        :ptype max_delay: float
        :param sleep: how a pause is taken, or ``None``
        :ptype sleep: Callable[[float], Awaitable[None]] | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``attempts`` is below 1
        """
        if attempts < 1:
            raise ValueError("attempts must be at least 1")
        self.attempts = attempts
        self.first_delay = first_delay
        self.max_delay = max(max_delay, first_delay)
        self.sleep = sleep


class ResponseTooLargeError(RuntimeError):
    """an image server's answer was larger than the transport reads.

    :param message: what was refused
    :ptype message: str
    :param limit: the cap, in bytes
    :ptype limit: int
    :param size: the bytes declared or read when it was refused
    :ptype size: int
    """

    def __init__(self, message: str, *, limit: int, size: int) -> None:
        """record the message, the cap and the size.

        :param message: what was refused
        :ptype message: str
        :param limit: the cap, in bytes
        :ptype limit: int
        :param size: the bytes declared or read
        :ptype size: int
        :return: nothing
        :rtype: None
        """
        super().__init__(message)
        self.limit = limit
        self.size = size


class _ServerError(Exception):
    """a 5xx answer to a request that is safe to send again."""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"answered {response.status_code}")
        self.response = response


def _any_failure(exc: Exception) -> bool:
    """whether a failed read is worth another send.

    :param exc: the failure
    :ptype exc: Exception
    :return: True for a transport failure or a 5xx answer
    :rtype: bool
    """
    return isinstance(exc, (httpx.TransportError, _ServerError))


def _never_left(exc: Exception) -> bool:
    """whether a failed request that starts work is worth another send.

    :param exc: the failure
    :ptype exc: Exception
    :return: True only when the request never reached the server
    :rtype: bool
    """
    return isinstance(exc, NOT_SENT)


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    """a URL's origin: scheme, host and port (the scheme's default when none is given).

    :param url: the URL
    :ptype url: httpx.URL
    :return: the origin
    :rtype: tuple[str, str, int | None]
    """
    port = url.port if url.port is not None else {"http": 80, "https": 443}.get(url.scheme)
    return url.scheme, url.host, port


class ImageServerTransport:
    """one self-hosted image server's HTTP connection, open for the length of an ``async with``.

    :param base_url: the server's address (``http://host:8188``), or a prefix that proxies it; set
        by the operator, never taken from request data
    :ptype base_url: str
    :param name: who uses it, for messages and logs (``ComfyUIClient``)
    :ptype name: str
    :param timeout: seconds a request may take unless the call sets its own
    :ptype timeout: float
    :param retry: sends per request and the pauses between them; ``None`` is three sends
    :ptype retry: RetryPolicy | None
    :param max_response_bytes: the largest body read; a larger one raises ResponseTooLargeError
    :ptype max_response_bytes: int
    :param transport: an httpx transport (a test's ``httpx.MockTransport``); ``None`` is the network
    :ptype transport: httpx.AsyncBaseTransport | None
    """

    def __init__(
        self,
        base_url: str,
        *,
        name: str,
        timeout: float,
        retry: RetryPolicy | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """capture the settings; the connection opens on ``async with``.

        :param base_url: the server's address
        :ptype base_url: str
        :param name: who uses it
        :ptype name: str
        :param timeout: seconds a request may take
        :ptype timeout: float
        :param retry: sends per request and pauses, or ``None``
        :ptype retry: RetryPolicy | None
        :param max_response_bytes: the largest body read
        :ptype max_response_bytes: int
        :param transport: an httpx transport, or ``None``
        :ptype transport: httpx.AsyncBaseTransport | None
        :return: nothing
        :rtype: None
        :raises ValueError: when the base URL isn't http(s) with a host, or the cap is below 1
        """
        base = httpx.URL(base_url.rstrip("/"))
        if base.scheme not in {"http", "https"} or not base.host:
            raise ValueError(f"{name} needs an http(s) address with a host, got {base_url!r}")
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be at least 1")
        self.base_url = str(base)
        self.name = name
        self.timeout = timeout
        self.retry = retry or RetryPolicy()
        self.max_response_bytes = max_response_bytes
        self._origin = _origin(base)
        self._transport = transport
        self._stack: contextlib.AsyncExitStack | None = None
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> ImageServerTransport:
        """open the connection.

        :return: self
        :rtype: ImageServerTransport
        """
        stack = contextlib.AsyncExitStack()
        self._client = await stack.enter_async_context(
            httpx.AsyncClient(timeout=self.timeout, transport=self._transport, follow_redirects=False)
        )
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
        stack, self._stack, self._client = self._stack, None, None
        if stack is not None:
            await stack.aclose()

    def url_for(self, path: str) -> str:
        """the full URL of a path under the base URL; a full URL in its place is refused.

        :param path: the path, starting with ``/`` (``/history/p1``)
        :ptype path: str
        :return: the URL
        :rtype: str
        :raises ValueError: when ``path`` is not a plain path, or the result leaves the base's origin
        """
        if not path.startswith("/") or path.startswith("//") or "://" in path or "\\" in path:
            raise ValueError(f"{self.name} takes a path under its base URL, not {path!r}")
        url = httpx.URL(f"{self.base_url}{path}")
        if _origin(url) != self._origin:
            raise ValueError(f"{self.name} refuses {path!r}: it leaves {self.base_url}")
        return str(url)

    async def request(
        self,
        method: str,
        path: str,
        *,
        repeatable: bool,
        timeout: float | None = None,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        files: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        """send one request, again as far as ``repeatable`` allows. Returns the last answer, read.

        a 4xx or a 5xx comes back as the answer (the caller reads the server's own words in it);
        a 5xx to a repeatable request only once the attempts are spent.

        :param method: ``GET`` or ``POST``
        :ptype method: str
        :param path: the path under the base URL
        :ptype path: str
        :param repeatable: True when sending it twice changes nothing (a read)
        :ptype repeatable: bool
        :param timeout: this request's timeout in seconds; ``None`` is the transport's
        :ptype timeout: float | None
        :param params: the query string
        :ptype params: Mapping[str, Any] | None
        :param json: a JSON body
        :ptype json: Any
        :param files: a multipart upload, as httpx takes it
        :ptype files: Mapping[str, Any] | None
        :return: the server's answer, its body read
        :rtype: httpx.Response
        :raises RuntimeError: when used outside ``async with``
        :raises ValueError: when ``path`` is not a path under the base URL
        :raises ResponseTooLargeError: when the answer's body is over the cap
        :raises httpx.TransportError: when the server can't be reached past the attempts
        """
        client = self._client
        if client is None:
            raise RuntimeError(f"{self.name} is used inside `async with` only")
        url = self.url_for(path)
        build: dict[str, Any] = {"params": params, "json": json, "files": files}
        if timeout is not None:
            build["timeout"] = timeout
        policy = self.retry
        attempts = 0

        def account(outcome: str, started: float) -> None:
            log.debug(
                "image server request attempt",
                extra={
                    "extra_data": {
                        "server": self.name,
                        "method": method,
                        "path": path,
                        "attempt": attempts,
                        "outcome": outcome,
                        "seconds": round(time.monotonic() - started, 3),
                    }
                },
            )

        async def attempt() -> httpx.Response:
            nonlocal attempts
            attempts += 1
            started = time.monotonic()
            try:
                request = client.build_request(method, url, **build)
                if _origin(request.url) != self._origin:
                    raise ValueError(f"{self.name} refuses a request that leaves {self.base_url}")
                response = await self._send_capped(client, request)
            except Exception as exc:
                account(type(exc).__name__, started)
                raise
            account(str(response.status_code), started)
            if repeatable and response.status_code >= 500:
                raise _ServerError(response)
            return response

        def on_retry(exc: Exception, number: int, pause: float) -> None:
            log.info(
                "image server request sent again",
                extra={
                    "extra_data": {
                        "server": self.name,
                        "method": method,
                        "path": path,
                        "attempt": number,
                        "pause": pause,
                        "error": str(exc),
                    }
                },
            )

        retry_on = _any_failure if repeatable else _never_left
        try:
            return await retry_bounded(
                attempt,
                retry_on=retry_on,
                first_delay=policy.first_delay,
                max_delay=policy.max_delay,
                max_attempts=policy.attempts,
                on_retry=on_retry,
                sleep=policy.sleep,
            )
        except _ServerError as exc:
            self._gave_up(method, path, attempts, str(exc), spent=True)
            return exc.response
        except Exception as exc:
            self._gave_up(method, path, attempts, f"{type(exc).__name__}: {exc}", spent=retry_on(exc))
            raise

    def _gave_up(self, method: str, path: str, attempts: int, error: str, *, spent: bool) -> None:
        """log a request that failed for good.

        :param method: its method
        :ptype method: str
        :param path: its path
        :ptype path: str
        :param attempts: how many times it was sent
        :ptype attempts: int
        :param error: the last failure
        :ptype error: str
        :param spent: True when the attempts ran out, False when the failure was not one to retry
        :ptype spent: bool
        :return: nothing
        :rtype: None
        """
        log.error(
            "image server request failed after its attempts" if spent else "image server request failed",
            extra={
                "extra_data": {
                    "server": self.name,
                    "method": method,
                    "path": path,
                    "attempts": attempts,
                    "error": error,
                }
            },
        )

    async def _send_capped(self, client: httpx.AsyncClient, request: httpx.Request) -> httpx.Response:
        """send a request and read its body under the cap. Returns an answer whose body is read.

        :param client: the open client
        :ptype client: httpx.AsyncClient
        :param request: the request
        :ptype request: httpx.Request
        :return: the answer, its body in memory and decoded
        :rtype: httpx.Response
        :raises ResponseTooLargeError: when the body is over the cap
        """
        cap = self.max_response_bytes
        streamed = await client.send(request, stream=True)
        try:
            declared = streamed.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > cap:
                raise self._too_large(request, int(declared))
            chunks: list[bytes] = []
            total = 0
            async for chunk in streamed.aiter_bytes():
                total += len(chunk)
                if total > cap:
                    raise self._too_large(request, total)
                chunks.append(chunk)
        finally:
            await streamed.aclose()
        headers = [(k, v) for k, v in streamed.headers.multi_items() if k.lower() not in _WIRE_HEADERS]
        return httpx.Response(streamed.status_code, headers=headers, content=b"".join(chunks), request=request)

    def _too_large(self, request: httpx.Request, size: int) -> ResponseTooLargeError:
        """the refusal of a body over the cap.

        :param request: the request answered
        :ptype request: httpx.Request
        :param size: the bytes declared or read so far
        :ptype size: int
        :return: the error to raise
        :rtype: ResponseTooLargeError
        """
        return ResponseTooLargeError(
            f"{self.name} {request.method} {request.url.path} answered more than "
            f"{self.max_response_bytes} bytes (at least {size})",
            limit=self.max_response_bytes,
            size=size,
        )
