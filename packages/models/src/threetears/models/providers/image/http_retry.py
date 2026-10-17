"""one request to a self-hosted image server, retried as far as it is safe to.

the clients for ComfyUI and SwarmUI share this. a read (``repeatable=True``) is
sent again when the server can't be reached, times out, drops the connection or
answers 5xx; a request that starts work (``repeatable=False``: queue a graph,
draw a picture, upload a file) is sent again only when it never left, because a
slow answer may mean the server took it and a second send would do it twice.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from threetears.observe import get_logger, retry_bounded

__all__ = ["NOT_SENT", "RetryPolicy", "send"]

log = get_logger(__name__)

#: failures that mean the request never reached the server: safe to send again whatever it does
NOT_SENT: tuple[type[Exception], ...] = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


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


async def send(
    http: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    repeatable: bool,
    policy: RetryPolicy,
    **kwargs: Any,
) -> httpx.Response:
    """send one request, again as far as ``repeatable`` allows. Returns the last answer.

    a 4xx or a 5xx comes back as the answer (the caller reads the server's own
    words in it); a 5xx to a repeatable request only once the attempts are spent.

    :param http: the open httpx client
    :ptype http: httpx.AsyncClient
    :param method: ``GET`` or ``POST``
    :ptype method: str
    :param url: the full URL
    :ptype url: str
    :param repeatable: True when sending it twice changes nothing (a read)
    :ptype repeatable: bool
    :param policy: attempts and pauses
    :ptype policy: RetryPolicy
    :param kwargs: what httpx's ``get``/``post`` take (``json``, ``params``, ``files``, ``timeout``)
    :ptype kwargs: Any
    :return: the server's answer
    :rtype: httpx.Response
    :raises httpx.TransportError: when the server can't be reached past the attempts
    """
    request = http.get if method == "GET" else http.post

    async def attempt() -> httpx.Response:
        response = await request(url, **kwargs)
        if repeatable and response.status_code >= 500:
            raise _ServerError(response)
        return response

    def on_retry(exc: Exception, number: int, pause: float) -> None:
        log.info(
            "image server request sent again",
            extra={"extra_data": {"url": url, "attempt": number, "pause": pause, "error": str(exc)}},
        )

    try:
        return await retry_bounded(
            attempt,
            retry_on=_any_failure if repeatable else _never_left,
            first_delay=policy.first_delay,
            max_delay=policy.max_delay,
            max_attempts=policy.attempts,
            on_retry=on_retry,
            sleep=policy.sleep,
        )
    except _ServerError as exc:
        return exc.response
