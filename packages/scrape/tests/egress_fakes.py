"""The one stand-in for an egress exit, shared by every suite that asserts which exit a request took.

An exit's whole contract to an HTTP consumer is the transport it hands over, so this one hands
over a transport that records every request reaching it and answers each with a fixed response.
A request that reached it is a request that left by this exit -- the claim "the fetch went to
the proxy", observed one layer down, without a network call to a third party's address-echo
service and without reading the proxy URL back out of httpx's private connection pool.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
from threetears.core.config import DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS
from threetears.core.egress import EgressHealth

__all__ = ["FakeEgress"]


# parity-with: threetears.core.egress.EgressDriver
class FakeEgress:
    """An exit whose transport records every request that leaves by it."""

    def __init__(
        self,
        name: str = "recording",
        *,
        respond: Callable[[httpx.Request], httpx.Response] | None = None,
    ) -> None:
        """
        :param name: the exit's stable name
        :ptype name: str
        :param respond: builds the response for each request; a ``200`` with an empty JSON
            list when omitted
        :ptype respond: Callable[[httpx.Request], httpx.Response] | None
        """
        self.requests: list[httpx.Request] = []
        self._name = name
        answer = respond if respond is not None else (lambda _request: httpx.Response(200, json=[]))

        def _record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return answer(request)

        self._transport = httpx.MockTransport(_record)

    @property
    def name(self) -> str:
        """Stable identifier of this exit."""
        return self._name

    def httpx_transport(self) -> httpx.AsyncBaseTransport | None:
        """The recording transport every request through this exit is sent on."""
        return self._transport

    def browser_proxy_arg(self) -> str | None:
        """No opinion about a browser's proxy; this exit is for HTTP consumers."""
        return None

    async def health(self, *, timeout: float = DEFAULT_EGRESS_HEALTH_TIMEOUT_SECONDS) -> EgressHealth:
        """Always reachable: it is in-process."""
        return EgressHealth(reachable=True)
