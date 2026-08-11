"""SearchTransport -- the one injected transport seam (SR-N1, P9, D19, D21).

An adapter reaches an upstream through an injected transport, never a
client it opens itself. The protocol is *structural* so the two rules that
both refuse to bend can both hold: the family's sanctioned transport is
``threetears.core.http_client.TracedHttpClient``, and this leaf must not
import core (SR-L7). A thin host-side adapter over ``TracedHttpClient``
satisfies this shape for every host that has core (it lives with the host,
where core is already a hard dependency); hosts without core use the
``[standalone]`` extra's bare-httpx implementation -- the one module path
sanctioned by the D19 norm widening.

What an implementation owes, behind this one ``request`` method:

- **configurable timeout** -- never a constant (SR-G1); the per-call
  ``timeout_seconds`` override exists so a caller's remaining deadline can
  bound the call (SR-G2);
- **bounded retry with backoff** -- finite attempts, 5xx/connect/timeout
  retried, 4xx not (SR-G4); per-attempt accounting surfaces in
  :attr:`TransportResponse.attempts` so "budget follows the bill" can sit
  below the retry boundary (D4);
- **circuit-breaking** per upstream, so provider exhaustion
  short-circuits (SR-D3);
- **a per-call span** where telemetry exists, zero-cost where it does not
  (SR-I1);
- **egress selection** at construction (D20/SR-N2): which exit requests
  leave by is the transport's configuration, reported per response and as
  :attr:`SearchTransport.egress_name`;
- **the SSRF rulings** (D21): base URLs come from deployment config only,
  redirect policy and private-address guards live here, not per call site.

Ports are parameters, never payload (SR-L4): a transport is handed to
adapters and Call at construction and never appears in any wire type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import JsonValue

__all__ = ["SearchTransport", "TransportResponse"]


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """What one transport call yielded, with per-attempt accounting.

    A seam value, not a wire type: it crosses the transport/adapter seam
    in process and never rides a result payload, so it is a plain frozen
    dataclass rather than a :class:`ContractModel` (``body`` is raw bytes).
    """

    #: final HTTP status code.
    status_code: int
    #: response body bytes. Adapters parse; the transport never interprets.
    body: bytes
    #: the URL that actually answered, after any redirects the
    #: implementation's policy allowed (D21).
    final_url: str
    #: which egress the request left by (D20) -- ``direct`` is a named
    #: value, never an absence.
    egress: str
    #: wall-clock the call took, across all attempts, in seconds.
    elapsed_seconds: float
    #: attempts made, including the successful one (SR-G4). Spend's
    #: bill-following rule reads this: retried-but-unbilled attempts
    #: never count as billed calls (D4).
    attempts: int = 1
    #: response headers, case-insensitivity already normalised to lower-case
    #: keys by the implementation.
    headers: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class SearchTransport(Protocol):
    """Structural protocol for the injected outbound transport.

    Implementations: the host's thin adapter over
    ``threetears.core.http_client.TracedHttpClient`` (embedded or
    pod-resident hosts that have core), or this package's ``standalone``
    module (hosts that inject nothing). Adapters and Call depend on this
    shape only.
    """

    @property
    def egress_name(self) -> str:
        """Name of the egress this transport's requests leave by (D20).

        :return: the configured exit's name; ``direct`` for the default
            route
        :rtype: str
        """
        ...

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Perform one bounded, traced HTTP exchange.

        :param method: HTTP method (``GET``, ``POST``, ...)
        :ptype method: str
        :param url: absolute URL on a deployment-configured base -- a
            transport must never accept a caller-supplied base URL (D21)
        :ptype url: str
        :param headers: request headers, if any
        :ptype headers: Mapping[str, str] | None
        :param params: query parameters, if any
        :ptype params: Mapping[str, str] | None
        :param json_body: JSON request body, if any; mutually exclusive
            with other body kinds by construction of this seam
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: per-call override of the configured
            timeout, for deadline-derived bounds (SR-G2); None uses the
            implementation's configured value (SR-G1)
        :ptype timeout_seconds: float | None
        :return: the completed exchange with per-attempt accounting
        :rtype: TransportResponse
        :raises Exception: implementations surface transport-level
            failures however they choose; adapters map them onto the
            typed taxonomy in :mod:`threetears.search.contracts.errors`
            (SR-J1) -- the taxonomy mapping is adapter business, not
            transport business
        """
        ...
