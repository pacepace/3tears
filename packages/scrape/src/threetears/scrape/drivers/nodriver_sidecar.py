"""NodriverSidecarDriver — httpx client for the ``services/nodriver-sidecar`` container.

Talks to the sidecar exclusively over HTTP (``POST /v1/render``, per
``scrape-api-contract.md``) — never imports ``nodriver`` itself, which stays
inside the sidecar's own AGPL-3.0-licensed process. This module has zero
faidh imports (see ``scrape/__init__.py``); the sidecar's base URL is a
plain constructor argument, resolved by faidh-side wiring code.
"""

from __future__ import annotations

from dataclasses import asdict

import httpx
from threetears.observe import get_logger

from ..driver import NavStep, NetworkCall, RenderedPage, ScrapeDriver

__all__ = ["NodriverSidecarDriver", "NodriverSidecarError"]

log = get_logger(__name__)


class NodriverSidecarError(Exception):
    """Raised when the sidecar's ``/v1/render`` call fails.

    Carries the error-model fields the sidecar returns on 4xx/5xx
    (``{"error": {"code": str, "message": str}}``), or ``code="transport"``
    when the failure never reached the sidecar (connection refused, DNS
    failure, client-side timeout).
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class NodriverSidecarDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by the nodriver sidecar's HTTP API.

    Chromium-direct-CDP rendering via a separate, AGPL-3.0-licensed
    container — this class only ever speaks plain HTTP to it, never
    imports ``nodriver`` as a library.
    """

    def __init__(self, base_url: str, *, client: httpx.AsyncClient | None = None) -> None:
        """
        :param base_url: the sidecar's base URL (e.g. ``"http://localhost:8088"``),
            with no trailing slash assumed either way.
        :ptype base_url: str
        :param client: an already-constructed httpx client to reuse (e.g. for
            test injection); a fresh one is created per call when omitted.
        :ptype client: httpx.AsyncClient | None
        """
        self._base_url = base_url.rstrip("/")
        self._client = client

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "nodriver"

    async def render(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        wait_for: str | None = None,
        capture_network: bool = False,
        nav_steps: list[NavStep] | None = None,
        results_path: str | None = None,
        fragment_field: str | None = None,
    ) -> RenderedPage:
        """Render *url* through the sidecar's ``POST /v1/render`` endpoint.

        :param url: the page to fetch
        :ptype url: str
        :param timeout: seconds to wait for the render before failing
        :ptype timeout: float
        :param wait_for: optional CSS selector to wait for before the
            sidecar considers the page rendered
        :ptype wait_for: str | None
        :param capture_network: when true, ask the sidecar to capture
            XHR/fetch calls with JSON-shaped response bodies
        :ptype capture_network: bool
        :param nav_steps: ordered browser actions the sidecar performs
            before *wait_for*'s settle-wait
        :ptype nav_steps: list[NavStep] | None
        :param results_path: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype results_path: str | None
        :param fragment_field: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype fragment_field: str | None
        :return: the rendered page
        :rtype: RenderedPage
        :raises NodriverSidecarError: on a sidecar-reported error (4xx/5xx
            with the documented error body, including a nav step that
            couldn't be executed) or a transport-level failure (connection
            refused, DNS failure, client-side timeout)
        """
        payload = {
            "url": url,
            "timeout": timeout,
            "wait_for": wait_for,
            "capture_network": capture_network,
            "nav_steps": [asdict(step) for step in nav_steps] if nav_steps else None,
        }
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=timeout + 5.0)
        try:
            response = await client.post(f"{self._base_url}/v1/render", json=payload)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("sidecar transport failure", extra={"extra_data": {"url": url, "error": str(exc)}})
            raise NodriverSidecarError("transport", str(exc)) from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            body = response.json()
            error = body.get("error", {})
            log.warning(
                "sidecar render failed",
                extra={"extra_data": {"url": url, "status": response.status_code, "code": error.get("code")}},
            )
            raise NodriverSidecarError(
                error.get("code", "unknown"),
                error.get("message", response.text),
            )

        data = response.json()
        return RenderedPage(
            html=data["html"],
            status=data["status"],
            final_url=data["final_url"],
            timing_ms=data["timing_ms"],
            network_calls=[NetworkCall(**call) for call in data.get("network_calls", [])],
        )
