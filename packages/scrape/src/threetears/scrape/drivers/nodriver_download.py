"""NodriverDownloadDriver -- httpx client for the sidecar's ``POST /v1/download``
endpoint, for a document behind a real bot challenge a plain HTTP client can't
pass.

**Design (scrape-task-04, 2026-07-15):** West Virginia's real WARN notice PDFs
sit behind an active Cloudflare managed challenge that blocks any plain HTTP
client. A real browser session passes it on its own -- no active
challenge-solving is involved here, genuine browser JS execution is enough --
but Chrome's own built-in PDF viewer then intercepts the navigation before
any bytes are otherwise reachable. The sidecar's ``/v1/download`` endpoint
(a real browser, forced-download behavior via
``plugins.always_open_pdf_externally`` + ``Browser.setDownloadBehavior``, an
isolated browser context per request) solves that; this driver is the thin
HTTP client for it, feeding the resulting bytes through the exact same
parse-and-convert step :class:`~threetears.scrape.drivers.document.
DocumentDriver` already has (:func:`~threetears.scrape.drivers.document.
parse_document_bytes_to_html`) rather than a second copy of that logic.

Talks to the sidecar exclusively over HTTP -- never imports ``nodriver``
itself, which stays inside the sidecar's own AGPL-3.0-licensed process.
Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

import base64
import time

import httpx
from threetears.agent.tools.document import OcrConfig
from threetears.observe import get_logger

from ..driver import NavStep, RenderedPage, ScrapeDriver
from .document import parse_document_bytes_to_html

__all__ = ["NodriverDownloadDriver", "NodriverDownloadError"]

log = get_logger(__name__)


class NodriverDownloadError(Exception):
    """Raised when the sidecar's ``/v1/download`` call fails.

    Mirrors ``NodriverSidecarError``'s ``code``/``message`` shape -- carries
    the sidecar's own error body on 4xx/5xx, or ``code="transport"`` when
    the failure never reached the sidecar.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class NodriverDownloadDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by the sidecar's browser-forced-download endpoint."""

    def __init__(
        self, base_url: str, *, client: httpx.AsyncClient | None = None, ocr_config: OcrConfig | None = None
    ) -> None:
        """
        :param base_url: the sidecar's base URL (e.g. ``"http://localhost:8088"``),
            with no trailing slash assumed either way
        :ptype base_url: str
        :param client: an already-constructed httpx client to reuse (test
            injection); a fresh one is created per call when omitted
        :ptype client: httpx.AsyncClient | None
        :param ocr_config: OCR fallback config for scanned PDF pages, passed
            straight through to ``parse_document``
        :ptype ocr_config: OcrConfig | None
        """
        self._base_url = base_url.rstrip("/")
        self._client = client
        self._ocr_config = ocr_config

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "nodriver_download"

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
        link_selector: str | None = None,
    ) -> RenderedPage:
        """Download *url*'s real file bytes through the sidecar, then parse and convert them.

        :param url: the document's direct download URL
        :ptype url: str
        :param timeout: seconds to wait for the download before failing
        :ptype timeout: float
        :param wait_for: accepted for interface conformance; not applicable
            (no CSS selector concept for a forced file download)
        :ptype wait_for: str | None
        :param capture_network: accepted for interface conformance; not applicable
        :ptype capture_network: bool
        :param nav_steps: accepted for interface conformance; not applicable
        :ptype nav_steps: list[NavStep] | None
        :param results_path: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype results_path: str | None
        :param fragment_field: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype fragment_field: str | None
        :param link_selector: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.multi_document.MultiDocumentDriver` uses it)
        :ptype link_selector: str | None
        :return: the parsed document's content as synthetic HTML
        :rtype: RenderedPage
        :raises NodriverDownloadError: on a sidecar-reported error (4xx/5xx
            with the documented error body, including a download that never
            completed) or a transport-level failure
        :raises DocumentDriverError: the download succeeded but
            ``parse_document`` couldn't handle the file -- propagated
            directly from :func:`~threetears.scrape.drivers.document.
            parse_document_bytes_to_html`, not wrapped, since it's already
            the right semantic type for "downloaded fine, couldn't parse"
        """
        start = time.monotonic()
        payload = {"url": url, "timeout": timeout}
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=timeout + 5.0)
        try:
            response = await client.post(f"{self._base_url}/v1/download", json=payload)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("sidecar download transport failure", extra={"extra_data": {"url": url, "error": str(exc)}})
            raise NodriverDownloadError("transport", str(exc)) from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            body = response.json()
            error = body.get("error", {})
            log.warning(
                "sidecar download failed",
                extra={"extra_data": {"url": url, "status": response.status_code, "code": error.get("code")}},
            )
            raise NodriverDownloadError(error.get("code", "unknown"), error.get("message", response.text))

        data = response.json()
        file_bytes = base64.b64decode(data["content_base64"])
        parsed = await parse_document_bytes_to_html(
            file_bytes, content_type=data["content_type"], filename=data["filename"], ocr_config=self._ocr_config
        )
        return RenderedPage(
            html=parsed.html,
            status=data["status"],
            final_url=url,
            timing_ms=(time.monotonic() - start) * 1000,
            was_ocr=parsed.was_ocr,
        )
