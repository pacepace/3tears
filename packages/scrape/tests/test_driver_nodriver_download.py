"""Unit tests for NodriverDownloadDriver.

All tests are fully mocked/in-memory -- no network calls, no sidecar
container, no real document parsing (``parse_document_bytes_to_html`` is
monkeypatched). The real, live sidecar proof lives in
tests/integration/test_scrape_nodriver_sidecar_live.py.
"""

from __future__ import annotations

import base64
import json as _json
from unittest.mock import AsyncMock

import httpx
import pytest

from threetears.scrape.driver import RenderedPage
from threetears.scrape.drivers.document import DocumentDriverError, ParsedDocumentHtml
from threetears.scrape.drivers.nodriver_download import NodriverDownloadDriver, NodriverDownloadError


class TestNodriverDownloadDriver:
    def test_name(self):
        driver = NodriverDownloadDriver("http://localhost:8088")
        assert driver.name == "nodriver_download"

    async def test_render_success_parses_response(self, monkeypatch):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = request.content
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "filename": "notice.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"%PDF-fake-bytes").decode("ascii"),
                    "timing_ms": 789.1,
                },
            )

        monkeypatch.setattr(
            "threetears.scrape.drivers.nodriver_download.parse_document_bytes_to_html",
            AsyncMock(return_value=ParsedDocumentHtml(html="<html>parsed</html>", was_ocr=False)),
        )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        page = await driver.render("https://workforcewv.org/notice.pdf", timeout=15.0)

        assert captured["url"] == "http://sidecar.test/v1/download"
        assert page.html == "<html>parsed</html>"
        assert page.status == 200
        assert page.final_url == "https://workforcewv.org/notice.pdf"
        assert page.was_ocr is False
        await client.aclose()

    async def test_render_sends_request_payload_shape(self, monkeypatch):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = _json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "filename": "notice.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"x").decode("ascii"),
                    "timing_ms": 1.0,
                },
            )

        monkeypatch.setattr(
            "threetears.scrape.drivers.nodriver_download.parse_document_bytes_to_html",
            AsyncMock(return_value=ParsedDocumentHtml(html="", was_ocr=False)),
        )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        await driver.render("https://workforcewv.org/notice.pdf", timeout=9.5)

        assert captured["payload"] == {"url": "https://workforcewv.org/notice.pdf", "timeout": 9.5}
        await client.aclose()

    async def test_render_decodes_base64_and_hands_off_to_parse_document_bytes_to_html(self, monkeypatch):
        mock_parse = AsyncMock(return_value=ParsedDocumentHtml(html="<html>ok</html>", was_ocr=False))

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "filename": "WARN-Notice.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"%PDF-1.7 real bytes").decode("ascii"),
                    "timing_ms": 1.0,
                },
            )

        monkeypatch.setattr("threetears.scrape.drivers.nodriver_download.parse_document_bytes_to_html", mock_parse)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        await driver.render("https://workforcewv.org/notice.pdf")

        mock_parse.assert_awaited_once_with(
            b"%PDF-1.7 real bytes",
            content_type="application/pdf",
            filename="WARN-Notice.pdf",
            ocr_config=None,
            force_images=False,
            merge_wrapped_table_rows=False,
        )
        await client.aclose()

    async def test_render_forwards_ocr_config_to_parse_step(self, monkeypatch):
        mock_parse = AsyncMock(return_value=ParsedDocumentHtml(html="", was_ocr=False))
        sentinel_ocr = object()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "filename": "notice.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"x").decode("ascii"),
                    "timing_ms": 1.0,
                },
            )

        monkeypatch.setattr("threetears.scrape.drivers.nodriver_download.parse_document_bytes_to_html", mock_parse)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client, ocr_config=sentinel_ocr)  # type: ignore[arg-type]

        await driver.render("https://workforcewv.org/notice.pdf")

        assert mock_parse.await_args.kwargs["ocr_config"] is sentinel_ocr
        await client.aclose()

    async def test_render_propagates_was_ocr_true_onto_rendered_page(self, monkeypatch):
        monkeypatch.setattr(
            "threetears.scrape.drivers.nodriver_download.parse_document_bytes_to_html",
            AsyncMock(return_value=ParsedDocumentHtml(html="<html>scanned</html>", was_ocr=True)),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "filename": "notice.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"x").decode("ascii"),
                    "timing_ms": 1.0,
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        page = await driver.render("https://workforcewv.org/notice.pdf")

        assert page.was_ocr is True
        await client.aclose()

    async def test_render_propagates_document_driver_error_unwrapped(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "filename": "notice.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"x").decode("ascii"),
                    "timing_ms": 1.0,
                },
            )

        async def failing_parse(*args, **kwargs):
            raise DocumentDriverError("parse_failed", "[Parsing failed: corrupt pdf]")

        monkeypatch.setattr("threetears.scrape.drivers.nodriver_download.parse_document_bytes_to_html", failing_parse)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        with pytest.raises(DocumentDriverError) as exc_info:
            await driver.render("https://workforcewv.org/notice.pdf")

        assert exc_info.value.code == "parse_failed"
        await client.aclose()

    async def test_render_raises_on_sidecar_error_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                504, json={"error": {"code": "download_timeout", "message": "download timed out after 30.0s"}}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        with pytest.raises(NodriverDownloadError) as exc_info:
            await driver.render("https://workforcewv.org/notice.pdf")

        assert exc_info.value.code == "download_timeout"
        assert exc_info.value.message == "download timed out after 30.0s"
        await client.aclose()

    async def test_render_raises_on_driver_crash_error_body(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, json={"error": {"code": "driver_crash", "message": "boom"}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        with pytest.raises(NodriverDownloadError) as exc_info:
            await driver.render("https://workforcewv.org/notice.pdf")

        assert exc_info.value.code == "driver_crash"
        assert exc_info.value.message == "boom"
        await client.aclose()

    async def test_render_raises_on_transport_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        with pytest.raises(NodriverDownloadError) as exc_info:
            await driver.render("https://workforcewv.org/notice.pdf")

        assert exc_info.value.code == "transport"
        await client.aclose()

    async def test_render_computes_timing_ms(self, monkeypatch):
        monkeypatch.setattr(
            "threetears.scrape.drivers.nodriver_download.parse_document_bytes_to_html",
            AsyncMock(return_value=ParsedDocumentHtml(html="", was_ocr=False)),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "filename": "notice.pdf",
                    "content_type": "application/pdf",
                    "content_base64": base64.b64encode(b"x").decode("ascii"),
                    "timing_ms": 999.0,
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = NodriverDownloadDriver("http://sidecar.test", client=client)

        page = await driver.render("https://workforcewv.org/notice.pdf")

        assert isinstance(page, RenderedPage)
        assert page.timing_ms >= 0.0
