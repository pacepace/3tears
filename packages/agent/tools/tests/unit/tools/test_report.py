"""Tests for the general report tool (render -> object store -> handle).

Covers the first real producer: rendering Markdown (inline or compiled from
findings), streaming it to the object store on the per-call scope, returning
the handle in ``ToolResult.metadata``, and minting a best-effort presigned
delivery URL. Fail-closed paths (bad input / no store / PDF toolchain absent)
and the presign-degradation path are exercised explicitly.
"""

from __future__ import annotations

from typing import AsyncIterator
from uuid import UUID

import pytest

from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.tools.reports import PandocNotFoundError
from threetears.media.contracts import OBJECT_HANDLE_METADATA_KEY, ObjectListing

from threetears.agent.tools import report as report_module
from threetears.agent.tools.report import ReportTool, _strip_emoji

_CUSTOMER = UUID("06a41d51-a6d5-7824-8000-29ab66754fc0")
_CONVERSATION = UUID("019f1900-0000-7000-8000-0000000000cc")

_PRESIGN_URL = "https://minio.example/aibots-objects/key?sig=abc"


# parity-with: threetears.media.contracts.ObjectStore
class _FakeStore:
    """Records the single ``put`` and hands back a fixed presigned URL."""

    def __init__(self, presign_raises: bool = False) -> None:
        self.puts: list[dict[str, object]] = []
        self.presigned: list[dict[str, object]] = []
        self._presign_raises = presign_raises

    async def put(
        self,
        key: str,
        body: AsyncIterator[bytes],
        *,
        content_type: str,
        size: int | None = None,
    ) -> None:
        collected = b"".join([chunk async for chunk in body])
        self.puts.append({"key": key, "body": collected, "content_type": content_type, "size": size})

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        if self._presign_raises:
            raise RuntimeError("presign backend unavailable")
        self.presigned.append({"key": key, "expires_in": expires_in})
        return _PRESIGN_URL

    def open_read(self, key: str) -> AsyncIterator[bytes]:  # pragma: no cover
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def delete_many(self, keys: list[str]) -> None:  # pragma: no cover
        raise NotImplementedError

    def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:  # pragma: no cover
        raise NotImplementedError

    def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:  # pragma: no cover
        raise NotImplementedError


def _scope(store: object | None) -> ToolCallScope:
    """Build a call scope carrying ``store`` + a customer/conversation context."""
    context = CallContext(customer_id=_CUSTOMER, conversation_id=_CONVERSATION)
    return ToolCallScope(context=context, object_store=store)  # type: ignore[arg-type]


_FINDINGS = [
    {
        "severity": "high",
        "title": "Outdated TLS",
        "affected_target": "example.com",
        "description": "The server negotiates TLS 1.0.",
        "remediation": "Disable TLS < 1.2.",
    }
]


def test_default_and_custom_name() -> None:
    """The registered name is constructor-parameterised for the serving pod."""
    assert ReportTool().mcp_name() == "threetears.report"
    assert ReportTool(name="pentest.report").mcp_name() == "pentest.report"


async def test_content_mode_markdown_produces_and_delivers() -> None:
    """Inline Markdown is streamed to the store and a delivery URL returns."""
    store = _FakeStore()
    tool = ReportTool(name="pentest.report")
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(content="# Report\n\nBody text.", report_format="markdown", title="My Report")

    assert result.success is True
    assert result.metadata is not None
    assert OBJECT_HANDLE_METADATA_KEY in result.metadata
    assert result.metadata["delivery_url"] == _PRESIGN_URL
    # the bytes streamed are exactly the inline Markdown, under a reports key.
    assert len(store.puts) == 1
    put = store.puts[0]
    assert put["body"] == b"# Report\n\nBody text."
    assert put["content_type"] == "text/markdown"
    assert str(put["key"]).startswith(f"{_CUSTOMER}/conversation-{_CONVERSATION}/reports/")
    assert str(put["key"]).endswith("/my-report-" + _today() + ".md")
    handle = result.metadata[OBJECT_HANDLE_METADATA_KEY]
    assert handle["category"] == "reports"
    assert handle["mime_type"] == "text/markdown"


async def test_findings_mode_compiles_markdown() -> None:
    """Structured findings are compiled to Markdown then streamed."""
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(findings=_FINDINGS, title="Assessment")

    assert result.success is True
    body = store.puts[0]["body"]
    assert isinstance(body, bytes)
    text = body.decode("utf-8")
    # the compiler produced a real report body from the findings.
    assert "# Assessment" in text
    assert "Executive Summary" in text
    assert "Outdated TLS" in text


async def test_requires_content_or_findings() -> None:
    """Neither content nor findings -> fail closed, nothing stored."""
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(report_format="markdown", title="Empty")
    assert result.success is False
    assert "content" in result.error and "findings" in result.error
    assert store.puts == []


async def test_malformed_findings_fail_closed() -> None:
    """Findings missing compiler-required keys refuse cleanly, nothing stored."""
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(findings=[{"title": "incomplete"}], report_format="markdown")
    assert result.success is False
    assert "could not compile findings" in result.error
    assert store.puts == []


def test_strip_emoji_preserves_non_latin() -> None:
    """Emoji are stripped for pdflatex, but CJK / Hangul text is preserved."""
    out = _strip_emoji("Findings 中文 한글 😀 done ✅")
    assert "😀" not in out
    assert "✅" not in out
    assert "中文" in out
    assert "한글" in out


async def test_unknown_format_fails_closed() -> None:
    """An unsupported report_format is refused before any rendering."""
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(content="# x", report_format="html")
    assert result.success is False
    assert "report_format" in result.error
    assert store.puts == []


async def test_no_store_fails_closed() -> None:
    """With no object store on the scope, the produce seam refuses."""
    tool = ReportTool()
    async with enter_call_scope(_scope(None)):
        result = await tool.execute(content="# x\n\ny", report_format="markdown")
    assert result.success is False
    assert "cannot store the report" in result.error


async def test_pdf_without_pandoc_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF requested but the pandoc toolchain is absent -> clean refusal."""

    def _boom(_markdown: str) -> bytes:
        raise PandocNotFoundError("pandoc not found")

    monkeypatch.setattr(report_module, "_render_markdown_to_pdf_bytes", _boom)
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(findings=_FINDINGS, report_format="pdf")
    assert result.success is False
    assert "pandoc" in result.error.lower()
    assert "markdown" in result.error.lower()
    assert store.puts == []


async def test_pdf_render_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero pandoc exit surfaces as a clean refusal, nothing stored."""

    def _boom(_markdown: str) -> bytes:
        raise RuntimeError("Pandoc failed (exit 43): boom")

    monkeypatch.setattr(report_module, "_render_markdown_to_pdf_bytes", _boom)
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(content="# x\n\ny", report_format="pdf")
    assert result.success is False
    assert "PDF rendering failed" in result.error
    assert store.puts == []


async def test_pdf_success_streams_pdf_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the toolchain is present, PDF bytes stream under a .pdf key."""
    monkeypatch.setattr(report_module, "_render_markdown_to_pdf_bytes", lambda _md: b"%PDF-1.7 fake")
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(findings=_FINDINGS, report_format="pdf", title="Pdf Report")
    assert result.success is True
    put = store.puts[0]
    assert put["body"] == b"%PDF-1.7 fake"
    assert put["content_type"] == "application/pdf"
    assert str(put["key"]).endswith("/pdf-report-" + _today() + ".pdf")


async def test_presign_failure_still_succeeds() -> None:
    """A presign error degrades to no URL but keeps the report a success."""
    store = _FakeStore(presign_raises=True)
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(content="# x\n\ny", report_format="markdown")
    assert result.success is True
    assert result.metadata is not None
    # the handle is still present so the catalog seam fires ...
    assert OBJECT_HANDLE_METADATA_KEY in result.metadata
    # ... but no delivery URL when presign failed.
    assert "delivery_url" not in result.metadata
    assert "could not be generated" in result.content


async def test_presign_ttl_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The delivery-URL TTL is read from REPORT_PRESIGN_TTL_SECONDS."""
    monkeypatch.setenv("REPORT_PRESIGN_TTL_SECONDS", "120")
    store = _FakeStore()
    tool = ReportTool()
    async with enter_call_scope(_scope(store)):
        result = await tool.execute(content="# x\n\ny", report_format="markdown")
    assert result.success is True
    assert store.presigned[0]["expires_in"] == 120
    assert "120s" in result.content


def test_schema_timeout_honors_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool advertises a render+store timeout, overridable via env."""
    assert ReportTool().mcp_schema().timeout_seconds == 120.0
    monkeypatch.setenv("REPORT_TIMEOUT_SECONDS", "45")
    assert ReportTool().mcp_schema().timeout_seconds == 45.0


def _today() -> str:
    """The UTC date string the tool stamps into filenames."""
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%d")
