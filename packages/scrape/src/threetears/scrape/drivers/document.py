"""DocumentDriver -- ``ScrapeDriver`` backend for PDF/DOCX/XLSX/TXT/Markdown/
LaTeX targets, reusing 3tears' own published document reader instead of
reinventing one.

**Design (2026-07-14, document reader integration):** ``threetears.agent.
tools.document.parse_document`` turns a document's bytes into clean
markdown -- and, critically, turns any table it finds (a PDF table, a DOCX
table, an XLSX sheet) into a GitHub-flavored markdown pipe-table. That's
close enough in shape to an HTML ``<table>`` that converting it into one
lets a document target flow through the *exact same* AI eval loop
(``extraction.py``/``eval_loop.py``, CSS-selector-based candidate
generation and structural validation) that every HTML-page target already
uses -- zero new extraction/validation code, per the user's own framing
("we go through that same scraper loop for improving and testing it").
``wait_for``/``nav_steps``/``capture_network`` are accepted for
``ScrapeDriver`` interface conformance but are no-ops here: this is a plain
HTTP GET of a static file, no browser, no JS, nothing to wait for or click.
"""

from __future__ import annotations

import html as html_lib
import re
import time
from pathlib import PurePosixPath
from urllib.parse import urlparse

import httpx
from threetears.agent.tools.document import OcrConfig, detect_mime_from_filename, parse_document
from threetears.observe import get_logger

from ..driver import NavStep, RenderedPage, ScrapeDriver

__all__ = ["DocumentDriver", "DocumentDriverError"]

log = get_logger(__name__)

#: GitHub-flavored markdown pipe-table shape -- exactly what parse_document's
#: own PDF/DOCX/XLSX table extractors emit (see that module's
#: _extract_pdf_tables/_docx_table_to_markdown/_parse_xlsx). Tolerant of
#: alignment colons in the separator row (`:---`, `---:`, `:---:`), which
#: CommonMark-flavored table generators commonly emit.
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


class DocumentDriverError(Exception):
    """Raised when a document fetch or parse fails.

    Mirrors ``NodriverSidecarError``/``CamoufoxDriverError``'s ``code``/
    ``message`` shape.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _merge_broken_pipe_rows(lines: list[str]) -> list[str]:
    """Rejoin a table row a source cell's own embedded newline split across
    physical lines.

    Live-found (2026-07-14, California's real WARN Excel file): a header
    cell like "Notice Date" word-wraps inside its own XLSX cell, and
    ``parse_document``'s markdown rendering preserves that literal newline
    -- splitting one logical pipe-table row (``"| County/Parish | Notice
    Date | ... |"``) across several physical lines (``"| County/Parish |
    Notice"``, then ``"Date | ..."``), none of which alone starts AND ends
    with ``|``. Undetected, the whole row (and the table it belongs to)
    silently degrades to plain ``<p>`` paragraphs instead of a real
    ``<table>`` -- wrong, not just incomplete. A line that starts with
    ``|`` but doesn't end with one is treated as unterminated and merged
    (space-joined) with following lines until it closes.
    """
    merged: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped.startswith("|") and not stripped.endswith("|"):
            accumulated = stripped
            i += 1
            while i < n and not accumulated.endswith("|"):
                accumulated = f"{accumulated} {lines[i].strip()}"
                i += 1
            merged.append(accumulated)
        else:
            merged.append(lines[i])
            i += 1
    return merged


def _split_table_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _table_to_html(header: list[str], rows: list[list[str]]) -> str:
    thead = "<tr>" + "".join(f"<th>{html_lib.escape(cell)}</th>" for cell in header) + "</tr>"
    tbody = "".join("<tr>" + "".join(f"<td>{html_lib.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table>{thead}{tbody}</table>"


def document_text_to_html(text: str) -> str:
    """Convert a parsed document's markdown text into synthetic HTML.

    Every GFM-style pipe-table block (a header row, a ``| --- | --- |``
    separator, then body rows) becomes a real ``<table>`` -- the eval loop's
    existing CSS-selector-based candidate generation/validation runs on this
    completely unmodified, exactly as it would on a real HTML page's table.
    Headings (``#`` through ``######``) become ``<h1>``-``<h6>``; every other
    non-blank line becomes a ``<p>`` -- not critical for extraction accuracy
    (documents with no table have nothing selector-shaped to extract from
    either way), just keeps the synthetic page well-formed.

    :param text: a ``DocumentResult.text`` value (clean markdown)
    :ptype text: str
    :return: a minimal ``<html><body>...</body></html>`` document
    :rtype: str
    """
    lines = _merge_broken_pipe_rows(text.split("\n"))
    parts: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _TABLE_ROW_RE.match(line) and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            header = _split_table_row(line)
            i += 2  # skip header + separator rows
            rows: list[list[str]] = []
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                rows.append(_split_table_row(lines[i]))
                i += 1
            parts.append(_table_to_html(header, rows))
            continue
        stripped = line.strip()
        if stripped:
            heading_match = _HEADING_RE.match(stripped)
            if heading_match:
                level = len(heading_match.group(1))
                parts.append(f"<h{level}>{html_lib.escape(heading_match.group(2))}</h{level}>")
            else:
                parts.append(f"<p>{html_lib.escape(stripped)}</p>")
        i += 1
    return "<html><body>" + "\n".join(parts) + "</body></html>"


class DocumentDriver(ScrapeDriver):
    """``ScrapeDriver`` backed by 3tears' document reader.

    Fetches *url*'s raw bytes over plain HTTP (no browser -- a document
    target is a static file, not a page needing JS rendering), parses it via
    ``threetears.agent.tools.document.parse_document``, and converts the
    result to synthetic HTML the rest of the pipeline already knows how to
    handle.
    """

    def __init__(self, *, client: httpx.AsyncClient | None = None, ocr_config: OcrConfig | None = None) -> None:
        """
        :param client: an already-constructed httpx client to reuse (test
            injection); a fresh one is created per call when omitted.
        :ptype client: httpx.AsyncClient | None
        :param ocr_config: OCR fallback config for scanned PDF pages, passed
            straight through to ``parse_document``.
        :ptype ocr_config: OcrConfig | None
        """
        self._client = client
        self._ocr_config = ocr_config

    @property
    def name(self) -> str:
        """Stable string key for this driver."""
        return "document"

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
        """Fetch and parse the document at *url*.

        :param url: the document's direct download URL
        :ptype url: str
        :param timeout: seconds to wait for the HTTP fetch before failing
        :ptype timeout: float
        :param wait_for: accepted for interface conformance; not applicable
            (no browser, nothing to wait for)
        :ptype wait_for: str | None
        :param capture_network: accepted for interface conformance; not
            applicable (a single plain HTTP GET, not a rendered page)
        :ptype capture_network: bool
        :param nav_steps: accepted for interface conformance; not applicable
            (no browser to drive)
        :ptype nav_steps: list[NavStep] | None
        :param results_path: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype results_path: str | None
        :param fragment_field: accepted for interface conformance; not
            applicable (only :class:`~threetears.scrape.drivers.api.ApiDriver` uses it)
        :ptype fragment_field: str | None
        :return: the parsed document's content as synthetic HTML
        :rtype: RenderedPage
        :raises DocumentDriverError: on a transport failure, a non-2xx HTTP
            response, or a document the parser couldn't handle
        """
        start = time.monotonic()
        client = self._client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        try:
            try:
                response = await client.get(url)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                log.warning("document driver transport failure", extra={"extra_data": {"url": url, "error": str(exc)}})
                raise DocumentDriverError("transport", str(exc)) from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            raise DocumentDriverError("fetch_failed", f"HTTP {response.status_code} fetching {url}")

        filename = PurePosixPath(urlparse(str(response.url)).path).name or "document"
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        # parse_document falls back to filename-extension detection itself
        # when the given mime_type isn't one it recognizes (its own
        # documented behavior) -- passing the server's declared content-type
        # verbatim, even a generic one, is enough; no need to duplicate that
        # fallback logic here.
        mime_type = content_type or (detect_mime_from_filename(filename) or "")

        result = await parse_document(response.content, mime_type, filename, ocr_config=self._ocr_config)
        if result.text.startswith("[Unsupported document type:") or result.text.startswith("[Parsing failed:"):
            raise DocumentDriverError("parse_failed", result.text)

        html = document_text_to_html(result.text)
        return RenderedPage(
            html=html,
            status=response.status_code,
            final_url=str(response.url),
            timing_ms=(time.monotonic() - start) * 1000,
        )
