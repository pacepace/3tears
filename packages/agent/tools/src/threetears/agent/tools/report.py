"""General report tool: render findings -> object store -> delivered handle.

The first real object-store *producer*. A consuming pod registers this tool;
the agent (or a graph step) calls it with either pre-compiled Markdown
(``content``) or structured ``findings``. The tool renders the report, streams
the bytes to the pod's object store via the produce seam
(:func:`~threetears.agent.tools.produce.stream_result_to_object_store`), and
returns a small :class:`~threetears.media.contracts.ObjectHandle` in
``ToolResult.metadata`` -- the rendered bytes never cross NATS. The agent's
catalog seam persists the handle into the hub-owned object catalog.

Delivery (design 5): the tool also mints a presigned GET URL for the produced
object so a human can download it without the bytes passing through the agent.
Delivery is best-effort -- a presign error never fails an already-produced,
cataloged report; the handle still returns and the object stays retrievable by
its id.

The tool is *general*: the render engine (:mod:`threetears.agent.tools.reports`)
and the produce seam are framework primitives, and this tool only wires them.
Its registered name is constructor-parameterised so a consuming pod names it in
its own namespace (e.g. a pentest pod registers ``ReportTool(name="pentest.report")``).

PDF output requires ``pandoc`` + ``pdflatex`` + ``mermaid-filter`` on the pod
image's PATH; where they are absent, PDF fails closed with a clear message and
Markdown (which needs no toolchain) remains available.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from datetime import UTC, datetime
from typing import Any, AsyncIterator

from threetears.media.contracts import ObjectHandle
from threetears.observe import get_logger

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.produce import (
    ProduceObjectError,
    object_handle_result_metadata,
    stream_result_to_object_store,
)
from threetears.agent.tools.reports import (
    MarkdownCompiler,
    PandocNotFoundError,
    PdfRenderer,
    ReportMetadata,
)

__all__ = ["ReportTool"]

_log = get_logger(__name__)

#: object-store category for produced reports (scope-first key scheme, design 8).
_CATEGORY = "reports"

#: default registered name; a consuming pod overrides it for its namespace.
_DEFAULT_NAME = "threetears.report"

#: presigned delivery-URL validity; override with REPORT_PRESIGN_TTL_SECONDS.
_DEFAULT_PRESIGN_TTL = 3600
_PRESIGN_TTL_ENV = "REPORT_PRESIGN_TTL_SECONDS"

#: byte chunk size for streaming the rendered report to the store.
_STREAM_CHUNK = 65536

#: expected max render+store time; PDF (pandoc/pdflatex) is the slow path.
#: override with REPORT_TIMEOUT_SECONDS.
_DEFAULT_TIMEOUT_SECONDS = 120.0
_TIMEOUT_ENV = "REPORT_TIMEOUT_SECONDS"

_VALID_FORMATS = ("markdown", "pdf")

# Emoji codepoints pdflatex cannot render; stripped before PDF rendering.
# Only genuine emoji blocks are listed -- a broad catch-all range would also
# delete CJK / Hangul / kana, silently corrupting non-Latin reports.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # regional indicator symbols (flags)
    "\U00002600-\U000026ff"  # miscellaneous symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U0001fa00-\U0001fa6f"  # symbols & pictographs extended-A (chess)
    "\U0001fa70-\U0001faff"  # symbols & pictographs extended-A
    "]+",
    flags=re.UNICODE,
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "description": (
                "Pre-rendered Markdown report body. Provide this OR 'findings'. "
                "Use when you have already compiled the report from tool results "
                "in context. Do NOT include emojis when requesting PDF output."
            ),
        },
        "findings": {
            "type": "array",
            "items": {"type": "object"},
            "description": (
                "Structured findings to compile into a report. Provide this OR "
                "'content'. Each finding: severity (required), title, "
                "affected_target, description, cvss_score, remediation, cve_ids, "
                "affected_component."
            ),
        },
        "report_format": {
            "type": "string",
            "enum": list(_VALID_FORMATS),
            "default": "markdown",
            "description": (
                "Output format. 'markdown' is always available; 'pdf' requires "
                "the pod image to ship pandoc + pdflatex and refuses cleanly "
                "otherwise."
            ),
        },
        "title": {
            "type": "string",
            "description": "Report title; used in the header and the object filename.",
        },
        "client_name": {
            "type": "string",
            "description": "Client / organization name for the report header.",
        },
        "target_scope": {
            "type": "string",
            "description": "Assessed target scope; derived from findings when omitted.",
        },
        "assessor_name": {
            "type": "string",
            "description": "Assessor or team name for the report header.",
        },
        "chains": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Optional vulnerability chains (name, escalated_severity, attack_flow_mermaid).",
        },
    },
    "required": [],
}


def _timeout_seconds() -> float:
    """Resolve the tool's expected-max render+store time from the environment.

    :return: timeout in seconds (falls back to the default on unset/invalid)
    :rtype: float
    """
    raw = os.environ.get(_TIMEOUT_ENV)
    if raw:
        try:
            value = float(raw)
        except ValueError:
            _log.warning("invalid %s=%r; using default", _TIMEOUT_ENV, raw)
        else:
            if value > 0:
                return value
    return _DEFAULT_TIMEOUT_SECONDS


def _presign_ttl() -> int:
    """Resolve the presigned-URL validity (seconds) from the environment.

    :return: TTL in seconds (falls back to the default on unset/invalid)
    :rtype: int
    """
    raw = os.environ.get(_PRESIGN_TTL_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            _log.warning("invalid %s=%r; using default", _PRESIGN_TTL_ENV, raw)
        else:
            if value > 0:
                return value
    return _DEFAULT_PRESIGN_TTL


def _strip_emoji(text: str) -> str:
    """Remove emoji characters that pdflatex cannot render.

    :param text: text potentially containing emojis
    :ptype text: str
    :return: text with emojis removed
    :rtype: str
    """
    return _EMOJI_PATTERN.sub("", text)


def _render_markdown_to_pdf_bytes(markdown: str) -> bytes:
    """Render Markdown to PDF bytes via the Pandoc renderer.

    Strips emoji (pdflatex cannot render them), renders to a temp file, and
    reads the bytes back. Synchronous + blocking (a pandoc subprocess); call
    it via :func:`asyncio.to_thread`.

    :param markdown: Markdown report body
    :ptype markdown: str
    :return: rendered PDF bytes
    :rtype: bytes
    :raises PandocNotFoundError: when pandoc is not installed on the pod
    :raises RuntimeError: when pandoc exits non-zero
    """
    cleaned = _strip_emoji(markdown)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        output_path = tmp.name
    try:
        PdfRenderer().render(markdown_content=cleaned, output_path=output_path)
        with open(output_path, "rb") as handle:
            return handle.read()
    finally:
        os.unlink(output_path)


def _build_filename(title: str | None, extension: str) -> str:
    """Build a descriptive, dated filename from a report title.

    Slugifies the title and appends the UTC date; falls back to
    ``report-<date>.<ext>`` when no title is given.

    :param title: report title (may be None)
    :ptype title: str | None
    :param extension: file extension without the dot (e.g. ``md`` / ``pdf``)
    :ptype extension: str
    :return: filename like ``whatweb-scan-example-com-2026-07-01.pdf``
    :rtype: str
    """
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    if not title:
        return f"report-{date_str}.{extension}"
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if len(slug) > 80:
        slug = slug[:80].rstrip("-")
    if not slug:
        return f"report-{date_str}.{extension}"
    return f"{slug}-{date_str}.{extension}"


def _extract_targets(findings: list[dict[str, Any]]) -> str:
    """Derive a target-scope string from findings' affected targets.

    :param findings: list of finding dicts
    :ptype findings: list[dict[str, Any]]
    :return: comma-separated unique targets, or ``Unknown`` when none
    :rtype: str
    """
    targets = {str(f.get("affected_target", "")) for f in findings if f.get("affected_target")}
    return ", ".join(sorted(targets)) if targets else "Unknown"


def _summarize(title: str | None, findings: Any, report_format: str) -> str:
    """Build the short handle summary shown to the model without the bytes.

    :param title: report title (may be None)
    :ptype title: str | None
    :param findings: findings list when compiled from structured input
    :ptype findings: Any
    :param report_format: rendered format label
    :ptype report_format: str
    :return: a short human/model-facing summary
    :rtype: str
    """
    base = title or "Security report"
    if isinstance(findings, list) and findings:
        return f"{base}: {len(findings)} findings ({report_format})"
    return f"{base} ({report_format})"


async def _aiter(data: bytes) -> AsyncIterator[bytes]:
    """Yield ``data`` as a chunked async byte stream for the store.

    :param data: the rendered report bytes
    :ptype data: bytes
    :return: async iterator over the bytes in chunks
    :rtype: AsyncIterator[bytes]
    """
    for i in range(0, len(data), _STREAM_CHUNK):
        yield data[i : i + _STREAM_CHUNK]


async def _presign_delivery_url(s3_key: str, ttl: int) -> str | None:
    """Best-effort presigned GET URL for the produced report.

    Reads the object store off the current call scope and mints a presigned
    URL. Returns ``None`` (never raises) when there is no store or the store
    cannot presign -- delivery is a convenience on top of an already-produced,
    cataloged object and must not fail the produce.

    :param s3_key: the produced object's key
    :ptype s3_key: str
    :param ttl: URL validity in seconds
    :ptype ttl: int
    :return: a presigned GET URL, or None when unavailable
    :rtype: str | None
    """
    scope = current_scope()
    if scope is None or scope.object_store is None:
        return None
    try:
        return await scope.object_store.presigned_get_url(s3_key, expires_in=ttl)
    except Exception:  # prawduct:allow prawduct/broad-except -- delivery is best-effort on an already-produced+cataloged object; the producer stays impl-free (media-contracts only) so it cannot import the store impl's concrete exception types. Logged with context below; the report still returns successfully.
        _log.warning(
            "could not presign a delivery URL for a produced report; returning the handle without a URL",
            exc_info=True,
        )
        return None


def _result_message(handle: ObjectHandle, filename: str, report_format: str, url: str | None, ttl: int) -> str:
    """Build the model/human-facing result content for a produced report.

    :param handle: the produced object's handle
    :ptype handle: ObjectHandle
    :param filename: the report filename
    :ptype filename: str
    :param report_format: rendered format label
    :ptype report_format: str
    :param url: presigned delivery URL, or None when unavailable
    :ptype url: str | None
    :param ttl: presigned-URL validity in seconds
    :ptype ttl: int
    :return: the result content string
    :rtype: str
    """
    parts = [
        f"Report generated: {filename} ({handle.size_bytes} bytes, {report_format}).",
        f"Stored as object {handle.object_id} (category={_CATEGORY}).",
    ]
    if url is not None:
        parts.append(f"Download link (expires in {ttl}s): {url}")
    else:
        parts.append(
            "A download link could not be generated; the report is stored and "
            "cataloged and can be retrieved by its object id."
        )
    return " ".join(parts)


class ReportTool(TearsTool):
    """Render a report, store it in the object store, and return its handle.

    Takes pre-compiled Markdown (``content``) or structured ``findings``,
    renders Markdown or PDF, streams the bytes to the pod's object store via
    the produce seam, and returns an :class:`ObjectHandle` (plus a best-effort
    presigned delivery URL) in ``ToolResult.metadata``. The rendered bytes
    never cross NATS.

    :param name: the registered MCP tool name; defaults to ``threetears.report``.
        A consuming pod passes its own namespace (e.g. ``pentest.report``).
    :ptype name: str
    """

    def __init__(self, name: str = _DEFAULT_NAME) -> None:
        """Initialize the tool with its registered name.

        :param name: registered MCP tool name (namespace-qualified)
        :ptype name: str
        """
        super().__init__()
        self._name = name

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Render, store, and deliver a report.

        :param kwargs: tool input (``content`` or ``findings`` required;
            ``report_format`` / ``title`` / ``client_name`` / ``target_scope`` /
            ``assessor_name`` / ``chains`` optional)
        :ptype kwargs: Any
        :return: the produced report's handle + delivery URL, or a fail-closed
            refusal (bad input / no store / PDF toolchain absent)
        :rtype: ToolResult
        """
        report_format = str(kwargs.get("report_format") or "markdown").strip().lower()
        if report_format not in _VALID_FORMATS:
            return ToolResult(
                success=False,
                content="",
                error=f"unknown report_format {report_format!r}; choose one of {list(_VALID_FORMATS)}",
            )

        title = kwargs.get("title")
        findings = kwargs.get("findings")

        markdown = self._build_markdown(kwargs, findings, title)
        if markdown is None:
            return ToolResult(
                success=False,
                content="",
                error="provide either 'content' (pre-rendered markdown) or a non-empty 'findings' list",
            )
        if isinstance(markdown, ToolResult):  # a compile error was surfaced as a refusal
            return markdown

        try:
            body_bytes, filename, mime = await self._render(markdown, report_format, title)
        except PandocNotFoundError as exc:
            _log.warning("PDF report refused: pandoc toolchain absent", extra={"extra_data": {"error": str(exc)}})
            return ToolResult(
                success=False,
                content="",
                error=(
                    "PDF rendering is unavailable in this pod (pandoc/pdflatex not installed): "
                    f"{exc}. Use report_format='markdown'."
                ),
            )
        except RuntimeError as exc:
            _log.error("PDF report render failed", extra={"extra_data": {"error": str(exc)}})
            return ToolResult(success=False, content="", error=f"PDF rendering failed: {exc}")

        summary = _summarize(title, findings, report_format)
        try:
            handle = await stream_result_to_object_store(
                _aiter(body_bytes),
                filename=filename,
                content_type=mime,
                category=_CATEGORY,
                summary=summary,
                size_hint=len(body_bytes),
            )
        except ProduceObjectError as exc:
            _log.warning("report refused: cannot store", extra={"extra_data": {"error": str(exc)}})
            return ToolResult(success=False, content="", error=f"refused: cannot store the report ({exc})")

        ttl = _presign_ttl()
        url = await _presign_delivery_url(handle.s3_key, ttl)
        metadata = object_handle_result_metadata(handle)
        if url is not None:
            metadata["delivery_url"] = url
        return ToolResult(
            success=True,
            content=_result_message(handle, filename, report_format, url, ttl),
            metadata=metadata,
        )

    def _build_markdown(self, kwargs: dict[str, Any], findings: Any, title: str | None) -> str | ToolResult | None:
        """Resolve the report Markdown from inline content or structured findings.

        :param kwargs: raw tool input
        :ptype kwargs: dict[str, Any]
        :param findings: the findings input, if any
        :ptype findings: Any
        :param title: report title, if any
        :ptype title: str | None
        :return: the Markdown string; a ``ToolResult`` refusal on a compile
            error; or ``None`` when neither content nor findings were supplied
        :rtype: str | ToolResult | None
        """
        content = kwargs.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(findings, list) and findings:
            metadata = ReportMetadata(
                title=title or "Security Assessment Report",
                client_name=str(kwargs.get("client_name") or "Security Team"),
                target_scope=str(kwargs.get("target_scope") or _extract_targets(findings)),
                report_date=datetime.now(UTC),
                assessor_name=str(kwargs.get("assessor_name") or "Security Team"),
            )
            chains = kwargs.get("chains")
            chains = chains if isinstance(chains, list) else None
            try:
                return MarkdownCompiler.compile(findings, metadata, chains=chains)
            except (KeyError, TypeError, ValueError) as exc:
                _log.warning("report refused: findings did not compile", extra={"extra_data": {"error": str(exc)}})
                return ToolResult(
                    success=False,
                    content="",
                    error=f"could not compile findings into a report: {exc}",
                )
        return None

    async def _render(self, markdown: str, report_format: str, title: str | None) -> tuple[bytes, str, str]:
        """Render the Markdown to the requested format's bytes.

        PDF rendering runs off the event loop (pandoc is a blocking subprocess).

        :param markdown: the report Markdown
        :ptype markdown: str
        :param report_format: ``markdown`` or ``pdf``
        :ptype report_format: str
        :param title: report title (drives the filename)
        :ptype title: str | None
        :return: (rendered bytes, filename, MIME type)
        :rtype: tuple[bytes, str, str]
        :raises PandocNotFoundError: when a PDF is requested but pandoc is absent
        :raises RuntimeError: when PDF rendering fails
        """
        if report_format == "pdf":
            body_bytes = await asyncio.to_thread(_render_markdown_to_pdf_bytes, markdown)
            return body_bytes, _build_filename(title, "pdf"), "application/pdf"
        return markdown.encode("utf-8"), _build_filename(title, "md"), "text/markdown"

    def mcp_schema(self) -> MCPToolDefinition:
        """Return the MCP-compatible tool definition.

        :return: the tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description=(
                "Generate a professional report from findings and store it out-of-band. "
                "Provide 'content' (Markdown you compiled from tool results in context) OR "
                "'findings' (structured items). Set report_format='pdf' for a PDF (requires "
                "the pod's PDF toolchain) or 'markdown' (always available). Returns a stored "
                "object handle plus a presigned download link; the report bytes are not returned inline."
            ),
            input_schema=_INPUT_SCHEMA,
            timeout_seconds=_timeout_seconds(),
        )

    def mcp_name(self) -> str:
        """Return the namespaced tool name.

        :return: the registered tool name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """Return the tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"
