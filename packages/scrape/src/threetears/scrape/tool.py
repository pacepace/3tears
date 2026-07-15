"""ScrapeTool -- exposes 3tears-scrape as a callable ``TearsTool`` (MCP tool).

**Design (2026-07-14, MCP exposure):** an ad-hoc, one-off scrape -- "fetch
this URL, extract these fields" -- with no pre-registered ``ScrapeTarget``,
matching the exact use case ``StaticTargetSource``/a single inline
``ScrapeTarget`` construction was already designed for (Chunk 13's own
design decision). Runs through the *same* unmodified AI eval loop
(``eval_loop.run_eval_loop``/``run_eval_loop_multi_row``) every configured
target already uses -- no separate "MCP extraction path" to keep in sync.

Repeated calls against the same URL + field schema benefit from the eval
loop's own self-healing recipe reuse (zero further LLM calls once a recipe
wins) via a *deterministic* ``target_id`` derived from ``(url,
field_schema)`` when the caller doesn't supply one explicitly -- an LLM
caller shouldn't have to invent and remember target IDs itself for this to
work.

Zero faidh imports (see ``scrape/__init__.py``): all real dependencies
(collections, drivers, API key) are constructor-injected, never resolved
internally (no env-var reads, no ``faidh.config``/``faidh.store`` calls) --
mirrors every other driver in this package (e.g. ``NodriverSidecarDriver``'s
``base_url``). The faidh-side registration wrapper that resolves real
config/collections/drivers and constructs this class lives in
``src/faidh/tools/scrape_tool.py``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.observe import get_logger

from .collections import ScrapeExtractionCollection, ScrapeRecipeCollection, decode_field_schema, decode_nav_steps
from .driver import NavStep, RenderedPage, ScrapeDriver
from .eval_loop import StrategyType, run_eval_loop, run_eval_loop_multi_row
from .extraction import FieldSchema

__all__ = ["ScrapeTool"]

log = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30.0


def _derive_target_id(url: str, field_schema: dict[str, Any]) -> str:
    """Deterministic recipe-reuse key for an ad-hoc scrape with no caller-supplied target_id.

    Hashes *url* plus the field schema's own field names (sorted -- dict
    iteration order isn't part of the schema's identity) so the exact same
    ``(url, field_schema)`` call always resolves to the same key, letting
    repeated calls reuse a winning recipe instead of re-running candidate
    generation every time. Field *types* aren't part of the hash input --
    two calls differing only in a field's declared type are rare enough
    (and a schema drift within the same fields is itself informative, not
    something to route around) that keying on names alone is the simpler,
    still-correct choice.
    """
    digest = hashlib.sha256(f"{url}|{','.join(sorted(field_schema))}".encode()).hexdigest()
    return f"adhoc_{digest[:16]}"


class ScrapeTool(TearsTool):
    """Ad-hoc "fetch this URL, extract these fields" tool, backed by 3tears-scrape.

    All state (collections, drivers, API key) is injected at construction --
    no internal env-var or faidh-config resolution, per this module's own
    zero-faidh-imports discipline.
    """

    def __init__(
        self,
        *,
        recipe_collection: ScrapeRecipeCollection,
        extraction_collection: ScrapeExtractionCollection,
        drivers: dict[str, ScrapeDriver],
        api_key: str,
        default_timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        :param recipe_collection: shared recipe store for the eval loop's self-healing reuse
        :ptype recipe_collection: ScrapeRecipeCollection
        :param extraction_collection: where each call's extraction result is persisted
        :ptype extraction_collection: ScrapeExtractionCollection
        :param drivers: ``driver_backend`` name -> ``ScrapeDriver`` instance
            (e.g. ``{"nodriver": ..., "camoufox": ..., "document": ...}``)
        :ptype drivers: dict[str, ScrapeDriver]
        :param api_key: OpenRouter API key for the eval loop's candidate-generation and judge calls
        :ptype api_key: str
        :param default_timeout: seconds to wait for a render when the caller doesn't specify one
        :ptype default_timeout: float
        """
        self._recipe_collection = recipe_collection
        self._extraction_collection = extraction_collection
        self._drivers = drivers
        self._api_key = api_key
        self._default_timeout = default_timeout

    def mcp_name(self) -> str:
        """Return the namespaced tool name.

        A generic ``3tears.scrape`` identity, since this class is a
        reusable 3tears component, not faidh-specific -- a consuming
        wrapper (e.g. faidh's own ``FaidhScrapeTool``) is free to
        override this with its own namespaced name.
        """
        return "3tears.scrape"

    def mcp_version(self) -> str:
        """Return the tool version string."""
        return "1.0.0"

    def mcp_schema(self) -> MCPToolDefinition:
        """Return the MCP tool definition."""
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description=(
                "Scrape a URL (a web page, or a document -- PDF/DOCX/XLSX/TXT/Markdown/LaTeX -- "
                "when driver_backend='document') and extract structured fields via the real, "
                "AI-driven eval loop: LLM-proposed extraction strategy, structural validation, "
                "an LLM judge, and self-healing recipe reuse on repeated calls against the same "
                "URL and field schema -- the same system every pre-configured scrape target uses, "
                "not a separate one-shot extraction path."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "the page or document to scrape"},
                    "field_schema": {
                        "type": "object",
                        "description": (
                            "field name -> type name ('str' | 'int' | 'float' | 'bool') to extract, "
                            'e.g. {"employer": "str", "affected_count": "int"}'
                        ),
                        "additionalProperties": {"type": "string"},
                    },
                    "multi_row": {
                        "type": "boolean",
                        "description": "true if the page/document lists many records (a table/listing) rather than one",
                        "default": False,
                    },
                    "driver_backend": {
                        "type": "string",
                        "enum": ["nodriver", "camoufox", "document"],
                        "description": "which ScrapeDriver renders the URL; 'document' for a PDF/DOCX/XLSX/TXT/Markdown/LaTeX file",
                        "default": "nodriver",
                    },
                    "wait_for": {
                        "type": "string",
                        "description": "CSS selector to wait for before considering the page rendered (ignored by driver_backend='document')",
                    },
                    "nav_steps": {
                        "type": "array",
                        "description": (
                            "ordered browser actions driving the page to its real content before "
                            "extraction (a search form, a second page in a listing) -- each item is "
                            '{"action": "click"|"fill"|"wait_for"|"wait_ms", "selector": str, '
                            '"value": str, "ms": int} (fields other than action are optional per '
                            "action type). Ignored by driver_backend='document'."
                        ),
                        "items": {"type": "object"},
                    },
                    "target_id": {
                        "type": "string",
                        "description": (
                            "stable key for recipe reuse across repeated calls; derived from "
                            "(url, field_schema) when omitted, so repeated identical calls reuse "
                            "a winning recipe without the caller needing to invent one"
                        ),
                    },
                    "strategy_type": {
                        "type": "string",
                        "enum": ["css", "regex"],
                        "description": (
                            "'css' (default) for an HTML table -- CSS-selector candidates. 'regex' for "
                            "a page whose real content is prose/list text with no <table> structure at "
                            "all -- regex-pattern candidates (one (?P<field_name>...) named group per "
                            "field) matched against the page's plain text instead."
                        ),
                        "default": "css",
                    },
                },
                "required": ["url", "field_schema"],
            },
            timeout_seconds=self._default_timeout + 60.0,  # render timeout + eval loop's own LLM-call budget
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Fetch *url*, then extract *field_schema* via the real AI eval loop.

        :param kwargs: tool input; requires ``url``/``field_schema``, optional
            ``multi_row``/``driver_backend``/``wait_for``/``target_id``
        :ptype kwargs: Any
        :return: JSON-encoded ``{"target_id", "validation_status", "records"}``,
            or an error result
        :rtype: ToolResult
        """
        # Single-exit structure (an `error` sentinel gates each stage) rather than
        # early-returning per validation failure -- this function has several
        # independent, sequential things that can go wrong (bad input, an
        # unsupported backend, a transport failure), and this repo's own
        # single-return convention wants one exit point, not one per failure mode.
        error: str | None = None

        url = kwargs.get("url") or ""
        if not url:
            error = "url is required"

        schema: FieldSchema = {}
        raw_schema = kwargs.get("field_schema") or {}
        if error is None:
            try:
                schema = decode_field_schema(raw_schema)
            except ValueError as exc:
                error = str(exc)
            if error is None and not schema:
                error = "field_schema must declare at least one field"

        driver_backend = kwargs.get("driver_backend") or "nodriver"
        driver: ScrapeDriver | None = None
        if error is None:
            driver = self._drivers.get(driver_backend)
            if driver is None:
                error = f"unsupported driver_backend {driver_backend!r}; available: {sorted(self._drivers)}"

        nav_steps: list[NavStep] | None = None
        raw_nav_steps = kwargs.get("nav_steps")
        if error is None and raw_nav_steps:
            try:
                nav_steps = decode_nav_steps(raw_nav_steps)
            except TypeError as exc:
                error = f"invalid nav_steps: {exc}"

        strategy_type: StrategyType = kwargs.get("strategy_type") or "css"
        if error is None and strategy_type not in ("css", "regex"):
            error = f"unsupported strategy_type {strategy_type!r}; must be 'css' or 'regex'"

        multi_row = bool(kwargs.get("multi_row", False))
        wait_for = kwargs.get("wait_for") or None
        target_id = kwargs.get("target_id") or _derive_target_id(url, raw_schema)

        page: RenderedPage | None = None
        if error is None:
            assert driver is not None  # narrowed by `error is None` above
            try:
                page = await driver.render(url, timeout=self._default_timeout, wait_for=wait_for, nav_steps=nav_steps)
            except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- any backend-specific driver error surfaces as a ToolResult, never crashes the tool call
                log.warning(
                    "scrape tool: render failed",
                    extra={"extra_data": {"url": url, "driver_backend": driver_backend}},
                )
                error = f"fetch failed: {exc}"

        if error is not None:
            result = ToolResult(success=False, content="", error=error)
        else:
            assert page is not None  # narrowed by `error is None` above
            eval_loop_fn = run_eval_loop_multi_row if multi_row else run_eval_loop
            extraction = await eval_loop_fn(
                target_id,
                page.html,
                page.final_url,
                schema,
                recipe_collection=self._recipe_collection,
                extraction_collection=self._extraction_collection,
                api_key=self._api_key,
                strategy_type=strategy_type,
            )
            records: list[dict[str, Any]] = extraction.structured_fields.get("records", [])
            content = json.dumps(
                {"target_id": target_id, "validation_status": extraction.validation_status, "records": records},
                default=str,
            )
            result = ToolResult(
                success=extraction.validation_status == "validated",
                content=content,
                metadata={
                    "target_id": target_id,
                    "validation_status": extraction.validation_status,
                    "record_count": len(records),
                    "source_url": page.final_url,
                },
            )
        return result
