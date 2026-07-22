"""Bounded-turn page-finding research agent (scrape-task-02).

Given a plain-language query ("Ohio WARN Act notices"), searches and fetches
candidate pages via a bounded WebSearch/WebFetch tool-calling loop
(``threetears.agent.tools.executor.ToolExecutor`` -- this module is its
first real production caller anywhere in the monorepo), then deterministically
verifies the winning candidate has real extractable structure (a table, a
document link, a JSON API response) before returning it. Independently
callable: takes a query, returns plain data (``PageFinderResult``) -- never
persists a ``ScrapeTarget`` itself, never forces extraction to follow. See
``docs/scrape-task-02-page-finder-agent.md`` for the full design and the
reasoning behind reusing ``ToolExecutor``/``WebSearchTool``/``WebFetchTool``
as-is rather than building a new agent-loop primitive.

Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.agent.tools import ToolExecutor
from threetears.agent.tools.builtin.web_fetch import create_web_fetch_tool
from threetears.agent.tools.builtin.web_search import create_web_search_tool
from threetears.models import LlmPurpose, create_chat_model
from threetears.observe import get_logger

from .llm_retry import bounded_retry_structured_call

__all__ = [
    "DEFAULT_PAGE_FINDER_MODEL_ID",
    "PageFinderResult",
    "find_target_page",
]

log = get_logger(__name__)

# Same default and reliability posture as extraction.py's DEFAULT_EXTRACTION_MODEL_ID
# (~50% single-call structured-output failure rate via OpenRouter, live-measured) --
# one shared default rather than a second, independently drifting choice.
DEFAULT_PAGE_FINDER_MODEL_ID = "deepseek/deepseek-chat-v3-0324"

_DEFAULT_MAX_TURNS = 6
_COERCION_TIMEOUT_SECONDS = 30
_COERCION_ATTEMPTS = 6
_COERCION_BACKOFF_SECONDS = 2.0
_VERIFY_TIMEOUT_SECONDS = 15.0
_DOCUMENT_EXTENSIONS = (".pdf", ".doc", ".docx", ".xlsx", ".csv")

# Verified backends only -- a stateless structural check can't tell whether a page needs
# JS rendering (camoufox) or an authenticated in-session XHR (network_capture), so this
# module never guesses either (see docs/scrape-task-02-page-finder-agent.md's Design §4).
_VERIFIABLE_BACKENDS = frozenset({"nodriver", "document", "api"})


class _CandidatePage(BaseModel):
    """Structured coercion of the search-loop's free-text final answer."""

    url: str = PydanticField(description="the URL of the page the agent concluded is correct")
    driver_backend_guess: str | None = PydanticField(
        default=None,
        description="the agent's own guess at nodriver/document/api/camoufox/network_capture, if it has one",
    )
    wait_for_guess: str | None = PydanticField(
        default=None, description="a CSS selector the agent believes the page needs to settle on, if any"
    )
    summary: str = PydanticField(
        description="one or two sentences on what the page contains and why it's the right one"
    )


@dataclass
class PageFinderResult:
    """Plain-data result of a page-finding run -- never a persisted ``ScrapeTarget``.

    A caller decides whether/how to turn this into a real target row (mirrors
    ``ScrapeTool``'s own constructor-injection philosophy: no hidden I/O a
    caller didn't ask for).
    """

    url: str
    driver_backend: str
    wait_for: str | None
    verified: bool
    verification_note: str
    reasoning: str
    turns_used: int
    search_queries_tried: list[str] = field(default_factory=list)


def _build_search_messages(query: str) -> list[Any]:
    system = SystemMessage(
        content=(
            "You are finding the correct, real webpage for a data-extraction task. Use the "
            "web_search tool to find candidates, then use the web_fetch tool to inspect a "
            "candidate BEFORE concluding it's correct -- never conclude from a search snippet "
            "alone. Once you've verified a real candidate page, respond with plain text (no "
            "tool call) stating: the URL, your best guess at whether it needs a real browser to "
            "render (say so if the fetched content looked like an empty JS shell) or is a "
            "document (PDF/DOCX/etc.) or a JSON API, any CSS selector you believe the page needs "
            "to wait for/settle on, and a one- or two-sentence summary of what the page contains."
        )
    )
    human = HumanMessage(content=f"Find the real page for: {query}")
    return [system, human]


def _extract_search_queries(tool_calls_made: list[dict[str, Any]], search_tool_name: str) -> list[str]:
    """Pull every ``query`` arg from calls to the search tool.

    Takes the search tool's actual bound name rather than hardcoding
    ``"web_search"`` -- ``WebSearchTool.mcp_name()`` returns
    ``"threetears.web_search"``, the name ``ToolExecutor`` actually records
    in ``tool_calls_made``, not the bare string (a real bug caught by
    Critic review before this shipped: the original hardcoded filter never
    matched, so this always returned ``[]`` in production despite unit
    tests passing against fabricated call names).

    :param tool_calls_made: ``ToolExecutionResult.tool_calls_made``
    :ptype tool_calls_made: list[dict[str, Any]]
    :param search_tool_name: the search tool's actual bound ``.name``
    :ptype search_tool_name: str
    :return: every search query the loop tried, in order
    :rtype: list[str]
    """
    return [
        str(call["args"]["query"])
        for call in tool_calls_made
        if call.get("name") == search_tool_name and "query" in call.get("args", {})
    ]


async def _verify_candidate_page(url: str, *, client: httpx.AsyncClient | None = None) -> tuple[bool, str, str]:
    """Deterministic (no LLM) structural check -- does this page have real extractable structure.

    A direct, stateless HTTP fetch -- no nodriver sidecar, no browser -- so
    ``find_target_page`` stays independently callable without any running
    container (Design Rule 4). Checks, in order: a real HTML table, a
    document link, a JSON API response. Never verifies to ``camoufox``/
    ``network_capture`` -- see this module's own docstring and
    ``docs/scrape-task-02-page-finder-agent.md``.

    :param url: the candidate URL to check
    :ptype url: str
    :param client: injectable HTTP client (``ApiDriver``'s own DI shape) -- built fresh if omitted
    :ptype client: httpx.AsyncClient | None
    :return: (verified, driver_backend guess, human-readable note on what was found)
    :rtype: tuple[bool, str, str]
    """
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(follow_redirects=True, timeout=_VERIFY_TIMEOUT_SECONDS)
    try:
        try:
            response = await client.get(url)
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- honest-unverified
            # a fetch failure here must degrade to "unverified," never raise into the caller --
            # same "surface for review, never silently drop" discipline as bounded_retry_structured_call.
            log.warning("page-finder verification fetch failed for %s: %s", url, exc)
            return False, "nodriver", f"could not fetch candidate page for verification: {exc}"
    finally:
        if owns_client:
            await client.aclose()

    content_type = response.headers.get("content-type", "")

    if "json" in content_type:
        try:
            body = response.json()
        except ValueError:
            body = None
        has_list = isinstance(body, list) or (
            isinstance(body, dict) and any(isinstance(v, list) for v in body.values())
        )
        if has_list:
            return True, "api", "response is JSON containing a list -- looks like a real API"

    soup = BeautifulSoup(response.text, "html.parser")

    # Table checked before document link: a real page can carry an incidental PDF link
    # (privacy policy, a related-regulations reference) alongside its actual notices table --
    # live-verified against Maryland's real WARN page, which has exactly this shape. A real
    # table is the stronger, more specific signal of "this is the actual data source."
    for table in soup.find_all("table"):
        if len(table.find_all("tr")) >= 2:
            return True, "nodriver", "found a real HTML table with multiple rows"

    for link in soup.find_all("a", href=True):
        href = str(link["href"]).lower()
        if href.endswith(_DOCUMENT_EXTENSIONS):
            return True, "document", f"found a document link ({href})"

    return False, "nodriver", "no table, document link, or JSON list found on the fetched page"


async def find_target_page(
    query: str,
    *,
    api_key: str,
    searxng_url: str,
    model_id: str = DEFAULT_PAGE_FINDER_MODEL_ID,
    max_turns: int = _DEFAULT_MAX_TURNS,
) -> PageFinderResult:
    """Search for, fetch, and self-verify a candidate page for *query*.

    Never raises and never returns ``None`` -- a query that never converges
    (turn exhaustion) or a candidate that fails structural verification still
    returns a ``PageFinderResult`` with ``verified=False`` and an honest
    ``reasoning``/``verification_note`` (the same "surface for review, never
    silently drop" discipline ``ScrapeExtraction.validation_status``
    establishes for the extraction path).

    :param query: plain-language description of the page to find
    :ptype query: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param searxng_url: base URL of the SearXNG instance backing web_search
    :ptype searxng_url: str
    :param model_id: the tool-calling and structured-coercion model
    :ptype model_id: str
    :param max_turns: bounded round cap for the search/fetch loop
    :ptype max_turns: int
    :return: the finding, verified or not
    :rtype: PageFinderResult
    """
    web_search_lc = create_web_search_tool({"base_url": searxng_url}, "Search the web for candidate pages.")
    web_fetch_lc = create_web_fetch_tool({}, "Fetch a candidate page's readable content to inspect it.")

    chat_model = create_chat_model(model_id, api_key=api_key, purpose=LlmPurpose.TOOL_SELECTION).bind_tools(
        [web_search_lc, web_fetch_lc]
    )
    messages = _build_search_messages(query)
    loop_result = await ToolExecutor(max_rounds=max_turns).invoke_with_tools(
        chat_model, messages, [web_search_lc, web_fetch_lc]
    )
    queries_tried = _extract_search_queries(loop_result.tool_calls_made, web_search_lc.name)

    if loop_result.error is not None and not loop_result.output.strip():
        return PageFinderResult(
            url="",
            driver_backend="nodriver",
            wait_for=None,
            verified=False,
            verification_note="search loop exhausted its turn budget with no usable answer",
            reasoning=f"page-finder gave up after {loop_result.rounds_used} turns: {loop_result.error}",
            turns_used=loop_result.rounds_used,
            search_queries_tried=queries_tried,
        )

    coercion_prompt = (
        f"The following is a research agent's free-text conclusion about which page answers "
        f'"{query}". Extract the structured fields from it:\n\n{loop_result.output}'
    )
    candidate = await bounded_retry_structured_call(
        coercion_prompt,
        _CandidatePage,
        model_id=model_id,
        api_key=api_key,
        purpose=LlmPurpose.EXTRACTION,
        temperature=0.0,
        timeout=_COERCION_TIMEOUT_SECONDS,
        attempts=_COERCION_ATTEMPTS,
        backoff_seconds=_COERCION_BACKOFF_SECONDS,
        log_label="page-finder candidate coercion",
        degraded_to="no resolvable candidate",
        is_acceptable=lambda c: bool(c.url) and c.url.startswith(("http://", "https://")),
    )
    if candidate is None:
        return PageFinderResult(
            url="",
            driver_backend="nodriver",
            wait_for=None,
            verified=False,
            verification_note="could not coerce the search loop's answer into a URL",
            reasoning=loop_result.output,
            turns_used=loop_result.rounds_used,
            search_queries_tried=queries_tried,
        )

    verified, structural_backend, verification_note = await _verify_candidate_page(candidate.url)
    if verified:
        driver_backend = structural_backend
    elif candidate.driver_backend_guess in _VERIFIABLE_BACKENDS:
        driver_backend = candidate.driver_backend_guess
    else:
        driver_backend = "nodriver"

    return PageFinderResult(
        url=candidate.url,
        driver_backend=driver_backend,
        wait_for=candidate.wait_for_guess,
        verified=verified,
        verification_note=verification_note,
        reasoning=candidate.summary,
        turns_used=loop_result.rounds_used,
        search_queries_tried=queries_tried,
    )
