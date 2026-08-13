"""Bounded-turn page-finding research agent (scrape-task-02).

Given a plain-language query ("Ohio WARN Act notices"), searches and fetches
candidate pages via a bounded WebSearch/WebFetch tool-calling loop
(``threetears.agent.tools.executor.ToolExecutor`` -- this module is its
first real production caller anywhere in the monorepo), then deterministically
verifies the winning candidate has real extractable structure (a table, a
document link, a JSON API response) before returning it. Independently
callable: takes a query, returns plain data (``PageFinderResult``) -- never
persists a ``ScrapeTarget`` itself, never forces extraction to follow.

Built on ``ToolExecutor``/``WebSearchTool``/``WebFetchTool`` exactly as they
ship rather than on a new agent-loop primitive: what this needs is a capped
number of search/fetch turns over plain-data tools, which is precisely what
those already do. A page-finder-specific loop would have been the same
mechanism with a narrower blast radius of reuse. See
``docs/scrape-task-02-page-finder-agent.md`` for the full design.

**Structure, not just prose (search-spec.md check 4).** Since the builtins
moved onto the search leaf, every ``threetears.web_search`` turn deposits its
typed result on ``ToolMessage.artifact`` under
:data:`~threetears.search.contracts.SEARCH_RESULTS_METADATA_KEY` (D22), and
``ToolExecutor`` keeps that artifact rather than stringifying it (§4.7). This
module reads it. The loop's free-text answer is still what names the winning
page -- an LLM chose it, and only prose carries that choice -- but the
structure is what lets the answer be *qualified* rather than merely believed:
which URLs the search actually returned, what the provider said it degraded,
and whether a search refused outright. Callers are unaffected either way,
which is what check 4 asks: the new facts arrive as additive
:class:`PageFinderResult` fields with defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel
from pydantic import Field as PydanticField
from threetears.agent.tools import ToolExecutor
from threetears.agent.tools.builtin.web_fetch import create_web_fetch_tool
from threetears.agent.tools.builtin.web_search import create_web_search_tool
from threetears.models import LlmPurpose, create_chat_model
from threetears.observe import get_logger
from threetears.search.aggregate import aggregate
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    Candidate,
    CandidateSet,
    Corpus,
    SearchResultsMetadata,
)

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
#: hard cap on the verification fetch, matching ``extract.py``'s own
#: ``DEFAULT_MAX_BYTES`` (SR-G5) rather than inventing a second bound. The fetch
#: used to be unbounded -- ``client.get`` buffered the whole body and
#: BeautifulSoup then built a parse tree from it, which measured **77x** the
#: served size (19 MiB of HTML peaked at ~1.5 GiB of heap). ``find_target_page``
#: fetches a URL an LLM picked out of search results, so the size of that body
#: is not this process's to choose. Same defect class as search-spec.md §10
#: defect 7, which the gutting removed from ``web_fetch``; it survived here
#: because this is scrape's own fetch rather than the leaf's.
_VERIFY_MAX_BYTES = 2 * 1024 * 1024
_DOCUMENT_EXTENSIONS = (".pdf", ".doc", ".docx", ".xlsx", ".csv")

# Verified backends only -- a stateless structural check can't tell whether a page needs
# JS rendering (camoufox) or an authenticated in-session XHR (network_capture), so this
# module never guesses either -- an unverifiable guess in a returned
# `driver_backend` is worse than an absent one, since the caller can't tell
# the two apart (see docs/scrape-task-02-page-finder-agent.md's Design section).
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
    #: every candidate the search turns actually returned, in provider order,
    #: deduplicated by identity across turns. Typed, off ``metadata`` -- never
    #: re-parsed out of the prose the LLM read (check 4). Where several turns
    #: returned one identity, the first turn's candidate is the one here; every
    #: contribution is on :attr:`candidate_corpus`.
    candidates_seen: tuple[Candidate, ...] = ()
    #: the same candidates before that projection, one entry per identity with
    #: every contributing turn intact. Each turn searched a different query and
    #: provenance records which, so an entry with two contributions is a URL
    #: two differently-worded searches both found -- corroboration the flat
    #: tuple cannot express. ``None`` only on a result built without a search.
    candidate_corpus: Corpus | None = None
    #: whether :attr:`url` was among :attr:`candidates_seen`. ``False`` on a
    #: real finding means the coercion step produced a URL no search returned
    #: -- the page may still be correct (the loop can reach it by following a
    #: fetched link) but it was not *found*, and that difference was invisible
    #: before structure crossed the border.
    url_was_a_search_result: bool = False
    #: provider-reported degradations gathered across the search turns (SR-L2,
    #: P8) -- engines that did not answer, a partial fan-in, output known to be
    #: unranked. A page found over a degraded search is still a finding; it is
    #: just one whose thinness has a stated cause.
    search_notices: tuple[str, ...] = ()
    #: the first typed search failure the loop hit, rendered as
    #: ``"<failure-class>: <message>"``, or ``None`` when no search failed. This
    #: is how a refusal stops being indistinguishable from a fruitless search.
    search_failure: str | None = None


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


def _read_search_structure(messages: list[Any], search_tool_name: str) -> list[SearchResultsMetadata]:
    """Read the typed search results off the loop's ``ToolMessage`` artifacts.

    ``ToolExecutor`` mutates the message list in place and appends each tool's
    ``ToolMessage`` with its artifact intact (§4.7), so the structure the
    search leaf put on ``ToolResult.metadata`` is sitting in the conversation
    this module already owns -- no second call, no re-parse of prose.

    **Filtered by bound tool name, and that is load-bearing.** ``web_fetch``
    writes its own projection under the *same*
    :data:`SEARCH_RESULTS_METADATA_KEY`, so an unfiltered scan would read
    fetched pages as though they were search results and report a candidate
    the search never returned. The name is passed in rather than hardcoded for
    the reason ``_extract_search_queries`` documents directly above: the bound
    name is ``threetears.web_search``, not ``web_search``, and hardcoding the
    bare string is a bug that unit tests on fabricated names do not catch.

    A payload this reader is too old to understand is skipped with a warning
    rather than raised: :meth:`SearchResultsMetadata.from_metadata` refuses a
    newer ``schema_version`` loudly (D13), which is right for a reader that
    can fail, and wrong here -- ``find_target_page`` promises never to raise,
    so a structure it cannot read degrades to the prose path it used before
    structure existed.

    :param messages: the message list ``invoke_with_tools`` mutated in place
    :ptype messages: list[Any]
    :param search_tool_name: the search tool's actual bound ``.name``
    :ptype search_tool_name: str
    :return: one projection per search turn that carried readable structure
    :rtype: list[SearchResultsMetadata]
    """
    found: list[SearchResultsMetadata] = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != search_tool_name:
            continue
        artifact = message.artifact
        if not isinstance(artifact, dict):
            continue
        payload = artifact.get(SEARCH_RESULTS_METADATA_KEY)
        if not isinstance(payload, dict):
            continue
        try:
            found.append(SearchResultsMetadata.from_metadata(payload))
        except ValueError as exc:
            log.warning("page-finder could not read search structure, falling back to prose: %s", exc)
    return found


def _corpus(projections: list[SearchResultsMetadata]) -> Corpus:
    """Accumulate every turn's candidates into one corpus.

    This module used to hand-roll the accumulation -- flatten the turns, drop
    any identity already seen -- which is what ``aggregate`` now owns, and
    which lost something in the dropping. Each turn searches a *different
    query*, recorded on every candidate's provenance, so the second turn to
    return a URL was carrying the fact that a differently-worded search also
    found it. That is corroboration a page-finding loop should weigh, and
    discarding the later contribution threw it away silently.

    Only successful turns are aggregated. A failed turn's projection carries a
    ``FailureRecord`` rather than a live exception, and its failure is already
    reported by :func:`_first_failure` and :func:`_every_search_failed`;
    passing it through here as well would double-report it in the corpus
    notices, which ``search_notices`` is not for.

    Turn order is preserved and is emphatically not a ranking (SR-L2) -- just
    the honest record of what came back, in the order it came.

    :param projections: one projection per search turn
    :ptype projections: list[SearchResultsMetadata]
    :return: the accumulated corpus, one entry per distinct identity
    :rtype: Corpus
    """
    return aggregate(
        CandidateSet(
            candidates=projection.candidates,
            dispositions=projection.dispositions,
            spend=projection.spend,
        )
        for projection in projections
        if projection.failure is None
    )


def _first_seen(corpus: Corpus) -> tuple[Candidate, ...]:
    """Project the corpus back to one candidate per identity, first turn winning.

    Preserves exactly what :attr:`PageFinderResult.candidates_seen` has always
    meant, so no caller changes. The contributions the projection drops are
    still on :attr:`PageFinderResult.candidate_corpus` for anything that wants
    them.

    :param corpus: the accumulated corpus
    :ptype corpus: Corpus
    :return: the candidates, first occurrence of each identity winning
    :rtype: tuple[Candidate, ...]
    """
    return tuple(entry.contributions[0] for entry in corpus.entries)


def _candidate_urls(candidates: tuple[Candidate, ...]) -> set[str]:
    """Every URL a candidate can be reached at -- identity plus locators.

    Both, because ``identity`` is the canonical URL *by convention* rather
    than by guarantee (a provider without URLs uses its native id), and the
    URL an LLM names is whichever one the prose rendering showed it.

    :param candidates: the deduplicated candidates
    :ptype candidates: tuple[Candidate, ...]
    :return: the set of URLs those candidates account for
    :rtype: set[str]
    """
    urls = {candidate.identity for candidate in candidates}
    urls.update(locator.url for candidate in candidates for locator in candidate.locators)
    return urls


def _first_failure(projections: list[SearchResultsMetadata]) -> str | None:
    """Render the first typed search failure across the turns, if any.

    The failure *class* leads, because that is the fact worth acting on --
    ``rate-limited`` and ``local-cap-exceeded`` want different responses from
    an operator, and telling them apart used to require matching on an error
    prefix in prose, which is the defect the gutting removed.

    :param projections: one projection per search turn
    :ptype projections: list[SearchResultsMetadata]
    :return: ``"<failure-class>: <message>"``, or None when every search
        completed (including the ones that completed with zero results --
        that is a success, SR-J2)
    :rtype: str | None
    """
    for projection in projections:
        if projection.failure is not None:
            return f"{projection.failure.failure_class}: {projection.failure.message}"
    return None


def _every_search_failed(projections: list[SearchResultsMetadata]) -> bool:
    """Whether every search turn failed, as opposed to merely one of them.

    The distinction decides what a fruitless run is *told* it was. A run whose
    first turn was rate-limited and whose next four searched fine did not fail
    for want of searching -- it failed to converge -- and blaming the provider
    would send an operator after a quota problem that had already cleared. The
    first failure is still reported on its own field either way.

    :param projections: one projection per search turn
    :ptype projections: list[SearchResultsMetadata]
    :return: True only when there was at least one turn and all of them failed
    :rtype: bool
    """
    return bool(projections) and all(projection.failure is not None for projection in projections)


def _all_notices(projections: list[SearchResultsMetadata]) -> tuple[str, ...]:
    """Gather provider-reported degradations across turns, order-preserved.

    :param projections: one projection per search turn
    :ptype projections: list[SearchResultsMetadata]
    :return: the distinct notices, first occurrence winning
    :rtype: tuple[str, ...]
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for projection in projections:
        for notice in projection.notices:
            if notice in seen:
                continue
            seen.add(notice)
            ordered.append(notice)
    return tuple(ordered)


def _decode(raw: bytes, declared_charset: str | None) -> str:
    """Decode a fetched body, tolerating a charset Python has never heard of.

    ``httpx`` hands back the ``charset=`` parameter verbatim; it never checks it
    against the codec registry. A server declaring ``charset=utf8mb4`` -- a real
    MySQL-ism that appears in the wild -- or any typo makes :meth:`bytes.decode`
    raise :class:`LookupError`, which is **not** a :class:`ValueError` and so
    would sail past the fetch's own guard and out of ``find_target_page``, whose
    contract is that it never raises. The header comes off a page an LLM picked
    out of search results, so it is third-party input on the strength of a
    string match.

    An unknown charset falls back to UTF-8 rather than refusing: the caller
    wants to know whether the page has a table, and every marker that answers
    that is ASCII.

    :param raw: the body bytes, already capped
    :ptype raw: bytes
    :param declared_charset: the charset the response declared, if any
    :ptype declared_charset: str | None
    :return: the decoded text, never raising
    :rtype: str
    """
    if declared_charset:
        try:
            return raw.decode(declared_charset, errors="replace")
        except LookupError:
            log.warning("page-finder ignoring unknown declared charset %r; decoding as utf-8", declared_charset)
    return raw.decode("utf-8", errors="replace")


async def _verify_candidate_page(url: str, *, client: httpx.AsyncClient | None = None) -> tuple[bool, str, str]:
    """Deterministic (no LLM) structural check -- does this page have real extractable structure.

    A direct, stateless HTTP fetch -- no nodriver sidecar, no browser -- so
    ``find_target_page`` stays independently callable without any running
    container (Design Rule 4). Checks, in order: a real HTML table, a
    document link, a JSON API response. Never verifies to ``camoufox``/
    ``network_capture`` -- see this module's own docstring for why those two
    are structurally unreachable from a stateless fetch, and
    ``docs/scrape-task-02-page-finder-agent.md`` for the full design.

    The GET is unconditional -- no ``If-None-Match``, no ``If-Modified-Since``
    -- and that is not an oversight. Conditional requests are how a caller
    revalidates a copy it already holds; this function holds none, keeps none,
    and runs once per candidate at discovery time, so a 304 would leave it with
    nothing to inspect. Revalidating an *already-onboarded* target is a real
    and valuable thing -- a 304 there skips a render and an LLM extraction --
    and it is ruled at SR-M4 / D30 with a build sequence in
    ``docs/search-task-01-conditional-revalidation.md``. It belongs to the
    scrape pipeline and to Extract, never to this discovery-time check.

    The body is read under :data:`_VERIFY_MAX_BYTES` and may therefore be
    truncated; the returned note distinguishes "no structure in what I read"
    from "no structure on the page" rather than conflating them.

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
            async with client.stream("GET", url) as response:
                content_type = response.headers.get("content-type", "")
                declared_charset = response.charset_encoding
                chunks: list[bytes] = []
                read = 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    read += len(chunk)
                    if read > _VERIFY_MAX_BYTES:
                        break
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- honest-unverified
            # a fetch failure here must degrade to "unverified," never raise into the caller --
            # same "surface for review, never silently drop" discipline as bounded_retry_structured_call.
            log.warning("page-finder verification fetch failed for %s: %s", url, exc)
            return False, "nodriver", f"could not fetch candidate page for verification: {exc}"
    finally:
        if owns_client:
            await client.aclose()

    truncated = read > _VERIFY_MAX_BYTES
    raw = b"".join(chunks)[:_VERIFY_MAX_BYTES]

    if "json" in content_type:
        # A truncated body is not parseable JSON, so this correctly declines to
        # call a cut-off document an API rather than guessing at the missing half.
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        has_list = isinstance(body, list) or (
            isinstance(body, dict) and any(isinstance(v, list) for v in body.values())
        )
        if has_list:
            return True, "api", "response is JSON containing a list -- looks like a real API"

    # Decoded with the charset the server declared, falling back to UTF-8, and
    # never strictly: a page whose bytes do not match its declared charset (or
    # that got cut mid-character at the cap) must still be inspectable for
    # structure. It always is, because every marker below -- <table>, <tr>, href
    # -- is ASCII, so structure detection survives text this cannot decode.
    text = _decode(raw, declared_charset)
    soup = BeautifulSoup(text, "html.parser")

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

    # "Nothing in the part I read" is a weaker claim than "nothing on the page",
    # and a note that conflated them would send the next reader looking for a
    # structure bug that is really a size cap.
    if truncated:
        return (
            False,
            "nodriver",
            f"no table, document link, or JSON list in the first {_VERIFY_MAX_BYTES} bytes "
            "of the page, which was longer than the verification cap",
        )
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

    # Structure off metadata, read once and threaded through every exit below --
    # including the failure exits, which is the point: a run that ends with
    # nothing now says whether the search *refused* or merely came up empty.
    projections = _read_search_structure(messages, web_search_lc.name)
    candidate_corpus = _corpus(projections)
    candidates_seen = _first_seen(candidate_corpus)
    search_notices = _all_notices(projections)
    search_failure = _first_failure(projections)

    if loop_result.error is not None and not loop_result.output.strip():
        return PageFinderResult(
            url="",
            driver_backend="nodriver",
            wait_for=None,
            verified=False,
            verification_note=(
                f"every search turn was refused ({search_failure})"
                if _every_search_failed(projections)
                else "search loop exhausted its turn budget with no usable answer"
            ),
            reasoning=f"page-finder gave up after {loop_result.rounds_used} turns: {loop_result.error}",
            turns_used=loop_result.rounds_used,
            search_queries_tried=queries_tried,
            candidates_seen=candidates_seen,
            candidate_corpus=candidate_corpus,
            search_notices=search_notices,
            search_failure=search_failure,
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
            candidates_seen=candidates_seen,
            candidate_corpus=candidate_corpus,
            search_notices=search_notices,
            search_failure=search_failure,
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
        candidates_seen=candidates_seen,
        candidate_corpus=candidate_corpus,
        url_was_a_search_result=candidate.url in _candidate_urls(candidates_seen),
        search_notices=search_notices,
        search_failure=search_failure,
    )
