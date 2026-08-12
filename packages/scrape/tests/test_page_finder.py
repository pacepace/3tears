"""Unit tests for threetears.scrape.page_finder -- the bounded-turn page-finding
agent (mocking approach mirrors test_extraction.py's create_chat_model patching,
plus httpx.MockTransport for _verify_candidate_page per test_driver_api.py's
own convention).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    Candidate,
    CandidateSet,
    FailureRecord,
    Locator,
    Provenance,
    SearchResultsMetadata,
    Spend,
)

from threetears.scrape.page_finder import (
    _VERIFY_MAX_BYTES,
    PageFinderResult,
    _all_notices,
    _candidate_urls,
    _CandidatePage,
    _dedupe_candidates,
    _extract_search_queries,
    _first_failure,
    _read_search_structure,
    _verify_candidate_page,
    find_target_page,
)

_SEARCH_TOOL = "threetears.web_search"
_FETCH_TOOL = "threetears.web_fetch"


def _candidate(url: str, *, title: str = "a page") -> Candidate:
    """A minimally-valid real Candidate -- the contract type, never a stand-in."""
    return Candidate(
        identity=url,
        locators=(Locator(url=url),),
        provenance=Provenance(
            query="Ohio WARN notices",
            provider_instance="searxng-local",
            retrieved_at=datetime(2026, 8, 12, tzinfo=UTC),
        ),
        title=title,
    )


def _search_tool_message(
    *candidates: Candidate, name: str = _SEARCH_TOOL, notices: tuple[str, ...] = ()
) -> ToolMessage:
    """A ToolMessage shaped exactly as the leaf + langchain_adapter produce one."""
    projection = SearchResultsMetadata.from_candidate_set(
        query="Ohio WARN notices",
        candidate_set=CandidateSet(candidates=tuple(candidates), notices=notices),
    )
    return ToolMessage(
        content="1. a page\n   URL: ...",
        tool_call_id="tc-1",
        name=name,
        artifact={SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()},
    )


# ===========================================================================
# _verify_candidate_page
# ===========================================================================


def _client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestVerifyCandidatePage:
    async def test_real_table_verifies_as_nodriver(self):
        html = "<html><body><table><tr><td>a</td></tr><tr><td>b</td></tr></table></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html.encode())

        verified, backend, note = await _verify_candidate_page("https://example.gov/x", client=_client_for(handler))
        assert verified is True
        assert backend == "nodriver"
        assert "table" in note

    async def test_single_row_table_does_not_verify(self):
        html = "<html><body><table><tr><td>only one row</td></tr></table></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html.encode())

        verified, backend, note = await _verify_candidate_page("https://example.gov/x", client=_client_for(handler))
        assert verified is False
        assert backend == "nodriver"

    async def test_document_link_verifies_as_document(self):
        html = '<html><body><a href="/notices/2026-warn.pdf">WARN notices</a></body></html>'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html.encode())

        verified, backend, note = await _verify_candidate_page("https://example.gov/x", client=_client_for(handler))
        assert verified is True
        assert backend == "document"
        assert ".pdf" in note

    async def test_table_wins_over_an_incidental_document_link_on_the_same_page(self):
        # Live-discovered (Maryland's real WARN page): a page can carry both a real notices
        # table AND an unrelated PDF link elsewhere (e.g. federal WARN regulations reference).
        # The table is the actual data source and must win.
        html = (
            "<html><body>"
            '<a href="/about/warn-act-regulations.pdf">Federal WARN Act regulations</a>'
            "<table><tr><td>Acme Corp</td></tr><tr><td>Beta Inc</td></tr></table>"
            "</body></html>"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html.encode())

        verified, backend, note = await _verify_candidate_page("https://example.gov/x", client=_client_for(handler))
        assert verified is True
        assert backend == "nodriver"
        assert "table" in note

    async def test_json_list_response_verifies_as_api(self):
        body = json.dumps({"records": [{"employer": "Acme"}, {"employer": "Beta"}]}).encode()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})

        verified, backend, note = await _verify_candidate_page("https://example.gov/api", client=_client_for(handler))
        assert verified is True
        assert backend == "api"

    async def test_json_object_with_no_list_does_not_verify_as_api(self):
        body = json.dumps({"status": "ok"}).encode()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=body, headers={"content-type": "application/json"})

        verified, backend, _ = await _verify_candidate_page("https://example.gov/api", client=_client_for(handler))
        assert verified is False

    async def test_no_structure_found_does_not_verify(self):
        html = "<html><body><p>Nothing to see here.</p></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html.encode())

        verified, backend, note = await _verify_candidate_page("https://example.gov/x", client=_client_for(handler))
        assert verified is False
        assert backend == "nodriver"
        assert "no table" in note.lower()

    async def test_fetch_failure_degrades_to_unverified_not_a_crash(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        verified, backend, note = await _verify_candidate_page("https://example.gov/x", client=_client_for(handler))
        assert verified is False
        assert backend == "nodriver"
        assert "could not fetch" in note

    async def test_no_injected_client_constructs_and_closes_its_own(self):
        html = "<html><body><p>none</p></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=html.encode())

        owned_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("threetears.scrape.page_finder.httpx.AsyncClient", return_value=owned_client) as ctor:
            verified, _, _ = await _verify_candidate_page("https://example.gov/x")

        ctor.assert_called_once()
        assert owned_client.is_closed
        assert verified is False


# ===========================================================================
# _extract_search_queries
# ===========================================================================


class TestExtractSearchQueries:
    def test_pulls_query_args_from_web_search_calls_only(self):
        # "threetears.web_search" is the ACTUAL name ToolExecutor records (WebSearchTool.mcp_name()),
        # not the bare "web_search" -- using the real name here is the regression test for the bug
        # Critic caught: the original filter hardcoded the bare string and never matched in production.
        calls = [
            {"name": "threetears.web_search", "args": {"query": "Ohio WARN notices"}},
            {"name": "threetears.web_fetch", "args": {"url": "https://example.gov"}},
            {"name": "threetears.web_search", "args": {"query": "Ohio layoff notices"}},
        ]
        assert _extract_search_queries(calls, "threetears.web_search") == ["Ohio WARN notices", "Ohio layoff notices"]

    def test_bare_web_search_name_does_not_match_the_real_bound_name(self):
        # Regression test: a call recorded under the bare "web_search" string (what the original
        # bug hardcoded) must NOT match when the real bound name is "threetears.web_search".
        calls = [{"name": "web_search", "args": {"query": "should not match"}}]
        assert _extract_search_queries(calls, "threetears.web_search") == []

    def test_no_search_calls_returns_empty(self):
        assert (
            _extract_search_queries([{"name": "threetears.web_fetch", "args": {"url": "x"}}], "threetears.web_search")
            == []
        )


# ===========================================================================
# find_target_page -- composition
# ===========================================================================


def _fake_tool_chat_model(response):
    """A fake chat model supporting .bind_tools(...).ainvoke(...) for ToolExecutor."""
    ainvoke_mock = AsyncMock(return_value=response)
    bound = SimpleNamespace(ainvoke=ainvoke_mock)
    unbound = SimpleNamespace(bind_tools=lambda tools: bound)
    return unbound, ainvoke_mock


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=text, tool_calls=[])


class TestFindTargetPage:
    async def test_converged_and_verified_candidate(self):
        loop_model, _ = _fake_tool_chat_model(_text_response("https://example.gov/warn is the page."))
        candidate = _CandidatePage(
            url="https://example.gov/warn",
            driver_backend_guess="nodriver",
            wait_for_guess=None,
            summary="the real WARN page",
        )
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model") as coercion_create,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "nodriver", "found a real HTML table with multiple rows")),
            ),
        ):
            coercion_create.return_value = SimpleNamespace(
                with_structured_output=lambda schema, **kw: SimpleNamespace(ainvoke=AsyncMock(return_value=candidate))
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert isinstance(result, PageFinderResult)
        assert result.url == "https://example.gov/warn"
        assert result.driver_backend == "nodriver"
        assert result.verified is True
        assert result.turns_used == 1

    async def test_verification_fails_falls_back_to_agents_verifiable_guess(self):
        loop_model, _ = _fake_tool_chat_model(_text_response("https://example.gov/notices.pdf is the page."))
        candidate = _CandidatePage(
            url="https://example.gov/notices.pdf", driver_backend_guess="document", wait_for_guess=None, summary="a PDF"
        )
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model") as coercion_create,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(False, "nodriver", "no table, document link, or JSON list found")),
            ),
        ):
            coercion_create.return_value = SimpleNamespace(
                with_structured_output=lambda schema, **kw: SimpleNamespace(ainvoke=AsyncMock(return_value=candidate))
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.verified is False
        assert result.driver_backend == "document"  # agent's own guess is a verifiable backend, so it's used

    async def test_verification_fails_and_guess_unverifiable_defaults_to_nodriver(self):
        loop_model, _ = _fake_tool_chat_model(_text_response("https://example.gov/dashboard is the page."))
        candidate = _CandidatePage(
            url="https://example.gov/dashboard",
            driver_backend_guess="camoufox",
            wait_for_guess=None,
            summary="a JS dashboard",
        )
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model") as coercion_create,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(False, "nodriver", "no table, document link, or JSON list found")),
            ),
        ):
            coercion_create.return_value = SimpleNamespace(
                with_structured_output=lambda schema, **kw: SimpleNamespace(ainvoke=AsyncMock(return_value=candidate))
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.verified is False
        assert result.driver_backend == "nodriver"  # camoufox is never guessable -- falls back

    async def test_coercion_failure_degrades_without_crashing(self):
        loop_model, _ = _fake_tool_chat_model(_text_response("I couldn't find a clear answer."))
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model") as coercion_create,
        ):
            coercion_create.return_value = SimpleNamespace(
                with_structured_output=lambda schema, **kw: SimpleNamespace(
                    ainvoke=AsyncMock(side_effect=RuntimeError("boom"))
                )
            )
            with patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()):
                result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.verified is False
        assert result.url == ""
        assert "could not coerce" in result.verification_note

    async def test_turn_exhaustion_with_no_usable_output_returns_honest_result(self):
        # ToolExecutor sets error="max rounds exhausted" only when at least one tool call was
        # made; simulate that shape directly rather than re-driving the real round loop (already
        # covered by packages/agent/tools/tests/test_executor.py -- not this module's job to retest).
        exhausted_loop_result = SimpleNamespace(
            output="",
            rounds_used=3,
            tool_calls_made=[{"name": "threetears.web_search", "args": {"query": "Ohio WARN"}}],
            error="max rounds exhausted",
        )
        with patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls:
            executor_cls.return_value.invoke_with_tools = AsyncMock(return_value=exhausted_loop_result)
            with patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ):
                result = await find_target_page(
                    "Ohio WARN notices", api_key="k", searxng_url="http://searx.local", max_turns=3
                )

        assert result.verified is False
        assert result.turns_used == 3
        assert result.search_queries_tried == ["Ohio WARN"]
        assert "exhausted" in result.verification_note


# ===========================================================================
# _read_search_structure -- structure off metadata (search-spec.md check 4)
# ===========================================================================


class TestReadSearchStructure:
    def test_reads_the_typed_projection_off_a_tool_message_artifact(self):
        messages = [HumanMessage(content="find it"), _search_tool_message(_candidate("https://example.gov/warn"))]

        projections = _read_search_structure(messages, _SEARCH_TOOL)

        assert len(projections) == 1
        assert projections[0].candidates[0].identity == "https://example.gov/warn"

    def test_web_fetch_structure_is_not_read_as_a_search_result(self):
        # web_fetch writes its own projection under the SAME metadata key, so an
        # unfiltered scan would report a fetched page as a candidate the search
        # returned. The bound-name filter is the whole defence.
        messages = [_search_tool_message(_candidate("https://example.gov/a"), name=_FETCH_TOOL)]

        assert _read_search_structure(messages, _SEARCH_TOOL) == []

    def test_bare_tool_name_does_not_match_the_real_bound_name(self):
        # The same name-grain bug _extract_search_queries documents, one layer up.
        messages = [_search_tool_message(_candidate("https://example.gov/a"))]

        assert _read_search_structure(messages, "web_search") == []

    def test_a_message_with_no_artifact_is_skipped(self):
        messages = [ToolMessage(content="plain prose", tool_call_id="tc-1", name=_SEARCH_TOOL)]

        assert _read_search_structure(messages, _SEARCH_TOOL) == []

    def test_a_schema_version_newer_than_this_reader_degrades_instead_of_raising(self):
        # from_metadata refuses a newer schema loudly (D13). find_target_page
        # promises never to raise, so here that refusal must become a skip.
        payload = SearchResultsMetadata.from_candidate_set(
            query="q", candidate_set=CandidateSet(candidates=(_candidate("https://example.gov/a"),))
        ).to_metadata()
        payload["schema_version"] = 9999
        messages = [
            ToolMessage(
                content="prose",
                tool_call_id="tc-1",
                name=_SEARCH_TOOL,
                artifact={SEARCH_RESULTS_METADATA_KEY: payload},
            )
        ]

        assert _read_search_structure(messages, _SEARCH_TOOL) == []


class TestDedupeCandidates:
    def test_identity_dedupes_across_turns_and_keeps_first_order(self):
        turn_one = SearchResultsMetadata.from_candidate_set(
            query="q",
            candidate_set=CandidateSet(candidates=(_candidate("https://a.gov"), _candidate("https://b.gov"))),
        )
        turn_two = SearchResultsMetadata.from_candidate_set(
            query="q",
            candidate_set=CandidateSet(candidates=(_candidate("https://b.gov"), _candidate("https://c.gov"))),
        )

        assert [c.identity for c in _dedupe_candidates([turn_one, turn_two])] == [
            "https://a.gov",
            "https://b.gov",
            "https://c.gov",
        ]


class TestFirstFailure:
    def test_a_typed_failure_is_rendered_class_first(self):
        projection = SearchResultsMetadata.from_failure(
            query="q",
            failure=FailureRecord(failure_class="rate-limited", message="slow down", spend=Spend()),
        )

        assert _first_failure([projection]) == "rate-limited: slow down"

    def test_zero_results_is_not_a_failure(self):
        # SR-J2: an empty candidate set is a success value.
        projection = SearchResultsMetadata.from_candidate_set(query="q", candidate_set=CandidateSet())

        assert _first_failure([projection]) is None


# ===========================================================================
# find_target_page -- the structure seam, end to end
# ===========================================================================


def _executor_that_deposits(*tool_messages: ToolMessage, output: str, error: str | None = None):
    """A ToolExecutor stand-in that mutates `messages` in place, as the real one does.

    ToolExecutor appends each tool's ToolMessage (artifact intact, §4.7) to the
    caller-supplied list rather than returning them on ToolExecutionResult, so
    that in-place mutation IS page_finder's structure seam. Faking the executor
    without reproducing it would test a path production does not have.
    """

    async def invoke_with_tools(chat_model, messages, service_tools):
        messages.extend(tool_messages)
        return SimpleNamespace(
            output=output,
            rounds_used=2,
            tool_calls_made=[{"name": _SEARCH_TOOL, "args": {"query": "Ohio WARN"}}],
            error=error,
        )

    return invoke_with_tools


def _coercion_returning(candidate):
    return SimpleNamespace(
        with_structured_output=lambda schema, **kw: SimpleNamespace(ainvoke=AsyncMock(return_value=candidate))
    )


class TestFindTargetPageReadsStructure:
    async def test_typed_candidates_reach_the_result_and_the_url_is_recognised(self):
        found = _CandidatePage(
            url="https://example.gov/warn",
            driver_backend_guess="nodriver",
            wait_for_guess=None,
            summary="the real WARN page",
        )
        deposited = _search_tool_message(_candidate("https://example.gov/warn"), _candidate("https://other.gov/x"))
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "nodriver", "found a real HTML table with multiple rows")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                deposited, output="https://example.gov/warn is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert [c.identity for c in result.candidates_seen] == ["https://example.gov/warn", "https://other.gov/x"]
        assert result.url_was_a_search_result is True
        assert result.verified is True

    async def test_a_url_no_search_returned_is_marked_as_such(self):
        # The coercion step can name a page the loop reached by following a
        # fetched link -- or one it invented. Either way it was not *found*,
        # and before structure crossed the border the two were indistinguishable.
        found = _CandidatePage(
            url="https://invented.gov/nope", driver_backend_guess=None, wait_for_guess=None, summary="hmm"
        )
        deposited = _search_tool_message(_candidate("https://example.gov/warn"))
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(False, "nodriver", "no table, document link, or JSON list found")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                deposited, output="https://invented.gov/nope is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.url == "https://invented.gov/nope"
        assert result.url_was_a_search_result is False

    async def test_provider_notices_survive_to_the_result(self):
        found = _CandidatePage(
            url="https://example.gov/warn", driver_backend_guess=None, wait_for_guess=None, summary="ok"
        )
        deposited = _search_tool_message(
            _candidate("https://example.gov/warn"), notices=("two engines did not answer",)
        )
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "nodriver", "found a real HTML table with multiple rows")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                deposited, output="https://example.gov/warn is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.search_notices == ("two engines did not answer",)
        assert result.verified is True  # a degraded search still yields a real finding

    async def test_a_refused_search_is_named_rather_than_blamed_on_turn_exhaustion(self):
        # The behaviour change structure buys: before, every empty run reported
        # "exhausted its turn budget" regardless of why it actually came up dry.
        refused = SearchResultsMetadata.from_failure(
            query="Ohio WARN notices",
            failure=FailureRecord(failure_class="rate-limited", message="slow down", spend=Spend()),
        )
        deposited = ToolMessage(
            content="search failed",
            tool_call_id="tc-1",
            name=_SEARCH_TOOL,
            artifact={SEARCH_RESULTS_METADATA_KEY: refused.to_metadata()},
        )
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                deposited, output="", error="max rounds exhausted"
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.search_failure == "rate-limited: slow down"
        assert "search refused" in result.verification_note
        assert result.url == ""

    async def test_no_structure_at_all_leaves_the_new_fields_at_their_defaults(self):
        # A loop that never called search (or an older tool that carried no
        # metadata) must still produce exactly the result it did before.
        found = _CandidatePage(
            url="https://example.gov/warn", driver_backend_guess=None, wait_for_guess=None, summary="ok"
        )
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "nodriver", "found a real HTML table with multiple rows")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                output="https://example.gov/warn is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.candidates_seen == ()
        assert result.url_was_a_search_result is False
        assert result.search_notices == ()
        assert result.search_failure is None
        assert result.verified is True


# ===========================================================================
# The seam, with nothing stubbed at the interesting joints
# ===========================================================================


def _searxng_body(*urls: str) -> bytes:
    """A minimal real-shaped SearXNG ``format=json`` envelope.

    Written here rather than imported: `packages/search/tests/_searxng_payloads.py`
    is that package's own test module, not published surface this one may reach
    into. Only the fields the adapter needs to build a Candidate appear.
    """
    return json.dumps(
        {
            "query": "Ohio WARN notices",
            "number_of_results": len(urls),
            "results": [
                {
                    "url": url,
                    "title": "Ohio WARN notices",
                    "content": "A list of WARN notices.",
                    "engine": "duckduckgo",
                    "engines": ["duckduckgo"],
                    "positions": [i],
                    "score": 1.0 / i,
                    "category": "general",
                    "template": "default.html",
                }
                for i, url in enumerate(urls, 1)
            ],
            "unresponsive_engines": [],
        }
    ).encode()


def _chat_model_calling_search_once(tool_name: str):
    """A chat model that calls the search tool on round 1, then answers in prose."""
    responses = [
        SimpleNamespace(
            content="",
            tool_calls=[{"name": tool_name, "args": {"query": "Ohio WARN notices"}, "id": "tc-1", "type": "tool_call"}],
        ),
        SimpleNamespace(content="https://example.gov/warn is the page.", tool_calls=[]),
    ]
    return SimpleNamespace(ainvoke=AsyncMock(side_effect=responses))


class TestTheStructureSeamAgainstARealSocket:
    """Drive the real tool, the real adapter and the real ToolExecutor.

    Every test above builds its ToolMessage by hand, which pins what
    `_read_search_structure` does with a message but assumes the two facts it
    depends on: that LangChain stamps the bound tool name onto `ToolMessage.name`
    and that ToolExecutor lets the artifact through. Assuming those is exactly
    the gap #321's review found -- a tool and its transport each tested alone,
    with the seam between them tested nowhere -- so they get asserted here
    against a real socket instead.
    """

    async def test_a_real_search_turn_lands_readable_structure_in_the_messages(self):
        from threetears.search.testing import LocalHttpServer, Reply

        from threetears.agent.tools import ToolExecutor
        from threetears.agent.tools.builtin.web_search import create_web_search_tool

        async with LocalHttpServer(
            (Reply(body=_searxng_body("https://example.gov/warn"), headers={"content-type": "application/json"}),)
        ) as server:
            search_tool = create_web_search_tool({"base_url": server.base_url}, "Search the web.")
            messages: list = [HumanMessage(content="Find the Ohio WARN page")]
            result = await ToolExecutor(max_rounds=3).invoke_with_tools(
                _chat_model_calling_search_once(search_tool.name), messages, [search_tool]
            )

        assert result.output == "https://example.gov/warn is the page."

        # The two assumed facts, now asserted: the bound name is stamped on the
        # message, and the artifact survived the executor.
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert [m.name for m in tool_messages] == ["threetears.web_search"]
        assert isinstance(tool_messages[0].artifact, dict)

        projections = _read_search_structure(messages, search_tool.name)
        assert len(projections) == 1
        assert [c.identity for c in _dedupe_candidates(projections)] == ["https://example.gov/warn"]


# ===========================================================================
# The helpers' remaining branches
# ===========================================================================


def _candidate_with_locators(identity: str, *urls: str) -> Candidate:
    """A candidate whose reachable URLs deliberately differ from its identity."""
    return Candidate(
        identity=identity,
        locators=tuple(Locator(url=u, rel="direct-file") for u in urls),
        provenance=Provenance(
            query="Ohio WARN notices",
            provider_instance="searxng-local",
            retrieved_at=datetime(2026, 8, 12, tzinfo=UTC),
        ),
    )


class TestCandidateUrls:
    def test_identity_counts_as_a_reachable_url(self):
        assert "https://a.gov" in _candidate_urls((_candidate("https://a.gov"),))

    def test_a_non_canonical_locator_counts_too(self):
        # identity is the canonical URL *by convention*, not by guarantee, and the
        # URL an LLM names is whichever one the prose rendering showed it -- so a
        # direct-file locator must count as "the search returned this".
        candidates = (_candidate_with_locators("provider-native-id-42", "https://a.gov/notices.pdf"),)

        urls = _candidate_urls(candidates)

        assert "https://a.gov/notices.pdf" in urls
        assert "provider-native-id-42" in urls

    def test_no_candidates_is_an_empty_set_not_a_crash(self):
        assert _candidate_urls(()) == set()


class TestDedupeCandidatesEdges:
    def test_no_projections_yields_nothing(self):
        assert _dedupe_candidates([]) == ()

    def test_a_zero_result_turn_contributes_nothing(self):
        # SR-J2: zero results is a success, and must not become a phantom candidate.
        empty = SearchResultsMetadata.from_candidate_set(query="q", candidate_set=CandidateSet())

        assert _dedupe_candidates([empty]) == ()

    def test_first_occurrence_wins_so_the_earlier_turns_locators_survive(self):
        first = SearchResultsMetadata.from_candidate_set(
            query="q",
            candidate_set=CandidateSet(candidates=(_candidate_with_locators("same-id", "https://first.gov"),)),
        )
        second = SearchResultsMetadata.from_candidate_set(
            query="q",
            candidate_set=CandidateSet(candidates=(_candidate_with_locators("same-id", "https://second.gov"),)),
        )

        deduped = _dedupe_candidates([first, second])

        assert len(deduped) == 1
        assert [loc.url for loc in deduped[0].locators] == ["https://first.gov"]


class TestFirstFailureEdges:
    def test_the_first_failure_wins_when_several_turns_failed(self):
        def failed(cls_: str, msg: str) -> SearchResultsMetadata:
            return SearchResultsMetadata.from_failure(
                query="q", failure=FailureRecord(failure_class=cls_, message=msg, spend=Spend())
            )

        assert _first_failure([failed("rate-limited", "first"), failed("timeout", "second")]) == "rate-limited: first"

    def test_a_failure_after_a_successful_turn_is_still_reported(self):
        ok = SearchResultsMetadata.from_candidate_set(
            query="q", candidate_set=CandidateSet(candidates=(_candidate("https://a.gov"),))
        )
        bad = SearchResultsMetadata.from_failure(
            query="q", failure=FailureRecord(failure_class="timeout", message="too slow", spend=Spend())
        )

        assert _first_failure([ok, bad]) == "timeout: too slow"

    def test_no_projections_is_no_failure(self):
        assert _first_failure([]) is None


class TestReadSearchStructureEdges:
    def test_several_search_turns_each_yield_a_projection(self):
        messages = [
            _search_tool_message(_candidate("https://a.gov")),
            HumanMessage(content="keep looking"),
            _search_tool_message(_candidate("https://b.gov")),
        ]

        assert len(_read_search_structure(messages, _SEARCH_TOOL)) == 2

    def test_an_artifact_that_is_not_a_dict_is_skipped(self):
        messages = [ToolMessage(content="prose", tool_call_id="tc-1", name=_SEARCH_TOOL, artifact="not a dict")]

        assert _read_search_structure(messages, _SEARCH_TOOL) == []

    def test_an_artifact_without_the_named_key_is_skipped(self):
        messages = [
            ToolMessage(content="prose", tool_call_id="tc-1", name=_SEARCH_TOOL, artifact={"something_else": {}})
        ]

        assert _read_search_structure(messages, _SEARCH_TOOL) == []

    def test_a_payload_that_is_not_a_dict_is_skipped(self):
        messages = [
            ToolMessage(
                content="prose", tool_call_id="tc-1", name=_SEARCH_TOOL, artifact={SEARCH_RESULTS_METADATA_KEY: "nope"}
            )
        ]

        assert _read_search_structure(messages, _SEARCH_TOOL) == []

    def test_the_current_schema_version_is_accepted(self):
        # The boundary the refusal test above does not pin: equal-to-current must
        # read, or the reader would refuse every payload the family actually emits.
        payload = SearchResultsMetadata.from_candidate_set(
            query="q", candidate_set=CandidateSet(candidates=(_candidate("https://a.gov"),))
        ).to_metadata()
        messages = [
            ToolMessage(
                content="prose",
                tool_call_id="tc-1",
                name=_SEARCH_TOOL,
                artifact={SEARCH_RESULTS_METADATA_KEY: payload},
            )
        ]

        assert len(_read_search_structure(messages, _SEARCH_TOOL)) == 1

    def test_messages_that_are_not_tool_messages_are_ignored(self):
        assert (
            _read_search_structure([HumanMessage(content="hi"), SimpleNamespace(name=_SEARCH_TOOL)], _SEARCH_TOOL) == []
        )


class TestAllNotices:
    def test_notices_dedupe_across_turns_and_keep_first_order(self):
        def with_notices(*notices: str) -> SearchResultsMetadata:
            return SearchResultsMetadata.from_candidate_set(query="q", candidate_set=CandidateSet(notices=notices))

        gathered = _all_notices([with_notices("engine A down", "unranked"), with_notices("unranked", "engine B down")])

        assert gathered == ("engine A down", "unranked", "engine B down")

    def test_no_notices_is_an_empty_tuple(self):
        assert _all_notices([]) == ()


# ===========================================================================
# check 4's own wording: "without its callers changing"
# ===========================================================================


class TestExistingCallersAreUnaffected:
    def test_the_result_still_constructs_from_only_its_original_fields(self):
        # This IS success check 4's second clause, as an assertion rather than a
        # claim in a PR body: every field structure added carries a default, so
        # code written before the metadata border existed still builds a result.
        result = PageFinderResult(
            url="https://example.gov/warn",
            driver_backend="nodriver",
            wait_for=None,
            verified=True,
            verification_note="found a real HTML table with multiple rows",
            reasoning="the real WARN page",
            turns_used=2,
        )

        assert result.candidates_seen == ()
        assert result.url_was_a_search_result is False
        assert result.search_notices == ()
        assert result.search_failure is None
        assert result.search_queries_tried == []


class TestFindTargetPageStructureEdges:
    async def test_a_url_matching_only_a_locator_still_counts_as_found(self):
        # The PDF a provider lists as a direct-file locator is a page the search
        # genuinely returned, even though identity names the containing page.
        found = _CandidatePage(
            url="https://example.gov/notices.pdf", driver_backend_guess=None, wait_for_guess=None, summary="the PDF"
        )
        deposited = _search_tool_message(
            _candidate_with_locators("https://example.gov/warn", "https://example.gov/notices.pdf")
        )
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "document", "found a document link (/notices.pdf)")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                deposited, output="https://example.gov/notices.pdf is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.url_was_a_search_result is True
        assert result.driver_backend == "document"

    async def test_the_coercion_failure_path_still_reports_what_the_search_found(self):
        # A run that could not turn prose into a URL has still learned real
        # things about the web, and dropping them would waste the spend.
        deposited = _search_tool_message(_candidate("https://example.gov/warn"), notices=("one engine did not answer",))
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model") as coercion_create,
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            coercion_create.return_value = SimpleNamespace(
                with_structured_output=lambda schema, **kw: SimpleNamespace(
                    ainvoke=AsyncMock(side_effect=RuntimeError("boom"))
                )
            )
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                deposited, output="I found something but cannot say what."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.url == ""
        assert "could not coerce" in result.verification_note
        assert [c.identity for c in result.candidates_seen] == ["https://example.gov/warn"]
        assert result.search_notices == ("one engine did not answer",)

    async def test_notices_from_several_turns_are_gathered_and_deduplicated(self):
        found = _CandidatePage(
            url="https://example.gov/warn", driver_backend_guess=None, wait_for_guess=None, summary="ok"
        )
        turn_one = _search_tool_message(_candidate("https://a.gov"), notices=("unranked", "engine A down"))
        turn_two = _search_tool_message(_candidate("https://example.gov/warn"), notices=("unranked", "engine B down"))
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "nodriver", "found a real HTML table with multiple rows")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                turn_one, turn_two, output="https://example.gov/warn is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.search_notices == ("unranked", "engine A down", "engine B down")
        assert [c.identity for c in result.candidates_seen] == ["https://a.gov", "https://example.gov/warn"]

    async def test_a_failed_turn_is_reported_even_when_a_later_turn_saved_the_run(self):
        # search_failure is a record of what happened, not a verdict on the run:
        # the loop recovered and found a page, and both facts are true at once.
        found = _CandidatePage(
            url="https://example.gov/warn", driver_backend_guess=None, wait_for_guess=None, summary="ok"
        )
        refused = SearchResultsMetadata.from_failure(
            query="Ohio WARN notices",
            failure=FailureRecord(failure_class="rate-limited", message="slow down", spend=Spend()),
        )
        failed_turn = ToolMessage(
            content="search failed",
            tool_call_id="tc-1",
            name=_SEARCH_TOOL,
            artifact={SEARCH_RESULTS_METADATA_KEY: refused.to_metadata()},
        )
        good_turn = _search_tool_message(_candidate("https://example.gov/warn"))
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "nodriver", "found a real HTML table with multiple rows")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                failed_turn, good_turn, output="https://example.gov/warn is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.search_failure == "rate-limited: slow down"
        assert result.verified is True
        assert result.url_was_a_search_result is True

    async def test_web_fetch_structure_never_becomes_a_search_candidate_end_to_end(self):
        # The filter's real consequence: a page the loop FETCHED must not be
        # reported as one the search RETURNED, or url_was_a_search_result lies.
        found = _CandidatePage(
            url="https://fetched.gov/page", driver_backend_guess=None, wait_for_guess=None, summary="ok"
        )
        fetch_turn = _search_tool_message(_candidate("https://fetched.gov/page"), name=_FETCH_TOOL)
        with (
            patch(
                "threetears.scrape.page_finder.create_chat_model",
                return_value=SimpleNamespace(bind_tools=lambda t: None),
            ),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
            patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
            patch(
                "threetears.scrape.page_finder._verify_candidate_page",
                AsyncMock(return_value=(True, "nodriver", "found a real HTML table with multiple rows")),
            ),
        ):
            executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
                fetch_turn, output="https://fetched.gov/page is the page."
            )
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.candidates_seen == ()
        assert result.url_was_a_search_result is False
        assert result.verified is True

    async def test_zero_results_over_a_real_socket_is_a_success_with_no_candidates(self):
        # SR-J2: an empty answer is a success value. It must reach the reader as a
        # projection carrying nothing -- never as a failure, never as no projection.
        from threetears.search.testing import LocalHttpServer, Reply

        from threetears.agent.tools import ToolExecutor
        from threetears.agent.tools.builtin.web_search import create_web_search_tool

        async with LocalHttpServer(
            (Reply(body=_searxng_body(), headers={"content-type": "application/json"}),)
        ) as server:
            search_tool = create_web_search_tool({"base_url": server.base_url}, "Search the web.")
            messages: list = [HumanMessage(content="Find the Ohio WARN page")]
            await ToolExecutor(max_rounds=3).invoke_with_tools(
                _chat_model_calling_search_once(search_tool.name), messages, [search_tool]
            )

        projections = _read_search_structure(messages, search_tool.name)
        assert len(projections) == 1
        assert _dedupe_candidates(projections) == ()
        assert _first_failure(projections) is None

    async def test_a_provider_error_over_a_real_socket_arrives_as_a_typed_failure(self):
        # D10: nothing raises across the border. The failure must reach the reader
        # as a named class -- which is what lets page_finder tell "refused" from
        # "found nothing" without matching on an error prefix in prose.
        from threetears.search.testing import LocalHttpServer, Reply

        from threetears.agent.tools import ToolExecutor
        from threetears.agent.tools.builtin.web_search import create_web_search_tool

        async with LocalHttpServer((Reply(status=500, body=b"upstream is unwell"),)) as server:
            search_tool = create_web_search_tool({"base_url": server.base_url}, "Search the web.")
            messages: list = [HumanMessage(content="Find the Ohio WARN page")]
            await ToolExecutor(max_rounds=3).invoke_with_tools(
                _chat_model_calling_search_once(search_tool.name), messages, [search_tool]
            )

        projections = _read_search_structure(messages, search_tool.name)
        assert len(projections) == 1
        # The class is named, not merely present: "transport-failed" is what an
        # operator acts on, and it is the fact the old [TOOL ERROR] prefix could
        # not carry. Three attempts happened underneath -- the transport's own
        # bounded retry -- and the border still reports one typed outcome.
        assert _first_failure(projections).startswith("transport-failed: ")
        assert _dedupe_candidates(projections) == ()

    def test_a_structurally_invalid_payload_degrades_the_same_way(self):
        # from_metadata validates as well as version-checks, and pydantic's
        # ValidationError is a ValueError -- so a malformed payload takes the same
        # degrade-to-prose path as a too-new one rather than escaping as a crash.
        messages = [
            ToolMessage(
                content="prose",
                tool_call_id="tc-1",
                name=_SEARCH_TOOL,
                artifact={SEARCH_RESULTS_METADATA_KEY: {"schema_version": 1, "candidates": "not a tuple"}},
            )
        ]

        assert _read_search_structure(messages, _SEARCH_TOOL) == []


# ===========================================================================
# The verification fetch: size cap, encodings, and lookalike links
# ===========================================================================

_TABLE = "<table><tr><td>{}</td></tr><tr><td>b</td></tr></table>"


def _serving(body: bytes, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers=headers or {})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestVerificationSizeCap:
    async def test_a_body_over_the_cap_is_truncated_and_says_so(self):
        # The fetch used to be unbounded: client.get buffered the whole body and
        # BeautifulSoup built a parse tree from it, measured at 77x the served
        # size. find_target_page fetches a URL an LLM picked out of search
        # results, so that size is not this process's to choose.
        oversized = b"<html><body>" + b"<p>pad</p>" * 300_000 + b"</body></html>"
        assert len(oversized) > _VERIFY_MAX_BYTES

        verified, backend, note = await _verify_candidate_page(
            "https://example.gov/huge", client=_serving(oversized, {"content-type": "text/html"})
        )

        assert verified is False
        assert backend == "nodriver"
        # "nothing in the part I read" is a weaker claim than "nothing on the page".
        assert str(_VERIFY_MAX_BYTES) in note
        assert "longer than the verification cap" in note

    async def test_structure_inside_the_cap_still_verifies_on_an_oversized_page(self):
        # The cap must not cost a real finding: a table in the first 2 MiB is
        # found regardless of how much padding follows it.
        body = b"<html><body>" + _TABLE.format("Acme Corp").encode() + b"<p>pad</p>" * 300_000 + b"</body></html>"
        assert len(body) > _VERIFY_MAX_BYTES

        verified, backend, note = await _verify_candidate_page(
            "https://example.gov/big-but-real", client=_serving(body, {"content-type": "text/html"})
        )

        assert verified is True
        assert backend == "nodriver"
        assert "table" in note

    async def test_a_body_exactly_at_the_cap_is_not_reported_as_truncated(self):
        # Boundary: read > cap is truncation, read == cap is a whole document.
        filler = b"x" * (_VERIFY_MAX_BYTES - len(b"<html><body></body></html>"))
        body = b"<html><body>" + filler + b"</body></html>"
        body = body[:_VERIFY_MAX_BYTES]

        verified, _, note = await _verify_candidate_page(
            "https://example.gov/exact", client=_serving(body, {"content-type": "text/html"})
        )

        assert verified is False
        assert "verification cap" not in note

    async def test_a_truncated_json_body_is_not_called_an_api(self):
        # A cut-off document is not parseable JSON, and guessing at the missing
        # half would verify a page nobody has seen the end of.
        head = b'{"records": [' + b'{"employer": "Acme"},' * 200_000
        assert len(head) > _VERIFY_MAX_BYTES

        verified, backend, _ = await _verify_candidate_page(
            "https://example.gov/api", client=_serving(head, {"content-type": "application/json"})
        )

        assert verified is False
        assert backend == "nodriver"


class TestVerificationEncodings:
    async def test_declared_shift_jis_is_decoded_and_structure_found(self):
        body = ("<html><body>" + _TABLE.format("日本語のデータ") + "</body></html>").encode("shift_jis")

        verified, backend, _ = await _verify_candidate_page(
            "https://example.jp/warn", client=_serving(body, {"content-type": "text/html; charset=shift_jis"})
        )

        assert verified is True
        assert backend == "nodriver"

    async def test_declared_windows_1256_arabic_is_decoded_and_structure_found(self):
        body = ("<html><body>" + _TABLE.format("بيانات التسريح") + "</body></html>").encode("cp1256")

        verified, backend, _ = await _verify_candidate_page(
            "https://example.gov/ar", client=_serving(body, {"content-type": "text/html; charset=windows-1256"})
        )

        assert verified is True
        assert backend == "nodriver"

    async def test_declared_utf_16_is_decoded_and_structure_found(self):
        body = ("<html><body>" + _TABLE.format("data") + "</body></html>").encode("utf-16")

        verified, _, _ = await _verify_candidate_page(
            "https://example.gov/u16", client=_serving(body, {"content-type": "text/html; charset=utf-16"})
        )

        assert verified is True

    async def test_undeclared_non_utf8_still_finds_structure(self):
        # No charset header, bytes that are not UTF-8. The text mis-decodes and
        # that is fine: every marker this function looks for -- <table>, <tr>,
        # href -- is ASCII, so structure detection does not depend on getting
        # the human-readable text right.
        body = ("<html><body>" + _TABLE.format("日本語") + "</body></html>").encode("shift_jis")

        verified, _, _ = await _verify_candidate_page(
            "https://example.jp/nocharset", client=_serving(body, {"content-type": "text/html"})
        )

        assert verified is True

    async def test_invalid_byte_sequences_do_not_raise(self):
        # errors="replace", never strict: a page whose bytes contradict its
        # declared charset must still be inspectable rather than crash a run
        # that promises never to raise.
        body = b"<html><body>\xff\xfe\x00\x81" + _TABLE.format("x").encode() + b"</body></html>"

        verified, _, _ = await _verify_candidate_page(
            "https://example.gov/broken", client=_serving(body, {"content-type": "text/html; charset=utf-8"})
        )

        assert verified is True

    async def test_a_body_cut_mid_multibyte_character_does_not_raise(self):
        # The cap can slice a UTF-8 sequence in half. That must degrade to a
        # replacement character, not an exception.
        body = b"<html><body>" + ("日" * 2_000_000).encode() + b"</body></html>"
        assert len(body) > _VERIFY_MAX_BYTES

        verified, _, note = await _verify_candidate_page(
            "https://example.jp/multibyte", client=_serving(body, {"content-type": "text/html; charset=utf-8"})
        )

        assert verified is False
        assert "verification cap" in note


class TestVerificationLookalikeLinks:
    async def test_a_bidi_override_link_that_merely_looks_like_a_pdf_is_not_one(self):
        # U+202E renders "/annexgnp.fdp" as if it ended in .pdf. The extension
        # check reads the real characters, so the spoof does not verify -- and
        # this pins that it stays that way.
        body = '<html><body><a href="/annex‮gnp.fdp">Notices</a></body></html>'.encode()

        verified, backend, _ = await _verify_candidate_page(
            "https://example.gov/spoof", client=_serving(body, {"content-type": "text/html"})
        )

        assert verified is False
        assert backend == "nodriver"

    async def test_a_genuine_rtl_named_pdf_still_verifies(self):
        # The converse: an Arabic-named PDF is a real document link and must not
        # be collateral damage of the check above.
        body = '<html><body><a href="/إشعارات.pdf">PDF</a></body></html>'.encode()

        verified, backend, note = await _verify_candidate_page(
            "https://example.gov/ar-pdf", client=_serving(body, {"content-type": "text/html"})
        )

        assert verified is True
        assert backend == "document"
        assert ".pdf" in note


# ===========================================================================
# Exotic text across the metadata border
# ===========================================================================


class TestStructureCarriesExoticText:
    def test_rtl_cjk_and_emoji_survive_the_metadata_round_trip(self):
        # to_metadata is model_dump(mode="json") and from_metadata revalidates,
        # so this crosses the same projection a NATS hop would. JSON is unicode
        # end to end; nothing here should need escaping to survive.
        title = "إشعارات التسريح 日本語 🏛️ notices"
        messages = [_search_tool_message(_candidate("https://example.gov/warn", title=title))]

        projections = _read_search_structure(messages, _SEARCH_TOOL)

        assert projections[0].candidates[0].title == title

    def test_a_bidi_override_in_a_title_is_carried_verbatim_not_stripped(self):
        # The border's job is to carry what the provider said, exactly. Deciding
        # what to do about a title that renders deceptively is a rendering
        # concern, and silently mutating it here would hide the fact from
        # whoever does have to make that call.
        title = "annex‮gnp.fdp"
        messages = [_search_tool_message(_candidate("https://example.gov/x", title=title))]

        assert _read_search_structure(messages, _SEARCH_TOOL)[0].candidates[0].title == title

    def test_a_non_ascii_url_matches_itself_when_checking_membership(self):
        # url_was_a_search_result is a string comparison, so an IDN or a
        # percent-encoded path must match the form the candidate carries.
        url = "https://例え.jp/通知/warn%20notices.html"
        candidates = _dedupe_candidates(_read_search_structure([_search_tool_message(_candidate(url))], _SEARCH_TOOL))

        assert url in _candidate_urls(candidates)

    def test_percent_encoded_and_decoded_forms_are_not_treated_as_the_same_url(self):
        # Deliberately NOT normalised: this module does not own URL canonicalisation,
        # and quietly equating the two forms would report "the search returned this"
        # about a URL the search did not return. Recorded as a pin so the choice is
        # visible if someone later wants normalisation -- it belongs upstream, at the
        # adapter that mints identity, not in a membership check.
        candidates = _dedupe_candidates(
            _read_search_structure(
                [_search_tool_message(_candidate("https://example.gov/warn%20notices"))], _SEARCH_TOOL
            )
        )

        assert "https://example.gov/warn notices" not in _candidate_urls(candidates)

    def test_a_lone_surrogate_in_a_payload_degrades_rather_than_crashing(self):
        # A surrogate is unencodable to UTF-8 and can reach a reader via a
        # provider that emitted it. Whatever pydantic makes of it, this must not
        # be how find_target_page learns to raise.
        messages = [
            ToolMessage(
                content="prose",
                tool_call_id="tc-1",
                name=_SEARCH_TOOL,
                artifact={SEARCH_RESULTS_METADATA_KEY: {"schema_version": 1, "query": "\ud800 broken"}},
            )
        ]

        projections = _read_search_structure(messages, _SEARCH_TOOL)

        assert len(projections) <= 1  # read or skipped, never raised
