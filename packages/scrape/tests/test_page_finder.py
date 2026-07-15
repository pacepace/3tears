"""Unit tests for threetears.scrape.page_finder -- the bounded-turn page-finding
agent (mocking approach mirrors test_extraction.py's create_chat_model patching,
plus httpx.MockTransport for _verify_candidate_page per test_driver_api.py's
own convention).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from threetears.scrape.page_finder import (
    PageFinderResult,
    _CandidatePage,
    _extract_search_queries,
    _verify_candidate_page,
    find_target_page,
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
        assert _extract_search_queries([{"name": "threetears.web_fetch", "args": {"url": "x"}}], "threetears.web_search") == []


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
            url="https://example.gov/warn", driver_backend_guess="nodriver", wait_for_guess=None, summary="the real WARN page"
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
            url="https://example.gov/dashboard", driver_backend_guess="camoufox", wait_for_guess=None, summary="a JS dashboard"
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
            output="", rounds_used=3, tool_calls_made=[{"name": "threetears.web_search", "args": {"query": "Ohio WARN"}}],
            error="max rounds exhausted",
        )
        with patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls:
            executor_cls.return_value.invoke_with_tools = AsyncMock(return_value=exhausted_loop_result)
            with patch("threetears.scrape.page_finder.create_chat_model", return_value=SimpleNamespace(bind_tools=lambda t: None)):
                result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local", max_turns=3)

        assert result.verified is False
        assert result.turns_used == 3
        assert result.search_queries_tried == ["Ohio WARN"]
        assert "exhausted" in result.verification_note
