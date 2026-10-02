"""Unit tests for threetears.scrape.page_finder -- the bounded-turn page-finding agent.

Everything here goes through :func:`find_target_page`, the module's one entry point. The
search loop is faked at the seams production hands it -- ``create_chat_model``, the
``ToolExecutor`` that deposits tool messages into the caller's list, and the coercion model
in ``llm_retry`` -- and the structural verification fetch is driven through the
``verify_client`` it accepts (``httpx.MockTransport``, per test_driver_api.py's own
convention). What each test asserts is a field of the :class:`PageFinderResult` a caller
reads, so a helper's shape can change freely as long as the result does not.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
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

from threetears.scrape.page_finder import PageFinderResult, find_target_page

_SEARCH_TOOL = "threetears.web_search"
_FETCH_TOOL = "threetears.web_fetch"

#: the production cap on the verification fetch, 2 MiB.
_PRODUCTION_VERIFY_CAP = 2 * 1024 * 1024


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


def _search_tool_message(
    *candidates: Candidate, name: str = _SEARCH_TOOL, notices: tuple[str, ...] = ()
) -> ToolMessage:
    """A ToolMessage shaped exactly as the leaf + langchain_adapter produce one."""
    projection = SearchResultsMetadata.from_candidate_set(
        query="Ohio WARN notices",
        candidate_set=CandidateSet(candidates=tuple(candidates), notices=notices),
    )
    return _projection_message(projection, name=name)


def _projection_message(projection: SearchResultsMetadata, *, name: str = _SEARCH_TOOL) -> ToolMessage:
    """A search ToolMessage carrying *projection* as its metadata artifact."""
    return ToolMessage(
        content="1. a page\n   URL: ...",
        tool_call_id="tc-1",
        name=name,
        artifact={SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()},
    )


def _artifact_message(artifact: Any) -> ToolMessage:
    """A search ToolMessage carrying *artifact* verbatim, however malformed."""
    return ToolMessage(content="prose", tool_call_id="tc-1", name=_SEARCH_TOOL, artifact=artifact)


def _refused(cls_: str = "rate-limited", message: str = "slow down") -> ToolMessage:
    """A search turn the provider refused, as a typed failure."""
    return _projection_message(
        SearchResultsMetadata.from_failure(
            query="Ohio WARN notices",
            failure=FailureRecord(failure_class=cls_, message=message, spend=Spend()),
        )
    )


def _client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _serving(body: bytes, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers=headers or {})

    return _client_for(handler)


_TABLE = "<table><tr><td>{}</td></tr><tr><td>b</td></tr></table>"
_VERIFYING_PAGE = ("<html><body>" + _TABLE.format("a") + "</body></html>").encode()
_UNSTRUCTURED_PAGE = b"<html><body><p>Nothing to see here.</p></body></html>"


def _found(url: str = "https://example.gov/warn", *, guess: str | None = None, summary: str = "ok") -> dict[str, Any]:
    """What the coercion model concludes, as the field names a real model answers with."""
    return {"url": url, "driver_backend_guess": guess, "wait_for_guess": None, "summary": summary}


def _coercion_returning(found: dict[str, Any]) -> SimpleNamespace:
    """A coercion model answering *found*, as whatever schema the caller binds it to."""
    return SimpleNamespace(
        with_structured_output=lambda schema, **kw: SimpleNamespace(
            ainvoke=AsyncMock(return_value=schema.model_validate(found))
        )
    )


def _coercion_failing() -> SimpleNamespace:
    return SimpleNamespace(
        with_structured_output=lambda schema, **kw: SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("boom")))
    )


def _executor_that_deposits(
    *tool_messages: Any,
    output: str,
    error: str | None = None,
    tool_calls_made: list[dict[str, Any]] | None = None,
):
    """A ToolExecutor stand-in that mutates `messages` in place, as the real one does.

    ToolExecutor appends each tool's ToolMessage (artifact intact, §4.7) to the
    caller-supplied list rather than returning them on ToolExecutionResult, so
    that in-place mutation IS page_finder's structure seam. Faking the executor
    without reproducing it would test a path production does not have.
    """
    calls = tool_calls_made if tool_calls_made is not None else [{"name": _SEARCH_TOOL, "args": {"query": "Ohio WARN"}}]

    async def invoke_with_tools(chat_model, messages, service_tools):
        messages.extend(tool_messages)
        return SimpleNamespace(output=output, rounds_used=2, tool_calls_made=calls, error=error)

    return invoke_with_tools


class _VerifyingClient:
    """Marks "serve a page that verifies", which ``None`` (build your own client) cannot."""


_A_VERIFYING_CLIENT = _VerifyingClient()


async def _find(
    *deposited: Any,
    output: str = "https://example.gov/warn is the page.",
    error: str | None = None,
    found: dict[str, Any] | None = None,
    coercion: SimpleNamespace | None = None,
    verify_client: httpx.AsyncClient | _VerifyingClient | None = _A_VERIFYING_CLIENT,
    verify_max_bytes: int | None = None,
    tool_calls_made: list[dict[str, Any]] | None = None,
) -> PageFinderResult:
    """Run ``find_target_page`` with a search loop that deposits *deposited* and answers *output*."""
    client = (
        _serving(_VERIFYING_PAGE, {"content-type": "text/html"})
        if isinstance(verify_client, _VerifyingClient)
        else verify_client
    )
    extra: dict[str, Any] = {} if verify_max_bytes is None else {"verify_max_bytes": verify_max_bytes}
    with (
        patch(
            "threetears.scrape.page_finder.create_chat_model", return_value=SimpleNamespace(bind_tools=lambda t: None)
        ),
        patch(
            "threetears.scrape.llm_retry.create_chat_model",
            return_value=coercion if coercion is not None else _coercion_returning(found or _found()),
        ),
        patch("threetears.scrape.page_finder.ToolExecutor") as executor_cls,
        patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
    ):
        executor_cls.return_value.invoke_with_tools = _executor_that_deposits(
            *deposited, output=output, error=error, tool_calls_made=tool_calls_made
        )
        return await find_target_page(
            "Ohio WARN notices", api_key="k", searxng_url="http://searx.local", verify_client=client, **extra
        )


async def _verify(
    body: bytes, headers: dict[str, str] | None = None, *, max_bytes: int | None = None
) -> PageFinderResult:
    """Run a whole finding whose candidate page serves *body*; the coercion made no backend guess.

    With no guess, an unverified result's ``driver_backend`` is the structural fallback,
    ``nodriver`` -- the same value the verification itself reports for every unverified page.
    """
    return await _find(verify_client=_serving(body, headers), verify_max_bytes=max_bytes)


# ===========================================================================
# structural verification of the candidate page
# ===========================================================================


class TestVerifyCandidatePage:
    async def test_real_table_verifies_as_nodriver(self):
        result = await _verify(_VERIFYING_PAGE)
        assert result.verified is True
        assert result.driver_backend == "nodriver"
        assert "table" in result.verification_note

    async def test_single_row_table_does_not_verify(self):
        result = await _verify(b"<html><body><table><tr><td>only one row</td></tr></table></body></html>")
        assert result.verified is False
        assert result.driver_backend == "nodriver"

    async def test_document_link_verifies_as_document(self):
        result = await _verify(b'<html><body><a href="/notices/2026-warn.pdf">WARN notices</a></body></html>')
        assert result.verified is True
        assert result.driver_backend == "document"
        assert ".pdf" in result.verification_note

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
        result = await _verify(html.encode())
        assert result.verified is True
        assert result.driver_backend == "nodriver"
        assert "table" in result.verification_note

    async def test_json_list_response_verifies_as_api(self):
        body = json.dumps({"records": [{"employer": "Acme"}, {"employer": "Beta"}]}).encode()
        result = await _verify(body, {"content-type": "application/json"})
        assert result.verified is True
        assert result.driver_backend == "api"

    async def test_json_object_with_no_list_does_not_verify_as_api(self):
        result = await _verify(json.dumps({"status": "ok"}).encode(), {"content-type": "application/json"})
        assert result.verified is False

    async def test_no_structure_found_does_not_verify(self):
        result = await _verify(_UNSTRUCTURED_PAGE)
        assert result.verified is False
        assert result.driver_backend == "nodriver"
        assert "no table" in result.verification_note.lower()

    async def test_fetch_failure_degrades_to_unverified_not_a_crash(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        result = await _find(verify_client=_client_for(handler))
        assert result.verified is False
        assert result.driver_backend == "nodriver"
        assert "could not fetch" in result.verification_note

    async def test_no_injected_client_constructs_and_closes_its_own(self):
        owned_client = _serving(b"<html><body><p>none</p></body></html>")
        with patch("threetears.scrape.page_finder.httpx.AsyncClient", return_value=owned_client) as ctor:
            result = await _find(verify_client=None)

        ctor.assert_called_once()
        assert owned_client.is_closed
        assert result.verified is False

    async def test_an_injected_client_is_left_open_for_its_owner(self):
        client = _serving(_VERIFYING_PAGE)
        result = await _find(verify_client=client)
        assert result.verified is True
        assert not client.is_closed


# ===========================================================================
# which search queries the result reports
# ===========================================================================


class TestExtractSearchQueries:
    async def test_pulls_query_args_from_web_search_calls_only(self):
        # "threetears.web_search" is the ACTUAL name ToolExecutor records (WebSearchTool.mcp_name()),
        # not the bare "web_search" -- using the real name here is the regression test for the bug
        # Critic caught: the original filter hardcoded the bare string and never matched in production.
        calls = [
            {"name": "threetears.web_search", "args": {"query": "Ohio WARN notices"}},
            {"name": "threetears.web_fetch", "args": {"url": "https://example.gov"}},
            {"name": "threetears.web_search", "args": {"query": "Ohio layoff notices"}},
        ]
        result = await _find(tool_calls_made=calls)
        assert result.search_queries_tried == ["Ohio WARN notices", "Ohio layoff notices"]

    async def test_bare_web_search_name_does_not_match_the_real_bound_name(self):
        # Regression test: a call recorded under the bare "web_search" string (what the original
        # bug hardcoded) must NOT match when the real bound name is "threetears.web_search".
        result = await _find(tool_calls_made=[{"name": "web_search", "args": {"query": "should not match"}}])
        assert result.search_queries_tried == []

    async def test_no_search_calls_returns_empty(self):
        result = await _find(tool_calls_made=[{"name": "threetears.web_fetch", "args": {"url": "x"}}])
        assert result.search_queries_tried == []


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
        found = _found(guess="nodriver", summary="the real WARN page")
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
        ):
            result = await find_target_page(
                "Ohio WARN notices",
                api_key="k",
                searxng_url="http://searx.local",
                verify_client=_serving(_VERIFYING_PAGE),
            )

        assert isinstance(result, PageFinderResult)
        assert result.url == "https://example.gov/warn"
        assert result.driver_backend == "nodriver"
        assert result.verified is True
        assert result.turns_used == 1

    async def test_verification_fails_falls_back_to_agents_verifiable_guess(self):
        loop_model, _ = _fake_tool_chat_model(_text_response("https://example.gov/notices.pdf is the page."))
        found = _found("https://example.gov/notices.pdf", guess="document", summary="a PDF")
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
        ):
            result = await find_target_page(
                "Ohio WARN notices",
                api_key="k",
                searxng_url="http://searx.local",
                verify_client=_serving(_UNSTRUCTURED_PAGE),
            )

        assert result.verified is False
        assert result.driver_backend == "document"  # agent's own guess is a verifiable backend, so it's used

    async def test_verification_fails_and_guess_unverifiable_defaults_to_nodriver(self):
        loop_model, _ = _fake_tool_chat_model(_text_response("https://example.gov/dashboard is the page."))
        found = _found("https://example.gov/dashboard", guess="camoufox", summary="a JS dashboard")
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(found)),
        ):
            result = await find_target_page(
                "Ohio WARN notices",
                api_key="k",
                searxng_url="http://searx.local",
                verify_client=_serving(_UNSTRUCTURED_PAGE),
            )

        assert result.verified is False
        assert result.driver_backend == "nodriver"  # camoufox is never guessable -- falls back

    async def test_coercion_failure_degrades_without_crashing(self):
        loop_model, _ = _fake_tool_chat_model(_text_response("I couldn't find a clear answer."))
        with (
            patch("threetears.scrape.page_finder.create_chat_model", return_value=loop_model),
            patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_failing()),
            patch("threetears.scrape.llm_retry.asyncio.sleep", AsyncMock()),
        ):
            result = await find_target_page("Ohio WARN notices", api_key="k", searxng_url="http://searx.local")

        assert result.verified is False
        assert result.url == ""
        assert "could not coerce" in result.verification_note
        # The note names the failure, so it is not read as "the agent named no page".
        assert "RuntimeError: boom" in result.verification_note

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
# reading search structure off the loop's messages (search-spec.md check 4)
# ===========================================================================


class TestReadSearchStructure:
    async def test_reads_the_typed_projection_off_a_tool_message_artifact(self):
        result = await _find(
            HumanMessage(content="find it"), _search_tool_message(_candidate("https://example.gov/warn"))
        )

        assert [c.identity for c in result.candidates_seen] == ["https://example.gov/warn"]

    async def test_web_fetch_structure_is_not_read_as_a_search_result(self):
        # web_fetch writes its own projection under the SAME metadata key, so an
        # unfiltered scan would report a fetched page as a candidate the search
        # returned. The bound-name filter is the whole defence.
        result = await _find(_search_tool_message(_candidate("https://example.gov/a"), name=_FETCH_TOOL))

        assert result.candidates_seen == ()

    async def test_bare_tool_name_does_not_match_the_real_bound_name(self):
        # The same name-grain bug the search-query extraction documents, one layer up: the
        # bound name is "threetears.web_search", so a message under the bare name is not it.
        result = await _find(_search_tool_message(_candidate("https://example.gov/a"), name="web_search"))

        assert result.candidates_seen == ()

    async def test_a_message_with_no_artifact_is_skipped(self):
        result = await _find(ToolMessage(content="plain prose", tool_call_id="tc-1", name=_SEARCH_TOOL))

        assert result.candidates_seen == ()

    async def test_a_schema_version_newer_than_this_reader_degrades_instead_of_raising(self):
        # from_metadata refuses a newer schema loudly (D13). find_target_page
        # promises never to raise, so here that refusal must become a skip.
        payload = SearchResultsMetadata.from_candidate_set(
            query="q", candidate_set=CandidateSet(candidates=(_candidate("https://example.gov/a"),))
        ).to_metadata()
        payload["schema_version"] = 9999

        result = await _find(_artifact_message({SEARCH_RESULTS_METADATA_KEY: payload}))

        assert result.candidates_seen == ()


class TestDedupeCandidates:
    async def test_identity_dedupes_across_turns_and_keeps_first_order(self):
        result = await _find(
            _search_tool_message(_candidate("https://a.gov"), _candidate("https://b.gov")),
            _search_tool_message(_candidate("https://b.gov"), _candidate("https://c.gov")),
        )

        assert [c.identity for c in result.candidates_seen] == ["https://a.gov", "https://b.gov", "https://c.gov"]


class TestFirstFailure:
    async def test_a_typed_failure_is_rendered_class_first(self):
        result = await _find(_refused("rate-limited", "slow down"))

        assert result.search_failure == "rate-limited: slow down"

    async def test_zero_results_is_not_a_failure(self):
        # SR-J2: an empty candidate set is a success value.
        result = await _find(_search_tool_message())

        assert result.search_failure is None


# ===========================================================================
# find_target_page -- the structure seam, end to end
# ===========================================================================


class TestFindTargetPageReadsStructure:
    async def test_typed_candidates_reach_the_result_and_the_url_is_recognised(self):
        result = await _find(
            _search_tool_message(_candidate("https://example.gov/warn"), _candidate("https://other.gov/x")),
            found=_found(guess="nodriver", summary="the real WARN page"),
        )

        assert [c.identity for c in result.candidates_seen] == ["https://example.gov/warn", "https://other.gov/x"]
        assert result.url_was_a_search_result is True
        assert result.verified is True

    async def test_a_url_no_search_returned_is_marked_as_such(self):
        # The coercion step can name a page the loop reached by following a
        # fetched link -- or one it invented. Either way it was not *found*,
        # and before structure crossed the border the two were indistinguishable.
        result = await _find(
            _search_tool_message(_candidate("https://example.gov/warn")),
            output="https://invented.gov/nope is the page.",
            found=_found("https://invented.gov/nope", summary="hmm"),
            verify_client=_serving(_UNSTRUCTURED_PAGE),
        )

        assert result.url == "https://invented.gov/nope"
        assert result.url_was_a_search_result is False

    async def test_provider_notices_survive_to_the_result(self):
        result = await _find(
            _search_tool_message(_candidate("https://example.gov/warn"), notices=("two engines did not answer",))
        )

        assert result.search_notices == ("two engines did not answer",)
        assert result.verified is True  # a degraded search still yields a real finding

    async def test_a_refused_search_is_named_rather_than_blamed_on_turn_exhaustion(self):
        # The behaviour change structure buys: before, every empty run reported
        # "exhausted its turn budget" regardless of why it actually came up dry.
        result = await _find(_refused(), output="", error="max rounds exhausted")

        assert result.search_failure == "rate-limited: slow down"
        assert "every search turn was refused" in result.verification_note
        assert result.url == ""

    async def test_no_structure_at_all_leaves_the_new_fields_at_their_defaults(self):
        # A loop that never called search (or an older tool that carried no
        # metadata) must still produce exactly the result it did before.
        result = await _find()

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

    Written here rather than imported: `packages/search/tests/searxng_payloads.py`
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


async def _find_over_a_real_search(searxng_url: str) -> PageFinderResult:
    """Run ``find_target_page`` with the real search tool, adapter and ToolExecutor.

    Only the chat model's decisions and the coercion answer are canned: the model calls the
    search tool once (by the name the real tool is bound under) and then answers in prose.
    """

    def _bind(tools: list[Any]) -> SimpleNamespace:
        (search_tool,) = (tool for tool in tools if tool.name == _SEARCH_TOOL)
        return _chat_model_calling_search_once(search_tool.name)

    with (
        patch("threetears.scrape.page_finder.create_chat_model", return_value=SimpleNamespace(bind_tools=_bind)),
        patch("threetears.scrape.llm_retry.create_chat_model", return_value=_coercion_returning(_found())),
    ):
        return await find_target_page(
            "Ohio WARN notices", api_key="k", searxng_url=searxng_url, verify_client=_serving(_VERIFYING_PAGE)
        )


class TestTheStructureSeamAgainstARealSocket:
    """Drive the real tool, the real adapter and the real ToolExecutor.

    Every test above builds its ToolMessage by hand, which pins what
    page_finder does with a message but assumes the two facts it depends on:
    that LangChain stamps the bound tool name onto `ToolMessage.name` and that
    ToolExecutor lets the artifact through. Assuming those is exactly the gap
    #321's review found -- a tool and its transport each tested alone, with the
    seam between them tested nowhere -- so they get asserted here against a
    real socket instead: a candidate reaches the result only if both hold.
    """

    async def test_a_real_search_turn_lands_readable_structure_in_the_result(self):
        from threetears.search.testing import LocalHttpServer, Reply

        async with LocalHttpServer(
            (Reply(body=_searxng_body("https://example.gov/warn"), headers={"content-type": "application/json"}),)
        ) as server:
            result = await _find_over_a_real_search(server.base_url)

        assert result.search_queries_tried == ["Ohio WARN notices"]
        assert [c.identity for c in result.candidates_seen] == ["https://example.gov/warn"]
        assert result.url_was_a_search_result is True


# ===========================================================================
# The helpers' remaining branches, through the result
# ===========================================================================


class TestCandidateUrls:
    async def test_identity_counts_as_a_reachable_url(self):
        result = await _find(
            _search_tool_message(_candidate("https://a.gov")), output="https://a.gov", found=_found("https://a.gov")
        )
        assert result.url_was_a_search_result is True

    async def test_a_non_canonical_locator_counts_too(self):
        # identity is the canonical URL *by convention*, not by guarantee, and the
        # URL an LLM names is whichever one the prose rendering showed it -- so a
        # direct-file locator must count as "the search returned this".
        result = await _find(
            _search_tool_message(_candidate_with_locators("provider-native-id-42", "https://a.gov/notices.pdf")),
            output="https://a.gov/notices.pdf",
            found=_found("https://a.gov/notices.pdf"),
        )

        assert result.url_was_a_search_result is True

    async def test_no_candidates_is_not_a_search_result_and_not_a_crash(self):
        result = await _find()
        assert result.url_was_a_search_result is False


class TestDedupeCandidatesEdges:
    async def test_no_projections_yields_nothing(self):
        assert (await _find()).candidates_seen == ()

    async def test_a_zero_result_turn_contributes_nothing(self):
        # SR-J2: zero results is a success, and must not become a phantom candidate.
        assert (await _find(_search_tool_message())).candidates_seen == ()

    async def test_first_occurrence_wins_so_the_earlier_turns_locators_survive(self):
        result = await _find(
            _search_tool_message(_candidate_with_locators("same-id", "https://first.gov")),
            _search_tool_message(_candidate_with_locators("same-id", "https://second.gov")),
        )

        assert len(result.candidates_seen) == 1
        assert [loc.url for loc in result.candidates_seen[0].locators] == ["https://first.gov"]


class TestFirstFailureEdges:
    async def test_the_first_failure_wins_when_several_turns_failed(self):
        result = await _find(_refused("rate-limited", "first"), _refused("timeout", "second"))
        assert result.search_failure == "rate-limited: first"

    async def test_a_failure_after_a_successful_turn_is_still_reported(self):
        result = await _find(_search_tool_message(_candidate("https://a.gov")), _refused("timeout", "too slow"))
        assert result.search_failure == "timeout: too slow"

    async def test_no_projections_is_no_failure(self):
        assert (await _find()).search_failure is None


class TestReadSearchStructureEdges:
    async def test_several_search_turns_each_yield_a_projection(self):
        result = await _find(
            _search_tool_message(_candidate("https://a.gov")),
            HumanMessage(content="keep looking"),
            _search_tool_message(_candidate("https://b.gov")),
        )

        assert [c.identity for c in result.candidates_seen] == ["https://a.gov", "https://b.gov"]

    async def test_an_artifact_that_is_not_a_dict_is_skipped(self):
        assert (await _find(_artifact_message("not a dict"))).candidates_seen == ()

    async def test_an_artifact_without_the_named_key_is_skipped(self):
        assert (await _find(_artifact_message({"something_else": {}}))).candidates_seen == ()

    async def test_a_payload_that_is_not_a_dict_is_skipped(self):
        assert (await _find(_artifact_message({SEARCH_RESULTS_METADATA_KEY: "nope"}))).candidates_seen == ()

    async def test_the_current_schema_version_is_accepted(self):
        # The boundary the refusal test above does not pin: equal-to-current must
        # read, or the reader would refuse every payload the family actually emits.
        payload = SearchResultsMetadata.from_candidate_set(
            query="q", candidate_set=CandidateSet(candidates=(_candidate("https://a.gov"),))
        ).to_metadata()

        result = await _find(_artifact_message({SEARCH_RESULTS_METADATA_KEY: payload}))

        assert [c.identity for c in result.candidates_seen] == ["https://a.gov"]

    async def test_messages_that_are_not_tool_messages_are_ignored(self):
        result = await _find(HumanMessage(content="hi"), SimpleNamespace(name=_SEARCH_TOOL))
        assert result.candidates_seen == ()


class TestAllNotices:
    async def test_notices_dedupe_across_turns_and_keep_first_order(self):
        result = await _find(
            _search_tool_message(notices=("engine A down", "unranked")),
            _search_tool_message(notices=("unranked", "engine B down")),
        )

        assert result.search_notices == ("engine A down", "unranked", "engine B down")

    async def test_no_notices_is_an_empty_tuple(self):
        assert (await _find()).search_notices == ()


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
        result = await _find(
            _search_tool_message(
                _candidate_with_locators("https://example.gov/warn", "https://example.gov/notices.pdf")
            ),
            output="https://example.gov/notices.pdf is the page.",
            found=_found("https://example.gov/notices.pdf", summary="the PDF"),
            verify_client=_serving(b'<html><body><a href="/notices.pdf">notices</a></body></html>'),
        )

        assert result.url_was_a_search_result is True
        assert result.driver_backend == "document"

    async def test_the_coercion_failure_path_still_reports_what_the_search_found(self):
        # A run that could not turn prose into a URL has still learned real
        # things about the web, and dropping them would waste the spend.
        result = await _find(
            _search_tool_message(_candidate("https://example.gov/warn"), notices=("one engine did not answer",)),
            output="I found something but cannot say what.",
            coercion=_coercion_failing(),
        )

        assert result.url == ""
        assert "could not coerce" in result.verification_note
        assert [c.identity for c in result.candidates_seen] == ["https://example.gov/warn"]
        assert result.search_notices == ("one engine did not answer",)

    async def test_notices_from_several_turns_are_gathered_and_deduplicated(self):
        result = await _find(
            _search_tool_message(_candidate("https://a.gov"), notices=("unranked", "engine A down")),
            _search_tool_message(_candidate("https://example.gov/warn"), notices=("unranked", "engine B down")),
        )

        assert result.search_notices == ("unranked", "engine A down", "engine B down")
        assert [c.identity for c in result.candidates_seen] == ["https://a.gov", "https://example.gov/warn"]

    async def test_a_failed_turn_is_reported_even_when_a_later_turn_saved_the_run(self):
        # search_failure is a record of what happened, not a verdict on the run:
        # the loop recovered and found a page, and both facts are true at once.
        result = await _find(_refused(), _search_tool_message(_candidate("https://example.gov/warn")))

        assert result.search_failure == "rate-limited: slow down"
        assert result.verified is True
        assert result.url_was_a_search_result is True

    async def test_web_fetch_structure_never_becomes_a_search_candidate_end_to_end(self):
        # The filter's real consequence: a page the loop FETCHED must not be
        # reported as one the search RETURNED, or url_was_a_search_result lies.
        result = await _find(
            _search_tool_message(_candidate("https://fetched.gov/page"), name=_FETCH_TOOL),
            output="https://fetched.gov/page is the page.",
            found=_found("https://fetched.gov/page"),
        )

        assert result.candidates_seen == ()
        assert result.url_was_a_search_result is False
        assert result.verified is True

    async def test_zero_results_over_a_real_socket_is_a_success_with_no_candidates(self):
        # SR-J2: an empty answer is a success value. It must reach the reader as a
        # projection carrying nothing -- never as a failure, never as no projection.
        from threetears.search.testing import LocalHttpServer, Reply

        async with LocalHttpServer(
            (Reply(body=_searxng_body(), headers={"content-type": "application/json"}),)
        ) as server:
            result = await _find_over_a_real_search(server.base_url)

        assert result.candidates_seen == ()
        assert result.search_failure is None

    async def test_a_provider_error_over_a_real_socket_arrives_as_a_typed_failure(self):
        # D10: nothing raises across the border. The failure must reach the reader
        # as a named class -- which is what lets page_finder tell "refused" from
        # "found nothing" without matching on an error prefix in prose.
        from threetears.search.testing import LocalHttpServer, Reply

        async with LocalHttpServer((Reply(status=500, body=b"upstream is unwell"),)) as server:
            result = await _find_over_a_real_search(server.base_url)

        # The class is named, not merely present: "transport-failed" is what an
        # operator acts on, and it is the fact the old [TOOL ERROR] prefix could
        # not carry. Three attempts happened underneath -- the transport's own
        # bounded retry -- and the border still reports one typed outcome.
        assert result.search_failure is not None
        assert result.search_failure.startswith("transport-failed: ")
        assert result.candidates_seen == ()

    async def test_a_structurally_invalid_payload_degrades_the_same_way(self):
        # from_metadata validates as well as version-checks, and pydantic's
        # ValidationError is a ValueError -- so a malformed payload takes the same
        # degrade-to-prose path as a too-new one rather than escaping as a crash.
        result = await _find(
            _artifact_message({SEARCH_RESULTS_METADATA_KEY: {"schema_version": 1, "candidates": "not a tuple"}})
        )

        assert result.candidates_seen == ()


# ===========================================================================
# The verification fetch: size cap, encodings, and lookalike links
# ===========================================================================

#: a small cap for the size-cap tests, so they exercise the bound without serving megabytes.
_SMALL_CAP = 1024


class TestVerificationSizeCap:
    async def test_the_production_cap_is_two_mebibytes(self):
        # The fetch used to be unbounded: client.get buffered the whole body and
        # BeautifulSoup built a parse tree from it, measured at 77x the served
        # size. find_target_page fetches a URL an LLM picked out of search
        # results, so that size is not this process's to choose.
        oversized = b"<html><body>" + b"<p>pad</p>" * 300_000 + b"</body></html>"
        assert len(oversized) > _PRODUCTION_VERIFY_CAP

        result = await _verify(oversized, {"content-type": "text/html"})

        assert result.verified is False
        assert f"first {_PRODUCTION_VERIFY_CAP} bytes" in result.verification_note

    async def test_a_body_over_the_cap_is_truncated_and_says_so(self):
        oversized = b"<html><body>" + b"<p>pad</p>" * 300 + b"</body></html>"
        assert len(oversized) > _SMALL_CAP

        result = await _verify(oversized, {"content-type": "text/html"}, max_bytes=_SMALL_CAP)

        assert result.verified is False
        assert result.driver_backend == "nodriver"
        # "nothing in the part I read" is a weaker claim than "nothing on the page".
        assert str(_SMALL_CAP) in result.verification_note
        assert "longer than the verification cap" in result.verification_note

    async def test_structure_inside_the_cap_still_verifies_on_an_oversized_page(self):
        # The cap must not cost a real finding: a table inside the cap is found
        # regardless of how much padding follows it.
        body = b"<html><body>" + _TABLE.format("Acme Corp").encode() + b"<p>pad</p>" * 300 + b"</body></html>"
        assert len(body) > _SMALL_CAP

        result = await _verify(body, {"content-type": "text/html"}, max_bytes=_SMALL_CAP)

        assert result.verified is True
        assert result.driver_backend == "nodriver"
        assert "table" in result.verification_note

    async def test_a_body_exactly_at_the_cap_is_not_reported_as_truncated(self):
        # Boundary: read > cap is truncation, read == cap is a whole document.
        filler = b"x" * (_SMALL_CAP - len(b"<html><body></body></html>"))
        body = (b"<html><body>" + filler + b"</body></html>")[:_SMALL_CAP]
        assert len(body) == _SMALL_CAP

        result = await _verify(body, {"content-type": "text/html"}, max_bytes=_SMALL_CAP)

        assert result.verified is False
        assert "verification cap" not in result.verification_note

    async def test_a_truncated_json_body_is_not_called_an_api(self):
        # A cut-off document is not parseable JSON, and guessing at the missing
        # half would verify a page nobody has seen the end of.
        head = b'{"records": [' + b'{"employer": "Acme"},' * 200
        assert len(head) > _SMALL_CAP

        result = await _verify(head, {"content-type": "application/json"}, max_bytes=_SMALL_CAP)

        assert result.verified is False
        assert result.driver_backend == "nodriver"


class TestVerificationEncodings:
    async def test_declared_shift_jis_is_decoded_and_structure_found(self):
        body = ("<html><body>" + _TABLE.format("日本語のデータ") + "</body></html>").encode("shift_jis")

        result = await _verify(body, {"content-type": "text/html; charset=shift_jis"})

        assert result.verified is True
        assert result.driver_backend == "nodriver"

    async def test_declared_windows_1256_arabic_is_decoded_and_structure_found(self):
        body = ("<html><body>" + _TABLE.format("بيانات التسريح") + "</body></html>").encode("cp1256")

        result = await _verify(body, {"content-type": "text/html; charset=windows-1256"})

        assert result.verified is True
        assert result.driver_backend == "nodriver"

    async def test_declared_utf_16_is_decoded_and_structure_found(self):
        body = ("<html><body>" + _TABLE.format("data") + "</body></html>").encode("utf-16")

        result = await _verify(body, {"content-type": "text/html; charset=utf-16"})

        assert result.verified is True

    async def test_undeclared_non_utf8_still_finds_structure(self):
        # No charset header, bytes that are not UTF-8. The text mis-decodes and
        # that is fine: every marker this function looks for -- <table>, <tr>,
        # href -- is ASCII, so structure detection does not depend on getting
        # the human-readable text right.
        body = ("<html><body>" + _TABLE.format("日本語") + "</body></html>").encode("shift_jis")

        result = await _verify(body, {"content-type": "text/html"})

        assert result.verified is True

    async def test_invalid_byte_sequences_do_not_raise(self):
        # errors="replace", never strict: a page whose bytes contradict its
        # declared charset must still be inspectable rather than crash a run
        # that promises never to raise.
        body = b"<html><body>\xff\xfe\x00\x81" + _TABLE.format("x").encode() + b"</body></html>"

        result = await _verify(body, {"content-type": "text/html; charset=utf-8"})

        assert result.verified is True

    async def test_a_body_cut_mid_multibyte_character_does_not_raise(self):
        # The cap can slice a UTF-8 sequence in half. That must degrade to a
        # replacement character, not an exception. 1024 is not a multiple of 3,
        # so the cap lands inside a three-byte character.
        body = b"<html><body>" + ("日" * 2_000).encode() + b"</body></html>"
        assert len(body) > _SMALL_CAP

        result = await _verify(body, {"content-type": "text/html; charset=utf-8"}, max_bytes=_SMALL_CAP)

        assert result.verified is False
        assert "verification cap" in result.verification_note


class TestVerificationLookalikeLinks:
    async def test_a_bidi_override_link_that_merely_looks_like_a_pdf_is_not_one(self):
        # U+202E renders "/annexgnp.fdp" as if it ended in .pdf. The extension
        # check reads the real characters, so the spoof does not verify -- and
        # this pins that it stays that way.
        body = '<html><body><a href="/annex‮gnp.fdp">Notices</a></body></html>'.encode()

        result = await _verify(body, {"content-type": "text/html"})

        assert result.verified is False
        assert result.driver_backend == "nodriver"

    async def test_a_genuine_rtl_named_pdf_still_verifies(self):
        # The converse: an Arabic-named PDF is a real document link and must not
        # be collateral damage of the check above.
        body = '<html><body><a href="/إشعارات.pdf">PDF</a></body></html>'.encode()

        result = await _verify(body, {"content-type": "text/html"})

        assert result.verified is True
        assert result.driver_backend == "document"
        assert ".pdf" in result.verification_note


# ===========================================================================
# Exotic text across the metadata border
# ===========================================================================


class TestStructureCarriesExoticText:
    async def test_rtl_cjk_and_emoji_survive_the_metadata_round_trip(self):
        # to_metadata is model_dump(mode="json") and from_metadata revalidates,
        # so this crosses the same projection a NATS hop would. JSON is unicode
        # end to end; nothing here should need escaping to survive.
        title = "إشعارات التسريح 日本語 🏛️ notices"

        result = await _find(_search_tool_message(_candidate("https://example.gov/warn", title=title)))

        assert result.candidates_seen[0].title == title

    async def test_a_bidi_override_in_a_title_is_carried_verbatim_not_stripped(self):
        # The border's job is to carry what the provider said, exactly. Deciding
        # what to do about a title that renders deceptively is a rendering
        # concern, and silently mutating it here would hide the fact from
        # whoever does have to make that call.
        title = "annex‮gnp.fdp"

        result = await _find(_search_tool_message(_candidate("https://example.gov/x", title=title)))

        assert result.candidates_seen[0].title == title

    async def test_a_non_ascii_url_matches_itself_when_checking_membership(self):
        # url_was_a_search_result is a string comparison, so an IDN or a
        # percent-encoded path must match the form the candidate carries.
        url = "https://例え.jp/通知/warn%20notices.html"

        result = await _find(_search_tool_message(_candidate(url)), output=url, found=_found(url))

        assert result.url_was_a_search_result is True

    async def test_percent_encoded_and_decoded_forms_are_not_treated_as_the_same_url(self):
        # Deliberately NOT normalised: this module does not own URL canonicalisation,
        # and quietly equating the two forms would report "the search returned this"
        # about a URL the search did not return. Recorded as a pin so the choice is
        # visible if someone later wants normalisation -- it belongs upstream, at the
        # adapter that mints identity, not in a membership check.
        result = await _find(
            _search_tool_message(_candidate("https://example.gov/warn%20notices")),
            output="https://example.gov/warn notices",
            found=_found("https://example.gov/warn notices"),
        )

        assert result.url_was_a_search_result is False

    async def test_a_lone_surrogate_in_a_payload_degrades_rather_than_crashing(self):
        # A surrogate is unencodable to UTF-8 and can reach a reader via a
        # provider that emitted it. Whatever pydantic makes of it, this must not
        # be how find_target_page learns to raise.
        result = await _find(
            _artifact_message({SEARCH_RESULTS_METADATA_KEY: {"schema_version": 1, "query": "\ud800 broken"}})
        )

        assert isinstance(result, PageFinderResult)  # read or skipped, never raised
        assert result.candidates_seen == ()


class TestUnknownDeclaredCharset:
    async def test_a_charset_python_does_not_know_falls_back_instead_of_raising(self):
        # httpx hands back the charset= parameter verbatim without checking it
        # against the codec registry, so bytes.decode raises LookupError -- which
        # is NOT a ValueError and so escaped the fetch's own guard, out of a
        # function whose contract is that it never raises. The header comes off a
        # page an LLM picked out of search results.
        body = ("<html><body>" + _TABLE.format("data") + "</body></html>").encode()

        result = await _verify(body, {"content-type": "text/html; charset=utf8mb4"})

        assert result.verified is True
        assert result.driver_backend == "nodriver"

    async def test_find_target_page_survives_an_unknown_charset_end_to_end(self):
        # The contract that was actually at risk: find_target_page never raises.
        result = await _verify(
            b"<html><body><p>nothing structural</p></body></html>", {"content-type": "text/html; charset=unknown-8bit"}
        )

        assert result.verified is False
        assert result.url == "https://example.gov/warn"

    async def test_an_empty_charset_parameter_falls_back_too(self):
        body = ("<html><body>" + _TABLE.format("data") + "</body></html>").encode()

        result = await _verify(body, {"content-type": "text/html; charset="})

        assert result.verified is True


class TestEverySearchFailed:
    """Whether an empty run is blamed on the provider, as the verification note reports it."""

    async def test_all_failed_is_named_as_refused(self):
        result = await _find(_refused(), _refused("timeout", "m"), output="", error="max rounds exhausted")

        assert "every search turn was refused" in result.verification_note

    async def test_no_turns_is_not_blamed_on_the_provider(self):
        result = await _find(output="", error="max rounds exhausted")

        assert "exhausted its turn budget" in result.verification_note
        assert "refused" not in result.verification_note

    async def test_a_recovered_search_does_not_get_blamed_on_the_provider(self):
        # The run did not fail for want of searching; it failed to converge.
        # Blaming the provider would send an operator after a cleared quota.
        result = await _find(
            _refused(),
            _search_tool_message(_candidate("https://example.gov/warn")),
            output="",
            error="max rounds exhausted",
        )

        assert "exhausted its turn budget" in result.verification_note
        assert "refused" not in result.verification_note
        # still carried as a fact, just not as the verdict
        assert result.search_failure == "rate-limited: slow down"


class TestTheCorpusKeepsWhatTheFlatTupleDropped:
    """The migration's actual gain: corroboration across differently-worded turns."""

    def _turn(self, query: str, url: str) -> ToolMessage:
        """One search turn, with its own query on the candidate's provenance."""
        candidate = Candidate(
            identity=url,
            locators=(Locator(url=url),),
            provenance=Provenance(
                query=query,
                provider_instance="searxng-local",
                retrieved_at=datetime(2026, 8, 12, tzinfo=UTC),
            ),
        )
        return _projection_message(
            SearchResultsMetadata.from_candidate_set(query=query, candidate_set=CandidateSet(candidates=(candidate,)))
        )

    async def test_two_queries_finding_one_url_keep_both_provenances(self):
        result = await _find(
            self._turn("Ohio WARN notices", "https://example.gov/warn"),
            self._turn("Ohio layoff filings", "https://example.gov/warn"),
        )

        corpus = result.candidate_corpus
        assert corpus is not None
        assert len(corpus.entries) == 1
        entry = corpus.entries[0]
        assert len(entry.contributions) == 2
        assert [p.query for p in entry.provenances] == ["Ohio WARN notices", "Ohio layoff filings"]

    async def test_the_flat_projection_still_shows_one_candidate_per_identity(self):
        result = await _find(
            self._turn("Ohio WARN notices", "https://example.gov/warn"),
            self._turn("Ohio layoff filings", "https://example.gov/warn"),
        )

        assert [c.identity for c in result.candidates_seen] == ["https://example.gov/warn"]
        assert result.candidates_seen[0].provenance.query == "Ohio WARN notices", "first turn wins, as it always has"

    async def test_a_failed_turn_contributes_no_corpus_notice(self):
        """`search_failure` already reports it; the corpus must not double-report."""
        failed = SearchResultsMetadata(
            query="Ohio WARN notices",
            failure=FailureRecord(failure_class="rate-limited", message="429", spend=Spend()),
        )

        result = await _find(_projection_message(failed), self._turn("Ohio layoff filings", "https://example.gov/warn"))

        assert result.candidate_corpus is not None
        assert result.candidate_corpus.notices == ()
        assert len(result.candidate_corpus.entries) == 1

    async def test_spend_rolls_up_across_turns(self):
        """A loop that searched six times now has one number for what that cost."""
        result = await _find(
            _projection_message(
                SearchResultsMetadata.from_candidate_set(query="q1", candidate_set=CandidateSet(spend=Spend(calls=1)))
            ),
            _projection_message(
                SearchResultsMetadata.from_candidate_set(query="q2", candidate_set=CandidateSet(spend=Spend(calls=1)))
            ),
        )

        assert result.candidate_corpus is not None
        assert result.candidate_corpus.spend.calls == 2
