"""Tests for LangGraph context enrichment and save nodes."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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

from threetears.agent.tools.graph_nodes import (
    _DEFAULT_SAVEABLE_TOOLS,
    _MAX_SAVED_CANDIDATES,
    create_context_enrichment_node,
    create_context_save_node,
)


# -- Enrichment node tests --


class TestContextEnrichmentNode:
    """Tests for create_context_enrichment_node."""

    @pytest.mark.asyncio
    async def test_injects_system_message_when_entities_found(self) -> None:
        """Enrichment injects a SystemMessage with formatted entities."""

        async def searcher(text: str) -> list[dict]:
            """Mock searcher returning one entity.

            :param text: query text
            :ptype text: str
            :return: list of entity dicts
            :rtype: list[dict]
            """
            return [{"title": "XSS", "severity": "high"}]

        def formatter(entities: list) -> str:
            """Mock formatter.

            :param entities: entities to format
            :ptype entities: list
            :return: formatted string
            :rtype: str
            """
            return f"Found {len(entities)} items"

        node = create_context_enrichment_node(
            entity_searcher=searcher,
            entity_formatter=formatter,
        )
        state = {"messages": [HumanMessage(content="any vulnerability?")]}
        result = await node(state)
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)
        assert "Found 1 items" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_skips_when_no_entities(self) -> None:
        """Returns empty messages when searcher finds nothing."""

        async def searcher(text: str) -> list:
            """Empty searcher.

            :param text: query text
            :ptype text: str
            :return: empty list
            :rtype: list
            """
            return []

        node = create_context_enrichment_node(
            entity_searcher=searcher,
            entity_formatter=lambda e: "",
        )
        state = {"messages": [HumanMessage(content="hello")]}
        result = await node(state)
        assert result["messages"] == []

    @pytest.mark.asyncio
    async def test_keyword_gating(self) -> None:
        """Skips enrichment when no keywords match."""
        called = []

        async def searcher(text: str) -> list:
            """Tracking searcher.

            :param text: query text
            :ptype text: str
            :return: empty list
            :rtype: list
            """
            called.append(True)
            return []

        node = create_context_enrichment_node(
            entity_searcher=searcher,
            entity_formatter=lambda e: "",
            keywords=frozenset({"vulnerability", "exploit"}),
        )
        state = {"messages": [HumanMessage(content="what is the weather today?")]}
        result = await node(state)
        assert result["messages"] == []
        assert len(called) == 0  # searcher never called

    @pytest.mark.asyncio
    async def test_keyword_match_triggers_search(self) -> None:
        """Triggers search when keywords match."""
        called = []

        async def searcher(text: str) -> list:
            """Tracking searcher.

            :param text: query text
            :ptype text: str
            :return: empty list
            :rtype: list
            """
            called.append(True)
            return []

        node = create_context_enrichment_node(
            entity_searcher=searcher,
            entity_formatter=lambda e: "",
            keywords=frozenset({"vulnerability"}),
        )
        state = {"messages": [HumanMessage(content="any vulnerability found?")]}
        await node(state)
        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_ledger_coverage_skips(self) -> None:
        """Skips enrichment when surfaced-refs projection has sufficient coverage."""
        mock_cm = MagicMock()
        mock_cm.memory_refs_count = 5

        called = []

        async def searcher(text: str) -> list:
            """Tracking searcher.

            :param text: query text
            :ptype text: str
            :return: empty list
            :rtype: list
            """
            called.append(True)
            return []

        node = create_context_enrichment_node(
            entity_searcher=searcher,
            entity_formatter=lambda e: "",
            context_manager=mock_cm,
            min_ledger_coverage=3,
        )
        state = {"messages": [HumanMessage(content="find vulnerabilities")]}
        await node(state)
        assert len(called) == 0


# -- Context save node tests --


class TestContextSaveNode:
    """Tests for create_context_save_node."""

    @pytest.mark.asyncio
    async def test_saves_tool_results_for_saveable_tools(self) -> None:
        """Tool results from saveable tools are saved to context."""
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-123")

        node = create_context_save_node(
            context_manager=mock_cm,
            saveable_tools=frozenset({"web_fetch"}),
        )
        state = {
            "messages": [
                ToolMessage(content="Page content here", tool_call_id="tc1", name="web_fetch"),
            ]
        }
        await node(state)
        mock_cm.save_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ignores_non_saveable_tools(self) -> None:
        """Tool results from non-saveable tools are not saved."""
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock()

        node = create_context_save_node(
            context_manager=mock_cm,
            saveable_tools=frozenset({"web_fetch"}),
        )
        state = {
            "messages": [
                ToolMessage(content="Some result", tool_call_id="tc1", name="calculator"),
            ]
        }
        await node(state)
        mock_cm.save_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_suffix_matching(self) -> None:
        """Tools matching a saveable suffix are saved."""
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-456")

        node = create_context_save_node(
            context_manager=mock_cm,
            saveable_tools=frozenset(),
            saveable_suffixes=("_scan",),
        )
        state = {
            "messages": [
                ToolMessage(content="Scan results", tool_call_id="tc1", name="nmap_scan"),
            ]
        }
        await node(state)
        mock_cm.save_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_truncates_long_content(self) -> None:
        """Long tool results are truncated before saving."""
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-789")

        node = create_context_save_node(
            context_manager=mock_cm,
            saveable_tools=frozenset({"big_tool"}),
            max_content=100,
        )
        state = {
            "messages": [
                ToolMessage(content="x" * 200, tool_call_id="tc1", name="big_tool"),
            ]
        }
        await node(state)
        call_kwargs = mock_cm.save_tool_result.call_args
        saved_result = call_kwargs.kwargs.get("result") or call_kwargs[1].get("result")
        assert len(saved_result) <= 120  # 100 + truncation notice
        assert "[Content truncated]" in saved_result

    @pytest.mark.asyncio
    async def test_empty_messages_returns_empty(self) -> None:
        """No messages returns empty result."""
        mock_cm = AsyncMock()
        node = create_context_save_node(context_manager=mock_cm)
        result = await node({"messages": []})
        assert result == {"messages": []}


# ===========================================================================
# C8 -- the name grain, the type binding, and the structure that gets saved
# ===========================================================================


def _candidate(url: str, *, title: str | None = "a page") -> Candidate:
    return Candidate(
        identity=url,
        locators=(Locator(url=url),),
        provenance=Provenance(
            query="ohio warn notices",
            provider_instance="searxng-local",
            retrieved_at=datetime(2026, 8, 12, tzinfo=UTC),
        ),
        title=title,
    )


def _structured_message(
    *candidates: Candidate,
    name: str = "threetears.web_search",
    query: str = "ohio warn notices",
    notices: tuple[str, ...] = (),
    content: str = "1. a page\n   URL: https://example.gov/warn",
) -> ToolMessage:
    """A ToolMessage shaped as the search leaf + langchain_adapter produce one."""
    projection = SearchResultsMetadata.from_candidate_set(
        query=query, candidate_set=CandidateSet(candidates=tuple(candidates), notices=notices)
    )
    return ToolMessage(
        content=content,
        tool_call_id="tc1",
        name=name,
        artifact={SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()},
    )


def _saved_kwargs(mock_cm) -> dict:
    return mock_cm.save_tool_result.call_args.kwargs


class TestTheDefaultSetMatchesWhatIsActuallyBound:
    """C8's regression pin, at the grain the defect actually had.

    The node was inert because the default set held bare names while the
    adapter binds namespaced ones -- and the suite could not see it because
    every test passed its own bare names explicitly. These read the real
    tools' real bound names rather than restating a string.
    """

    def test_the_builtins_bound_names_are_in_the_default_set(self) -> None:
        from threetears.agent.tools.builtin.web_fetch import WebFetchTool
        from threetears.agent.tools.builtin.web_search import WebSearchTool

        assert WebSearchTool(base_url="http://searx.local").mcp_name() in _DEFAULT_SAVEABLE_TOOLS
        assert WebFetchTool().mcp_name() in _DEFAULT_SAVEABLE_TOOLS

    def test_the_bare_names_are_not_what_the_default_set_holds(self) -> None:
        # The exact shape of the bug: these look right and match nothing.
        assert "web_search" not in _DEFAULT_SAVEABLE_TOOLS
        assert "web_fetch" not in _DEFAULT_SAVEABLE_TOOLS

    @pytest.mark.asyncio
    async def test_a_bound_name_saves_under_the_defaults(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-1")

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [ToolMessage(content="page", tool_call_id="tc1", name="threetears.web_fetch")]})

        mock_cm.save_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_bare_name_with_no_structure_saves_nothing_under_the_defaults(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock()

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [ToolMessage(content="page", tool_call_id="tc1", name="web_fetch")]})

        mock_cm.save_tool_result.assert_not_awaited()


class TestBindingOnResultTypeNotName:
    @pytest.mark.asyncio
    async def test_structure_is_saved_even_under_an_unlisted_tool_name(self) -> None:
        # "A rename is a data-retention change" -- unless the binding is on what
        # the result IS. A tool renamed or split per carrier keeps being saved.
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-2")

        node = create_context_save_node(context_manager=mock_cm, saveable_tools=frozenset({"something_else"}))
        await node({"messages": [_structured_message(_candidate("https://example.gov/warn"), name="web.search.v2")]})

        mock_cm.save_tool_result.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_empty_set_means_no_tool_is_saveable_by_name(self) -> None:
        # Changed deliberately: an empty frozenset used to be falsy and fall
        # back to the defaults, so it silently meant its own opposite.
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock()

        node = create_context_save_node(context_manager=mock_cm, saveable_tools=frozenset())
        await node({"messages": [ToolMessage(content="page", tool_call_id="tc1", name="threetears.web_fetch")]})

        mock_cm.save_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_but_structure_still_saves_with_an_empty_set(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-3")

        node = create_context_save_node(context_manager=mock_cm, saveable_tools=frozenset())
        await node({"messages": [_structured_message(_candidate("https://example.gov/warn"))]})

        mock_cm.save_tool_result.assert_awaited_once()


class TestWhatStructureAddsToWhatIsStored:
    @pytest.mark.asyncio
    async def test_provenance_rides_the_saved_metadata(self) -> None:
        # SR-A3: a stored 4000-char truncation with no provenance is a claim
        # nobody can trace back. This is what turns it back into a citation.
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-4")

        node = create_context_save_node(context_manager=mock_cm)
        await node(
            {"messages": [_structured_message(_candidate("https://example.gov/warn", title="Ohio WARN notices"))]}
        )

        record = _saved_kwargs(mock_cm)["metadata"]["search_results"]
        assert record["query"] == "ohio warn notices"
        assert record["candidate_count"] == 1
        assert record["candidates"] == [{"identity": "https://example.gov/warn", "title": "Ohio WARN notices"}]

    @pytest.mark.asyncio
    async def test_the_query_becomes_the_dedup_fingerprint(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-5")

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [_structured_message(_candidate("https://example.gov/warn"))]})

        assert _saved_kwargs(mock_cm)["input_fingerprint"] == "ohio warn notices"

    @pytest.mark.asyncio
    async def test_notices_and_failures_are_recorded(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-6")
        failed = SearchResultsMetadata.from_failure(
            query="ohio warn notices",
            failure=FailureRecord(failure_class="rate-limited", message="slow down", spend=Spend()),
        )
        msg = ToolMessage(
            content="search failed (rate-limited)",
            tool_call_id="tc1",
            name="threetears.web_search",
            artifact={SEARCH_RESULTS_METADATA_KEY: failed.to_metadata()},
        )

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [msg]})

        # A failed search is worth remembering AS a failure -- prose alone would
        # leave a row that reads like a thin answer rather than like no answer.
        assert _saved_kwargs(mock_cm)["metadata"]["search_results"]["failure_class"] == "rate-limited"

    @pytest.mark.asyncio
    async def test_the_candidate_record_is_bounded(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-7")
        many = [_candidate(f"https://example.gov/{i}") for i in range(_MAX_SAVED_CANDIDATES + 5)]

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [_structured_message(*many)]})

        record = _saved_kwargs(mock_cm)["metadata"]["search_results"]
        assert len(record["candidates"]) == _MAX_SAVED_CANDIDATES
        assert record["candidate_count"] == _MAX_SAVED_CANDIDATES + 5
        assert record["candidates_truncated"] is True

    @pytest.mark.asyncio
    async def test_a_result_without_structure_saves_no_metadata(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-8")

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [ToolMessage(content="page", tool_call_id="tc1", name="threetears.web_fetch")]})

        kwargs = _saved_kwargs(mock_cm)
        assert kwargs["metadata"] is None
        assert kwargs["input_fingerprint"] is None

    @pytest.mark.asyncio
    async def test_an_unreadable_payload_still_saves_the_prose(self) -> None:
        # This node runs post-turn inside a graph; raising here would fail a
        # turn whose real work already succeeded.
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-9")
        payload = SearchResultsMetadata.from_candidate_set(
            query="q", candidate_set=CandidateSet(candidates=(_candidate("https://a.gov"),))
        ).to_metadata()
        payload["schema_version"] = 9999
        msg = ToolMessage(
            content="the prose survives",
            tool_call_id="tc1",
            name="threetears.web_search",
            artifact={SEARCH_RESULTS_METADATA_KEY: payload},
        )

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [msg]})

        kwargs = _saved_kwargs(mock_cm)
        assert kwargs["result"] == "the prose survives"
        assert kwargs["metadata"] is None


class TestTheSeamThatCouldNotSeeTheDefect:
    """Drive the real tool and the real adapter, not a hand-built message.

    Every test above constructs its own ToolMessage, which is exactly how C8
    survived: the suite asserted the node's logic against names the suite chose
    rather than against the name production binds. This one lets the adapter
    pick the name and the leaf build the artifact, then asserts the node saves
    it under its own defaults with structure intact.
    """

    @pytest.mark.asyncio
    async def test_a_real_search_result_is_saved_with_its_structure(self) -> None:
        import json as _json

        from threetears.search.testing import LocalHttpServer, Reply

        from threetears.agent.tools.builtin.web_search import create_web_search_tool

        body = _json.dumps(
            {
                "query": "ohio warn notices",
                "number_of_results": 1,
                "results": [
                    {
                        "url": "https://example.gov/warn",
                        "title": "Ohio WARN notices",
                        "content": "A list of WARN notices.",
                        "engine": "duckduckgo",
                        "engines": ["duckduckgo"],
                        "positions": [1],
                        "score": 1.0,
                        "category": "general",
                        "template": "default.html",
                    }
                ],
                "unresponsive_engines": [],
            }
        ).encode()

        async with LocalHttpServer((Reply(body=body, headers={"content-type": "application/json"}),)) as server:
            tool = create_web_search_tool({"base_url": server.base_url}, "Search the web.")
            message = await tool.ainvoke(
                {"name": tool.name, "args": {"query": "ohio warn notices"}, "id": "tc1", "type": "tool_call"}
            )

        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-real")

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [message]})

        mock_cm.save_tool_result.assert_awaited_once()
        kwargs = _saved_kwargs(mock_cm)
        # the adapter's own name, not one this test chose
        assert kwargs["tool_name"] == "threetears.web_search"
        record = kwargs["metadata"]["search_results"]
        assert record["query"] == "ohio warn notices"
        assert record["candidates"][0]["identity"] == "https://example.gov/warn"


class TestTheDedupKeyActuallyDedups:
    """The fingerprint is only worth passing if the key it lands in can collide.

    The node used to pass ``tool_name=f"{tool_name}:{msg.tool_call_id}"``, and
    ``save_tool_result`` derives ``tool_name:sha256(fingerprint)`` from that --
    so the per-call tool_call_id was baked into the dedup key and two identical
    queries could never collide. The original test asserted only that the kwarg
    was passed, which is exactly why it passed.
    """

    @pytest.mark.asyncio
    async def test_the_tool_name_carries_no_per_call_id(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-a")

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [_structured_message(_candidate("https://example.gov/warn"))]})

        assert _saved_kwargs(mock_cm)["tool_name"] == "threetears.web_search"

    @pytest.mark.asyncio
    async def test_two_identical_queries_derive_the_same_row_key(self) -> None:
        # The property that matters, asserted through the real key derivation
        # rather than through the kwarg.
        from threetears.agent.tools.context import make_tool_result_dedup_key

        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-b")
        node = create_context_save_node(context_manager=mock_cm)

        keys = []
        for call_id in ("tc-first", "tc-second"):
            msg = _structured_message(_candidate("https://example.gov/warn"))
            msg.tool_call_id = call_id
            await node({"messages": [msg]})
            kwargs = _saved_kwargs(mock_cm)
            keys.append(make_tool_result_dedup_key(kwargs["tool_name"], kwargs["input_fingerprint"]))

        assert keys[0] == keys[1]

    @pytest.mark.asyncio
    async def test_two_different_queries_derive_different_row_keys(self) -> None:
        from threetears.agent.tools.context import make_tool_result_dedup_key

        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-c")
        node = create_context_save_node(context_manager=mock_cm)

        keys = []
        for query in ("ohio warn notices", "texas warn notices"):
            await node({"messages": [_structured_message(_candidate("https://example.gov/a"), query=query)]})
            kwargs = _saved_kwargs(mock_cm)
            keys.append(make_tool_result_dedup_key(kwargs["tool_name"], kwargs["input_fingerprint"]))

        assert keys[0] != keys[1]


class TestSaveStructuredIsALever:
    @pytest.mark.asyncio
    async def test_disabling_it_returns_the_node_to_name_only_matching(self) -> None:
        # SR-K4: fetched page text is third-party content whose retention the
        # deployment's agreement with the site governs. "Stop using the node" is
        # not a lever; this is.
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock()

        node = create_context_save_node(
            context_manager=mock_cm, saveable_tools=frozenset({"my_tool"}), save_structured=False
        )
        await node({"messages": [_structured_message(_candidate("https://example.gov/warn"), name="web.search.v2")]})

        mock_cm.save_tool_result.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_disabling_it_does_not_disable_saving_by_name(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-d")

        node = create_context_save_node(context_manager=mock_cm, save_structured=False)
        await node({"messages": [_structured_message(_candidate("https://example.gov/warn"))]})

        # matched by name, so still saved -- but with no structure recorded
        mock_cm.save_tool_result.assert_awaited_once()
        assert _saved_kwargs(mock_cm)["metadata"] is None

    @pytest.mark.asyncio
    async def test_it_defaults_to_the_c8_posture(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-e")

        node = create_context_save_node(context_manager=mock_cm, saveable_tools=frozenset({"my_tool"}))
        await node({"messages": [_structured_message(_candidate("https://example.gov/warn"), name="web.search.v2")]})

        mock_cm.save_tool_result.assert_awaited_once()


class TestShortDescStaysWithinItsDocumentedBound:
    @pytest.mark.asyncio
    async def test_a_single_candidate_keeps_its_prose_preview(self) -> None:
        # web_fetch projects one candidate whose query is the URL, so the
        # structured summary would read "1 result(s) for 'https://...'" -- worse
        # than the page's own opening text.
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-f")

        node = create_context_save_node(context_manager=mock_cm)
        await node(
            {
                "messages": [
                    _structured_message(
                        _candidate("https://example.gov/warn"),
                        name="threetears.web_fetch",
                        query="https://example.gov/warn",
                        content="The Ohio WARN notice list for 2026 begins here.",
                    )
                ]
            }
        )

        assert _saved_kwargs(mock_cm)["short_desc"] == "The Ohio WARN notice list for 2026 begins here."

    @pytest.mark.asyncio
    async def test_a_long_query_cannot_write_past_the_200_char_bound(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-g")

        node = create_context_save_node(context_manager=mock_cm)
        await node(
            {
                "messages": [
                    _structured_message(
                        _candidate("https://a.gov"), _candidate("https://b.gov"), query="warn notices " * 40
                    )
                ]
            }
        )

        assert len(_saved_kwargs(mock_cm)["short_desc"]) <= 200

    @pytest.mark.asyncio
    async def test_a_real_result_set_still_gets_the_structured_summary(self) -> None:
        mock_cm = AsyncMock()
        mock_cm.save_tool_result = AsyncMock(return_value="ctx-h")

        node = create_context_save_node(context_manager=mock_cm)
        await node({"messages": [_structured_message(_candidate("https://a.gov"), _candidate("https://b.gov"))]})

        assert _saved_kwargs(mock_cm)["short_desc"] == "2 result(s) for 'ohio warn notices'"
