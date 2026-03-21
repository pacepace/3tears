"""Tests for LangGraph context enrichment and save nodes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from threetears.agent.tools.graph_nodes import (
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
        """Skips enrichment when ledger has sufficient items."""
        mock_cm = MagicMock()
        mock_ledger = MagicMock()
        mock_ledger.__len__ = MagicMock(return_value=5)
        mock_cm.ledger = mock_ledger

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
