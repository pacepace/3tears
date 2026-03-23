"""Tests for the add_memory tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.memory.tools import AddMemoryInput, load_add_memory_tool


_TEST_UID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_TEST_CID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_TEST_MID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")


def _make_pool():
    pool = AsyncMock()
    pool.fetch.return_value = []  # no similar memories by default
    pool.execute.return_value = None
    return pool


def _make_embedding_provider():
    provider = AsyncMock()
    provider.embed_text.return_value = ([0.1] * 768, 10)
    return provider


class TestAddMemoryInput:
    """Input validation for AddMemoryInput."""

    def test_valid_input(self):
        inp = AddMemoryInput(content="User likes Rust", memory_type="preference")
        assert inp.content == "User likes Rust"
        assert inp.memory_type == "preference"

    def test_default_type(self):
        inp = AddMemoryInput(content="something")
        assert inp.memory_type == "preference"


class TestLoadAddMemoryTool:
    """add_memory tool behavior."""

    async def test_creates_tool(self):
        pool = _make_pool()
        provider = _make_embedding_provider()
        tools = await load_add_memory_tool(pool, _TEST_UID, _TEST_CID, _TEST_MID, provider)
        assert len(tools) == 1
        assert tools[0].name == "add_memory"

    async def test_stores_new_memory(self):
        pool = _make_pool()
        provider = _make_embedding_provider()
        tools = await load_add_memory_tool(pool, _TEST_UID, _TEST_CID, _TEST_MID, provider)

        result = await tools[0].ainvoke({"content": "User prefers Rust", "memory_type": "preference"})

        assert "Remembered" in result
        assert "Rust" in result
        provider.embed_text.assert_called_once_with("User prefers Rust")
        pool.execute.assert_called_once()

    async def test_invalid_type_returns_error(self):
        pool = _make_pool()
        provider = _make_embedding_provider()
        tools = await load_add_memory_tool(pool, _TEST_UID, _TEST_CID, _TEST_MID, provider)

        result = await tools[0].ainvoke({"content": "something", "memory_type": "bogus"})

        assert "Invalid memory_type" in result
        provider.embed_text.assert_not_called()

    async def test_dedup_updates_existing(self):
        pool = _make_pool()
        # Simulate a very similar existing memory
        pool.fetch.return_value = [
            {
                "memory_id": uuid4(),
                "content": "User likes Rust programming",
                "type_memory": "preference",
                "similarity": 0.95,
            }
        ]
        provider = _make_embedding_provider()
        tools = await load_add_memory_tool(pool, _TEST_UID, _TEST_CID, _TEST_MID, provider)

        result = await tools[0].ainvoke({"content": "User prefers Rust", "memory_type": "preference"})

        assert "Updated existing memory" in result
        assert "95%" in result
        # Should call execute (UPDATE), not execute (INSERT)
        pool.execute.assert_called_once()
        call_sql = pool.execute.call_args[0][0]
        assert "UPDATE memories" in call_sql

    async def test_no_dedup_below_threshold(self):
        pool = _make_pool()
        pool.fetch.return_value = [
            {
                "memory_id": uuid4(),
                "content": "User likes Python",
                "type_memory": "preference",
                "similarity": 0.5,  # below 0.90 threshold
            }
        ]
        provider = _make_embedding_provider()
        tools = await load_add_memory_tool(pool, _TEST_UID, _TEST_CID, _TEST_MID, provider)

        result = await tools[0].ainvoke({"content": "User prefers Rust", "memory_type": "preference"})

        assert "Remembered" in result
        call_sql = pool.execute.call_args[0][0]
        assert "INSERT INTO memories" in call_sql

    async def test_embedding_failure_returns_error(self):
        pool = _make_pool()
        provider = _make_embedding_provider()
        provider.embed_text.side_effect = RuntimeError("embedding service down")
        tools = await load_add_memory_tool(pool, _TEST_UID, _TEST_CID, _TEST_MID, provider)

        result = await tools[0].ainvoke({"content": "something", "memory_type": "fact"})

        assert "TOOL ERROR" in result
        assert "embed" in result

    async def test_all_memory_types_accepted(self):
        pool = _make_pool()
        provider = _make_embedding_provider()
        tools = await load_add_memory_tool(pool, _TEST_UID, _TEST_CID, _TEST_MID, provider)

        for mt in ["preference", "fact", "decision", "topical_context", "relational_context"]:
            provider.reset_mock()
            pool.reset_mock()
            pool.fetch.return_value = []
            result = await tools[0].ainvoke({"content": f"test {mt}", "memory_type": mt})
            assert "Remembered" in result, f"Failed for type {mt}"
