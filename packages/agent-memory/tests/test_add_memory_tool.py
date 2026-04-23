"""Tests for the add_memory tool.

Collection-parameterised (namespace-task-01 phase 8.5b): the factory
takes a :class:`MemoriesCollection` as a required parameter. tests
build a registry-bound collection around an in-memory mock pool.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4


from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.tools import AddMemoryInput, load_add_memory_tool
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig


_TEST_UID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_TEST_CID = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_TEST_MID = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
_TEST_AID = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
_TEST_CUID = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")


def _make_pool():
    """build a simple asyncpg-shape mock pool."""
    pool = AsyncMock()
    pool.fetch.return_value = []
    pool.fetchval.return_value = False  # no existing memories for user
    pool.fetchrow.return_value = None
    pool.execute.return_value = "INSERT 0 1"
    return pool


def _make_embedding_provider():
    """build a stub embedding provider."""
    provider = AsyncMock()
    provider.embed_text.return_value = ([0.1] * 768, 10)
    return provider


def _make_collection(
    pool: AsyncMock,
    authorizer: MemoryAuthorizerDependencies,
) -> MemoriesCollection:
    """build a registry-bound :class:`MemoriesCollection` around the pool."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    core_config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    return MemoriesCollection(
        registry=registry,
        config=core_config,
        authorizer=authorizer,
    )


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

    async def test_creates_tool(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ):
        pool = _make_pool()
        provider = _make_embedding_provider()
        memories = _make_collection(pool, permissive_memory_authorizer)
        tools = await load_add_memory_tool(
            _TEST_UID,
            _TEST_CID,
            _TEST_MID,
            provider,
            _TEST_AID,
            _TEST_CUID,
            permissive_memory_authorizer,
            memories,
        )
        assert len(tools) == 1
        assert tools[0].name == "add_memory"

    async def test_stores_new_memory(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ):
        pool = _make_pool()
        provider = _make_embedding_provider()
        memories = _make_collection(pool, permissive_memory_authorizer)
        tools = await load_add_memory_tool(
            _TEST_UID,
            _TEST_CID,
            _TEST_MID,
            provider,
            _TEST_AID,
            _TEST_CUID,
            permissive_memory_authorizer,
            memories,
        )

        result = await tools[0].ainvoke({"content": "User prefers Rust", "memory_type": "preference"})

        assert "Remembered" in result
        assert "Rust" in result
        provider.embed_text.assert_called_once_with("User prefers Rust")
        pool.execute.assert_called_once()

    async def test_invalid_type_returns_error(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ):
        pool = _make_pool()
        provider = _make_embedding_provider()
        memories = _make_collection(pool, permissive_memory_authorizer)
        tools = await load_add_memory_tool(
            _TEST_UID,
            _TEST_CID,
            _TEST_MID,
            provider,
            _TEST_AID,
            _TEST_CUID,
            permissive_memory_authorizer,
            memories,
        )

        result = await tools[0].ainvoke({"content": "something", "memory_type": "bogus"})

        assert "Invalid memory_type" in result
        provider.embed_text.assert_not_called()

    async def test_dedup_updates_existing(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ):
        # seed an existing very-similar memory row so the Collection's
        # find_similar_for_dedup surfaces it, then dedup triggers UPDATE
        existing_id = uuid4()
        existing_row = {
            "memory_id": existing_id,
            "agent_id": _TEST_AID,
            "customer_id": _TEST_CUID,
            "user_id": _TEST_UID,
            "conversation_id": uuid4(),
            "message_id_source": uuid4(),
            "type_memory": "preference",
            "content": "User likes Rust programming",
            "embedding": "[0.1, 0.2, 0.3]",
            "is_deleted": False,
            "media_id": None,
            "date_created": None,
            "date_deleted": None,
            "date_updated": None,
        }
        pool = _make_pool()

        async def _fetch(query: str, *args):
            # find_similar_for_dedup returns rows with the cosine score
            return [
                {
                    "memory_id": existing_id,
                    "content": "User likes Rust programming",
                    "type_memory": "preference",
                    "similarity": 0.95,
                }
            ]

        async def _fetchrow(query: str, *args):
            if args and str(args[0]) == str(existing_id):
                return existing_row
            return None

        pool.fetch.side_effect = _fetch
        pool.fetchrow.side_effect = _fetchrow
        provider = _make_embedding_provider()
        memories = _make_collection(pool, permissive_memory_authorizer)
        tools = await load_add_memory_tool(
            _TEST_UID,
            _TEST_CID,
            _TEST_MID,
            provider,
            _TEST_AID,
            _TEST_CUID,
            permissive_memory_authorizer,
            memories,
        )

        result = await tools[0].ainvoke({"content": "User prefers Rust", "memory_type": "preference"})

        assert "Updated existing memory" in result
        assert "95%" in result

    async def test_no_dedup_below_threshold(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ):
        pool = _make_pool()

        async def _fetch(query: str, *args):
            return [
                {
                    "memory_id": uuid4(),
                    "content": "User likes Python",
                    "type_memory": "preference",
                    "similarity": 0.5,
                }
            ]

        pool.fetch.side_effect = _fetch
        provider = _make_embedding_provider()
        memories = _make_collection(pool, permissive_memory_authorizer)
        tools = await load_add_memory_tool(
            _TEST_UID,
            _TEST_CID,
            _TEST_MID,
            provider,
            _TEST_AID,
            _TEST_CUID,
            permissive_memory_authorizer,
            memories,
        )

        result = await tools[0].ainvoke({"content": "User prefers Rust", "memory_type": "preference"})

        assert "Remembered" in result
        # the Collection's _save_to_postgres emitted an INSERT via execute
        pool.execute.assert_called()
        call_sql = pool.execute.call_args[0][0]
        assert "INSERT INTO memories" in call_sql

    async def test_embedding_failure_returns_error(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ):
        pool = _make_pool()
        provider = _make_embedding_provider()
        provider.embed_text.side_effect = RuntimeError("embedding service down")
        memories = _make_collection(pool, permissive_memory_authorizer)
        tools = await load_add_memory_tool(
            _TEST_UID,
            _TEST_CID,
            _TEST_MID,
            provider,
            _TEST_AID,
            _TEST_CUID,
            permissive_memory_authorizer,
            memories,
        )

        result = await tools[0].ainvoke({"content": "something", "memory_type": "fact"})

        assert "TOOL ERROR" in result
        assert "embed" in result

    async def test_all_memory_types_accepted(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ):
        pool = _make_pool()
        provider = _make_embedding_provider()
        memories = _make_collection(pool, permissive_memory_authorizer)
        tools = await load_add_memory_tool(
            _TEST_UID,
            _TEST_CID,
            _TEST_MID,
            provider,
            _TEST_AID,
            _TEST_CUID,
            permissive_memory_authorizer,
            memories,
        )

        for mt in ["preference", "fact", "decision", "topical_context", "relational_context"]:
            provider.reset_mock()
            pool.reset_mock()
            pool.fetch.return_value = []
            pool.fetchval.return_value = False
            result = await tools[0].ainvoke({"content": f"test {mt}", "memory_type": mt})
            assert "Remembered" in result, f"Failed for type {mt}"
