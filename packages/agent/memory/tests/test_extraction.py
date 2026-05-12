"""Tests for memory extraction -- gating, extraction, resolution, end-to-end.

Collection-parameterised (namespace-task-01 phase 8.5b): the extractor
takes a :class:`MemoriesCollection` as a required constructor
parameter and no longer accepts a raw pool. tests build a
registry-bound collection around an in-memory mock pool and pass it
through.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.extraction import MemoryExtractor
from threetears.agent.memory.types import MemoryConfig
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig


_TEST_AID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_TEST_CUID = uuid.UUID("22222222-2222-2222-2222-222222222222")


# -- Stubs/helpers ------------------------------------------------------------


class StubChatModel:
    """Stub chat model that returns a preconfigured response."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages: list[Any]) -> Any:
        resp = MagicMock()
        resp.content = self._content
        return resp


class ErrorChatModel:
    """Chat model that raises on ainvoke."""

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("LLM unavailable")

    async def ainvoke(self, messages: list[Any]) -> Any:
        raise self._exc


class StubChatModelFactory:
    """Factory that returns preconfigured chat models per purpose."""

    def __init__(
        self,
        default_content: str = "[]",
        worthiness_content: str | None = None,
        extraction_content: str | None = None,
        resolution_content: str | None = None,
        error_on: set[str] | None = None,
    ) -> None:
        self._models: dict[str, StubChatModel | ErrorChatModel] = {}
        error_purposes = error_on or set()
        for purpose in ("worthiness", "extraction", "resolution"):
            if purpose in error_purposes:
                self._models[purpose] = ErrorChatModel()
            else:
                content_map = {
                    "worthiness": worthiness_content,
                    "extraction": extraction_content,
                    "resolution": resolution_content,
                }
                self._models[purpose] = StubChatModel(
                    content_map[purpose] or default_content,
                )

    async def create_chat_model(self, purpose: str = "extraction") -> Any:
        return self._models.get(purpose, StubChatModel("[]"))


class StubEmbeddingProvider:
    """Stub LangChain ``Embeddings`` returning a fixed vector."""

    def __init__(
        self,
        embedding: list[float] | None = None,
        fail: bool = False,
    ) -> None:
        self._embedding = embedding or [1.0, 0.0, 0.0]
        self._fail = fail

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            return [[] for _ in texts]
        return [self._embedding for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        _ = text
        if self._fail:
            return []
        return self._embedding

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        if self._fail:
            raise RuntimeError("embedding service down")
        return self._embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._fail:
            raise RuntimeError("embedding service down")
        return [self._embedding for _ in texts]

    @property
    def dimensions(self) -> int:
        return len(self._embedding) if self._embedding else 3


def _make_pool(
    fetch_rows: list[dict[str, Any]] | None = None,
) -> AsyncMock:
    """build a pool whose fetch / fetchrow / fetchval / execute are mocks."""
    pool = AsyncMock()
    pool.fetch = AsyncMock(return_value=fetch_rows or [])
    pool.fetchrow = AsyncMock(return_value=None)
    pool.fetchval = AsyncMock(return_value=False)
    pool.execute = AsyncMock(return_value="INSERT 0 1")
    return pool


def _make_memories_collection(
    pool: AsyncMock,
    authorizer: MemoryAuthorizerDependencies,
) -> MemoriesCollection:
    """build a registry-bound :class:`MemoriesCollection` around the pool.

    :param pool: mock asyncpg pool
    :ptype pool: AsyncMock
    :param authorizer: rbac authorizer bundle
    :ptype authorizer: MemoryAuthorizerDependencies
    :return: collection instance
    :rtype: MemoriesCollection
    """
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


def _make_extractor(
    authorizer: MemoryAuthorizerDependencies,
    pool: AsyncMock | None = None,
    config: MemoryConfig | None = None,
    factory: StubChatModelFactory | None = None,
    embedding: StubEmbeddingProvider | None = None,
    nats_client: Any = None,
    summary_callback: Any = None,
) -> MemoryExtractor:
    """build a :class:`MemoryExtractor` with a registry-bound Collection."""
    real_pool = pool or _make_pool()
    memories = _make_memories_collection(real_pool, authorizer)
    return MemoryExtractor(
        config=config or MemoryConfig(),
        embedding_provider=embedding or StubEmbeddingProvider(),
        chat_model_factory=factory or StubChatModelFactory(),
        authorizer=authorizer,
        memories_collection=memories,
        nats_client=nats_client,
        summary_callback=summary_callback,
    )


# -- Heuristic gate tests ----------------------------------------------------


class TestHeuristicGates:
    def test_short_user_message_rejected(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        ext = _make_extractor(permissive_memory_authorizer)
        passed, reason = ext.check_heuristic_gates("hi", "x" * 200, 10)
        assert not passed
        assert "user_message_too_short" in reason

    def test_short_assistant_response_rejected(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        ext = _make_extractor(permissive_memory_authorizer)
        passed, reason = ext.check_heuristic_gates("x" * 50, "short", 10)
        assert not passed
        assert "assistant_response_too_short" in reason

    def test_low_turn_count_rejected(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        ext = _make_extractor(permissive_memory_authorizer)
        passed, reason = ext.check_heuristic_gates("x" * 50, "x" * 200, 1)
        assert not passed
        assert "too_few_turns" in reason

    def test_all_thresholds_met(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        ext = _make_extractor(permissive_memory_authorizer)
        passed, reason = ext.check_heuristic_gates("x" * 50, "x" * 200, 10)
        assert passed
        assert reason == "passed"

    def test_custom_thresholds(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        config = MemoryConfig(
            extraction_min_user_message_length=5,
            extraction_min_assistant_response_length=10,
            extraction_min_conversation_turns=1,
        )
        ext = _make_extractor(permissive_memory_authorizer, config=config)
        passed, _ = ext.check_heuristic_gates("hello", "y" * 10, 1)
        assert passed

    def test_whitespace_stripped_for_length_check(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        ext = _make_extractor(permissive_memory_authorizer)
        passed, reason = ext.check_heuristic_gates("     short", "x" * 200, 10)
        assert not passed
        assert "user_message_too_short" in reason


# -- Rate limit tests ---------------------------------------------------------


class TestRateLimit:
    async def test_no_nats_client_passes(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        ext = _make_extractor(permissive_memory_authorizer, nats_client=None)
        passed, cooldown = await ext.check_rate_limit(uuid.uuid7())
        assert passed
        assert cooldown == 0

    async def test_nats_create_true_passes(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        nats = MagicMock()
        nats.bucket_name = MagicMock(return_value="locks")
        nats.create = AsyncMock(return_value=True)
        ext = _make_extractor(permissive_memory_authorizer, nats_client=nats)
        passed, cooldown = await ext.check_rate_limit(uuid.uuid7())
        assert passed
        assert cooldown == 0

    async def test_nats_create_false_rejected(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        nats = MagicMock()
        nats.bucket_name = MagicMock(return_value="locks")
        nats.create = AsyncMock(return_value=False)
        ext = _make_extractor(permissive_memory_authorizer, nats_client=nats)
        passed, cooldown = await ext.check_rate_limit(uuid.uuid7())
        assert not passed
        assert cooldown == 300

    async def test_nats_error_passes_fail_open(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        nats = MagicMock()
        nats.bucket_name = MagicMock(side_effect=RuntimeError("nats down"))
        ext = _make_extractor(permissive_memory_authorizer, nats_client=nats)
        passed, cooldown = await ext.check_rate_limit(uuid.uuid7())
        assert passed
        assert cooldown == 0


# -- Worthiness gate tests ----------------------------------------------------


class TestWorthinessGate:
    async def test_worthy_true_passes(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": True, "reason": "has facts"}),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        worthy, reason = await ext.check_worthiness("hello world", "response text")
        assert worthy
        assert reason == "has facts"

    async def test_worthy_false_rejected(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": False}),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        worthy, _ = await ext.check_worthiness("hello world", "response text")
        assert not worthy

    async def test_invalid_json_passes_fail_open(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(worthiness_content="not json at all")
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        worthy, reason = await ext.check_worthiness("hello world", "response text")
        assert worthy
        assert reason == "parse_error"

    async def test_llm_error_passes_fail_open(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(error_on={"worthiness"})
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        worthy, reason = await ext.check_worthiness("hello world", "response text")
        assert worthy
        assert reason == "llm_error"


# -- Extraction tests ---------------------------------------------------------


class TestExtractCandidates:
    async def test_valid_json_array_parsed(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        memories = [
            {"type": "fact", "content": "Lives in Seattle"},
            {"type": "preference", "content": "Prefers Python"},
        ]
        factory = StubChatModelFactory(
            extraction_content=json.dumps(memories),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        result = await ext.extract_candidates("msg", "resp")
        assert len(result) == 2
        assert result[0]["type"] == "fact"
        assert result[1]["content"] == "Prefers Python"

    async def test_invalid_types_filtered(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        memories = [
            {"type": "fact", "content": "valid"},
            {"type": "invalid_type", "content": "should be filtered"},
        ]
        factory = StubChatModelFactory(
            extraction_content=json.dumps(memories),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        result = await ext.extract_candidates("msg", "resp")
        assert len(result) == 1
        assert result[0]["type"] == "fact"

    async def test_empty_array_no_candidates(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(extraction_content="[]")
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        result = await ext.extract_candidates("msg", "resp")
        assert result == []

    async def test_non_list_returns_empty(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(
            extraction_content=json.dumps({"type": "fact", "content": "x"}),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        result = await ext.extract_candidates("msg", "resp")
        assert result == []

    async def test_markdown_wrapped_json_parsed(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        memories = [{"type": "fact", "content": "Lives in Seattle"}]
        wrapped = "```json\n" + json.dumps(memories) + "\n```"
        factory = StubChatModelFactory(extraction_content=wrapped)
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        result = await ext.extract_candidates("msg", "resp")
        assert len(result) == 1
        assert result[0]["content"] == "Lives in Seattle"

    async def test_empty_content_filtered(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        memories = [
            {"type": "fact", "content": ""},
            {"type": "fact", "content": "   "},
            {"type": "fact", "content": "valid content"},
        ]
        factory = StubChatModelFactory(
            extraction_content=json.dumps(memories),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        result = await ext.extract_candidates("msg", "resp")
        assert len(result) == 1


# -- Resolution tests ---------------------------------------------------------


class TestResolveActions:
    async def test_no_similar_memories_fast_path_all_add(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        ext = _make_extractor(permissive_memory_authorizer)
        candidates = [
            {"type": "fact", "content": "x", "embedding": [1.0], "similar_memories": []},
            {"type": "fact", "content": "y", "embedding": [1.0], "similar_memories": []},
        ]
        actions = await ext.resolve_actions(candidates)
        assert len(actions) == 2
        assert all(a["action"] == "ADD" for a in actions)

    async def test_no_similar_memories_no_llm_call(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(error_on={"resolution"})
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        candidates = [
            {"type": "fact", "content": "x", "embedding": [1.0], "similar_memories": []},
        ]
        actions = await ext.resolve_actions(candidates)
        assert len(actions) == 1
        assert actions[0]["action"] == "ADD"

    async def test_with_similar_memories_llm_called(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        resolution = [
            {"index": 0, "action": "UPDATE", "memory_id": "abc-123", "content": "updated", "type": "fact"},
        ]
        factory = StubChatModelFactory(
            resolution_content=json.dumps(resolution),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        candidates = [
            {
                "type": "fact",
                "content": "x",
                "embedding": [1.0],
                "similar_memories": [
                    {"memory_id": "abc-123", "content": "old", "type_memory": "fact", "similarity": 0.9},
                ],
            },
        ]
        actions = await ext.resolve_actions(candidates)
        assert len(actions) == 1
        assert actions[0]["action"] == "UPDATE"
        assert actions[0]["memory_id"] == "abc-123"

    async def test_invalid_action_index_skipped(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        resolution = [
            {"index": 99, "action": "ADD"},
            {"index": 0, "action": "ADD"},
        ]
        factory = StubChatModelFactory(
            resolution_content=json.dumps(resolution),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        candidates = [
            {
                "type": "fact",
                "content": "x",
                "embedding": [1.0],
                "similar_memories": [
                    {"memory_id": "id1", "content": "y", "type_memory": "fact", "similarity": 0.8},
                ],
            },
        ]
        actions = await ext.resolve_actions(candidates)
        valid = [a for a in actions if a["action"] == "ADD"]
        assert len(valid) == 1
        assert valid[0]["index"] == 0

    async def test_missing_candidates_default_to_add(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        resolution = [{"index": 0, "action": "NOOP"}]
        factory = StubChatModelFactory(
            resolution_content=json.dumps(resolution),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        candidates = [
            {
                "type": "fact",
                "content": "x",
                "embedding": [1.0],
                "similar_memories": [{"memory_id": "id1", "content": "y", "type_memory": "fact", "similarity": 0.8}],
            },
            {
                "type": "fact",
                "content": "z",
                "embedding": [1.0],
                "similar_memories": [],
            },
        ]
        actions = await ext.resolve_actions(candidates)
        action_map = {a["index"]: a["action"] for a in actions}
        assert action_map[0] == "NOOP"
        assert action_map[1] == "ADD"

    async def test_invalid_update_no_memory_id_becomes_noop(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        resolution = [
            {"index": 0, "action": "UPDATE", "content": "updated but no memory_id"},
        ]
        factory = StubChatModelFactory(
            resolution_content=json.dumps(resolution),
        )
        ext = _make_extractor(permissive_memory_authorizer, factory=factory)
        candidates = [
            {
                "type": "fact",
                "content": "x",
                "embedding": [1.0],
                "similar_memories": [{"memory_id": "id1", "content": "y", "type_memory": "fact", "similarity": 0.8}],
            },
        ]
        actions = await ext.resolve_actions(candidates)
        assert actions[0]["action"] == "NOOP"


# -- End-to-end extract() tests -----------------------------------------------


class TestExtractE2E:
    async def test_happy_path(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        extraction_result = [{"type": "fact", "content": "Lives in Seattle"}]
        worthiness_result = {"worthy": True, "reason": "biographical"}
        factory = StubChatModelFactory(
            worthiness_content=json.dumps(worthiness_result),
            extraction_content=json.dumps(extraction_result),
        )
        pool = _make_pool()
        ext = _make_extractor(permissive_memory_authorizer, pool=pool, factory=factory)

        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )
        pool.execute.assert_called()

    async def test_heuristic_gate_fails_no_llm_calls(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(error_on={"worthiness", "extraction", "resolution"})
        pool = _make_pool()
        ext = _make_extractor(permissive_memory_authorizer, pool=pool, factory=factory)

        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="hi",
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )
        pool.execute.assert_not_called()
        pool.fetch.assert_not_called()

    async def test_fire_and_forget_no_raise(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """extract() must never raise, even if internals blow up."""
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": True}),
            extraction_content=json.dumps([{"type": "fact", "content": "x"}]),
        )
        pool = _make_pool()
        pool.execute = AsyncMock(side_effect=RuntimeError("db down"))

        ext = _make_extractor(permissive_memory_authorizer, pool=pool, factory=factory)
        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )

    async def test_worthiness_rejected_no_extraction(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": False}),
            extraction_content=json.dumps([{"type": "fact", "content": "x"}]),
        )
        pool = _make_pool()
        ext = _make_extractor(permissive_memory_authorizer, pool=pool, factory=factory)

        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )
        pool.execute.assert_not_called()

    async def test_no_candidates_extracted(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": True}),
            extraction_content="[]",
        )
        pool = _make_pool()
        ext = _make_extractor(permissive_memory_authorizer, pool=pool, factory=factory)

        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )
        pool.execute.assert_not_called()

    async def test_embedding_failure_skips_candidate(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": True}),
            extraction_content=json.dumps([{"type": "fact", "content": "x"}]),
        )
        pool = _make_pool()
        embedding = StubEmbeddingProvider(fail=True)
        ext = _make_extractor(
            permissive_memory_authorizer,
            pool=pool,
            factory=factory,
            embedding=embedding,
        )

        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )
        pool.execute.assert_not_called()


# -- Summary callback tests ---------------------------------------------------


class TestSummaryCallback:
    async def test_callback_called_after_add(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        extraction_result = [{"type": "fact", "content": "Lives in Seattle"}]
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": True}),
            extraction_content=json.dumps(extraction_result),
        )
        pool = _make_pool()
        callback = AsyncMock()
        ext = _make_extractor(
            permissive_memory_authorizer,
            pool=pool,
            factory=factory,
            summary_callback=callback,
        )

        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[1] == "Lives in Seattle"

    async def test_no_callback_no_error(
        self,
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        extraction_result = [{"type": "fact", "content": "Lives in Seattle"}]
        factory = StubChatModelFactory(
            worthiness_content=json.dumps({"worthy": True}),
            extraction_content=json.dumps(extraction_result),
        )
        pool = _make_pool()
        ext = _make_extractor(
            permissive_memory_authorizer,
            pool=pool,
            factory=factory,
            summary_callback=None,
        )

        await ext.extract(
            user_id=uuid.uuid7(),
            conversation_id=uuid.uuid7(),
            message_id_source=uuid.uuid7(),
            user_message="x" * 50,
            assistant_response="y" * 200,
            turn_count=10,
            agent_id=_TEST_AID,
            customer_id=_TEST_CUID,
        )
