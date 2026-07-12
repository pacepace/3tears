"""Unit tests for Dream consolidation (A5, v0.15.0).

Covers the pure clustering + modal-type helpers exhaustively, and the
:class:`DreamService.run_consolidation` orchestration against mocked
Collections + a stubbed reflector / embedder (no database). The live
end-to-end merge is in ``tests/integration/test_dream_consolidation.py``.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.agent.memory.dream import (
    ConsolidationResult,
    DreamService,
    _cluster_by_cosine,
    _modal_type,
)
from threetears.agent.memory.events import MemoryConsolidatedEvent
from threetears.agent.memory.types import MemoryConfig


_AID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_CUID = uuid.UUID("22222222-2222-2222-2222-222222222222")
_UID = uuid.UUID("33333333-3333-3333-3333-333333333333")


# -- Stubs --------------------------------------------------------------------


class StubChatModel:
    """Reflector stub returning a preconfigured content string."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages: list[Any], **kwargs: Any) -> Any:
        _ = kwargs
        resp = MagicMock()
        resp.content = self._content
        return resp


class StubReflectorFactory:
    """Factory returning one reflector model, or raising on invoke."""

    def __init__(self, content: str = '{"gist": "merged", "rationale": "near-dup"}', fail: bool = False) -> None:
        self._content = content
        self._fail = fail

    async def create_chat_model(self, purpose: str = "consolidation") -> Any:
        _ = purpose
        if self._fail:
            model = MagicMock()
            model.ainvoke = AsyncMock(side_effect=RuntimeError("reflector down"))
            return model
        return StubChatModel(self._content)


class StubEmbeddings:
    """LangChain ``Embeddings`` stub returning a fixed vector."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = vector or [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        _ = text
        return self._vector

    async def aembed_query(self, text: str) -> list[float]:
        _ = text
        return self._vector

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


def _candidate(
    *,
    content: str,
    embedding: list[float],
    type_memory: str = "fact",
    customer_id: uuid.UUID | None = _CUID,
    user_id: uuid.UUID | None = _UID,
    conversation_id: uuid.UUID | None = None,
    age_days: int = 0,
) -> dict[str, Any]:
    """build a candidate dict shaped like find_active_for_consolidation output."""
    return {
        "memory_id": uuid.uuid4(),
        "customer_id": customer_id,
        "user_id": user_id,
        "conversation_id": conversation_id or uuid.uuid4(),
        "type_memory": type_memory,
        "content": content,
        "embedding": embedding,
        "date_created": datetime.now(UTC) - timedelta(days=age_days),
    }


def _make_service(
    *,
    candidates: list[dict[str, Any]],
    reflector: StubReflectorFactory | None = None,
    config: MemoryConfig | None = None,
) -> tuple[DreamService, MagicMock, MagicMock]:
    """wire a DreamService over mocked collections; return (service, memories, edges)."""
    memories = MagicMock()
    memories.find_active_for_consolidation = AsyncMock(return_value=candidates)
    memories.create = MagicMock(side_effect=lambda data: MagicMock(data=data))
    memories.save_entity = AsyncMock()
    memories.mark_superseded = AsyncMock()

    edges = MagicMock()
    edges.assert_no_cycle = AsyncMock()
    edges.create = MagicMock(side_effect=lambda data: MagicMock(data=data))
    edges.save_entity = AsyncMock()

    service = DreamService(
        config=config or MemoryConfig(),
        embedding_provider=StubEmbeddings(),
        chat_model_factory=reflector or StubReflectorFactory(),
        memories_collection=memories,
        consolidations_collection=edges,
    )
    return service, memories, edges


# -- Pure helpers -------------------------------------------------------------


class TestClusterByCosine:
    def test_near_duplicates_group_dissimilar_separate(self) -> None:
        # two near-identical vectors + one orthogonal.
        embeddings = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
        clusters = _cluster_by_cosine(embeddings, threshold=0.85)
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [1, 2]
        # the pair is the near-duplicate one (indices 0 and 1).
        pair = next(c for c in clusters if len(c) == 2)
        assert sorted(pair) == [0, 1]

    def test_transitive_chain_forms_one_cluster(self) -> None:
        # a~b, b~c, but a not directly ~c: union-find still unions all three.
        embeddings = [[1.0, 0.0], [0.9, 0.44], [0.7, 0.71]]
        clusters = _cluster_by_cosine(embeddings, threshold=0.80)
        assert len(clusters) == 1
        assert sorted(clusters[0]) == [0, 1, 2]

    def test_all_singletons_when_below_threshold(self) -> None:
        embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        clusters = _cluster_by_cosine(embeddings, threshold=0.85)
        assert sorted(len(c) for c in clusters) == [1, 1, 1]

    def test_zero_vector_never_unions(self) -> None:
        embeddings = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        clusters = _cluster_by_cosine(embeddings, threshold=0.85)
        # the two identical non-zero vectors group; the zero vector is alone.
        assert sorted(len(c) for c in clusters) == [1, 2]

    def test_empty(self) -> None:
        assert _cluster_by_cosine([], threshold=0.85) == []


class TestModalType:
    def test_majority_wins(self) -> None:
        assert _modal_type(["fact", "fact", "preference"]) == "fact"

    def test_tie_breaks_by_declaration_order(self) -> None:
        # preference is declared before fact in MemoryType -> wins the tie.
        assert _modal_type(["fact", "preference"]) == "preference"


# -- Orchestration ------------------------------------------------------------


class TestRunConsolidation:
    async def test_below_min_cluster_size_no_op(self) -> None:
        service, memories, edges = _make_service(
            candidates=[_candidate(content="only one", embedding=[1.0, 0.0])],
        )
        result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)
        assert isinstance(result, ConsolidationResult)
        assert result.gists_created == 0
        assert result.clusters_formed == 0
        memories.create.assert_not_called()
        memories.mark_superseded.assert_not_called()

    async def test_cluster_creates_gist_edges_and_supersedes(self) -> None:
        sources = [
            _candidate(content="prefers postgres, cites jsonb", embedding=[1.0, 0.0], age_days=2),
            _candidate(content="picked postgres for pgvector", embedding=[0.99, 0.02], age_days=1),
        ]
        service, memories, edges = _make_service(candidates=sources)
        result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)

        assert result.gists_created == 1
        assert result.clusters_formed == 1
        assert result.sources_superseded == 2
        assert len(result.gist_ids) == 1

        # cycle-guard ran before writes; gist committed; two edges recorded.
        edges.assert_no_cycle.assert_awaited_once()
        memories.save_entity.assert_awaited()  # the gist
        assert edges.save_entity.await_count == 2

        # gist inherits scope + the NEWEST source's conversation_id.
        gist_data = memories.create.call_args.args[0]
        assert gist_data["customer_id"] == _CUID
        assert gist_data["user_id"] == _UID
        assert gist_data["content"] == "merged"
        assert gist_data["salience"] == MemoryConfig().consolidation_gist_salience_seed
        newest_conv = max(sources, key=lambda m: m["date_created"])["conversation_id"]
        assert gist_data["conversation_id"] == newest_conv

        # sources superseded by the committed gist.
        supersede_kwargs = memories.mark_superseded.call_args.kwargs
        assert supersede_kwargs["gist_id"] == gist_data["memory_id"]
        assert sorted(str(s) for s in supersede_kwargs["source_memory_ids"]) == sorted(
            str(s["memory_id"]) for s in sources
        )

    async def test_reflector_failure_skips_cluster(self) -> None:
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0]),
            _candidate(content="b", embedding=[0.99, 0.02]),
        ]
        service, memories, edges = _make_service(
            candidates=sources,
            reflector=StubReflectorFactory(fail=True),
        )
        result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)
        # cluster formed but no gist committed on reflector failure.
        assert result.clusters_formed == 1
        assert result.gists_created == 0
        memories.create.assert_not_called()
        memories.mark_superseded.assert_not_called()

    async def test_empty_gist_text_skips_cluster(self) -> None:
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0]),
            _candidate(content="b", embedding=[0.99, 0.02]),
        ]
        service, memories, edges = _make_service(
            candidates=sources,
            reflector=StubReflectorFactory(content=json.dumps({"gist": "  ", "rationale": "x"})),
        )
        result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)
        assert result.gists_created == 0
        memories.create.assert_not_called()

    async def test_malformed_json_reflector_skips_cluster(self) -> None:
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0]),
            _candidate(content="b", embedding=[0.99, 0.02]),
        ]
        service, memories, edges = _make_service(
            candidates=sources,
            reflector=StubReflectorFactory(content="not json at all {oops"),
        )
        result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)
        assert result.gists_created == 0
        memories.create.assert_not_called()

    async def test_db_write_error_skips_cluster_without_raising(self) -> None:
        # a DB failure mid-commit is logged + skipped so the pass still
        # returns (per the cluster-grain resilience contract).
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0]),
            _candidate(content="b", embedding=[0.99, 0.02]),
        ]
        service, memories, edges = _make_service(candidates=sources)
        memories.mark_superseded = AsyncMock(side_effect=RuntimeError("l3 down"))
        result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)
        # the gist committed but supersession failed -> the cluster is not
        # counted a success, and run_consolidation does not raise.
        assert result.gists_created == 0

    async def test_agent_scoped_run_passes_null_grain(self) -> None:
        # agent-scoped: both grains None; the gist inherits null scope, and
        # the load is asked for the null grain.
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0], customer_id=None, user_id=None),
            _candidate(content="b", embedding=[0.99, 0.02], customer_id=None, user_id=None),
        ]
        service, memories, edges = _make_service(candidates=sources)
        result = await service.run_consolidation(_AID)

        load_kwargs = memories.find_active_for_consolidation.call_args.kwargs
        assert load_kwargs["customer_id"] is None
        assert load_kwargs["user_id"] is None
        assert result.gists_created == 1
        gist_data = memories.create.call_args.args[0]
        assert gist_data["customer_id"] is None
        assert gist_data["user_id"] is None


class TestConsolidatedEvent:
    """the MemoryConsolidatedEvent payload -- incl. the null-scope guard."""

    async def test_user_scoped_event_carries_str_scope(self) -> None:
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0]),
            _candidate(content="b", embedding=[0.99, 0.02]),
        ]
        service, memories, edges = _make_service(candidates=sources)
        with patch(
            "threetears.agent.memory.dream.dispatch_event",
            new_callable=AsyncMock,
        ) as dispatch:
            result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)

        dispatch.assert_awaited_once()
        event = dispatch.await_args.args[0]
        assert isinstance(event, MemoryConsolidatedEvent)
        assert event.agent_id == str(_AID)
        assert event.user_id == str(_UID)
        assert event.customer_id == str(_CUID)
        assert event.source_count == 2
        assert event.consolidated_memory_id == str(result.gist_ids[0])
        assert sorted(event.source_memory_ids) == sorted(str(s["memory_id"]) for s in sources)
        assert event.content_preview == "merged"

    async def test_agent_scoped_event_carries_null_scope_not_str_none(self) -> None:
        # the A1 forward-NOTE guard: the first null-scoped writer must emit
        # user_id/customer_id as None, never the string "None".
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0], customer_id=None, user_id=None),
            _candidate(content="b", embedding=[0.99, 0.02], customer_id=None, user_id=None),
        ]
        service, memories, edges = _make_service(candidates=sources)
        with patch(
            "threetears.agent.memory.dream.dispatch_event",
            new_callable=AsyncMock,
        ) as dispatch:
            await service.run_consolidation(_AID)

        event = dispatch.await_args.args[0]
        assert event.user_id is None
        assert event.customer_id is None
        assert event.agent_id == str(_AID)

    async def test_missing_run_manager_swallowed(self) -> None:
        # dispatch_event raises RuntimeError with no langgraph run manager in
        # scope (background job / test): the merge is committed, the event is
        # best-effort, and run_consolidation must not raise.
        sources = [
            _candidate(content="a", embedding=[1.0, 0.0]),
            _candidate(content="b", embedding=[0.99, 0.02]),
        ]
        service, memories, edges = _make_service(candidates=sources)
        with patch(
            "threetears.agent.memory.dream.dispatch_event",
            new_callable=AsyncMock,
            side_effect=RuntimeError("no run manager"),
        ):
            result = await service.run_consolidation(_AID, customer_id=_CUID, user_id=_UID)
        assert result.gists_created == 1
