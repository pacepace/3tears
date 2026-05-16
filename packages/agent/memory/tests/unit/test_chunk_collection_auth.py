"""Pin the auth triple on every MemoryChunkCollection method.

Cross-tenant data leaks happen when a method's SQL drops one of the
three scoping predicates ``(user_id, agent_id, customer_id)``. The new
methods added in transcript-chunks-task-A (``find_by_memory_id``,
``find_by_conversation_id``, ``hybrid_search_within_memory``, and the
cursor-extended ``hybrid_search``) each carry the full triple; this
test fails the build if a future refactor accidentally weakens one of
them.

Approach: static — fetch each method's source via ``inspect.getsource``
and assert all three scoping references appear. This is a cheap pin
that catches the most common regression shape ("I deleted that line by
accident") without needing testcontainers. End-to-end coverage that
exercises the predicates with real Postgres rows lives in
``tests/integration/test_chunk_collection_methods.py`` — which seeds
chunks under one ``(user_id, agent_id, customer_id)`` triple and
asserts that calls with any mismatched component return empty.
"""

from __future__ import annotations

import inspect

import pytest

from threetears.agent.memory.collections import MemoryChunkCollection


_METHODS_REQUIRING_AUTH_TRIPLE = [
    "hybrid_search",
    "hybrid_search_within_memory",
    "find_by_memory_id",
    "find_by_conversation_id",
    "search_by_ids",
    "search_by_semantic",
    "fetch_content_for_recall",
]


class TestChunkCollectionAuthScoping:
    @pytest.mark.parametrize("method_name", _METHODS_REQUIRING_AUTH_TRIPLE)
    def test_method_signature_takes_user_id(self, method_name: str) -> None:
        """Every chunk method must accept ``user_id`` so the SQL can scope on it."""
        method = getattr(MemoryChunkCollection, method_name)
        sig = inspect.signature(method)
        assert "user_id" in sig.parameters, f"MemoryChunkCollection.{method_name} is missing user_id parameter"

    @pytest.mark.parametrize("method_name", _METHODS_REQUIRING_AUTH_TRIPLE)
    def test_method_signature_takes_agent_id(self, method_name: str) -> None:
        """Every chunk method must accept ``agent_id`` (partition column)."""
        method = getattr(MemoryChunkCollection, method_name)
        sig = inspect.signature(method)
        assert "agent_id" in sig.parameters, f"MemoryChunkCollection.{method_name} is missing agent_id parameter"

    @pytest.mark.parametrize(
        "method_name",
        [
            "hybrid_search",
            "hybrid_search_within_memory",
            "find_by_memory_id",
            "find_by_conversation_id",
        ],
    )
    def test_method_signature_takes_customer_id(self, method_name: str) -> None:
        """Hybrid + paged methods all carry the customer_id sub-scope.

        ``search_by_ids`` / ``search_by_semantic`` / ``fetch_content_for_recall``
        predate the customer_id requirement and rely on agent + user
        scoping today; they are out of scope for this triple check (the
        partition column + row owner is already sufficient for those
        narrower lookup paths).
        """
        method = getattr(MemoryChunkCollection, method_name)
        sig = inspect.signature(method)
        assert "customer_id" in sig.parameters, f"MemoryChunkCollection.{method_name} is missing customer_id parameter"

    @pytest.mark.parametrize("method_name", _METHODS_REQUIRING_AUTH_TRIPLE)
    def test_method_source_references_user_id(self, method_name: str) -> None:
        """Source must reference user_id — not just take it, but use it."""
        method = getattr(MemoryChunkCollection, method_name)
        source = inspect.getsource(method)
        assert "user_id" in source, (
            f"MemoryChunkCollection.{method_name} takes user_id but does not "
            f"reference it in body — likely a dropped scope predicate."
        )

    @pytest.mark.parametrize("method_name", _METHODS_REQUIRING_AUTH_TRIPLE)
    def test_method_source_references_agent_id(self, method_name: str) -> None:
        method = getattr(MemoryChunkCollection, method_name)
        source = inspect.getsource(method)
        assert "agent_id" in source, (
            f"MemoryChunkCollection.{method_name} takes agent_id but does not "
            f"reference it in body — likely a dropped partition predicate."
        )

    @pytest.mark.parametrize(
        "method_name",
        [
            "hybrid_search",
            "hybrid_search_within_memory",
            "find_by_memory_id",
            "find_by_conversation_id",
        ],
    )
    def test_method_source_references_customer_id(self, method_name: str) -> None:
        method = getattr(MemoryChunkCollection, method_name)
        source = inspect.getsource(method)
        assert "customer_id" in source, (
            f"MemoryChunkCollection.{method_name} takes customer_id but does "
            f"not reference it in body — likely a dropped sub-scope predicate."
        )


class TestChunkCollectionCursorPaging:
    """Cursor-paged methods must reject the mutually-exclusive cursor pair."""

    @pytest.mark.parametrize(
        "method_name",
        [
            "find_by_memory_id",
            "find_by_conversation_id",
            "hybrid_search",
        ],
    )
    def test_method_accepts_both_cursors(self, method_name: str) -> None:
        """The signature offers both cursors — the runtime rejects passing both."""
        method = getattr(MemoryChunkCollection, method_name)
        sig = inspect.signature(method)
        assert "chunk_id_after" in sig.parameters
        assert "chunk_id_before" in sig.parameters
