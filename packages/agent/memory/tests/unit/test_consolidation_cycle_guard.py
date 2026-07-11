"""Unit tests for the consolidation DAG cycle-guard (v026, A4).

Exercises the pure :func:`assert_no_consolidation_cycle` walker against
an in-memory ``sources_of`` map so the reachability logic is proven
without a database: self-merge, descendant-merge, the depth bound, and —
critically — that a valid diamond DAG (two sources sharing a common
ancestor) does NOT false-positive.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from threetears.agent.memory.collections import (
    ConsolidationCycleError,
    assert_no_consolidation_cycle,
)


def _sources_of_from_map(
    edges: dict[UUID, list[UUID]],
):
    """build an async ``sources_of`` lookup over a static edge map.

    :param edges: consolidated_memory_id -> its source ids
    :ptype edges: dict[UUID, list[UUID]]
    :return: async callable matching the guard's ``sources_of`` protocol
    :rtype: Callable
    """

    async def _lookup(memory_id: UUID) -> list[UUID]:
        return list(edges.get(memory_id, []))

    return _lookup


class TestConsolidationCycleGuard:
    async def test_fresh_gist_no_sources_passes(self) -> None:
        gist = uuid4()
        await assert_no_consolidation_cycle(
            consolidated_memory_id=gist,
            source_memory_ids=[],
            sources_of=_sources_of_from_map({}),
        )

    async def test_independent_sources_pass(self) -> None:
        gist = uuid4()
        s1, s2 = uuid4(), uuid4()
        # neither source has any provenance of its own.
        await assert_no_consolidation_cycle(
            consolidated_memory_id=gist,
            source_memory_ids=[s1, s2],
            sources_of=_sources_of_from_map({}),
        )

    async def test_self_merge_rejected(self) -> None:
        gist = uuid4()
        with pytest.raises(ConsolidationCycleError):
            await assert_no_consolidation_cycle(
                consolidated_memory_id=gist,
                source_memory_ids=[gist, uuid4()],
                sources_of=_sources_of_from_map({}),
            )

    async def test_direct_descendant_rejected(self) -> None:
        # gist was, in a prior round, a source of s1 (edge s1 -> gist).
        # consolidating gist FROM s1 now would close the loop.
        gist = uuid4()
        s1 = uuid4()
        edges = {s1: [gist]}
        with pytest.raises(ConsolidationCycleError):
            await assert_no_consolidation_cycle(
                consolidated_memory_id=gist,
                source_memory_ids=[s1],
                sources_of=_sources_of_from_map(edges),
            )

    async def test_transitive_descendant_rejected(self) -> None:
        # s1 <- a <- gist (gist reachable two hops down from s1).
        gist = uuid4()
        s1, a = uuid4(), uuid4()
        edges = {s1: [a], a: [gist]}
        with pytest.raises(ConsolidationCycleError):
            await assert_no_consolidation_cycle(
                consolidated_memory_id=gist,
                source_memory_ids=[s1],
                sources_of=_sources_of_from_map(edges),
            )

    async def test_diamond_dag_does_not_false_positive(self) -> None:
        # s1 and s2 both descend from a common ancestor `a` (a valid DAG,
        # NOT a cycle). The guard must accept it — a naive "any revisited
        # node is a cycle" walk would wrongly reject on the second visit
        # to `a`.
        gist = uuid4()
        s1, s2, a = uuid4(), uuid4(), uuid4()
        edges = {s1: [a], s2: [a]}
        await assert_no_consolidation_cycle(
            consolidated_memory_id=gist,
            source_memory_ids=[s1, s2],
            sources_of=_sources_of_from_map(edges),
        )

    async def test_preexisting_cycle_in_graph_terminates(self) -> None:
        # a corrupt stored graph with a cycle among non-gist nodes
        # (a <-> b) must still terminate (visited set) and, since the gist
        # is not reachable, pass.
        gist = uuid4()
        a, b = uuid4(), uuid4()
        edges = {a: [b], b: [a]}
        await assert_no_consolidation_cycle(
            consolidated_memory_id=gist,
            source_memory_ids=[a],
            sources_of=_sources_of_from_map(edges),
        )

    async def test_depth_bound_rejects_runaway_graph(self) -> None:
        # a long acyclic chain that exceeds max_nodes trips the defensive
        # bound rather than walking forever.
        gist = uuid4()
        chain = [uuid4() for _ in range(6)]
        edges = {chain[i]: [chain[i + 1]] for i in range(len(chain) - 1)}
        with pytest.raises(ConsolidationCycleError):
            await assert_no_consolidation_cycle(
                consolidated_memory_id=gist,
                source_memory_ids=[chain[0]],
                sources_of=_sources_of_from_map(edges),
                max_nodes=3,
            )
