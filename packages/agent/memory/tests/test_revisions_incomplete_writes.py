"""A revision whose new memory committed before a later step failed says so, with the memory's id.

The new memory is saved first, then the edges from what it replaces, then the mark that makes those
stop reading as current. A failure after the save is a :class:`WriteIncompleteError` carrying the id,
so a caller can say the memory is stored instead of having it written again. A failure before the
save, or one that says the commit did not land, is the failure itself.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.memory.revisions import WriteIncompleteError, correct_beside, supersede, write_once
from threetears.core.exceptions import (
    DataLayerUnavailableError,
    GenerationNotCommittedError,
    GenerationUnavailableError,
)
from threetears.nats.errors import KvError

pytestmark = pytest.mark.asyncio

_AGENT = uuid4()
_SOURCE = uuid4()
_FIELDS: dict[str, Any] = {"user_id": uuid4(), "content": "lives in York", "embedding": [0.1] * 8}


# parity-exempt: records the save and the mark the revision writers make on the memories collection; its SQL is covered by the collection tests
class _Memories:
    def __init__(self, fail: dict[str, BaseException] | None = None) -> None:
        self.fail = fail or {}
        self.saved: list[UUID] = []

    def create(self, data: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(**data)

    async def get(self, key: tuple[UUID, UUID]) -> SimpleNamespace:
        _ = key
        return SimpleNamespace(evergreen=False)

    async def save_entity(self, entity: SimpleNamespace) -> None:
        self.saved.append(entity.memory_id)
        if "save_entity" in self.fail:
            raise self.fail["save_entity"]

    async def mark_superseded(self, agent_id: UUID, *, source_memory_ids: list[UUID], gist_id: UUID) -> None:
        _ = agent_id, source_memory_ids, gist_id
        if "mark_superseded" in self.fail:
            raise self.fail["mark_superseded"]


# parity-exempt: records the cycle check and the edge saves the revision writers make on the edges collection
class _Edges:
    def __init__(self, fail: dict[str, BaseException] | None = None) -> None:
        self.fail = fail or {}

    async def assert_no_cycle(self, agent_id: UUID, **kwargs: Any) -> None:
        _ = agent_id, kwargs
        if "assert_no_cycle" in self.fail:
            raise self.fail["assert_no_cycle"]

    def create(self, data: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(**data)

    async def save_entity(self, entity: SimpleNamespace) -> None:
        _ = entity
        if "save_entity" in self.fail:
            raise self.fail["save_entity"]


async def _supersede(memories: _Memories, edges: _Edges) -> UUID:
    return await supersede(
        memories,  # type: ignore[arg-type]
        edges,  # type: ignore[arg-type]
        agent_id=_AGENT,
        source_ids=[_SOURCE],
        fields=dict(_FIELDS),
        rationale="replaced",
    )


@pytest.mark.parametrize(
    ("memories", "edges", "unfinished"),
    [
        (_Memories(fail={"mark_superseded": DataLayerUnavailableError("broker down")}), _Edges(), "mark"),
        (_Memories(), _Edges(fail={"save_entity": KvError("kv down")}), "edges"),
        (_Memories(fail={"save_entity": GenerationUnavailableError("generation store down")}), _Edges(), "generation"),
    ],
    ids=["the mark", "an edge", "the generation"],
)
async def test_a_step_after_the_commit_failing_names_the_memory_written(
    memories: _Memories, edges: _Edges, unfinished: str
) -> None:
    with pytest.raises(WriteIncompleteError) as raised:
        await _supersede(memories, edges)
    assert raised.value.memory_id == memories.saved[0]
    assert raised.value.unfinished == unfinished
    assert raised.value.source_ids == (_SOURCE,)
    assert isinstance(raised.value, DataLayerUnavailableError)  # a storage handler still catches it


async def test_a_save_that_did_not_commit_is_the_failure_itself() -> None:
    not_committed = GenerationNotCommittedError("rolled back")
    with pytest.raises(GenerationNotCommittedError):
        await _supersede(_Memories(fail={"save_entity": not_committed}), _Edges())


async def test_a_save_that_failed_outright_is_the_failure_itself() -> None:
    with pytest.raises(DataLayerUnavailableError) as raised:
        await _supersede(_Memories(fail={"save_entity": DataLayerUnavailableError("broker down")}), _Edges())
    assert not isinstance(raised.value, WriteIncompleteError)


@pytest.mark.parametrize("bug", [TypeError("mark_superseded() takes 2 arguments"), AttributeError("no attribute")])
async def test_a_bug_after_the_commit_is_not_dressed_as_an_incomplete_write(bug: Exception) -> None:
    with pytest.raises(type(bug)):
        await _supersede(_Memories(fail={"mark_superseded": bug}), _Edges())


async def test_a_correction_beside_whose_edge_failed_names_the_memory_written() -> None:
    memories = _Memories()
    with pytest.raises(WriteIncompleteError) as raised:
        await correct_beside(
            memories,  # type: ignore[arg-type]
            _Edges(fail={"save_entity": KvError("kv down")}),  # type: ignore[arg-type]
            agent_id=_AGENT,
            source_ids=[_SOURCE],
            fields=dict(_FIELDS),
            rationale="beside",
        )
    assert raised.value.memory_id == memories.saved[0]
    assert raised.value.unfinished == "edges"


async def test_a_plain_write_whose_generation_did_not_advance_names_the_memory_written() -> None:
    memories = _Memories(fail={"save_entity": GenerationUnavailableError("generation store down")})
    with pytest.raises(WriteIncompleteError) as raised:
        await write_once(memories, None, agent_id=_AGENT, fields=dict(_FIELDS))  # type: ignore[arg-type]
    assert raised.value.memory_id == memories.saved[0]
    assert raised.value.unfinished == "generation"
    assert raised.value.source_ids == ()
