"""Salience decay on read in memories (Pace's ruling 2026-11-11): off by default, a reinforcement adds to the stored
salience as before; on, it folds the decay so far into the base and stamps the row."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.core.collections import SalienceDecay
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

_DECAY = SalienceDecay(half_life=timedelta(days=60), floor=0.1, pinned="evergreen")


class _Recording:
    """an L3 that records each statement and answers none."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        self.statements.append(sql)
        return []

    async def execute(self, sql: str, *params: Any) -> str:
        self.statements.append(sql)
        return "UPDATE 1"


async def _bumped(decay: SalienceDecay | None, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(MemoriesCollection, "salience_decay", decay)
    pool = _Recording()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope="salience-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    memories = MemoriesCollection(
        registry=registry, config=config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies)
    )
    await memories.bump_salience([uuid.uuid4()], agent_id=uuid.uuid4(), access_bump=0.1)
    (statement,) = pool.statements
    return statement


async def test_off_is_the_default_and_a_reinforcement_adds_to_the_stored_salience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if decay on read is on by default, or off a reinforcement adds to anything but the stored salience."""
    assert MemoriesCollection.salience_decay is None
    assert "SET salience = LEAST(1.0, salience + $1), last_accessed = now()" in await _bumped(None, monkeypatch)


async def test_on_a_reinforcement_folds_the_decay_and_stamps_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if with decay on a reinforcement adds to the stored base (the decay since its stamp lost) or leaves the
    stamp where it was (the next read would decay the folded base again)."""
    folded = f"SET salience = LEAST(1.0, {_DECAY.effective_sql()} + $1), last_decayed_at = now()"
    assert folded in await _bumped(_DECAY, monkeypatch)
