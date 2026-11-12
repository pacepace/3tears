"""Salience decay on read in intentions (Pace's ruling 2026-11-11): off by default, the ranked reads order by the
stored salience and a reinforcement adds to it as before; on, they order by the decay's rank expression and a
reinforcement folds the decay so far into the base and stamps the row."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.core.collections import SalienceDecay
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

from threetears.agent.intention.collections import IntentionsCollection

_DECAY = SalienceDecay(half_life=timedelta(days=60), floor=0.1)


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


async def _statements(decay: SalienceDecay | None, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    monkeypatch.setattr(IntentionsCollection, "salience_decay", decay)
    pool = _Recording()
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool, kv_key_scope="salience-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    intentions = IntentionsCollection(registry, config, nats_client=FakeNatsClient())
    agent, user = uuid.uuid4(), uuid.uuid4()
    await intentions.find_by_user(user, agent_id=agent)
    await intentions.find_open_for_deliberation(user, agent_id=agent, cooldown_cutoff=datetime.now(UTC))
    await intentions.bump_salience([uuid.uuid4()], agent_id=agent, access_bump=0.1)
    return pool.statements


@pytest.mark.asyncio
async def test_off_is_the_default_and_every_statement_reads_the_stored_salience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if decay on read is on by default, or off the ranked reads order by anything but the stored salience or
    a reinforcement adds to anything but it."""
    assert IntentionsCollection.salience_decay is None
    listed, deliberated, bumped = await _statements(None, monkeypatch)
    assert "ORDER BY salience DESC, date_created DESC" in listed
    assert "ORDER BY salience DESC, date_created DESC" in deliberated
    assert "SET salience = LEAST(1.0, salience + $1) WHERE" in bumped


@pytest.mark.asyncio
async def test_on_the_reads_rank_by_the_decay_and_a_reinforcement_folds_and_stamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if with decay on a ranked read orders by the stored base, or a reinforcement adds to the stored base or
    leaves the stamp where it was."""
    listed, deliberated, bumped = await _statements(_DECAY, monkeypatch)
    assert f"ORDER BY {_DECAY.rank_sql()} DESC, date_created DESC" in listed
    assert f"ORDER BY {_DECAY.rank_sql()} DESC, date_created DESC" in deliberated
    assert f"SET salience = LEAST(1.0, {_DECAY.effective_sql()} + $1), last_decayed_at = now()" in bumped
