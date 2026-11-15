"""the memory writes moved onto ``update_keyed`` write what the statements they replaced wrote.

``mark_retracted``, ``clear_retracted`` and ``bump_salience`` once wrote their own ``UPDATE``; they now hand
``update_keyed`` their values as expressions (the tags' jsonb edits, the salience's CASE, a reinforcement's values from
``SalienceDecay.reinforced_values`` with decay on read, or the stored salience plus the bump with it off) and their
guards. Each is compared, through :func:`~threetears.core.testing.same_results.assert_same_results` with
``state_sql``, with the statement it replaced: the rows each returns and the table after, salience compared exactly
(``NUMERIC(5,4)``), every same_results case. The stamps a write sets to ``now()`` differ by the instant each side ran,
so the table after says whether each was stamped.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.revisions import RETRACTED_BECAUSE_PREFIX, RETRACTED_SALIENCE_PREFIX
from threetears.core.collections.salience import SalienceDecay
from threetears.core.testing.same_results import Seeded, assert_same_results

from .memory_same_results import AGENTS, PAGE, T0, Serial, collection, conn, ident, memory_row, text_vectors

__all__ = ["conn"]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: decay on read as a host opts in (the memory unit tests' configuration)
_DECAY = SalienceDecay(half_life=timedelta(days=60), floor=0.1, pinned="evergreen")

#: the table after a write: every value it may change, salience exactly, each now() stamp as whether it was set (the
#: seeds' stamps are weeks old; a write's are now)
_STATE = (
    "SELECT memory_id, salience, tags::text AS tags, evergreen, (last_accessed IS NOT NULL) AS accessed, "
    "(last_decayed_at > now() - interval '1 day') AS decay_stamped, "
    "(date_updated > now() - interval '1 day') AS updated FROM memories"
)


def _memory(seed: Seeded) -> Mapping[str, Any]:
    """a memory: permanent every fourth, retracted (with why and its salience before) every third, the rest current;
    salience and the decay stamp varied so a reinforcement's fold shows."""
    index = seed.index
    retracted = index % 3 == 1
    tags = (
        f'["kept", "retracted", "{RETRACTED_BECAUSE_PREFIX}wrong", "{RETRACTED_SALIENCE_PREFIX}0.{index % 9 + 1}"]'
        if retracted
        else '["kept"]'
    )
    return {
        **memory_row(seed),
        "salience": round(0.1 + 0.07 * (index % 12), 4),
        "evergreen": index % 4 == 2,
        "tags": None if seed.null and not retracted else tags,
        "last_decayed_at": None if seed.null else T0 - timedelta(days=index),
        "last_accessed": None,
    }


async def _same(conn: asyncpg.Connection, old_sql: str, params: list[Any], helper: Any) -> None:
    await text_vectors(conn)
    await assert_same_results(
        conn,
        table="memories",
        row=_memory,
        old_sql=old_sql,
        params=params,
        helper=helper,
        state_sql=_STATE,
        page_size=PAGE,
    )


_MARK_SQL = f"""
    UPDATE memories SET
        tags = COALESCE(tags, '[]'::jsonb) || jsonb_build_array(
            'retracted', $3::text, '{RETRACTED_SALIENCE_PREFIX}' || salience::text),
        salience = CASE WHEN evergreen THEN salience ELSE 0.0 END,
        date_updated = now()
    WHERE agent_id = $1 AND memory_id = $2
      AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')
      AND ($4::boolean OR NOT evergreen)
    RETURNING memory_id
"""


@pytest.mark.parametrize("index", [0, 1, 2, 30])
@pytest.mark.parametrize("allow_permanent", [False, True])
async def test_mark_retracted_writes_what_its_statement_wrote(
    conn: asyncpg.Connection, index: int, allow_permanent: bool
) -> None:
    """Flips if mark_retracted writes another row, other tags or salience, retracts one already retracted, a
    permanent one without leave, or says it changed a row it did not."""
    memory = ident("memory", index)

    async def helper(c: Any) -> list[dict[str, Any]]:
        changed = await collection(MemoriesCollection, Serial(c)).mark_retracted(
            AGENTS[0], memory_id=memory, reason="wrong", allow_permanent=allow_permanent
        )
        return [{"memory_id": memory}] if changed else []

    await _same(conn, _MARK_SQL, [AGENTS[0], memory, f"{RETRACTED_BECAUSE_PREFIX}wrong", allow_permanent], helper)


_CLEAR_SQL = f"""
    UPDATE memories SET
        salience = CASE WHEN evergreen THEN salience ELSE GREATEST(0.0, LEAST(1.0, COALESCE(
            (SELECT substring(t FROM {len(RETRACTED_SALIENCE_PREFIX) + 1})::numeric
             FROM jsonb_array_elements_text(tags) AS t
             WHERE starts_with(t, '{RETRACTED_SALIENCE_PREFIX}') LIMIT 1),
            $3))) END,
        tags = (SELECT COALESCE(jsonb_agg(t), '[]'::jsonb)
                FROM jsonb_array_elements_text(tags) AS t
                WHERE t <> 'retracted'
                  AND NOT starts_with(t, '{RETRACTED_BECAUSE_PREFIX}')
                  AND NOT starts_with(t, '{RETRACTED_SALIENCE_PREFIX}')),
        date_updated = now()
    WHERE agent_id = $1 AND memory_id = $2
      AND NOT (NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted'))
    RETURNING memory_id
"""


@pytest.mark.parametrize("index", [0, 1, 4, 10, 30])
async def test_clear_retracted_writes_what_its_statement_wrote(conn: asyncpg.Connection, index: int) -> None:
    """Flips if clear_retracted writes another row, restores another salience, keeps a retraction tag, drops a tag
    of the memory's own, clears one not retracted, or says it changed a row it did not."""
    memory = ident("memory", index)

    async def helper(c: Any) -> list[dict[str, Any]]:
        changed = await collection(MemoriesCollection, Serial(c)).clear_retracted(
            AGENTS[0], memory_id=memory, default_salience=0.5
        )
        return [{"memory_id": memory}] if changed else []

    await _same(conn, _CLEAR_SQL, [AGENTS[0], memory, 0.5], helper)


def _bump_sql(decay: SalienceDecay | None) -> str:
    folded = "salience" if decay is None else decay.effective_sql()
    stamp = "" if decay is None else ", last_decayed_at = now()"
    return (
        f"UPDATE memories SET salience = LEAST(1.0, {folded} + $1){stamp}, last_accessed = now() "
        "WHERE agent_id = $2 AND memory_id = ANY($3::uuid[]) AND NOT evergreen"
    )


@pytest.mark.parametrize("decay", [None, _DECAY], ids=["decay off", "decay on"])
async def test_bump_salience_writes_what_its_statement_wrote(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch, decay: SalienceDecay | None
) -> None:
    """Flips if bump_salience, with decay on read off or on, sets another salience (to the stored one plus the bump,
    or the decayed one plus the bump, clamped to 1), bumps a permanent memory or one not named, or leaves a stamp
    unset (last_accessed always; last_decayed_at with decay on)."""
    monkeypatch.setattr(MemoriesCollection, "salience_decay", decay)
    ids = [ident("memory", index) for index in (0, 1, 2, 3, 6, 11, 30)]

    async def helper(c: Any) -> list[dict[str, Any]]:
        await collection(MemoriesCollection, Serial(c)).bump_salience(ids, agent_id=AGENTS[0], access_bump=0.35)
        return []

    await _same(conn, _bump_sql(decay), [0.35, AGENTS[0], ids], helper)
