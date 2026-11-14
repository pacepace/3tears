"""integration: each agent-skills index with a purpose serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. The tables are built by the package's own
migrations and seeded past the size where the planner would rather scan; the statement the collection really sends
(run over a pool that records statements) is asked whether its plan found its rows through the index by an index
condition (:func:`threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from typing import Any

import asyncpg
import pytest
from threetears.agent.skills.collections import AgentSkillInvocationCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import explain_statement, on_yugabyte, plan_uses_index

pytestmark = pytest.mark.integration

#: the page the listing asks for, and the invocations of the conversation it reads (many pages of them)
_PAGE = 20
_IN_CONVERSATION = 2000


def _node_types(node: Mapping[str, Any]) -> Iterator[str]:
    yield str(node.get("Node Type", ""))
    for child in node.get("Plans", ()):
        yield from _node_types(child)


async def test_skill_invocations_conv_key(applied_schema: tuple[str, str]) -> None:
    """Flips if a conversation's page of invocations stops reading idx_skill_invocations_conv_key in its order
    (``invoked_at DESC`` completed by ``invocation_id``), so it sorts the conversation's invocations, or if on
    YugabyteDB it reads more than about its page."""
    url, schema = applied_schema
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        agent_id, skill_id, conversation_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        await conn.execute(
            "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body)"
            " VALUES ($1, $2, gen_random_uuid(), 'n', 's', 'b')",
            agent_id,
            skill_id,
        )
        # the listed conversation's invocations, a tenth of them sharing one instant (ties the key orders), beside
        # 200 other conversations' of the same agent
        await conn.execute(
            "INSERT INTO agent_skill_invocations (agent_id, invocation_id, skill_id, user_id, conversation_id,"
            " invocation_source, invoked_at)"
            " SELECT $1, gen_random_uuid(), $2, gen_random_uuid(), $3, 'invoke',"
            " CASE WHEN g % 10 = 0 THEN timestamptz '2026-11-14' ELSE timestamptz '2026-11-01' + g * interval '1 s'"
            " END FROM generate_series(1, $4) g",
            agent_id,
            skill_id,
            conversation_id,
            _IN_CONVERSATION,
        )
        await conn.execute(
            "INSERT INTO agent_skill_invocations (agent_id, invocation_id, skill_id, user_id, conversation_id,"
            " invocation_source, invoked_at)"
            " SELECT $1, gen_random_uuid(), $2, gen_random_uuid(), c.id, 'invoke', now() - g * interval '1 s'"
            " FROM (SELECT gen_random_uuid() AS id FROM generate_series(1, 200)) c, generate_series(1, 20) g",
            agent_id,
            skill_id,
        )
        await conn.execute("ANALYZE agent_skill_invocations")
        captured = CapturedL3()
        registry = CollectionRegistry()
        registry.configure(l3_pool=captured)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        invocations = AgentSkillInvocationCollection(registry=registry, config=cfg)
        await invocations.list_for_conversation(agent_id, conversation_id, limit=_PAGE)
        (sql, args), *_ = captured.asked
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_skill_invocations_conv_key"), plan.plan
        assert not set(_node_types(plan.plan)) & {"Sort", "Incremental Sort"}, plan.plan
        # the page is read with keyset_page's one row past it (whether a next page exists)
        assert plan.rows_returned == _PAGE + 1, plan
        if await on_yugabyte(conn):
            assert plan.rows_scanned <= 2 * (_PAGE + 1), plan
    finally:
        await conn.close()
