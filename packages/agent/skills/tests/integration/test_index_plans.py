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
from threetears.agent.skills.collections import AgentSkillCollection, AgentSkillInvocationCollection
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


#: the uses of the paged skill (many pages of them), beside other skills' of the same agent
_OF_SKILL = 2000


async def test_skill_invocations_skill_time_key(applied_schema: tuple[str, str]) -> None:
    """Flips if a skill's history page stops reading idx_skill_invocations_skill_time_key in its order (``invoked_at
    DESC`` completed by ``invocation_id DESC``), so it sorts the skill's every use, or reads more than about its page;
    and if the history's bounded count stops walking the same index to its bound."""
    url, schema = applied_schema
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        agent_id, skill_id, other = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        await conn.execute(
            "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body)"
            " SELECT $1, s, gen_random_uuid(), 'n' || s, 's', 'b' FROM unnest($2::uuid[]) s",
            agent_id,
            [skill_id, other],
        )
        # the paged skill's uses, a tenth of them at one instant (ties the key orders), beside another skill's
        for skill, count in ((skill_id, _OF_SKILL), (other, _OF_SKILL)):
            await conn.execute(
                "INSERT INTO agent_skill_invocations (agent_id, invocation_id, skill_id, user_id, conversation_id,"
                " invocation_source, invoked_at)"
                " SELECT $1, gen_random_uuid(), $2, gen_random_uuid(), gen_random_uuid(), 'invoke',"
                " CASE WHEN g % 10 = 0 THEN timestamptz '2026-11-14' ELSE timestamptz '2026-11-01' + g * interval '1 s'"
                " END FROM generate_series(1, $3) g",
                agent_id,
                skill,
                count,
            )
        await conn.execute("ANALYZE agent_skill_invocations")
        captured = CapturedL3(real=conn)
        registry = CollectionRegistry()
        registry.configure(l3_pool=captured)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        invocations = AgentSkillInvocationCollection(registry=registry, config=cfg)
        first = await invocations.list_for_skill(agent_id, skill_id, limit=_PAGE)
        await invocations.list_for_skill(agent_id, skill_id, limit=_PAGE, cursor=first.next_cursor)
        counted = await invocations.count_for_skill_at_most(agent_id, skill_id, at_most=100)
        assert (counted.count, counted.more) == (100, True)
        # the bounded count reads ``... LIMIT $n) AS held``; YugabyteDB's page past a cursor is legs chained by count
        page_reads = [asked for asked in captured.asked if ") AS held" not in asked[0]]
        assert len(page_reads) == 2, [sql for sql, _ in captured.asked]
        for number, (sql, args) in enumerate(page_reads):
            plan = await explain_statement(conn, sql, *args)
            assert plan_uses_index(plan, "idx_skill_invocations_skill_time_key"), plan.plan
            if number == 0:
                # the first page walks the index in its order; a later page on YugabyteDB sorts its legs' few rows
                assert not set(_node_types(plan.plan)) & {"Sort", "Incremental Sort"}, plan.plan
            # the page with keyset_page's one row past it
            assert plan.rows_returned == _PAGE + 1, plan
            assert plan.rows_scanned <= 2 * (_PAGE + 1), plan
        ((sql, args),) = [asked for asked in captured.asked if ") AS held" in asked[0]]
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_skill_invocations_skill_time_key"), plan.plan
        assert plan.rows_scanned <= 2 * 101, plan
    finally:
        await conn.close()


#: the listed user's skills, most of them disabled, beside other users' of the same agent
_SKILLS_OF_USER = 2000


async def test_skills_agent_user_enabled(applied_schema: tuple[str, str]) -> None:
    """Flips if a user's enabled skills, listed (recency first) and counted, stop reading idx_skills_agent_user_enabled
    by its whole key (agent, user, enabled), so they read the user's disabled skills too (here nine in ten)."""
    url, schema = applied_schema
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        agent_id, user_id = uuid.uuid4(), uuid.uuid4()
        await conn.execute(
            "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body, enabled, last_used_at)"
            " SELECT $1, gen_random_uuid(), $2, 'n' || g, 's', 'b', g % 10 = 0,"
            " CASE WHEN g % 3 = 0 THEN timestamptz '2026-11-01' + g * interval '1 s' END"
            " FROM generate_series(1, $3) g",
            agent_id,
            user_id,
            _SKILLS_OF_USER,
        )
        await conn.execute(
            "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body)"
            " SELECT $1, gen_random_uuid(), gen_random_uuid(), 'o' || g, 's', 'b' FROM generate_series(1, 30000) g",
            agent_id,
        )
        await conn.execute("ANALYZE agent_skills")
        captured = CapturedL3(real=conn)
        registry = CollectionRegistry()
        registry.configure(l3_pool=captured)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        skills = AgentSkillCollection(registry=registry, config=cfg)
        listed = await skills.list_for_user(agent_id, user_id, limit=_PAGE)
        assert len(listed) == _PAGE and all(skill.enabled for skill in listed)
        assert await skills.count_for_user(agent_id, user_id) == _SKILLS_OF_USER // 10
        assert len(captured.asked) == 2, captured.asked
        for sql, args in captured.asked:
            plan = await explain_statement(conn, sql, *args)
            # on YugabyteDB too: no declared index holds the listing's recency order, so the keyed read names the one
            # binding its three pins (keys.serving_index); unnamed, YugabyteDB read the user's prefix of
            # uq_skills_agent_user_name, all 2,000 of the user's skills (D5, 2026-11-15)
            assert plan_uses_index(plan, "idx_skills_agent_user_enabled"), plan.plan
            # the user's enabled skills (200), never the disabled 1,800 beside them
            assert plan.rows_scanned <= 2 * (_SKILLS_OF_USER // 10), plan
    finally:
        await conn.close()
