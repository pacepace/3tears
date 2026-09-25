"""Integration test: FTS trigger populates ``search_vector`` and ranks correctly.

Verifies that the trigger-maintained tsvector picks up content from
``name`` (weight A), ``trigger_keywords`` (B), and ``body`` (C), and
that a ``websearch_to_tsquery`` filter matches inserted rows.

The trigger runs ``BEFORE INSERT OR UPDATE OF name, trigger_keywords,
body`` so this test exercises both the insert path and the
column-list update path.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.collections import AgentSkillCollection
from threetears.agent.skills.migrations import register as register_skills
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


async def _apply(conn: asyncpg.Connection, schema: str) -> None:
    """Apply conversations + skills migrations to the schema."""
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    store = AsyncpgStore(conn)
    await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]


async def _insert_skill(
    conn: asyncpg.Connection,
    *,
    agent_id: UUID,
    user_id: UUID,
    name: str,
    summary: str,
    body: str | None,
    keywords: str = "",
) -> UUID:
    """Insert one skill row and return its ``skill_id``."""
    skill_id = _new_uuid()
    await conn.execute(
        "INSERT INTO agent_skills "
        "(agent_id, skill_id, user_id, name, summary, body, trigger_keywords) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7)",
        agent_id,
        skill_id,
        user_id,
        name,
        summary,
        body,
        keywords,
    )
    return skill_id


class TestSearchVectorPopulation:
    """The FTS trigger sets ``search_vector`` on insert and update."""

    async def test_insert_populates_search_vector(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A skill insert populates ``search_vector`` automatically."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            skill_id = await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="ruff-format",
                summary="Run ruff format on the workspace",
                body="Use the ./scripts/lint.sh --fix invocation",
                keywords="format style python",
            )
            row = await conn.fetchrow(
                "SELECT search_vector FROM agent_skills WHERE agent_id = $1 AND skill_id = $2",
                agent_id,
                skill_id,
            )
            assert row is not None
            assert row["search_vector"] is not None
        finally:
            await conn.close()

    async def test_update_refreshes_search_vector(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Updating ``body`` causes the trigger to re-emit the tsvector."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            skill_id = await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="deploy",
                summary="Deploy",
                body="initial",
                keywords="",
            )
            before = await conn.fetchval(
                "SELECT search_vector::text FROM agent_skills WHERE agent_id = $1 AND skill_id = $2",
                agent_id,
                skill_id,
            )
            await conn.execute(
                "UPDATE agent_skills SET body = $3 WHERE agent_id = $1 AND skill_id = $2",
                agent_id,
                skill_id,
                "kubernetes rollout helm chart",
            )
            after = await conn.fetchval(
                "SELECT search_vector::text FROM agent_skills WHERE agent_id = $1 AND skill_id = $2",
                agent_id,
                skill_id,
            )
            assert after is not None
            assert after != before
            assert "kubernet" in after
        finally:
            await conn.close()


class TestRanking:
    """``websearch_to_tsquery`` filter returns the expected rows."""

    async def test_query_matches_body_token(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A skill with body mentioning 'ruff format' matches a 'ruff' query."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            target_id = await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="format-helper",
                summary="Format files",
                body="Use ruff format to reformat changed Python files",
                keywords="lint code",
            )
            await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="deploy-helper",
                summary="Deploy services",
                body="Run terraform apply against the prod workspace",
                keywords="deploy ship",
            )
            row = await conn.fetchrow(
                "SELECT skill_id FROM agent_skills "
                "WHERE agent_id = $1 "
                "  AND search_vector @@ websearch_to_tsquery('english', 'ruff') "
                "ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', 'ruff')) DESC "
                "LIMIT 1",
                agent_id,
            )
            assert row is not None
            assert UUID(str(row["skill_id"])) == target_id
        finally:
            await conn.close()

    async def test_name_outranks_body(self, pg_schema: tuple[str, str]) -> None:
        """A skill whose name matches ranks above a skill whose body matches.

        ``name`` is weighted A; ``body`` is weighted C. ``ts_rank_cd``
        respects the weights so the name-match row sorts first.
        """
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            name_match_id = await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="terraform-helper",
                summary="Terraform workflow helper",
                body="Generic prose with no special token",
                keywords="",
            )
            await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="deploy-helper",
                summary="Generic deploy helper",
                body="Mentions terraform in the body once",
                keywords="",
            )
            rows = await conn.fetch(
                "SELECT skill_id FROM agent_skills "
                "WHERE agent_id = $1 "
                "  AND search_vector @@ websearch_to_tsquery('english', 'terraform') "
                "ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', 'terraform')) DESC",
                agent_id,
            )
            assert len(rows) == 2
            assert UUID(str(rows[0]["skill_id"])) == name_match_id
        finally:
            await conn.close()


class TestTypedSearchSyntaxThroughTheCollection:
    """``list_for_user`` keeps the user's OR / NOT and the tag overlap, filtered rather than indexed.

    Those predicates go through ``gin_filter`` because YugabyteDB's GIN index refuses them
    outright; these pin that the wrapped predicates still answer what the operators mean.
    """

    async def _collection(self, url: str, schema: str) -> tuple[AgentSkillCollection, asyncpg.Pool]:
        pool = await asyncpg.create_pool(url, min_size=1, max_size=2, server_settings={"search_path": schema})
        registry = CollectionRegistry()
        registry.configure(l3_pool=pool)
        cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        return AgentSkillCollection(registry=registry, config=cfg), pool

    async def test_or_and_not_in_the_query_mean_what_they_say(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            ruff = await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="format-helper",
                summary="Format",
                body="Use ruff format on changed files",
            )
            terraform = await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="deploy-helper",
                summary="Deploy",
                body="Run terraform apply on staging",
            )
            prod = await _insert_skill(
                conn,
                agent_id=agent_id,
                user_id=user_id,
                name="prod-helper",
                summary="Prod",
                body="Run terraform apply on prod",
            )
        finally:
            await conn.close()
        coll, pool = await self._collection(url, schema)
        try:
            either = {s.skill_id for s in await coll.list_for_user(agent_id, user_id, query="ruff or terraform")}
            not_prod = {s.skill_id for s in await coll.list_for_user(agent_id, user_id, query="terraform -prod")}
            counted = await coll.count_for_user(agent_id, user_id, query="ruff or terraform")
        finally:
            await pool.close()
        assert either == {ruff, terraform, prod}
        assert not_prod == {terraform}
        assert counted == 3

    async def test_tag_overlap_matches_any_tag(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _apply(conn, schema)
            agent_id = _new_uuid()
            user_id = _new_uuid()
            infra = await _insert_skill(conn, agent_id=agent_id, user_id=user_id, name="infra", summary="i", body="b")
            docs = await _insert_skill(conn, agent_id=agent_id, user_id=user_id, name="docs", summary="d", body="b")
            other = await _insert_skill(conn, agent_id=agent_id, user_id=user_id, name="other", summary="o", body="b")
            await conn.execute("UPDATE agent_skills SET tags = $2 WHERE skill_id = $1", infra, ["infra"])
            await conn.execute("UPDATE agent_skills SET tags = $2 WHERE skill_id = $1", docs, ["docs"])
            await conn.execute("UPDATE agent_skills SET tags = $2 WHERE skill_id = $1", other, ["misc"])
        finally:
            await conn.close()
        coll, pool = await self._collection(url, schema)
        try:
            found = {s.skill_id for s in await coll.list_for_user(agent_id, user_id, tag_filter=["infra", "docs"])}
        finally:
            await pool.close()
        assert found == {infra, docs}
