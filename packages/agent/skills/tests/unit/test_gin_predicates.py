"""the skills search predicates on GIN-indexed columns filter rows instead of scanning the index.

YugabyteDB's GIN index refuses any scan needing more than one required entry, which a tag
overlap (``tags && $n``) and a typed full-text query with OR or NOT both need; the query
fails outright rather than degrading (``threetears.core.data.gin``). These pin that each
such predicate reaches the SQL through ``gin_filter``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.skills.collections import AgentSkillCollection


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


class _SqlRecorder:
    """asyncpg-pool stand-in that records the SQL of ``fetch`` and ``fetchval`` and returns nothing."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        """Record the statement and return no rows."""
        self.statements.append(sql)
        return []

    async def fetchval(self, sql: str, *params: Any) -> int:
        """Record the statement and return a zero count."""
        self.statements.append(sql)
        return 0


def _collection(pool: _SqlRecorder) -> AgentSkillCollection:
    """Build a collection wired only to ``pool``; the search paths touch nothing else."""
    coll = object.__new__(AgentSkillCollection)
    coll.l3_pool = pool
    return coll


class TestSkillSearchFiltersInsteadOfScanningTheGinIndex:
    async def test_listing_filters_on_tag_overlap_and_the_typed_query(self) -> None:
        pool = _SqlRecorder()

        await _collection(pool).list_for_user(
            _new_uuid(), _new_uuid(), tag_filter=["deploy", "infra"], query="build or publish -draft"
        )

        (sql,) = pool.statements
        assert "(tags && $3) IS TRUE" in sql
        assert "(search_vector @@ websearch_to_tsquery('english', $4)) IS TRUE" in sql

    async def test_counting_filters_the_same_way(self) -> None:
        pool = _SqlRecorder()

        await _collection(pool).count_for_user(
            _new_uuid(), _new_uuid(), tag_filter=["deploy", "infra"], query="build or publish -draft"
        )

        (sql,) = pool.statements
        assert "(tags && $3) IS TRUE" in sql
        assert "(search_vector @@ websearch_to_tsquery('english', $4)) IS TRUE" in sql
