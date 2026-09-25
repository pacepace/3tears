"""memory's keyword search filters rows instead of scanning the GIN index.

``websearch_to_tsquery`` turns an "or" or a leading "-" in ordinary text into OR / NOT, and
YugabyteDB's GIN index refuses any scan needing more than one required entry -- the query
fails outright (``threetears.core.data.gin``). These pin that the keyword predicate reaches
the SQL through ``gin_filter`` on the memory and media-content keyword searches.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


class _SqlRecorder:
    """asyncpg-pool stand-in that records the SQL of ``fetch`` and returns no rows."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        """Record the statement and return no rows."""
        self.statements.append(sql)
        return []


class TestMemoryKeywordSearchFiltersInsteadOfScanningTheGinIndex:
    async def test_the_memory_keyword_search(self) -> None:
        pool = _SqlRecorder()
        coll = object.__new__(MemoriesCollection)
        coll.l3_pool = pool

        await coll.search_by_fts(
            user_id=_new_uuid(), agent_id=_new_uuid(), fts_text="build or publish -draft", max_results=5
        )

        (sql,) = pool.statements
        assert "(search_vector @@ websearch_to_tsquery('english', $1)) IS TRUE" in sql

    async def test_the_media_content_keyword_search(self) -> None:
        pool = _SqlRecorder()
        coll = object.__new__(MediaContentCollection)
        coll.l3_pool = pool

        await coll.search_by_fts(
            user_id=_new_uuid(), agent_id=_new_uuid(), fts_text="build or publish -draft", max_results=5
        )

        (sql,) = pool.statements
        assert "(mc.search_vector @@ websearch_to_tsquery('english', $1)) IS TRUE" in sql
