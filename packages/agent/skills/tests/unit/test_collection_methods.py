"""Unit tests for pure-logic helpers on the agent-skills collections.

The Collection classes are wired to a real Postgres pool in
integration tests; the unit suite exercises:

- ``_skill_insert_params`` / ``_invocation_insert_params`` projections
  preserve the declared column order so positional asyncpg parameters
  stay in sync with the SQL placeholders.
- ``_build_upsert_sql`` emits a syntactically valid upsert with the
  expected conflict target.
- Class attributes (``primary_key_column``, ``partition_column``)
  declare the documented contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
    _AGENT_SKILL_INVOCATIONS_UPSERT_SQL,
    _AGENT_SKILLS_UPSERT_CAS_SQL,
    _AGENT_SKILLS_UPSERT_SQL,
    _INVOCATION_INSERT_COLUMNS,
    _SKILL_INSERT_COLUMNS,
    _build_upsert_sql,
    _invocation_insert_params,
    _skill_insert_params,
)


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


class TestSkillInsertParams:
    """``_skill_insert_params`` preserves column order + applies defaults."""

    def test_full_row_round_trip(self) -> None:
        """Every column in the dict is bound at its declared position."""
        agent_id = _new_uuid()
        skill_id = _new_uuid()
        user_id = _new_uuid()
        now = datetime.now(UTC)
        data = {
            "agent_id": agent_id,
            "skill_id": skill_id,
            "user_id": user_id,
            "name": "deploy-helper",
            "summary": "Deploy",
            "body": "Steps",
            "prompt_mode": "additive",
            "tool_additions": ["mcp.shell"],
            "tool_restrictions": ["mcp.dangerous"],
            "trigger_keywords": "deploy",
            "tags": ["ops"],
            "source": "manual",
            "enabled": True,
            "use_count": 0,
            "last_used_at": None,
            "success_count": 0,
            "failure_count": 0,
            "last_failure_at": None,
            "date_created": now,
            "date_updated": now,
        }
        params = _skill_insert_params(data)
        assert len(params) == len(_SKILL_INSERT_COLUMNS)
        # spot-check positional mapping
        agent_idx = _SKILL_INSERT_COLUMNS.index("agent_id")
        name_idx = _SKILL_INSERT_COLUMNS.index("name")
        prompt_idx = _SKILL_INSERT_COLUMNS.index("prompt_mode")
        additions_idx = _SKILL_INSERT_COLUMNS.index("tool_additions")
        assert params[agent_idx] == agent_id
        assert params[name_idx] == "deploy-helper"
        assert params[prompt_idx] == "additive"
        assert params[additions_idx] == ["mcp.shell"]

    def test_defaults_applied_for_omitted_columns(self) -> None:
        """Missing ``prompt_mode`` / ``enabled`` / counters get sensible defaults."""
        data = {
            "agent_id": _new_uuid(),
            "skill_id": _new_uuid(),
            "user_id": _new_uuid(),
            "name": "minimal",
            "summary": "summary",
            "body": "body",
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        params = _skill_insert_params(data)
        idx_prompt = _SKILL_INSERT_COLUMNS.index("prompt_mode")
        idx_additions = _SKILL_INSERT_COLUMNS.index("tool_additions")
        idx_restrictions = _SKILL_INSERT_COLUMNS.index("tool_restrictions")
        idx_keywords = _SKILL_INSERT_COLUMNS.index("trigger_keywords")
        idx_tags = _SKILL_INSERT_COLUMNS.index("tags")
        idx_source = _SKILL_INSERT_COLUMNS.index("source")
        idx_enabled = _SKILL_INSERT_COLUMNS.index("enabled")
        idx_use_count = _SKILL_INSERT_COLUMNS.index("use_count")
        idx_success = _SKILL_INSERT_COLUMNS.index("success_count")
        idx_failure = _SKILL_INSERT_COLUMNS.index("failure_count")
        assert params[idx_prompt] == "additive"
        assert params[idx_additions] == []
        assert params[idx_restrictions] == []
        assert params[idx_keywords] == ""
        assert params[idx_tags] == []
        assert params[idx_source] == "manual"
        assert params[idx_enabled] is True
        assert params[idx_use_count] == 0
        assert params[idx_success] == 0
        assert params[idx_failure] == 0

    def test_tool_additions_coerced_to_list(self) -> None:
        """A tuple input is normalised to ``list`` for asyncpg's text[] codec."""
        data = {
            "agent_id": _new_uuid(),
            "skill_id": _new_uuid(),
            "user_id": _new_uuid(),
            "name": "x",
            "summary": "x",
            "body": "x",
            "tool_additions": ("mcp.a", "mcp.b"),
            "date_created": datetime.now(UTC),
            "date_updated": datetime.now(UTC),
        }
        params = _skill_insert_params(data)
        idx = _SKILL_INSERT_COLUMNS.index("tool_additions")
        assert params[idx] == ["mcp.a", "mcp.b"]


class TestInvocationInsertParams:
    """``_invocation_insert_params`` preserves column order."""

    def test_full_row_round_trip(self) -> None:
        """All eleven invocation columns map positionally."""
        agent_id = _new_uuid()
        invocation_id = _new_uuid()
        skill_id = _new_uuid()
        user_id = _new_uuid()
        conversation_id = _new_uuid()
        message_id = _new_uuid()
        now = datetime.now(UTC)
        data = {
            "agent_id": agent_id,
            "invocation_id": invocation_id,
            "skill_id": skill_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "invocation_source": "wake",
            "invoked_at": now,
            "outcome": "success",
            "outcome_source": "agent_marker",
            "notes": "n",
        }
        params = _invocation_insert_params(data)
        assert len(params) == len(_INVOCATION_INSERT_COLUMNS)
        assert params[_INVOCATION_INSERT_COLUMNS.index("agent_id")] == agent_id
        assert params[_INVOCATION_INSERT_COLUMNS.index("invocation_source")] == "wake"
        assert params[_INVOCATION_INSERT_COLUMNS.index("outcome")] == "success"


class TestBuildUpsertSql:
    """``_build_upsert_sql`` emits SQL with the expected shape."""

    def test_upsert_includes_conflict_clause(self) -> None:
        """Generated SQL carries ``ON CONFLICT (pk_cols) DO UPDATE SET ...``."""
        sql = _build_upsert_sql(
            "demo",
            ("a", "b", "c"),
            ("b", "c"),
            ("a",),
        )
        assert sql.startswith("INSERT INTO demo (a, b, c) VALUES ($1, $2, $3) ")
        assert "ON CONFLICT (a) DO UPDATE SET" in sql
        assert "b = EXCLUDED.b" in sql
        assert "c = EXCLUDED.c" in sql

    def test_agent_skills_upsert_targets_composite_pk(self) -> None:
        """The module-level upsert string conflict-targets ``(agent_id, skill_id)``."""
        assert "ON CONFLICT (agent_id, skill_id)" in _AGENT_SKILLS_UPSERT_SQL

    def test_agent_skill_invocations_upsert_targets_composite_pk(self) -> None:
        """The module-level invocation upsert conflict-targets ``(agent_id, invocation_id)``."""
        assert "ON CONFLICT (agent_id, invocation_id)" in _AGENT_SKILL_INVOCATIONS_UPSERT_SQL


class TestCollectionClassAttributes:
    """Class attributes match the spec's documented contract."""

    def test_skill_collection_primary_key_column(self) -> None:
        """``primary_key_column`` is ``(agent_id, skill_id)``."""
        assert AgentSkillCollection.primary_key_column == ("agent_id", "skill_id")

    def test_skill_collection_partition_column(self) -> None:
        """``partition_column`` is ``agent_id``."""
        assert AgentSkillCollection.partition_column == "agent_id"

    def test_invocation_collection_primary_key_column(self) -> None:
        """``primary_key_column`` is ``(agent_id, invocation_id)``."""
        assert AgentSkillInvocationCollection.primary_key_column == (
            "agent_id",
            "invocation_id",
        )

    def test_invocation_collection_partition_column(self) -> None:
        """``partition_column`` is ``agent_id``."""
        assert AgentSkillInvocationCollection.partition_column == "agent_id"


class _RecordingPool:
    """Minimal asyncpg-pool stand-in recording ``execute`` calls.

    Returns a fixed command-tag so ``save_to_store`` can exercise the
    :func:`parse_rowcount` return path without a live database.
    """

    def __init__(self, status: str = "UPDATE 1") -> None:
        self.status = status
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *params: Any) -> str:
        """Record the statement + bound params and return the fixed tag."""
        self.calls.append((sql, params))
        return self.status


def _bare_skill_collection(
    pool: _RecordingPool,
) -> tuple[AgentSkillCollection, list[Any]]:
    """Build an ``AgentSkillCollection`` bypassing the registry-bound init.

    ``object.__new__`` skips :meth:`BaseCollection.__init__` (which needs
    a live registry / config); the counter + save paths under test touch
    only ``l3_pool`` and ``invalidate_cache``, both wired here. Returns
    the collection plus the list that records each ``invalidate_cache``
    pk so a test can assert the cross-tier invalidation fired.
    """
    coll = object.__new__(AgentSkillCollection)
    coll.l3_pool = pool
    invalidated: list[Any] = []

    async def _record_invalidate(entity_id: Any) -> None:
        invalidated.append(entity_id)

    coll.invalidate_cache = _record_invalidate  # type: ignore[method-assign]
    return coll, invalidated


class TestCounterMutationInvalidation:
    """Counter bumps drop the pk from L1/L2 so stale reads cannot survive."""

    async def test_bump_use_count_invalidates_each_pk(self) -> None:
        """Every bumped skill's pk is invalidated after the bulk UPDATE."""
        pool = _RecordingPool()
        coll, invalidated = _bare_skill_collection(pool)
        agent_id = _new_uuid()
        skill_a = _new_uuid()
        skill_b = _new_uuid()
        await coll.bump_use_count(agent_id, [skill_a, skill_b])
        assert len(pool.calls) == 1
        assert invalidated == [(agent_id, skill_a), (agent_id, skill_b)]

    async def test_bump_use_count_empty_batch_no_invalidation(self) -> None:
        """An empty batch short-circuits: no UPDATE, no invalidation."""
        pool = _RecordingPool()
        coll, invalidated = _bare_skill_collection(pool)
        await coll.bump_use_count(_new_uuid(), [])
        assert pool.calls == []
        assert invalidated == []

    async def test_increment_outcome_counts_invalidates_pk(self) -> None:
        """A success/failure bump invalidates the single affected pk."""
        pool = _RecordingPool()
        coll, invalidated = _bare_skill_collection(pool)
        agent_id = _new_uuid()
        skill_id = _new_uuid()
        await coll.increment_outcome_counts(agent_id, skill_id, "success")
        assert len(pool.calls) == 1
        assert invalidated == [(agent_id, skill_id)]


class TestSkillSaveCasFence:
    """``save_to_store`` honours the optimistic-lock fence on edits."""

    def test_cas_sql_carries_date_updated_fence(self) -> None:
        """The CAS variant fences the DO UPDATE on the trailing param."""
        cas_param = f"${len(_SKILL_INSERT_COLUMNS) + 1}"
        assert f"WHERE agent_skills.date_updated = {cas_param}" in _AGENT_SKILLS_UPSERT_CAS_SQL
        # The unfenced insert variant carries no WHERE fence.
        assert "WHERE" not in _AGENT_SKILLS_UPSERT_SQL

    async def test_insert_path_uses_unfenced_sql(self) -> None:
        """No ``original_timestamp`` -> unfenced upsert, no trailing fence param."""
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        data = {"agent_id": _new_uuid(), "skill_id": _new_uuid(), "user_id": _new_uuid(), "name": "x", "summary": "s"}
        affected = await coll.save_to_store(data)
        assert affected == 1
        sql, params = pool.calls[0]
        assert sql == _AGENT_SKILLS_UPSERT_SQL
        assert len(params) == len(_SKILL_INSERT_COLUMNS)

    async def test_edit_path_uses_cas_sql_with_fence_param(self) -> None:
        """An ``original_timestamp`` selects the CAS SQL and binds the fence last."""
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        fence = datetime.now(UTC)
        data = {"agent_id": _new_uuid(), "skill_id": _new_uuid(), "user_id": _new_uuid(), "name": "x", "summary": "s"}
        affected = await coll.save_to_store(data, fence)
        assert affected == 1
        sql, params = pool.calls[0]
        assert sql == _AGENT_SKILLS_UPSERT_CAS_SQL
        assert len(params) == len(_SKILL_INSERT_COLUMNS) + 1
        assert params[-1] == fence

    async def test_cas_fence_mismatch_reports_zero_rows(self) -> None:
        """A conflicting DO UPDATE (0 rows) is surfaced as 0 for the lost-update guard."""
        pool = _RecordingPool(status="INSERT 0 0")
        coll, _ = _bare_skill_collection(pool)
        data = {"agent_id": _new_uuid(), "skill_id": _new_uuid(), "user_id": _new_uuid(), "name": "x", "summary": "s"}
        affected = await coll.save_to_store(data, datetime.now(UTC))
        assert affected == 0
