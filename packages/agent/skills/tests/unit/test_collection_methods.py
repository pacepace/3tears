"""Unit tests for the agent-skills collections' store writes, read off the SQL they send.

The Collection classes are wired to a real Postgres pool in
integration tests; the unit suite drives ``save_to_store`` /
``fetch_from_store`` over a recording pool and checks:

- every positional parameter binds to the column its placeholder names,
  with the documented defaults for omitted columns.
- the upsert conflict-targets the composite pk, updates every non-pk
  column from ``EXCLUDED``, and an edit fences on ``date_updated``.
- Class attributes (``primary_key_column``, ``partition_column``)
  declare the documented contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
    SkillShapeError,
)


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


_UPSERT = re.compile(
    r"^INSERT INTO (?P<table>\w+) \((?P<columns>[^)]*)\) VALUES \((?P<placeholders>[^)]*)\) "
    r"ON CONFLICT \((?P<conflict>[^)]*)\) DO UPDATE SET (?P<set>.*?)"
    r"(?: WHERE (?P<fence>\S+) = (?P<fence_param>\$\d+))?$"
)


@dataclass(frozen=True)
class _Upsert:
    """one recorded upsert, split into the parts a test asserts on."""

    table: str
    columns: list[str]
    placeholders: list[str]
    conflict: list[str]
    assignments: list[str]
    fence: str | None
    fence_param: str | None
    params: tuple[Any, ...]

    @property
    def bound(self) -> dict[str, Any]:
        """column name -> the value bound at that column's placeholder."""
        return dict(zip(self.columns, self.params[: len(self.columns)], strict=True))


def _upsert(sql: str, params: tuple[Any, ...]) -> _Upsert:
    match = _UPSERT.match(sql)
    assert match is not None, f"not an upsert: {sql}"
    return _Upsert(
        table=match["table"],
        columns=[c.strip() for c in match["columns"].split(",")],
        placeholders=[p.strip() for p in match["placeholders"].split(",")],
        conflict=[c.strip() for c in match["conflict"].split(",")],
        assignments=[a.strip() for a in match["set"].split(",")],
        fence=match["fence"],
        fence_param=match["fence_param"],
        params=params,
    )


async def _saved_skill(data: dict[str, Any], fence: datetime | None = None) -> _Upsert:
    pool = _RecordingPool(status="INSERT 0 1")
    coll, _ = _bare_skill_collection(pool)
    await coll.save_to_store(data, fence)
    [(sql, params)] = pool.calls
    return _upsert(sql, params)


class TestSkillInsertParams:
    """a skill write binds every column at its placeholder and applies defaults."""

    async def test_full_row_round_trip(self) -> None:
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
        upsert = await _saved_skill(data)
        assert upsert.placeholders == [f"${i + 1}" for i in range(len(upsert.columns))]
        assert len(upsert.params) == len(upsert.columns)
        bound = upsert.bound
        assert bound["agent_id"] == agent_id
        assert bound["skill_id"] == skill_id
        assert bound["name"] == "deploy-helper"
        assert bound["prompt_mode"] == "additive"
        assert bound["tool_additions"] == ["mcp.shell"]
        assert bound["date_updated"] == now

    async def test_defaults_applied_for_omitted_columns(self) -> None:
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
        bound = (await _saved_skill(data)).bound
        assert bound["prompt_mode"] == "additive"
        assert bound["tool_additions"] == []
        assert bound["tool_restrictions"] == []
        assert bound["trigger_keywords"] == ""
        assert bound["tags"] == []
        assert bound["source"] == "manual"
        assert bound["enabled"] is True
        assert bound["use_count"] == 0
        assert bound["success_count"] == 0
        assert bound["failure_count"] == 0

    async def test_tool_additions_coerced_to_list(self) -> None:
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
        assert (await _saved_skill(data)).bound["tool_additions"] == ["mcp.a", "mcp.b"]


class TestInvocationInsertParams:
    """an invocation write binds every column at its placeholder."""

    async def test_full_row_round_trip(self) -> None:
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
        pool = _RecordingPool(status="INSERT 0 1")
        coll = object.__new__(AgentSkillInvocationCollection)
        coll.l3_pool = pool
        await coll.save_to_store(data)
        [(sql, params)] = pool.calls
        upsert = _upsert(sql, params)
        assert upsert.table == "agent_skill_invocations"
        assert len(upsert.columns) == 11
        assert len(params) == len(upsert.columns)
        assert upsert.bound == data


class TestBuildUpsertSql:
    """the upsert SQL a save sends has the expected shape."""

    async def test_upsert_includes_conflict_clause(self) -> None:
        """``ON CONFLICT (pk) DO UPDATE SET`` every non-pk column from ``EXCLUDED``."""
        upsert = await _saved_skill(TestSkillSaveShape.row(body="steps"))
        assert upsert.table == "agent_skills"
        non_pk = [c for c in upsert.columns if c not in upsert.conflict]
        assert upsert.assignments
        assert set(upsert.assignments) <= {f"{c} = EXCLUDED.{c}" for c in non_pk}
        assert "name = EXCLUDED.name" in upsert.assignments
        assert not any(a.startswith(("agent_id ", "skill_id ")) for a in upsert.assignments)

    async def test_agent_skills_upsert_targets_composite_pk(self) -> None:
        """The skills upsert conflict-targets ``(agent_id, skill_id)``."""
        assert (await _saved_skill(TestSkillSaveShape.row(body="steps"))).conflict == ["agent_id", "skill_id"]

    async def test_agent_skill_invocations_upsert_targets_composite_pk(self) -> None:
        """The invocation upsert conflict-targets ``(agent_id, invocation_id)``."""
        pool = _RecordingPool(status="INSERT 0 1")
        coll = object.__new__(AgentSkillInvocationCollection)
        coll.l3_pool = pool
        await coll.save_to_store({"agent_id": _new_uuid(), "invocation_id": _new_uuid(), "skill_id": _new_uuid()})
        [(sql, params)] = pool.calls
        assert _upsert(sql, params).conflict == ["agent_id", "invocation_id"]


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

    async def fetchrow(self, sql: str, *params: Any) -> None:
        """Record a single-row read; the row is absent."""
        self.calls.append((sql, params))


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
        """Every bumped skill's pk is invalidated after the bulk UPDATE, in one call.

        One call: the UPDATE is one commit, so its rows are one write-generation advance whose
        broadcasts all carry one count; a call per row would advance once per row.
        """
        pool = _RecordingPool()
        coll, invalidated = _bare_skill_collection(pool)
        batches: list[list[Any]] = []

        async def _record_many(entity_ids: Any, **_: Any) -> None:
            batches.append(list(entity_ids))

        coll.invalidate_cache_many = _record_many  # type: ignore[method-assign]
        agent_id = _new_uuid()
        skill_a = _new_uuid()
        skill_b = _new_uuid()
        await coll.bump_use_count(agent_id, [skill_a, skill_b])
        assert len(pool.calls) == 1
        assert batches == [[(agent_id, skill_a), (agent_id, skill_b)]]
        assert invalidated == []

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

    async def test_cas_sql_carries_date_updated_fence(self) -> None:
        """The edit SQL fences the DO UPDATE on the trailing param; the insert SQL carries no fence."""
        data = TestSkillSaveShape.row(body="steps")
        edit = await _saved_skill(data, datetime.now(UTC))
        assert edit.fence == "agent_skills.date_updated"
        assert edit.fence_param == f"${len(edit.columns) + 1}"
        insert = await _saved_skill(data)
        assert insert.fence is None

    async def test_insert_path_uses_unfenced_sql(self) -> None:
        """No ``original_timestamp`` -> unfenced upsert, no trailing fence param."""
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        affected = await coll.save_to_store(TestSkillSaveShape.row(name="x", summary="s"))
        assert affected == 1
        sql, params = pool.calls[0]
        upsert = _upsert(sql, params)
        assert upsert.fence is None
        assert "WHERE" not in sql
        assert len(params) == len(upsert.columns)

    async def test_edit_path_uses_cas_sql_with_fence_param(self) -> None:
        """An ``original_timestamp`` selects the CAS SQL and binds the fence last."""
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        fence = datetime.now(UTC)
        affected = await coll.save_to_store(TestSkillSaveShape.row(name="x", summary="s"), fence)
        assert affected == 1
        sql, params = pool.calls[0]
        upsert = _upsert(sql, params)
        assert upsert.fence == "agent_skills.date_updated"
        assert len(params) == len(upsert.columns) + 1
        assert params[-1] == fence

    async def test_cas_fence_mismatch_reports_zero_rows(self) -> None:
        """A conflicting DO UPDATE (0 rows) is surfaced as 0 for the lost-update guard."""
        pool = _RecordingPool(status="INSERT 0 0")
        coll, _ = _bare_skill_collection(pool)
        data = {"agent_id": _new_uuid(), "skill_id": _new_uuid(), "user_id": _new_uuid(), "name": "x", "summary": "s"}
        affected = await coll.save_to_store(data, datetime.now(UTC))
        assert affected == 0


class TestSkillSaveShape:
    """``save_to_store`` refuses a row that is both kinds before any SQL runs."""

    @staticmethod
    def row(**fields: Any) -> dict[str, Any]:
        return {
            "agent_id": _new_uuid(),
            "skill_id": _new_uuid(),
            "user_id": _new_uuid(),
            "name": "x",
            "summary": "s",
        } | fields

    async def test_body_and_tool_refused_before_sql(self) -> None:
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        with pytest.raises(SkillShapeError, match="not both"):
            await coll.save_to_store(self.row(body="steps", tool="loki.query"))
        assert pool.calls == []

    async def test_arguments_without_tool_refused_before_sql(self) -> None:
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        with pytest.raises(SkillShapeError, match="arguments need a tool"):
            await coll.save_to_store(self.row(body="steps", arguments={"q": 1}))
        assert pool.calls == []

    async def test_non_object_arguments_refused_before_sql(self) -> None:
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        with pytest.raises(SkillShapeError, match="JSON object"):
            await coll.save_to_store(self.row(tool="loki.query", arguments=[1, 2]))
        assert pool.calls == []

    async def test_tool_only_row_binds_tool_and_arguments(self) -> None:
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        affected = await coll.save_to_store(self.row(tool="loki.query", arguments={"q": "error"}))
        assert affected == 1
        [(sql, params)] = pool.calls
        bound = _upsert(sql, params).bound
        assert bound["body"] is None
        assert bound["tool"] == "loki.query"
        # a native object for the jsonb codec, never pre-encoded text.
        assert bound["arguments"] == {"q": "error"}

    async def test_arguments_read_back_as_json_text_still_save(self) -> None:
        """A row read through a pool without the jsonb codec carries arguments as text."""
        pool = _RecordingPool(status="INSERT 0 1")
        coll, _ = _bare_skill_collection(pool)
        await coll.save_to_store(self.row(tool="loki.query", arguments='{"q": "error"}'))
        [(sql, params)] = pool.calls
        assert _upsert(sql, params).bound["arguments"] == {"q": "error"}

    async def test_every_written_column_is_read(self) -> None:
        """The fetch selects exactly the written columns, so ``tool`` / ``arguments`` reach every read."""
        written = await _saved_skill(self.row(tool="loki.query", arguments={"q": "error"}))
        pool = _RecordingPool()
        coll, _ = _bare_skill_collection(pool)
        assert await coll.fetch_from_store((_new_uuid(), _new_uuid())) is None
        [(read_sql, _params)] = pool.calls
        assert read_sql.startswith(f"SELECT {', '.join(written.columns)} FROM agent_skills")
        assert {"tool", "arguments"} <= set(written.columns)
