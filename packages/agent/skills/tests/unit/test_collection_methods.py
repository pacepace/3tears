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
from uuid import UUID

from uuid_utils import uuid7

from threetears.agent.skills.collections import (
    AgentSkillCollection,
    AgentSkillInvocationCollection,
    _AGENT_SKILL_INVOCATIONS_UPSERT_SQL,
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
