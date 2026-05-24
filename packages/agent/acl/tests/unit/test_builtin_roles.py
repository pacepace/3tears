"""unit tests for :mod:`threetears.agent.acl.builtin_roles`.

agent-tools-eligibility shard 01 (TE-12): the platform ships a
builtin role + a documented set of pre-check tool mcp_names so the
deploying app can wire default ACL grants for the wake pre-check
tools. The 3tears-side helper:

- guarantees the role exists in ``roles`` with the canonical name +
  permissions when the deploying app's bootstrap calls it,
- returns the existing row's id when present (idempotent),
- never overwrites a deploying-app-edited description on rerun.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid7

import pytest

from threetears.agent.acl import (
    PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES,
    PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME,
    PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS,
    ensure_platform_builtin_tool_user_role,
)


class _StubRoleCollection:
    """stand-in for :class:`RoleCollection`.

    parity-with: threetears.agent.acl.collections.RoleCollection
    -- the helper only touches ``l3_pool``; the rest of the
    Collection surface is irrelevant. Mocking only what's used
    keeps the test brittle to changes that matter and tolerant of
    everything else.

    :param pool: stand-in for the L3 asyncpg pool exposing
        ``fetchrow`` and ``execute``
    :ptype pool: Any
    """

    def __init__(self, pool: Any) -> None:
        """initialize with a pool stand-in.

        :param pool: stand-in pool
        :ptype pool: Any
        """
        self.l3_pool = pool


class TestPublishedConstants:
    """canonical values the deploying app's seed code references."""

    def test_role_name_is_camelcase_proper_noun(self) -> None:
        """name is the documented constant; no whitespace, no underscores."""
        assert PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME == "PlatformBuiltinToolUser"

    def test_pre_check_tool_names_cover_three_tools(self) -> None:
        """the three documented pre-check tool mcp_names are published."""
        assert set(PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES) == {
            "http_get",
            "loki_query",
            "postgres_query",
        }

    def test_permissions_grant_tool_call_action(self) -> None:
        """role grants exactly ``tool.call`` -- minimal surface."""
        assert PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS == {"tool": ["call"]}


class TestEnsurePlatformBuiltinToolUserRole:
    """idempotent insert-or-fetch helper."""

    @pytest.mark.asyncio
    async def test_returns_existing_role_id_without_insert(self) -> None:
        """existing row wins; no INSERT issued."""
        existing_id = uuid7()
        pool = AsyncMock()
        pool.fetchrow.return_value = {"role_id": existing_id}
        coll = _StubRoleCollection(pool=pool)
        result = await ensure_platform_builtin_tool_user_role(coll)
        assert result == existing_id
        pool.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inserts_when_absent_and_returns_new_id(self) -> None:
        """missing row triggers INSERT with the canonical payload."""
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        pool.execute.return_value = "INSERT 0 1"
        coll = _StubRoleCollection(pool=pool)
        result = await ensure_platform_builtin_tool_user_role(coll)
        assert isinstance(result, UUID)
        pool.execute.assert_awaited_once()
        # second positional arg after the SQL is the new UUID.
        call_args = pool.execute.await_args.args
        # args[0] is SQL; args[1..N] are parameters $1..$N
        # we passed (sql, role_id, name, description, perms_json, now)
        assert call_args[1] == result
        assert call_args[2] == PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME
        # description is a non-empty string carrying the canonical
        # "default access to pre-check tools" intent.
        assert "pre-check tools" in call_args[3].lower() or "pre_check" in call_args[3].lower()
        # the JSONB-encoded permissions string round-trips to the
        # published mapping.
        assert json.loads(call_args[4]) == PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS

    @pytest.mark.asyncio
    async def test_collection_without_pool_raises(self) -> None:
        """the helper fails fast if the Collection has no L3 pool."""
        coll = _StubRoleCollection(pool=None)
        with pytest.raises(RuntimeError):
            await ensure_platform_builtin_tool_user_role(coll)

    @pytest.mark.asyncio
    async def test_coerces_string_uuid_back_to_uuid(self) -> None:
        """existing-row branch coerces string UUIDs (NATS L3 backend)."""
        canonical_id = uuid7()
        pool = AsyncMock()
        # mimic the NATS-proxy L3 backend which returns UUIDs as strings.
        pool.fetchrow.return_value = {"role_id": str(canonical_id)}
        coll = _StubRoleCollection(pool=pool)
        result = await ensure_platform_builtin_tool_user_role(coll)
        assert result == canonical_id
        assert isinstance(result, UUID)
