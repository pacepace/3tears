"""unit tests for tool-eligibility query methods on NamespaceCollection.

agent-tools-eligibility shard 01 (TE-05 / TE-06): the two filter
columns added by the platform-scope migration ride on the
``namespaces`` rows; two new query methods on
:class:`NamespaceCollection` return the ACL-permitted subset for
each filter. ACL evaluation is mocked here because the wiring
contract (``evaluate_decision`` returning a bool per row) is
covered by the evaluator's own tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid7

import pytest

from threetears.agent.acl import NamespaceCollection


def _make_collection(
    cls: type,
    *,
    l3_pool: AsyncMock | None = None,
) -> Any:
    """build a Collection instance with mocked registry + config.

    parity-with: threetears.agent.acl.collections.NamespaceCollection
    -- same shape as the canonical ``_make_collection`` in
    ``test_collections.py`` (kept local rather than re-imported
    across test modules to avoid the import-order coupling that
    pattern would create).

    :param cls: Collection class to instantiate
    :ptype cls: type
    :param l3_pool: optional mocked pool
    :ptype l3_pool: AsyncMock | None
    :return: Collection instance with mocks wired in
    :rtype: Any
    """
    mock_registry = MagicMock()
    mock_registry.get_l1_backend.return_value = None
    mock_registry.get_l3_pool.return_value = l3_pool
    mock_registry.register.return_value = None
    mock_config = MagicMock()
    mock_config.collection_flush = "ALWAYS"
    mock_config.collection_flush_tables = ""
    return cls(registry=mock_registry, config=mock_config)


def _tool_namespace_row(
    *,
    customer_id: UUID | None = None,
    name: str = "tools.example.1-0",
    owner_agent_id: UUID | None = None,
    tool_eligible: bool = True,
    skill_eligible: bool = False,
    namespace_id: UUID | None = None,
) -> dict[str, Any]:
    """build a fake ``namespaces`` row of type ``tool``.

    :param customer_id: owning customer UUID (or None for platform)
    :ptype customer_id: UUID | None
    :param name: canonical sanitized namespace name
    :ptype name: str
    :param owner_agent_id: owning agent UUID (or None for platform tools)
    :ptype owner_agent_id: UUID | None
    :param tool_eligible: default-surface flag value
    :ptype tool_eligible: bool
    :param skill_eligible: skills-catalog flag value
    :ptype skill_eligible: bool
    :param namespace_id: optional id override
    :ptype namespace_id: UUID | None
    :return: row dict
    :rtype: dict[str, Any]
    """
    now = datetime.now(UTC)
    return {
        "row_scope": "platform" if customer_id is None else "customer",
        "namespace_id": namespace_id if namespace_id is not None else uuid7(),
        "name": name,
        "namespace_type": "tool",
        "owner_agent_id": owner_agent_id,
        "customer_id": customer_id,
        "schema_name": None,
        "metadata": None,
        "tool_eligible": tool_eligible,
        "skill_eligible": skill_eligible,
        "date_created": now,
        "date_updated": now,
    }


class TestListToolNamespacesForActor:
    """TE-06: default-surface query filters on ``tool_eligible=TRUE``."""

    @pytest.mark.asyncio
    async def test_query_filters_tool_eligible_column(self) -> None:
        """SQL where clause references ``tool_eligible = TRUE``."""
        pool = AsyncMock()
        pool.fetch.return_value = []
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        with patch(
            "threetears.agent.acl.evaluator.evaluate_decision",
            new=AsyncMock(return_value=True),
        ):
            await coll.list_tool_namespaces_for_actor(
                actor_user_id=uuid7(),
                actor_agent_id=uuid7(),
                cache=MagicMock(),
            )
        sql = pool.fetch.await_args.args[0]
        assert "tool_eligible = TRUE" in sql
        assert "skill_eligible" not in sql
        assert "namespace_type = 'tool'" in sql

    @pytest.mark.asyncio
    async def test_returns_only_acl_permitted_candidates(self) -> None:
        """tools the ACL evaluator denies are dropped from the result."""
        permitted_row = _tool_namespace_row(name="tools.permitted.1-0")
        denied_row = _tool_namespace_row(name="tools.denied.1-0")
        pool = AsyncMock()
        pool.fetch.return_value = [permitted_row, denied_row]
        coll = _make_collection(NamespaceCollection, l3_pool=pool)

        async def _decisions(ctx: Any, *, cache: Any) -> bool:
            return ctx.namespace.id == permitted_row["namespace_id"]

        with patch(
            "threetears.agent.acl.evaluator.evaluate_decision",
            new=AsyncMock(side_effect=_decisions),
        ):
            result = await coll.list_tool_namespaces_for_actor(
                actor_user_id=uuid7(),
                actor_agent_id=uuid7(),
                cache=MagicMock(),
            )
        assert [e.id for e in result] == [permitted_row["namespace_id"]]

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_list(self) -> None:
        """no candidate rows -> empty list, no evaluator calls."""
        pool = AsyncMock()
        pool.fetch.return_value = []
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        decide = AsyncMock(return_value=True)
        with patch(
            "threetears.agent.acl.evaluator.evaluate_decision",
            new=decide,
        ):
            result = await coll.list_tool_namespaces_for_actor(
                actor_user_id=uuid7(),
                actor_agent_id=uuid7(),
                cache=MagicMock(),
            )
        assert result == []
        decide.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_l3_pool_returns_empty_list(self) -> None:
        """collections wired without a pool return ``[]`` (defensive)."""
        coll = _make_collection(NamespaceCollection, l3_pool=None)
        result = await coll.list_tool_namespaces_for_actor(
            actor_user_id=uuid7(),
            actor_agent_id=uuid7(),
            cache=MagicMock(),
        )
        assert result == []


class TestListSkillEligibleToolNamespaces:
    """TE-05: skill-catalog query filters on ``skill_eligible=TRUE``."""

    @pytest.mark.asyncio
    async def test_query_filters_skill_eligible_column(self) -> None:
        """SQL where clause references ``skill_eligible = TRUE``."""
        pool = AsyncMock()
        pool.fetch.return_value = []
        coll = _make_collection(NamespaceCollection, l3_pool=pool)
        with patch(
            "threetears.agent.acl.evaluator.evaluate_decision",
            new=AsyncMock(return_value=True),
        ):
            await coll.list_skill_eligible_tool_namespaces(
                actor_user_id=uuid7(),
                actor_agent_id=uuid7(),
                cache=MagicMock(),
            )
        sql = pool.fetch.await_args.args[0]
        assert "skill_eligible = TRUE" in sql
        # filter columns are exclusive; only one column is referenced
        # in the WHERE clause per call.
        assert "tool_eligible = TRUE" not in sql
        assert "namespace_type = 'tool'" in sql

    @pytest.mark.asyncio
    async def test_returns_only_acl_permitted_skill_tools(self) -> None:
        """skill-eligible tools the ACL denies are dropped."""
        permitted_row = _tool_namespace_row(
            name="tools.loki_query.1-0",
            tool_eligible=False,
            skill_eligible=True,
        )
        denied_row = _tool_namespace_row(
            name="tools.postgres_query.1-0",
            tool_eligible=False,
            skill_eligible=True,
        )
        pool = AsyncMock()
        pool.fetch.return_value = [permitted_row, denied_row]
        coll = _make_collection(NamespaceCollection, l3_pool=pool)

        async def _decisions(ctx: Any, *, cache: Any) -> bool:
            return ctx.namespace.id == permitted_row["namespace_id"]

        with patch(
            "threetears.agent.acl.evaluator.evaluate_decision",
            new=AsyncMock(side_effect=_decisions),
        ):
            result = await coll.list_skill_eligible_tool_namespaces(
                actor_user_id=uuid7(),
                actor_agent_id=uuid7(),
                cache=MagicMock(),
            )
        assert [e.id for e in result] == [permitted_row["namespace_id"]]


class TestPrivateFilteredHelperRejectsUnknownColumn:
    """defense in depth: the f-string interpolation accepts only the
    whitelisted column names; anything else raises ``ValueError``."""

    @pytest.mark.asyncio
    async def test_unknown_filter_column_raises(self) -> None:
        coll = _make_collection(NamespaceCollection, l3_pool=AsyncMock())
        with pytest.raises(ValueError):
            await coll._list_tool_namespaces_filtered(  # noqa: SLF001
                actor_user_id=uuid7(),
                actor_agent_id=uuid7(),
                cache=MagicMock(),
                filter_column="DROP TABLE",
            )
