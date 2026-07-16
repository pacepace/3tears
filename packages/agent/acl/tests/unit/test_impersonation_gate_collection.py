"""unit tests for ``ImpersonationGateCollection`` -- build-plan.md Chunk 13
(identity-core repo), security-model.md's Impersonation paragraph: "the
gate (per-tenant on/off + optional TTL, request/grant audit trail) lives
in agent-acl".

mirrors ``test_collections.py``'s ``AsyncMock``-pool pattern (``_make_collection``)
for the same reason: these domain methods talk to ``l3_pool`` directly with
parameterized SQL (see the Collection's own docstring), so a mocked pool is
the right-sized fixture -- no real Postgres needed to prove the SQL shape
and the TTL-self-revert / state-transition logic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid7

import pytest

from threetears.agent.acl import ImpersonationGateCollection, ImpersonationGateStatus


def _make_collection(*, l3_pool: AsyncMock | None = None) -> ImpersonationGateCollection:
    mock_registry = MagicMock()
    mock_registry.get_l1_backend.return_value = None
    mock_registry.get_l3_pool.return_value = l3_pool
    mock_registry.register.return_value = None

    mock_config = MagicMock()
    mock_config.collection_flush = "ALWAYS"
    mock_config.collection_flush_tables = ""

    return ImpersonationGateCollection(registry=mock_registry, config=mock_config)


def _gate_row(
    *,
    customer_id: UUID,
    status: ImpersonationGateStatus,
    requested_at: datetime | None = None,
    requested_by: UUID | None = None,
    granted_at: datetime | None = None,
    granted_by: UUID | None = None,
    ttl_seconds: int | None = None,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "customer_id": customer_id,
        "status": status.value,
        "requested_at": requested_at,
        "requested_by": requested_by,
        "granted_at": granted_at,
        "granted_by": granted_by,
        "ttl_seconds": ttl_seconds,
        "expires_at": expires_at,
        "date_created": now,
        "date_updated": now,
    }


class TestGetEffectiveStatus:
    """``get_effective_status`` -- read-time TTL self-revert."""

    @pytest.mark.asyncio
    async def test_no_row_reads_as_disabled(self) -> None:
        """a tenant that never requested impersonation is gated off by default."""
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        coll = _make_collection(l3_pool=pool)
        status = await coll.get_effective_status(uuid7())
        assert status is ImpersonationGateStatus.DISABLED

    @pytest.mark.asyncio
    async def test_enabled_row_without_ttl_stays_enabled(self) -> None:
        cid = uuid7()
        row = _gate_row(customer_id=cid, status=ImpersonationGateStatus.ENABLED, ttl_seconds=None, expires_at=None)
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)
        status = await coll.get_effective_status(cid)
        assert status is ImpersonationGateStatus.ENABLED

    @pytest.mark.asyncio
    async def test_enabled_row_before_ttl_stays_enabled(self) -> None:
        cid = uuid7()
        now = datetime.now(UTC)
        row = _gate_row(
            customer_id=cid,
            status=ImpersonationGateStatus.ENABLED,
            ttl_seconds=3600,
            expires_at=now + timedelta(hours=1),
        )
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)
        status = await coll.get_effective_status(cid, now=now)
        assert status is ImpersonationGateStatus.ENABLED

    @pytest.mark.asyncio
    async def test_enabled_row_past_ttl_self_reverts_to_disabled(self) -> None:
        """test-specifications.md's Edge Case: "gate self-reverts after the
        TTL elapses" -- a read past `expires_at` sees `disabled` without a
        separate write."""
        cid = uuid7()
        now = datetime.now(UTC)
        row = _gate_row(
            customer_id=cid,
            status=ImpersonationGateStatus.ENABLED,
            ttl_seconds=3600,
            expires_at=now - timedelta(seconds=1),
        )
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)
        status = await coll.get_effective_status(cid, now=now)
        assert status is ImpersonationGateStatus.DISABLED

    @pytest.mark.asyncio
    async def test_requested_row_reads_as_requested(self) -> None:
        """a pending request is neither enabled nor disabled -- distinct status."""
        cid = uuid7()
        row = _gate_row(customer_id=cid, status=ImpersonationGateStatus.REQUESTED)
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)
        status = await coll.get_effective_status(cid)
        assert status is ImpersonationGateStatus.REQUESTED


class TestStateTransitions:
    """``disabled -> requested -> enabled`` -- test-specifications.md's Edge
    Case: "disabled -> requested -> enabled transitions recorded with
    requested_by/granted_by"."""

    @pytest.mark.asyncio
    async def test_request_enable_writes_requested_status(self) -> None:
        cid = uuid7()
        requester = uuid7()
        row = _gate_row(customer_id=cid, status=ImpersonationGateStatus.REQUESTED, requested_by=requester)
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)

        entity = await coll.request_enable(cid, requested_by=requester)

        assert entity.status == ImpersonationGateStatus.REQUESTED.value
        assert entity.requested_by == requester
        pool.fetchrow.assert_awaited_once()
        sql = pool.fetchrow.await_args.args[0]
        assert "INSERT INTO impersonation_gates" in sql
        assert "ON CONFLICT (customer_id) DO UPDATE" in sql
        assert pool.fetchrow.await_args.args[1] == cid
        assert pool.fetchrow.await_args.args[2] == ImpersonationGateStatus.REQUESTED.value
        assert pool.fetchrow.await_args.args[4] == requester

    @pytest.mark.asyncio
    async def test_grant_enable_writes_enabled_status_with_ttl(self) -> None:
        cid = uuid7()
        granter = uuid7()
        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=2)
        row = _gate_row(
            customer_id=cid,
            status=ImpersonationGateStatus.ENABLED,
            granted_by=granter,
            ttl_seconds=7200,
            expires_at=expires_at,
        )
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)

        entity = await coll.grant_enable(cid, granted_by=granter, ttl_seconds=7200, now=now)

        assert entity.status == ImpersonationGateStatus.ENABLED.value
        assert entity.granted_by == granter
        assert entity.ttl_seconds == 7200
        pool.fetchrow.assert_awaited_once()
        sql = pool.fetchrow.await_args.args[0]
        assert "INSERT INTO impersonation_gates" in sql
        assert pool.fetchrow.await_args.args[2] == ImpersonationGateStatus.ENABLED.value
        # expires_at computed from now + ttl_seconds, passed as the sixth bind param
        assert pool.fetchrow.await_args.args[6] == now + timedelta(seconds=7200)

    @pytest.mark.asyncio
    async def test_grant_enable_with_no_ttl_leaves_expires_at_none(self) -> None:
        cid = uuid7()
        granter = uuid7()
        row = _gate_row(customer_id=cid, status=ImpersonationGateStatus.ENABLED, granted_by=granter)
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)

        await coll.grant_enable(cid, granted_by=granter, ttl_seconds=None)

        assert pool.fetchrow.await_args.args[5] is None  # ttl_seconds bind
        assert pool.fetchrow.await_args.args[6] is None  # expires_at bind

    @pytest.mark.asyncio
    async def test_disable_writes_disabled_status(self) -> None:
        """mid-session revocation path -- test-specifications.md's Error
        Case: "gate revoked mid-session stops the next refresh"."""
        cid = uuid7()
        row = _gate_row(customer_id=cid, status=ImpersonationGateStatus.DISABLED)
        pool = AsyncMock()
        pool.fetchrow.return_value = row
        coll = _make_collection(l3_pool=pool)

        entity = await coll.disable(cid)

        assert entity is not None
        assert entity.status == ImpersonationGateStatus.DISABLED.value
        pool.fetchrow.assert_awaited_once()
        sql = pool.fetchrow.await_args.args[0]
        assert "UPDATE impersonation_gates" in sql
        assert pool.fetchrow.await_args.args[2] == ImpersonationGateStatus.DISABLED.value

    @pytest.mark.asyncio
    async def test_disable_with_no_existing_row_returns_none(self) -> None:
        """disabling a tenant that never had a gate row is a no-op, not an
        error -- the effective status was already `disabled`."""
        pool = AsyncMock()
        pool.fetchrow.return_value = None
        coll = _make_collection(l3_pool=pool)
        result = await coll.disable(uuid7())
        assert result is None
