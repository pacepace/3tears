"""unit tests for :class:`threetears.epoch.client.EpochClient`.

covers atomic-bump SQL parameterization, RETURNING value
parsing, broadcast happy-path, broadcast-fail tolerance (durable
write, log-and-swallow on PublishError), and the cold-row default
on :meth:`current`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.epoch.client import EpochClient
from threetears.epoch.wire import EpochBumpMessage
from threetears.nats.errors import PublishError
from threetears.nats.subjects import Subject


def _subject(path: str = "metallm.capabilities.epoch") -> Subject:
    """build a point Subject for tests."""
    return Subject(path=path, kind="point")


def _pool_with_bump(returning_epoch: int) -> Any:
    """build a pool stub whose fetchrow returns one ``epoch`` column."""
    pool = MagicMock()
    pool.fetchrow = AsyncMock(return_value={"epoch": returning_epoch})
    pool.fetchval = AsyncMock(return_value=None)
    return pool


def _nats_mock() -> Any:
    """build a NatsClient stub with publish + subscribe_typed AsyncMocks."""
    nats = MagicMock()
    nats.publish = AsyncMock()
    nats.subscribe_typed = AsyncMock()
    return nats


class TestEpochClientCurrent:
    """:meth:`current` reads from ``config_epochs`` with the row PK on subject path."""

    @pytest.mark.asyncio
    async def test_current_returns_zero_when_no_row(self) -> None:
        """fresh-database read with no row returns ``0`` (cold-start default)."""
        pool = _pool_with_bump(0)
        pool.fetchval = AsyncMock(return_value=None)
        client = EpochClient(pool, _nats_mock())

        result = await client.current(_subject())

        assert result == 0
        pool.fetchval.assert_awaited_once()
        sql, *args = pool.fetchval.await_args.args
        assert "SELECT epoch FROM config_epochs" in sql
        assert "WHERE subject_path" in sql
        assert args == ["metallm.capabilities.epoch"]

    @pytest.mark.asyncio
    async def test_current_returns_int_for_existing_row(self) -> None:
        """existing row returns its epoch as ``int``."""
        pool = _pool_with_bump(0)
        pool.fetchval = AsyncMock(return_value=42)
        client = EpochClient(pool, _nats_mock())

        result = await client.current(_subject())

        assert result == 42


class TestEpochClientBump:
    """:meth:`bump` runs the atomic upsert and broadcasts on success."""

    @pytest.mark.asyncio
    async def test_bump_returns_new_epoch_from_upsert(self) -> None:
        """RETURNING value flows through as the bump's return value."""
        pool = _pool_with_bump(returning_epoch=5)
        nats = _nats_mock()
        client = EpochClient(pool, nats)

        epoch = await client.bump(_subject(), payload={"hint": "x"})

        assert epoch == 5

    @pytest.mark.asyncio
    async def test_bump_publishes_typed_envelope(self) -> None:
        """publish carries an :class:`EpochBumpMessage` whose fields match the upsert."""
        pool = _pool_with_bump(returning_epoch=7)
        nats = _nats_mock()
        client = EpochClient(pool, nats)
        subject = _subject("aibots.gateway.catalog.epoch")

        await client.bump(subject, payload={"action": "create"})

        nats.publish.assert_awaited_once()
        call = nats.publish.await_args
        assert call.kwargs["subject"] is subject
        message = call.kwargs["message"]
        assert isinstance(message, EpochBumpMessage)
        assert message.subject_path == "aibots.gateway.catalog.epoch"
        assert message.epoch == 7
        assert message.payload == {"action": "create"}

    @pytest.mark.asyncio
    async def test_bump_uses_subject_path_as_row_pk(self) -> None:
        """upsert SQL is parameterized with the subject path as the PK column."""
        pool = _pool_with_bump(returning_epoch=1)
        client = EpochClient(pool, _nats_mock())
        subject = _subject("aibots.mcp.rbac.epoch")

        await client.bump(subject, payload=None)

        pool.fetchrow.assert_awaited_once()
        sql, *args = pool.fetchrow.await_args.args
        assert "INSERT INTO config_epochs" in sql
        assert "ON CONFLICT (subject_path)" in sql
        assert "epoch = config_epochs.epoch + 1" in sql
        assert "RETURNING epoch" in sql
        assert args[0] == "aibots.mcp.rbac.epoch"
        assert args[1] is None

    @pytest.mark.asyncio
    async def test_bump_swallows_publish_error(self) -> None:
        """broadcast failure logs + returns the durable epoch; row commit is the truth."""
        pool = _pool_with_bump(returning_epoch=3)
        nats = _nats_mock()
        nats.publish = AsyncMock(side_effect=PublishError("transport down"))
        client = EpochClient(pool, nats)

        epoch = await client.bump(_subject())

        assert epoch == 3
        nats.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_bump_raises_when_returning_clause_yields_no_row(self) -> None:
        """defensive guard for an impossible return path — programming error path."""
        pool = MagicMock()
        pool.fetchrow = AsyncMock(return_value=None)
        client = EpochClient(pool, _nats_mock())

        with pytest.raises(RuntimeError, match="returned no row"):
            await client.bump(_subject())

    @pytest.mark.asyncio
    async def test_bump_payload_default_is_none(self) -> None:
        """bumps without an explicit payload pass NULL to JSONB column."""
        pool = _pool_with_bump(returning_epoch=1)
        nats = _nats_mock()
        client = EpochClient(pool, nats)

        await client.bump(_subject())

        sql, *args = pool.fetchrow.await_args.args
        assert args[1] is None
        message = nats.publish.await_args.kwargs["message"]
        assert message.payload is None
