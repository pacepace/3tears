"""unit tests for :mod:`threetears.mcp.noop_epoch`.

guards the contract that consumer stdio MCP servers depend on:

- every method matches the canonical
  :class:`~threetears.epoch.EpochClient` /
  :class:`~threetears.epoch.EpochListener` signature.
- every method returns the documented no-op value (``0`` for ints
  / ``None`` for awaitable returns) without I/O.
- :class:`~threetears.mcp.auth.LocalGrantAuthorizer` constructed
  with the noops can ``start()`` + ``stop()`` cleanly -- the noop
  helpers don't break the lifecycle.

if the canonical EpochClient / EpochListener signatures drift
(e.g. add a required argument), these tests fail loudly rather
than silently breaking every consumer that imports the noops.
"""

from __future__ import annotations

import pytest
from threetears.mcp import (
    LocalGrantAuthorizer,
    NoopEpochClient,
    NoopEpochListener,
)
from threetears.nats.subjects import Subject


def _subject() -> Subject:
    """canonical test subject (mcp.rbac shape)."""
    return Subject(path="aibots.mcp.rbac.epoch", kind="point")


# ---------------------------------------------------------------------
# NoopEpochClient
# ---------------------------------------------------------------------


class TestNoopEpochClient:
    """``current`` and ``bump`` return ``0`` without I/O."""

    @pytest.mark.asyncio
    async def test_current_returns_zero(self) -> None:
        """current(subject) returns 0 -- the script never observes a non-zero epoch."""
        client = NoopEpochClient()
        assert await client.current(_subject()) == 0

    @pytest.mark.asyncio
    async def test_bump_returns_zero(self) -> None:
        """bump(subject) is a no-op; returns 0."""
        client = NoopEpochClient()
        assert await client.bump(_subject()) == 0

    @pytest.mark.asyncio
    async def test_bump_with_payload_returns_zero(self) -> None:
        """bump accepts a payload kwarg matching EpochClient.bump shape."""
        client = NoopEpochClient()
        assert await client.bump(_subject(), payload={"action": "create"}) == 0


# ---------------------------------------------------------------------
# NoopEpochListener
# ---------------------------------------------------------------------


class TestNoopEpochListener:
    """``subscribe`` / ``catch_up`` / ``echo`` / ``last_seen`` no-op."""

    @pytest.mark.asyncio
    async def test_subscribe_returns_none(self) -> None:
        """subscribe(subject, on_bump) is a no-op; returns None."""
        listener = NoopEpochListener()

        async def _cb(epoch: int, payload: object) -> None:  # noqa: ARG001
            raise AssertionError("noop listener must not invoke callback")

        result = await listener.subscribe(_subject(), _cb)
        assert result is None

    @pytest.mark.asyncio
    async def test_catch_up_returns_zero(self) -> None:
        """catch_up returns 0 (matches EpochListener.catch_up return type)."""
        listener = NoopEpochListener()

        async def _cb(epoch: int, payload: object) -> None:  # noqa: ARG001
            raise AssertionError("noop listener must not invoke callback")

        assert await listener.catch_up(_subject(), _cb) == 0

    @pytest.mark.asyncio
    async def test_echo_returns_none(self) -> None:
        """echo(subject, epoch, on_bump) is a no-op; returns None."""
        listener = NoopEpochListener()

        async def _cb(epoch: int, payload: object) -> None:  # noqa: ARG001
            raise AssertionError("noop listener must not invoke callback")

        result = await listener.echo(_subject(), 99, _cb)
        assert result is None

    def test_last_seen_returns_zero(self) -> None:
        """last_seen(subject) returns 0 (sync method matching EpochListener)."""
        listener = NoopEpochListener()
        assert listener.last_seen(_subject()) == 0


# ---------------------------------------------------------------------
# Integration with LocalGrantAuthorizer
# ---------------------------------------------------------------------


class TestNoopsCompatibleWithAuthorizerLifecycle:
    """LocalGrantAuthorizer.start()/stop() against noops completes cleanly.

    proves the duck-typing contract holds at the actual consumer
    call site -- if the canonical EpochClient / EpochListener
    signatures drift (added required arg, renamed method) the
    authorizer's start path would blow up here.
    """

    @pytest.mark.asyncio
    async def test_authorizer_start_and_stop_with_noops(self) -> None:
        """authorizer.start() reloads cache, subscribes (noop), spawns tick (noop). stop cancels."""

        async def _empty_loader() -> list[dict[str, object]]:
            return []

        authz = LocalGrantAuthorizer(
            grant_loader=_empty_loader,
            epoch_client=NoopEpochClient(),
            epoch_listener=NoopEpochListener(),
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            # confirm the lifecycle reached the steady state.
            assert authz._started is True  # noqa: SLF001
            assert authz._catchup_task is not None  # noqa: SLF001
        finally:
            await authz.stop()
        # second stop is idempotent.
        await authz.stop()
