"""coverage for the shared eager BIND of the ``{ns}-collections`` bucket.

This function is called at startup by every process that wires an L2-live
``CollectionRegistry`` -- the registry server, tool pods, and the agent SDK -- so the
distinction it draws between its two failure modes is load-bearing in all of them: a
``KvConfigMismatch`` must kill the process on the first attempt, and a ``KvError`` must
be retried, because on a cold cluster it means only that the declaring identity has not
run yet.

These tests were written against a per-consumer COPY of this function that has since been
deleted in favour of the shared one. They live here now because this is where the
behaviour lives; leaving them behind would have deleted the only coverage the retry
policy has.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.collections import bucket as bucket_module
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.bucket import COLLECTIONS_BIND_ATTEMPTS, bind_collections_bucket
from threetears.nats.errors import KvConfigMismatch, KvError


class TestBindsRatherThanDeclares:
    """the bind must never issue a ``STREAM.CREATE`` this principal cannot make."""

    async def test_binds_the_suffix_basecollection_opens(self) -> None:
        """one bucket name, taken from the collection base rather than re-typed."""
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(return_value=MagicMock())

        await bind_collections_bucket(nc)

        nc.ensure_kv_bucket.assert_awaited_once_with(
            name=BaseCollection.L2_BUCKET_SUFFIX,
            create_if_missing=False,
        )

    async def test_returns_the_bound_handle(self) -> None:
        """callers that need the handle get it rather than re-opening the bucket."""
        handle = MagicMock()
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(return_value=handle)

        result = await bind_collections_bucket(nc)

        assert result is handle


class TestTheTwoFailuresAreToldApart:
    """retrying the wrong one of these is how a process ends up silently broken."""

    async def test_config_mismatch_propagates_on_the_first_attempt(self) -> None:
        """config drift does not heal, so retrying it only delays the death.

        this is also why ``retry_with_backoff`` is not used here: it never raises, so it
        would downgrade the mismatch to a log line and carry the process on into exactly
        the state the distinct exception type exists to prevent.
        """
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(side_effect=KvConfigMismatch("allow_direct differs"))

        with pytest.raises(KvConfigMismatch):
            await bind_collections_bucket(nc)

        assert nc.ensure_kv_bucket.await_count == 1

    async def test_transient_kv_error_is_retried_then_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """a cold cluster races the declaring identity; that IS transient."""
        monkeypatch.setattr(bucket_module.asyncio, "sleep", AsyncMock())
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(
            side_effect=[KvError("bucket not found"), KvError("bucket not found"), MagicMock()],
        )

        await bind_collections_bucket(nc)

        assert nc.ensure_kv_bucket.await_count == 3

    async def test_kv_error_raises_once_the_budget_is_spent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """a bind that never succeeds must fail loud, not run on with L2 quietly off."""
        monkeypatch.setattr(bucket_module.asyncio, "sleep", AsyncMock())
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(side_effect=KvError("bucket not found"))

        with pytest.raises(KvError):
            await bind_collections_bucket(nc)

        assert nc.ensure_kv_bucket.await_count == COLLECTIONS_BIND_ATTEMPTS

    async def test_exhaustion_names_the_component_and_both_causes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """the raise is the operator's only clue, so it must say who failed and why.

        the two causes need different fixes -- start the hub, or grant the principal --
        and the bind cannot tell them apart, so it names both.
        """
        monkeypatch.setattr(bucket_module.asyncio, "sleep", AsyncMock())
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(side_effect=KvError("bucket not found"))

        with pytest.raises(KvError) as excinfo:
            await bind_collections_bucket(nc, component="tool-pod", attempts=2)

        message = str(excinfo.value)
        assert "tool-pod" in message
        assert "has not run" in message
        assert "grant does not cover" in message


class TestAttemptBudgetIsCallerOverridable:
    """the default is sized for a cold cluster; a test or a probe wants a short one."""

    async def test_attempts_override_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """without this, every failure test pays twenty real backoff steps."""
        monkeypatch.setattr(bucket_module.asyncio, "sleep", AsyncMock())
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(side_effect=KvError("bucket not found"))

        with pytest.raises(KvError):
            await bind_collections_bucket(nc, attempts=3)

        assert nc.ensure_kv_bucket.await_count == 3
