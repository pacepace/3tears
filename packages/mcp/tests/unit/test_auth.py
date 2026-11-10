"""unit tests for :mod:`threetears.mcp.auth` -- identity + authorizer."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
from threetears.mcp.auth import (
    EnvVarIdentityProvider,
    GrantCacheUnavailableError,
    Identity,
    LocalGrantAuthorizer,
)
from threetears.nats import Subjects


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

#: how long a test waits for the catch-up loop to tick; a bound on a hang, never a pacing sleep.
_TICK_DEADLINE_SECONDS = 5.0


def _counting_catch_up(ticks: int, *, first_raises: BaseException | None = None) -> tuple[AsyncMock, asyncio.Event]:
    """a ``catch_up`` stand-in that sets an event once it has been awaited *ticks* times.

    The catch-up loop runs on a timer, so a test that slept a fixed while and then counted
    ticks failed whenever the machine was slow enough to fit fewer ticks into the sleep. The
    test waits on this event instead, bounded by :data:`_TICK_DEADLINE_SECONDS`.

    :param ticks: the call count at which the event is set
    :ptype ticks: int
    :param first_raises: raised by the first call instead of returning, when given
    :ptype first_raises: BaseException | None
    :return: the mock and the event
    :rtype: tuple[AsyncMock, asyncio.Event]
    """
    reached = asyncio.Event()
    calls = 0

    async def _tick(*_args: Any, **_kwargs: Any) -> int:
        nonlocal calls
        calls += 1
        if calls >= ticks:
            reached.set()
        if calls == 1 and first_raises is not None:
            raise first_raises
        return 0

    return AsyncMock(side_effect=_tick), reached


def _make_listener_capturing_subscribe() -> tuple[Any, Any, list[Any], list[Any]]:
    """build a fake EpochClient + EpochListener; capture both callbacks.

    ``on_reset`` is captured separately from ``on_bump`` because they are not
    interchangeable: a reset carries no epoch, and a consumer that treated it
    as a bump would dedupe it away.
    """
    fake_client = MagicMock()
    fake_client.current = AsyncMock(return_value=0)

    fake_listener = MagicMock()
    captured: list[Any] = []
    captured_resets: list[Any] = []

    async def _subscribe(
        subject: Any,  # noqa: ARG001
        on_bump: Any,
        primed_epoch: Any = None,  # noqa: ARG001
        on_reset: Any = None,
    ) -> None:
        captured.append(on_bump)
        if on_reset is not None:
            captured_resets.append(on_reset)

    fake_listener.subscribe = AsyncMock(side_effect=_subscribe)
    fake_listener.catch_up = AsyncMock(return_value=0)
    fake_listener.last_seen = MagicMock(return_value=0)
    return fake_client, fake_listener, captured, captured_resets


def _row(principal_id: UUID | str, permission: str, grant_id: UUID | str | None = None) -> dict[str, Any]:
    """one ``mcp_tool_grants`` row as the loader and the reader return it.

    :param principal_id: the principal the grant gives to
    :ptype principal_id: UUID | str
    :param permission: what it gives
    :ptype permission: str
    :param grant_id: its key; a fresh one when omitted
    :ptype grant_id: UUID | str | None
    :return: the row
    :rtype: dict[str, Any]
    """
    return {"grant_id": grant_id or uuid4(), "principal_id": principal_id, "permission": permission}


async def _no_grant(_grant_id: UUID) -> dict[str, Any] | None:
    """a grant reader that finds no grant: every key it is asked for was revoked.

    :param _grant_id: the key
    :ptype _grant_id: UUID
    :return: ``None``
    :rtype: dict[str, Any] | None
    """
    return None


def _named(grant_id: UUID, action: str = "create") -> dict[str, Any]:
    """the payload the hub's grant endpoints bump with.

    :param grant_id: the grant written
    :ptype grant_id: UUID
    :param action: ``"create"`` or ``"delete"``
    :ptype action: str
    :return: the payload
    :rtype: dict[str, Any]
    """
    return {"grant_id": str(grant_id), "action": action}


_CATCHUP_TASK_NAME = "mcp-rbac-catchup-loop"


def _running_catchup_loops() -> list[asyncio.Task[Any]]:
    """the live catch-up loop tasks on the running event loop.

    the authorizer names its periodic task, so its presence is observable
    from the event loop without reaching into the authorizer.
    """
    return [task for task in asyncio.all_tasks() if task.get_name() == _CATCHUP_TASK_NAME and not task.done()]


async def _build_started_authorizer(
    *,
    loader: AsyncMock,
    reader: AsyncMock | None = None,
    admin_principal_ids: set[UUID] | None = None,
) -> tuple[LocalGrantAuthorizer, list[Any]]:
    """construct + start a LocalGrantAuthorizer; return (authz, captured_callbacks).

    every test that uses this helper MUST ``await authz.stop()`` at
    teardown so the spawned catch-up task is cancelled. catchup
    interval is set to a very large value so the task never fires
    during the test (deterministic).
    """
    client, listener, captured, captured_resets = _make_listener_capturing_subscribe()
    authz = LocalGrantAuthorizer(
        grant_loader=loader,
        grant_reader=reader if reader is not None else AsyncMock(side_effect=_no_grant),
        epoch_client=client,
        epoch_listener=listener,
        admin_principal_ids=admin_principal_ids,
        catchup_interval_seconds=3600.0,
    )
    await authz.start()
    return authz, captured


# ---------------------------------------------------------------------
# EnvVarIdentityProvider
# ---------------------------------------------------------------------


class TestEnvVarIdentityProvider:
    """env-var-derived identity for v1 stdio mode."""

    @pytest.mark.asyncio
    async def test_explicit_principal_id_bypasses_env(self) -> None:
        """when ``principal_id`` is set the env var is never read."""
        target = uuid4()
        provider = EnvVarIdentityProvider(principal_id=target)
        identity = await provider.identify()
        assert identity.principal_id == target
        assert identity.principal_type == "user"

    @pytest.mark.asyncio
    async def test_is_admin_defaults_deny(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_admin defaults False: a wirer that omits the flag gets a
        non-admin identity, not a total RBAC bypass. admin authority is
        granted only when the wirer passes is_admin=True explicitly."""
        target = uuid4()
        monkeypatch.setenv("MCP_ADMIN_USER_ID", str(target))
        assert (await EnvVarIdentityProvider().identify()).is_admin is False
        assert (await EnvVarIdentityProvider(principal_id=target).identify()).is_admin is False

    @pytest.mark.asyncio
    async def test_is_admin_granted_when_explicit(self) -> None:
        """admin authority is honoured when the wirer opts in explicitly."""
        provider = EnvVarIdentityProvider(principal_id=uuid4(), is_admin=True)
        identity = await provider.identify()
        assert identity.is_admin is True

    @pytest.mark.asyncio
    async def test_env_var_provides_principal_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """env var carrying a UUID populates principal_id."""
        target = uuid4()
        monkeypatch.setenv("MCP_ADMIN_USER_ID", str(target))
        provider = EnvVarIdentityProvider()
        identity = await provider.identify()
        assert identity.principal_id == target

    @pytest.mark.asyncio
    async def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no env var + no explicit principal_id raises a clear error."""
        monkeypatch.delenv("MCP_ADMIN_USER_ID", raising=False)
        provider = EnvVarIdentityProvider()
        with pytest.raises(RuntimeError, match="MCP_ADMIN_USER_ID"):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_invalid_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """malformed env var value raises a UUID-parse error."""
        monkeypatch.setenv("MCP_ADMIN_USER_ID", "not-a-uuid")
        provider = EnvVarIdentityProvider()
        with pytest.raises(RuntimeError, match="valid UUID"):
            await provider.identify()

    @pytest.mark.asyncio
    async def test_is_admin_flag_explicitly_disable(self) -> None:
        """non-admin identity construction works for future bearer-token paths."""
        provider = EnvVarIdentityProvider(principal_id=uuid4(), is_admin=False)
        identity = await provider.identify()
        assert identity.is_admin is False


# ---------------------------------------------------------------------
# LocalGrantAuthorizer
# ---------------------------------------------------------------------


class TestLocalGrantAuthorizer:
    """default-deny + admin auto-grant + cache reload on epoch bump."""

    @pytest.mark.asyncio
    async def test_admin_short_circuit_allows_without_grant(self) -> None:
        """is_admin=True identity allows every permission without DB lookup."""
        loader = AsyncMock(return_value=[])
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=uuid4(), is_admin=True)
            assert await authz.allows(identity, "anything.at.all") is True
            loader.assert_awaited_once()  # called by start() reload
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_default_deny_when_no_grant(self) -> None:
        """non-admin identity with no matching grant is denied."""
        loader = AsyncMock(return_value=[])
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=uuid4())
            assert await authz.allows(identity, "product.conv.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_principal_grant_allows(self) -> None:
        """direct principal_id+permission grant authorizes the call."""
        principal_id = uuid4()
        loader = AsyncMock(
            return_value=[
                _row(principal_id, "product.conv.read"),
            ]
        )
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "product.conv.read") is True
            assert await authz.allows(identity, "product.conv.write") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_group_grant_allows_via_membership(self) -> None:
        """grant on a group_id allows when identity carries that group."""
        group_id = uuid4()
        loader = AsyncMock(
            return_value=[
                _row(group_id, "audit.records.read"),
            ]
        )
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(
                principal_type="user",
                principal_id=uuid4(),
                groups=frozenset({group_id}),
            )
            assert await authz.allows(identity, "audit.records.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_role_grant_allows_via_assignment(self) -> None:
        """grant on a role_id allows when identity carries that role."""
        role_id = uuid4()
        loader = AsyncMock(
            return_value=[
                _row(role_id, "audit.records.read"),
            ]
        )
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(
                principal_type="user",
                principal_id=uuid4(),
                roles=frozenset({role_id}),
            )
            assert await authz.allows(identity, "audit.records.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_start_subscribes_to_rbac_epoch(self) -> None:
        """start() subscribes the listener to Subjects.mcp_rbac_epoch."""
        loader = AsyncMock(return_value=[])
        client, listener, _, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            listener.subscribe.assert_awaited_once()
            from threetears.nats import Subjects

            called_subject = listener.subscribe.await_args.args[0]
            assert called_subject == Subjects.mcp_rbac_epoch()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_start_subscribes_before_it_fills_and_primes_to_the_epoch_read_first(self) -> None:
        """the counter is read, the listener primed to it and subscribed, and only then the cache filled.

        a bump landing during the fill is then heard and applied by key; one subscribing after the
        fill would miss its broadcast, and the grant it changed could only be found by reading every
        grant again. the fill reads the counter once more before its rows, so a bump at or below that
        needs no read of its own.
        """
        order: list[str] = []

        async def _current(subject: Any) -> int:  # noqa: ARG001
            order.append("current")
            return 5

        async def _loader() -> list[dict[str, Any]]:
            order.append("fill")
            return []

        client, listener, _, _ = _make_listener_capturing_subscribe()
        client.current = AsyncMock(side_effect=_current)
        subscribe = listener.subscribe.side_effect

        async def _subscribe(*args: Any, **kwargs: Any) -> None:
            order.append("subscribe")
            await subscribe(*args, **kwargs)

        listener.subscribe = AsyncMock(side_effect=_subscribe)
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(side_effect=_loader),
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            assert order == ["current", "subscribe", "current", "fill"]
            assert listener.subscribe.await_args.kwargs["primed_epoch"] == 5
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_bump_applies_the_grant_it_names_by_key_and_never_reads_every_grant(self) -> None:
        """the grant the bump names is read by its key and put in the cache; the loader is not called again."""
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[])
        reader = AsyncMock(return_value=_row(principal_id, "product.conv.read", grant_id))
        authz, captured = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "product.conv.read") is False
            await captured[0](1, _named(grant_id))
            assert await authz.allows(identity, "product.conv.read") is True
            reader.assert_awaited_once_with(grant_id)
            assert loader.await_count == 1, "a bump read every grant again"
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_bump_for_a_revoked_grant_takes_it_out_of_the_cache(self) -> None:
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read", grant_id)])
        authz, captured = await _build_started_authorizer(loader=loader, reader=AsyncMock(return_value=None))
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is True
            await captured[0](1, _named(grant_id, "delete"))
            assert await authz.allows(identity, "audit.records.read") is False
            assert loader.await_count == 1
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_revoking_one_of_two_grants_of_a_permission_keeps_the_permission(self) -> None:
        """two grants (two tools) can give one permission; revoking one leaves the other's."""
        principal_id = uuid4()
        first, second = uuid4(), uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "p.read", first), _row(principal_id, "p.read", second)])
        authz, captured = await _build_started_authorizer(loader=loader, reader=AsyncMock(return_value=None))
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            await captured[0](1, _named(first, "delete"))
            assert await authz.allows(identity, "p.read") is True
            await captured[0](2, _named(second, "delete"))
            assert await authz.allows(identity, "p.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_bump_that_names_no_grant_refuses_and_never_reads_every_grant(self) -> None:
        """its grant is unknown: the cache may hold a revoked grant, and no whole reload stands in for it."""
        principal_id = uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read")])
        reader = AsyncMock(side_effect=_no_grant)
        authz, captured = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            await captured[0](1, {"hint": "something changed"})
            with pytest.raises(GrantCacheUnavailableError, match="unheard"):
                await authz.allows(identity, "audit.records.read")
            admin = Identity(principal_type="user", principal_id=uuid4(), is_admin=True)
            assert await authz.allows(admin, "audit.records.read") is True, "admins never read the cache"
            assert loader.await_count == 1, "a bump with no grant read every grant again"
            reader.assert_not_awaited()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_skipped_epoch_refuses_but_the_named_grant_is_still_applied(self) -> None:
        """epoch 1 went unheard and epoch 2 names its grant: 2 is applied, 1's grant is unknown."""
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[])
        reader = AsyncMock(return_value=_row(principal_id, "p.read", grant_id))
        authz, captured = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            await captured[0](2, _named(grant_id))
            reader.assert_awaited_once_with(grant_id)
            with pytest.raises(GrantCacheUnavailableError, match="unheard"):
                await authz.allows(Identity(principal_type="user", principal_id=principal_id), "p.read")
            assert loader.await_count == 1
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_bump_the_fill_already_holds_reads_nothing(self) -> None:
        """the fill read the counter at 3 before its rows, so bumps up to 3 are in them."""
        client, listener, captured, _ = _make_listener_capturing_subscribe()
        client.current = AsyncMock(side_effect=[1, 3])
        reader = AsyncMock(side_effect=_no_grant)
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(return_value=[]),
            grant_reader=reader,
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            await captured[0](2, _named(uuid4()))
            await captured[0](3, _named(uuid4()))
            reader.assert_not_awaited()
            assert await authz.allows(Identity(principal_type="user", principal_id=uuid4()), "p.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_failed_keyed_read_refuses_instead_of_serving_the_prior_cache(self) -> None:
        """the bump may have been a revocation, so a grant in the cache is not trusted until its read lands."""
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read", grant_id)])
        authz, captured = await _build_started_authorizer(
            loader=loader, reader=AsyncMock(side_effect=RuntimeError("L3 down"))
        )
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is True
            with pytest.raises(RuntimeError, match="L3 down"):
                await captured[0](1, _named(grant_id, "delete"))
            with pytest.raises(GrantCacheUnavailableError, match="not current"):
                await authz.allows(identity, "audit.records.read")
            admin = Identity(principal_type="user", principal_id=uuid4(), is_admin=True)
            assert await authz.allows(admin, "audit.records.read") is True, "admins never read the cache"
            assert loader.await_count == 1, "a failed keyed read fell back to reading every grant"
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_row_that_will_not_decode_fails_the_keyed_read_too(self) -> None:
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read", grant_id)])
        reader = AsyncMock(return_value=_row("not-a-uuid", "audit.records.read", grant_id))
        authz, captured = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            with pytest.raises(ValueError):
                await captured[0](1, _named(grant_id))
            with pytest.raises(GrantCacheUnavailableError):
                await authz.allows(Identity(principal_type="user", principal_id=principal_id), "audit.records.read")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_slow_read_cannot_put_back_a_grant_a_later_read_revoked(self) -> None:
        """bump 1 starts a slow read of the grant; bump 2 (its revocation) reads it gone at once; bump 1
        then lands with the row it read before the revocation. it must not put the grant back."""
        principal_id = uuid4()
        grant_id = uuid4()
        release = asyncio.Event()
        calls = 0

        async def _reader(_key: UUID) -> dict[str, Any] | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                await release.wait()
                return _row(principal_id, "audit.records.read", grant_id)
            return None

        authz, captured = await _build_started_authorizer(
            loader=AsyncMock(return_value=[]), reader=AsyncMock(side_effect=_reader)
        )
        try:
            slow = asyncio.create_task(captured[0](1, _named(grant_id)))
            await asyncio.sleep(0)
            await captured[0](2, _named(grant_id, "delete"))
            release.set()
            await slow
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_slow_read_that_started_before_a_failing_one_does_not_clear_what_is_owed(self) -> None:
        """bump 1 starts a slow read of the old row; bump 2 (a revocation) fails at once; bump 1 then
        finishes. without ordering, its success cleared what was owed and the revoked grant was
        allowed again from a row read before the revocation."""
        principal_id = uuid4()
        grant_id = uuid4()
        release = asyncio.Event()
        calls = 0

        async def _reader(_key: UUID) -> dict[str, Any] | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                await release.wait()
                return _row(principal_id, "audit.records.read", grant_id)
            raise RuntimeError("L3 down")

        authz, captured = await _build_started_authorizer(
            loader=AsyncMock(return_value=[]), reader=AsyncMock(side_effect=_reader)
        )
        try:
            slow = asyncio.create_task(captured[0](1, _named(grant_id)))
            await asyncio.sleep(0)
            with pytest.raises(RuntimeError):
                await captured[0](2, _named(grant_id, "delete"))
            release.set()
            await slow
            with pytest.raises(GrantCacheUnavailableError):
                await authz.allows(Identity(principal_type="user", principal_id=principal_id), "audit.records.read")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_failure_that_started_before_a_landed_read_owes_nothing(self) -> None:
        principal_id = uuid4()
        grant_id = uuid4()
        release = asyncio.Event()
        calls = 0

        async def _reader(_key: UUID) -> dict[str, Any] | None:
            nonlocal calls
            calls += 1
            if calls == 1:
                await release.wait()
                raise RuntimeError("L3 was down")
            return _row(principal_id, "audit.records.read", grant_id)

        authz, captured = await _build_started_authorizer(
            loader=AsyncMock(return_value=[]), reader=AsyncMock(side_effect=_reader)
        )
        try:
            slow = asyncio.create_task(captured[0](1, _named(grant_id)))
            await asyncio.sleep(0)
            await captured[0](2, _named(grant_id))
            release.set()
            with pytest.raises(RuntimeError):
                await slow
            assert await authz.allows(Identity(principal_type="user", principal_id=principal_id), "audit.records.read")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_read_that_lands_after_a_failure_answers_again(self) -> None:
        principal_id = uuid4()
        grant_id = uuid4()
        reader = AsyncMock(side_effect=[RuntimeError("L3 down"), _row(principal_id, "audit.records.read", grant_id)])
        authz, captured = await _build_started_authorizer(loader=AsyncMock(return_value=[]), reader=reader)
        try:
            with pytest.raises(RuntimeError):
                await captured[0](1, _named(grant_id))
            await captured[0](2, _named(grant_id))
            assert await authz.allows(Identity(principal_type="user", principal_id=principal_id), "audit.records.read")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_keyed_read_that_started_during_a_fill_outlives_it(self) -> None:
        """a reset's fill reads every row slowly; a revocation's keyed read starts after it and lands
        first. the fill's rows were read before the revocation, so the grant stays revoked."""
        principal_id = uuid4()
        grant_id = uuid4()
        release = asyncio.Event()
        fills = 0

        async def _loader() -> list[dict[str, Any]]:
            nonlocal fills
            fills += 1
            if fills == 2:
                await release.wait()
            return [_row(principal_id, "audit.records.read", grant_id)]

        client, listener, captured, captured_resets = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(side_effect=_loader),
            grant_reader=AsyncMock(return_value=None),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            reset = asyncio.create_task(captured_resets[0]())
            await asyncio.sleep(0)
            await captured[0](1, _named(grant_id, "delete"))
            release.set()
            await reset
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_reset_fills_again_and_clears_unheard_epochs(self) -> None:
        """the operator reset is the one fill after the cold start, and what makes unheard epochs known."""
        principal_id = uuid4()
        loader = AsyncMock(side_effect=[[], [_row(principal_id, "p.read")]])
        client, listener, captured, captured_resets = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            await captured[0](1, None)
            with pytest.raises(GrantCacheUnavailableError):
                await authz.allows(identity, "p.read")
            await captured_resets[0]()
            assert loader.await_count == 2
            assert await authz.allows(identity, "p.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_uuid_string_in_loader_row_is_normalised(self) -> None:
        """rows with stringified UUIDs (e.g. asyncpg .Record edge) match."""
        principal_id = uuid4()
        loader = AsyncMock(return_value=[_row(str(principal_id), "product.conv.read", str(uuid4()))])
        authz, _ = await _build_started_authorizer(loader=loader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "product.conv.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_double_start_is_no_op(self) -> None:
        """calling start() twice does not re-subscribe or re-load."""
        loader = AsyncMock(return_value=[])
        client, listener, _, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        await authz.start()
        try:
            loader.assert_awaited_once()
            listener.subscribe.assert_awaited_once()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_catchup_task_and_is_idempotent(self) -> None:
        """stop() cancels the spawned tick; second stop is a no-op."""
        loader = AsyncMock(return_value=[])
        authz, _ = await _build_started_authorizer(loader=loader)
        assert len(_running_catchup_loops()) == 1
        # first stop() cancels the task.
        await authz.stop()
        assert _running_catchup_loops() == []
        # second stop() is safe (no-op).
        await authz.stop()


class TestLocalGrantAuthorizerCatchupTick:
    """the periodic catch-up tick is the safety net for missed broadcasts."""

    @pytest.mark.asyncio
    async def test_catchup_loop_invokes_epoch_listener_catch_up(self) -> None:
        """the spawned tick calls EpochListener.catch_up on every interval.

        constructs the authorizer with a tiny interval and asserts
        catch_up was called at least twice before stop. proves the
        background task wires the right method, even if we don't
        wait for many ticks.
        """

        loader = AsyncMock(return_value=[])
        client, listener, _, _ = _make_listener_capturing_subscribe()
        listener.catch_up, ticked = _counting_catch_up(2)
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,  # 10ms -- fires almost immediately
        )
        await authz.start()
        try:
            # wait for a couple of ticks, however long the machine takes to schedule them.
            await asyncio.wait_for(ticked.wait(), timeout=_TICK_DEADLINE_SECONDS)
            assert listener.catch_up.await_count >= 2
            # the tick uses Subjects.mcp_rbac_epoch as the subject.
            from threetears.nats import Subjects

            for call in listener.catch_up.await_args_list:
                assert call.args[0] == Subjects.mcp_rbac_epoch()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_catchup_loop_swallows_transient_errors(self) -> None:
        """a tick that raises does not kill the loop; subsequent ticks fire.

        proves the narrow exception scope around catch_up keeps the
        safety net alive across a transient L3 / NATS hiccup.
        """

        loader = AsyncMock(return_value=[])
        client, listener, _, _ = _make_listener_capturing_subscribe()
        # first catch_up raises; subsequent ones return 0.
        listener.catch_up, survived = _counting_catch_up(2, first_raises=RuntimeError("transient"))
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,
        )
        await authz.start()
        try:
            # a second call means the loop survived the first one's error.
            await asyncio.wait_for(survived.wait(), timeout=_TICK_DEADLINE_SECONDS)
            assert listener.catch_up.await_count >= 2
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_keyed_read_that_failed_on_a_bump_is_retried_by_key_on_the_next_tick(self) -> None:
        """the bump is already counted as seen, so only the owed read can withdraw a revoked grant --
        and it is read again by its key, never by reading every grant."""
        principal_id = uuid4()
        grant_id = uuid4()
        retried = asyncio.Event()
        answers: list[Any] = [RuntimeError("L3 down"), None]  # the grant was revoked; the retry must land that

        async def _reader(_key: UUID) -> dict[str, Any] | None:
            answer = answers.pop(0)
            if not answers:
                retried.set()
            if isinstance(answer, BaseException):
                raise answer
            return answer

        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read", grant_id)])
        reader = AsyncMock(side_effect=_reader)
        client, listener, captured, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=reader,
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,
        )
        await authz.start()
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            with pytest.raises(RuntimeError):
                await captured[0](1, _named(grant_id, "delete"))
            await asyncio.wait_for(retried.wait(), timeout=_TICK_DEADLINE_SECONDS)
            await asyncio.sleep(0)
            assert await authz.allows(identity, "audit.records.read") is False
            assert [call.args[0] for call in reader.await_args_list] == [grant_id, grant_id]
            assert loader.await_count == 1, "the retry read every grant"
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_retry_that_keeps_failing_is_retried_every_tick_and_stays_refused(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """each failed retry raises out of the tick, so ``PeriodicTask`` logs every one."""
        retried_twice = asyncio.Event()
        calls = 0

        async def _reader(_key: UUID) -> dict[str, Any] | None:
            nonlocal calls
            calls += 1
            if calls >= 3:  # the bump's read, then two retries by the tick
                retried_twice.set()
            raise RuntimeError("L3 down")

        loader = AsyncMock(return_value=[])
        client, listener, captured, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_reader),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,
        )
        await authz.start()
        try:
            with pytest.raises(RuntimeError):
                await captured[0](1, _named(uuid4()))
            await asyncio.wait_for(retried_twice.wait(), timeout=_TICK_DEADLINE_SECONDS)
            with pytest.raises(GrantCacheUnavailableError):
                await authz.allows(Identity(principal_type="user", principal_id=uuid4()), "p.read")
            assert "catch-up tick errored" in caplog.text
            assert loader.await_count == 1
        finally:
            await authz.stop()


def _real_listener_over(current: list[int]) -> tuple[Any, Any, Any]:
    """a real EpochListener over a stubbed NATS client and an epoch client whose counter reads ``current[0]``.

    :param current: the counter; the test moves it by setting ``current[0]``
    :ptype current: list[int]
    :return: the listener, the epoch client, and the NATS stub (whose ``subscribe_typed`` call holds the
        typed callback a broadcast reaches)
    :rtype: tuple[Any, Any, Any]
    """
    from threetears.epoch import EpochListener

    nats = MagicMock()
    nats.subscribe_typed = AsyncMock()
    epoch_client = MagicMock()

    async def _current(_subject: Any) -> int:
        return current[0]

    epoch_client.current = AsyncMock(side_effect=_current)
    epoch_client.bucket_identity = AsyncMock(return_value="bucket-1")
    return EpochListener(nats, epoch_client), epoch_client, nats


async def _until(check: Any) -> None:
    """wait, bounded by :data:`_TICK_DEADLINE_SECONDS`, until ``check()`` is true.

    :param check: the condition
    :ptype check: Any
    :return: nothing
    :rtype: None
    """

    async def _poll() -> None:
        while not check():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=_TICK_DEADLINE_SECONDS)


class TestTheCatchUpTickFindsLostBroadcasts:
    """a broadcast the listener never heard: given one interval, then its grant is unknown -- never a whole reload."""

    @pytest.mark.asyncio
    async def test_a_lost_broadcast_refuses_and_never_reads_every_grant(self) -> None:
        current = [0]
        listener, epoch_client, _nats = _real_listener_over(current)
        loader = AsyncMock(return_value=[])
        reader = AsyncMock(side_effect=_no_grant)
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=reader,
            epoch_client=epoch_client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,
        )
        await authz.start()
        try:
            current[0] = 1  # a grant was written and bumped; its broadcast never arrives
            identity = Identity(principal_type="user", principal_id=uuid4())

            await _until(lambda: listener.last_seen(Subjects.mcp_rbac_epoch()) == 1)
            with pytest.raises(GrantCacheUnavailableError, match="unheard"):
                await authz.allows(identity, "p.read")
            assert loader.await_count == 1, "a lost broadcast read every grant again"
            reader.assert_not_awaited()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_broadcast_that_lands_after_the_tick_saw_its_epoch_is_still_applied_by_key(self) -> None:
        """the tick saw the counter at 1 before the listener heard 1. advancing the listener at once
        would make it drop the broadcast when it lands, payload and all, and the grant would be unknown."""
        from threetears.epoch import EpochBumpMessage

        current = [0]
        listener, epoch_client, nats = _real_listener_over(current)
        principal_id = uuid4()
        grant_id = uuid4()
        reader = AsyncMock(return_value=_row(principal_id, "p.read", grant_id))
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(return_value=[]),
            grant_reader=reader,
            epoch_client=epoch_client,
            epoch_listener=listener,
            catchup_interval_seconds=0.2,
        )
        await authz.start()
        try:
            reads_at_start = epoch_client.current.await_count
            current[0] = 1
            # the tick reads the counter ahead of what the listener heard
            await _until(lambda: epoch_client.current.await_count > reads_at_start)
            broadcast = nats.subscribe_typed.await_args.kwargs["cb"]
            await broadcast(
                EpochBumpMessage(subject_path=Subjects.mcp_rbac_epoch().path, epoch=1, payload=_named(grant_id))
            )
            await _until(lambda: epoch_client.current.await_count > reads_at_start + 2)
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "p.read") is True
            reader.assert_awaited_once_with(grant_id)
        finally:
            await authz.stop()


class TestLocalGrantAuthorizerAdminLogging:
    """admin auto-grant logs the specific principal IDs at start-time."""

    @pytest.mark.asyncio
    async def test_admin_principal_ids_recorded(self, caplog: pytest.LogCaptureFixture) -> None:
        """admin_principal_ids surface in the start-time log payload."""
        admin_id = uuid4()
        loader = AsyncMock(return_value=[])
        with caplog.at_level("INFO"):
            authz, _ = await _build_started_authorizer(
                loader=loader,
                admin_principal_ids={admin_id},
            )
            try:
                # caplog accumulates records; one of them carries the ids.
                joined = " ".join(rec.message for rec in caplog.records)
                assert str(admin_id) in joined or any(
                    str(admin_id) in str(rec.__dict__.get("extra_data", "")) for rec in caplog.records
                )
            finally:
                await authz.stop()


class TestLocalGrantAuthorizerOptionalEpoch:
    """epoch_client / epoch_listener are jointly optional (single-process mode)."""

    @pytest.mark.asyncio
    async def test_start_without_epoch_loads_cache_and_skips_subscribe(self) -> None:
        """no epoch deps: start() loads cache, skips subscribe + catchup."""
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            # no epoch_client / epoch_listener: single-process mode
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            # cache primed (loader called once at start)
            assert loader.await_count == 1
            # no catchup task in single-process mode
            assert _running_catchup_loops() == []
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_cold_load_that_fails_with_nothing_to_retry_it_fails_start(self) -> None:
        """single-process mode has no bump and no tick, so a failed first load would never be retried."""
        authz = LocalGrantAuthorizer(grant_loader=AsyncMock(side_effect=RuntimeError("L3 down")))
        with pytest.raises(RuntimeError, match="nothing would retry it"):
            await authz.start()

    @pytest.mark.asyncio
    async def test_a_cold_load_that_fails_in_epoch_mode_refuses_until_it_lands(self) -> None:
        """started, but not empty-and-answering: an empty cache would read as "no permission"."""
        principal_id = uuid4()
        gate = asyncio.Event()  # holds the retry back until the refusal has been observed
        landed = asyncio.Event()
        calls = 0

        async def _loader() -> list[dict[str, Any]]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("L3 down")
            await gate.wait()
            landed.set()
            return [_row(principal_id, "p.read")]

        client, listener, _, _ = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(side_effect=_loader),
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0.01,
        )
        await authz.start()
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            with pytest.raises(GrantCacheUnavailableError):
                await authz.allows(identity, "p.read")
            gate.set()
            await asyncio.wait_for(landed.wait(), timeout=_TICK_DEADLINE_SECONDS)
            await asyncio.sleep(0)
            assert await authz.allows(identity, "p.read") is True
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_stop_is_noop_when_no_catchup_task(self) -> None:
        """stop() handles the single-process case where no catchup task ever started."""
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(grant_loader=loader)
        await authz.start()
        # should not raise; catchup_task is None
        await authz.stop()

    def test_constructor_rejects_only_epoch_client(self) -> None:
        """passing exactly one of (client, listener) is a usage error."""
        loader = AsyncMock(return_value=[])
        client, _listener, _captured, _ = _make_listener_capturing_subscribe()
        with pytest.raises(ValueError, match="must be provided together"):
            LocalGrantAuthorizer(
                grant_loader=loader,
                epoch_client=client,
                # epoch_listener intentionally omitted
            )

    def test_constructor_rejects_only_epoch_listener(self) -> None:
        """passing exactly one of (client, listener) is a usage error."""
        loader = AsyncMock(return_value=[])
        _client, listener, _captured, _captured_resets = _make_listener_capturing_subscribe()
        with pytest.raises(ValueError, match="must be provided together"):
            LocalGrantAuthorizer(
                grant_loader=loader,
                epoch_listener=listener,
                # epoch_client intentionally omitted
            )

    def test_constructor_rejects_an_epoch_listener_without_a_grant_reader(self) -> None:
        """a bump is applied by reading its grant by key; without a reader the only way left is a whole reload."""
        client, listener, _captured, _ = _make_listener_capturing_subscribe()
        with pytest.raises(ValueError, match="grant_reader"):
            LocalGrantAuthorizer(grant_loader=AsyncMock(return_value=[]), epoch_client=client, epoch_listener=listener)


class TestLocalGrantAuthorizerReceivesResets:
    """A detected counter replacement must reload this pod's grant cache.

    Detection is inert without this. The listener clears its last-seen and the
    pod carries on serving authorization decisions from a cache nothing told it
    to refresh -- which is the failure the whole epoch mechanism exists to
    prevent, arriving through the one path nobody wired.
    """

    @pytest.mark.asyncio
    async def test_start_registers_a_reset_callback(self) -> None:
        client, listener, _captured, captured_resets = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(return_value=[]),
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            assert captured_resets, "subscribed without an on_reset; a replaced counter reloads nothing"
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_the_reset_callback_reloads_the_grant_cache(self) -> None:
        loads: list[int] = []

        async def _counting_loader() -> list[Any]:
            loads.append(1)
            return []

        client, listener, _captured, captured_resets = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(side_effect=_counting_loader),
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            before = len(loads)
            await captured_resets[0]()
            assert len(loads) == before + 1
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_the_reset_callback_takes_no_epoch(self) -> None:
        """It must not be interchangeable with on_bump.

        A reset dressed as a bump would carry a number below everything already
        acted on, and the framework teaches consumers to dedupe on exactly that.
        """
        client, listener, _captured, captured_resets = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(return_value=[]),
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            with pytest.raises(TypeError):
                await captured_resets[0](7, None)
        finally:
            await authz.stop()


class TestStopDeregistersFromTheListener:
    """A stopped authorizer must not keep receiving resets.

    ``EpochListener._registrations`` has no automatic removal, so without this
    the listener holds bound methods of a stopped object and hands them a reset
    on the next fan-out -- asking an authorizer that considers itself finished
    to reload a cache it no longer maintains.
    """

    @pytest.mark.asyncio
    async def test_stop_deregisters_the_rbac_subject(self) -> None:
        client, listener, _captured, _resets = _make_listener_capturing_subscribe()
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(return_value=[]),
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )
        from threetears.nats import Subjects

        await authz.start()
        await authz.stop()

        listener.deregister.assert_called_once()
        assert listener.deregister.call_args.args[0] == Subjects.mcp_rbac_epoch()


class TestStopReallyRemovesTheRegistration:
    """The sibling class asserts against a Mock listener, which cannot see this.

    `deregister.assert_called_once()` proves the CALL happened. It says nothing
    about whether anything was removed, and the interesting failures are all on
    that side: `deregister` first dropped every consumer on the subject, then
    (matching by identity, against a bound method that is rebuilt on every
    attribute access) dropped none. Both passed the Mock assertion.

    This drives a REAL `EpochListener` and asks the question behaviourally: after
    `stop()`, a reset fan-out must not reload this authorizer's cache.
    """

    @staticmethod
    def _real_listener() -> tuple[Any, Any]:
        """A real EpochListener over a stubbed NATS client and epoch client."""
        from threetears.epoch import EpochListener

        nats = MagicMock()
        nats.subscribe_typed = AsyncMock()
        epoch_client = MagicMock()
        epoch_client.current = AsyncMock(return_value=0)
        # signal_reset re-primes the bucket identity; without this the await
        # lands on a bare MagicMock and the test fails for the wrong reason.
        epoch_client.bucket_identity = AsyncMock(return_value="bucket-1")
        return EpochListener(nats, epoch_client), epoch_client

    @pytest.mark.asyncio
    async def test_a_reset_after_stop_does_not_reload_the_cache(self) -> None:
        """The registration is gone, so the fan-out cannot reach a stopped authorizer.

        :return: nothing
        :rtype: None
        """
        listener, epoch_client = self._real_listener()
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=epoch_client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )

        await authz.start()
        await authz.stop()
        loads_after_stop = loader.await_count

        await listener.signal_reset()

        assert loader.await_count == loads_after_stop, (
            "a reset reached a stopped authorizer: deregister did not remove the registration"
        )

    @pytest.mark.asyncio
    async def test_a_reset_before_stop_does_reload_the_cache(self) -> None:
        """The control. Without it the test above passes when the wiring never worked.

        :return: nothing
        :rtype: None
        """
        listener, epoch_client = self._real_listener()
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=epoch_client,
            epoch_listener=listener,
            catchup_interval_seconds=3600.0,
        )

        await authz.start()
        loads_after_start = loader.await_count

        await listener.signal_reset()

        assert loader.await_count > loads_after_start
        await authz.stop()


class TestStopIsReversible:
    """A stopped authorizer must be startable again.

    Separate from the reset-deregistration class above: that one is about a
    shut-down consumer still being called, this one is about it being unable to
    come back. They share a method and nothing else.
    """

    @pytest.mark.asyncio
    async def test_stop_is_reversible_without_epoch_mode(self) -> None:
        """A single-process authorizer must be startable again after stop.

        The previous version of this test built a mock listener the authorizer
        never received, so its ``assert_not_called()`` was true whatever the
        code did -- vacuous, and worse than the bare no-exception check it
        replaced, because the docstring claimed rigor it did not have. Passing
        the listener in cannot fix that: epoch mode is decided solely by
        ``self._epoch_listener is not None``, so it would invert the premise.

        So this asserts something the no-listener path can actually get wrong,
        and did: ``stop()`` set ``_started = False`` below an early return
        taken whenever there was no catch-up task, leaving a single-process
        authorizer permanently marked started and refused by ``start()``'s
        double-start guard.
        """
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(grant_loader=loader)
        await authz.start()
        await authz.stop()
        loads_after_first_cycle = loader.await_count

        await authz.start()

        assert loader.await_count == loads_after_first_cycle + 1, (
            "the restart was refused by the double-start guard, so the cache was never "
            "reloaded -- stop() left the authorizer marked started"
        )
        await authz.stop()


def test_a_non_positive_catchup_interval_is_refused_before_anything_is_primed() -> None:
    """Refusing at construction means start() never subscribes a listener it cannot then serve."""
    loader = AsyncMock(return_value=[])
    client, listener, _captured, _ = _make_listener_capturing_subscribe()
    with pytest.raises(ValueError, match="catchup_interval_seconds must be positive"):
        LocalGrantAuthorizer(
            grant_loader=loader,
            grant_reader=AsyncMock(side_effect=_no_grant),
            epoch_client=client,
            epoch_listener=listener,
            catchup_interval_seconds=0,
        )
