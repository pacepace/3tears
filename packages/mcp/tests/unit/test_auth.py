"""unit tests for :mod:`threetears.mcp.auth` -- identity + authorizer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from threetears.core.collections.epoch_change import ChangedRow, EpochChange
from threetears.core.collections.registry import CollectionRegistry
from threetears.mcp.auth import (
    EnvVarIdentityProvider,
    GrantCacheUnavailableError,
    Identity,
    LocalGrantAuthorizer,
)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

#: the table the grant cache is derived from
_GRANTS = "mcp_tool_grants"

#: how long a test waits for something to happen; a bound on a hang, never a pacing sleep.
_DEADLINE_SECONDS = 5.0


class _Watcher:
    """a generation reader whose table stands at ``generation``: read as that, watched as one push of it."""

    def __init__(self, generation: str | None = None) -> None:
        self.generation = generation
        self.read_for: list[str] = []

    async def read(self, table_name: str) -> str | None:
        self.read_for.append(table_name)
        return self.generation

    async def watch(self, table_name: str) -> AsyncGenerator[str | None]:
        yield self.generation
        await asyncio.Event().wait()


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


def _changed(registry: CollectionRegistry, grant_id: UUID) -> None:
    """what a row broadcast of one grant write does on the registry: tell the caches derived from the table.

    :param registry: the registry
    :ptype registry: CollectionRegistry
    :param grant_id: the grant written
    :ptype grant_id: UUID
    :return: nothing
    :rtype: None
    """
    registry.tell_derived_caches(registry.row_message(_GRANTS, grant_id))


async def _eventually(check: Callable[[], Awaitable[bool]], what: str) -> None:
    """wait, bounded by :data:`_DEADLINE_SECONDS`, until ``check()`` answers True.

    :param check: the condition
    :ptype check: Callable[[], Awaitable[bool]]
    :param what: what is waited for, for the failure
    :ptype what: str
    :return: nothing
    :rtype: None
    """
    deadline = asyncio.get_running_loop().time() + _DEADLINE_SECONDS
    while not await check():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


def _awaited(mock: AsyncMock, count: int) -> Callable[[], Awaitable[bool]]:
    """a check that ``mock`` has been awaited ``count`` times, and the awaits have run on.

    :param mock: the mock
    :ptype mock: AsyncMock
    :param count: how many awaits
    :ptype count: int
    :return: the check
    :rtype: Callable[[], Awaitable[bool]]
    """

    async def _check() -> bool:
        await asyncio.sleep(0)
        return mock.await_count >= count

    return _check


async def _build_started_authorizer(
    *,
    loader: AsyncMock,
    reader: AsyncMock | None = None,
    admin_principal_ids: set[UUID] | None = None,
    retry_interval_seconds: float = 3600.0,
) -> tuple[LocalGrantAuthorizer, CollectionRegistry]:
    """construct + start a LocalGrantAuthorizer that follows the grants on a registry of its own.

    every test that uses this helper MUST ``await authz.stop()`` at teardown. the retry interval
    is large unless a test asks for the retry, so the pass never fires otherwise.
    """
    registry = CollectionRegistry()
    authz = LocalGrantAuthorizer(
        grant_loader=loader,
        grant_reader=reader if reader is not None else AsyncMock(side_effect=_no_grant),
        registry=registry,
        generation_reader=_Watcher(),
        admin_principal_ids=admin_principal_ids,
        retry_interval_seconds=retry_interval_seconds,
    )
    await authz.start()
    return authz, registry


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
    """default-deny + admin auto-grant + group/role grants."""

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


class TestTheGrantsAreFollowedByKey:
    """each changed grant -- a row broadcast, this process's write, a walked epoch -- is read by its key."""

    @pytest.mark.asyncio
    async def test_start_takes_the_tables_mark_before_it_fills(self) -> None:
        """the derived cache is registered and the generation read taken as the mark before the fill reads rows.

        every write the fill may not hold is then an epoch after the mark, which the follow walks and reads
        by key. taking the mark after the fill would leave a write landing between them caught by nothing.
        """
        registry = CollectionRegistry()
        seen_at_fill: list[tuple[str | None, bool]] = []

        async def _loader() -> list[dict[str, Any]]:
            seen_at_fill.append((registry.generation_marks.recorded(_GRANTS), registry.has_derived_caches(_GRANTS)))
            return []

        watcher = _Watcher("inc:7")
        authz = LocalGrantAuthorizer(
            grant_loader=AsyncMock(side_effect=_loader),
            grant_reader=AsyncMock(side_effect=_no_grant),
            registry=registry,
            generation_reader=watcher,
            retry_interval_seconds=3600.0,
        )
        await authz.start()
        try:
            assert watcher.read_for == [_GRANTS]
            assert seen_at_fill == [("inc:7", True)]
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_follow_needs_the_invalidation_listener_running(self) -> None:
        """the follow judges each advance against the rows the listener heard; without it, follow refuses."""
        authz, _ = await _build_started_authorizer(loader=AsyncMock(return_value=[]))
        try:
            with pytest.raises(RuntimeError, match="invalidation listener"):
                authz.follow()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_changed_grant_is_read_by_its_key_and_never_every_grant(self) -> None:
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[])
        reader = AsyncMock(return_value=_row(principal_id, "product.conv.read", grant_id))
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "product.conv.read") is False
            _changed(registry, grant_id)
            await _eventually(lambda: authz.allows(identity, "product.conv.read"), "the grant read by key")
            reader.assert_awaited_once_with(grant_id)
            assert loader.await_count == 1, "a changed grant read every grant again"
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_revoked_grant_leaves_the_cache(self) -> None:
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read", grant_id)])
        reader = AsyncMock(return_value=None)
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is True
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the revocation's read")
            assert await authz.allows(identity, "audit.records.read") is False
            assert loader.await_count == 1
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_an_epoch_the_walk_applies_is_read_by_key(self) -> None:
        """a write whose broadcast never arrived -- or that bumped nothing at all -- is found by the
        generation walk, which applies its record's rows exactly as their broadcasts would."""
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[])
        reader = AsyncMock(return_value=_row(principal_id, "p.read", grant_id))
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            await registry.apply_epoch_change(
                EpochChange(table=_GRANTS, token="inc:1", rows=[ChangedRow(ids=[str(grant_id)])])
            )
            identity = Identity(principal_type="user", principal_id=principal_id)
            await _eventually(lambda: authz.allows(identity, "p.read"), "the walked grant read by key")
            reader.assert_awaited_once_with(grant_id)
            assert loader.await_count == 1, "a walked epoch read every grant"
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_checks_never_fail_closed_while_a_change_is_read(self) -> None:
        """steady state: while a changed grant's read is in flight the cache answers from what it holds."""
        principal_id = uuid4()
        grant_id = uuid4()
        release = asyncio.Event()

        async def _reader(_key: UUID) -> dict[str, Any] | None:
            await release.wait()
            return None

        loader = AsyncMock(return_value=[_row(principal_id, "p.read", grant_id)])
        reader = AsyncMock(side_effect=_reader)
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the read to start")
            assert await authz.allows(identity, "p.read") is True, "an in-flight read refused the check"
            release.set()
            await _eventually(lambda: _negated(authz.allows(identity, "p.read")), "the revocation to land")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_revoking_one_of_two_grants_of_a_permission_keeps_the_permission(self) -> None:
        """two grants (two tools) can give one permission; revoking one leaves the other's."""
        principal_id = uuid4()
        first, second = uuid4(), uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "p.read", first), _row(principal_id, "p.read", second)])
        reader = AsyncMock(return_value=None)
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            _changed(registry, first)
            await _eventually(_awaited(reader, 1), "the first revocation")
            assert await authz.allows(identity, "p.read") is True
            _changed(registry, second)
            await _eventually(_awaited(reader, 2), "the second revocation")
            assert await authz.allows(identity, "p.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_whole_table_change_fills_again(self) -> None:
        """a truncate or load changes the whole table: the operator's reset, the one fill after the cold start."""
        principal_id = uuid4()
        loader = AsyncMock(side_effect=[[], [_row(principal_id, "p.read")]])
        authz, registry = await _build_started_authorizer(loader=loader)
        try:
            registry.drop_table(_GRANTS, reason="a truncate")
            identity = Identity(principal_type="user", principal_id=principal_id)
            await _eventually(lambda: authz.allows(identity, "p.read"), "the fill after the drop")
            assert loader.await_count == 2
        finally:
            await authz.stop()


async def _negated(answer: Awaitable[bool]) -> bool:
    """``not await answer``.

    :param answer: the answer
    :ptype answer: Awaitable[bool]
    :return: its negation
    :rtype: bool
    """
    return not await answer


class TestReadsAreOrderedAndFailuresOwed:
    """reads are numbered when they start; a failed read refuses checks until a read of it lands."""

    @pytest.mark.asyncio
    async def test_a_failed_keyed_read_refuses_and_is_retried_by_key(self) -> None:
        """the change may have been a revocation, so the cache is not trusted until a read of it lands --
        and the retry reads that grant by key, never every grant."""
        principal_id = uuid4()
        grant_id = uuid4()
        answers: list[Any] = [RuntimeError("L3 down"), None]  # the grant was revoked; the retry lands that

        async def _reader(_key: UUID) -> dict[str, Any] | None:
            answer = answers.pop(0) if answers else None
            if isinstance(answer, BaseException):
                raise answer
            return answer

        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read", grant_id)])
        reader = AsyncMock(side_effect=_reader)
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader, retry_interval_seconds=0.05)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the failing read")
            with pytest.raises(GrantCacheUnavailableError, match="not current"):
                await authz.allows(identity, "audit.records.read")
            admin = Identity(principal_type="user", principal_id=uuid4(), is_admin=True)
            assert await authz.allows(admin, "audit.records.read") is True, "admins never read the cache"
            await _eventually(_awaited(reader, 2), "the retry")
            await _eventually(lambda: _negated(authz.allows(identity, "audit.records.read")), "the retry to land")
            assert [call.args[0] for call in reader.await_args_list[:2]] == [grant_id, grant_id]
            assert loader.await_count == 1, "the retry read every grant"
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_row_that_will_not_decode_fails_the_keyed_read_too(self) -> None:
        principal_id = uuid4()
        grant_id = uuid4()
        loader = AsyncMock(return_value=[_row(principal_id, "audit.records.read", grant_id)])
        reader = AsyncMock(return_value=_row("not-a-uuid", "audit.records.read", grant_id))
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the read")
            with pytest.raises(GrantCacheUnavailableError):
                await authz.allows(Identity(principal_type="user", principal_id=principal_id), "audit.records.read")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_slow_read_cannot_put_back_a_grant_a_later_read_revoked(self) -> None:
        """change 1 starts a slow read of the grant; change 2 (its revocation) reads it gone at once; the
        slow read then lands with the row it read before the revocation. it must not put the grant back."""
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

        reader = AsyncMock(side_effect=_reader)
        authz, registry = await _build_started_authorizer(loader=AsyncMock(return_value=[]), reader=reader)
        try:
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the slow read")
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 2), "the revocation's read")
            release.set()
            for _ in range(5):
                await asyncio.sleep(0)
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is False
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_slow_read_that_started_before_a_failing_one_does_not_clear_what_is_owed(self) -> None:
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

        reader = AsyncMock(side_effect=_reader)
        authz, registry = await _build_started_authorizer(loader=AsyncMock(return_value=[]), reader=reader)
        try:
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the slow read")
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 2), "the failing read")
            release.set()
            for _ in range(5):
                await asyncio.sleep(0)
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

        reader = AsyncMock(side_effect=_reader)
        authz, registry = await _build_started_authorizer(loader=AsyncMock(return_value=[]), reader=reader)
        try:
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the slow read")
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 2), "the landing read")
            release.set()
            for _ in range(5):
                await asyncio.sleep(0)
            assert await authz.allows(Identity(principal_type="user", principal_id=principal_id), "audit.records.read")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_keyed_read_that_started_during_a_fill_outlives_it(self) -> None:
        """a whole-table change's fill reads every row slowly; a revocation's keyed read starts after it and
        lands first. the fill's rows were read before the revocation, so the grant stays revoked."""
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

        loader = AsyncMock(side_effect=_loader)
        reader = AsyncMock(return_value=None)
        authz, registry = await _build_started_authorizer(loader=loader, reader=reader)
        try:
            registry.drop_table(_GRANTS, reason="a load")
            await _eventually(_awaited(loader, 2), "the slow fill")
            _changed(registry, grant_id)
            await _eventually(_awaited(reader, 1), "the revocation's read")
            release.set()
            for _ in range(5):
                await asyncio.sleep(0)
            identity = Identity(principal_type="user", principal_id=principal_id)
            assert await authz.allows(identity, "audit.records.read") is False
        finally:
            await authz.stop()


class TestStartAndStop:
    """the cold start, single-process mode, and a stop that undoes everything start did."""

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
        loader = AsyncMock(return_value=[])
        authz, registry = await _build_started_authorizer(loader=loader)
        try:
            await authz.start()
            loader.assert_awaited_once()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_stop_unregisters_so_a_later_change_reads_nothing(self) -> None:
        reader = AsyncMock(side_effect=_no_grant)
        authz, registry = await _build_started_authorizer(loader=AsyncMock(return_value=[]), reader=reader)
        await authz.stop()
        assert not registry.has_derived_caches(_GRANTS)
        _changed(registry, uuid4())
        for _ in range(5):
            await asyncio.sleep(0)
        reader.assert_not_awaited()
        await authz.stop()  # idempotent

    @pytest.mark.asyncio
    async def test_single_process_mode_fills_once_and_follows_nothing(self) -> None:
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(grant_loader=loader)
        await authz.start()
        try:
            assert loader.await_count == 1
            with pytest.raises(RuntimeError, match="no table to follow"):
                authz.follow()
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_a_cold_load_that_fails_with_nothing_to_retry_it_fails_start(self) -> None:
        """single-process mode has no retry, so a failed first load would never land."""
        authz = LocalGrantAuthorizer(grant_loader=AsyncMock(side_effect=RuntimeError("L3 down")))
        with pytest.raises(RuntimeError, match="nothing would retry it"):
            await authz.start()

    @pytest.mark.asyncio
    async def test_a_cold_load_that_fails_while_following_refuses_until_the_retry_lands(self) -> None:
        """started, but not empty-and-answering: an empty cache would read as "no permission"."""
        principal_id = uuid4()
        loader = AsyncMock(side_effect=[RuntimeError("L3 down"), [_row(principal_id, "p.read")]])
        authz, _ = await _build_started_authorizer(loader=loader, retry_interval_seconds=0.05)
        try:
            identity = Identity(principal_type="user", principal_id=principal_id)
            with pytest.raises(GrantCacheUnavailableError):
                await authz.allows(identity, "p.read")
            await _eventually(_awaited(loader, 2), "the retry")
            await _eventually(lambda: authz.allows(identity, "p.read"), "the retried fill to land")
        finally:
            await authz.stop()

    @pytest.mark.asyncio
    async def test_stop_is_reversible_without_a_registry(self) -> None:
        """a single-process authorizer must be startable again after stop."""
        loader = AsyncMock(return_value=[])
        authz = LocalGrantAuthorizer(grant_loader=loader)
        await authz.start()
        await authz.stop()
        await authz.start()
        assert loader.await_count == 2, "the restart was refused by the double-start guard"
        await authz.stop()

    def test_constructor_rejects_a_registry_without_a_generation_reader(self) -> None:
        with pytest.raises(ValueError, match="must be provided together"):
            LocalGrantAuthorizer(
                grant_loader=AsyncMock(return_value=[]),
                grant_reader=AsyncMock(side_effect=_no_grant),
                registry=CollectionRegistry(),
            )

    def test_constructor_rejects_a_generation_reader_without_a_registry(self) -> None:
        with pytest.raises(ValueError, match="must be provided together"):
            LocalGrantAuthorizer(
                grant_loader=AsyncMock(return_value=[]),
                grant_reader=AsyncMock(side_effect=_no_grant),
                generation_reader=_Watcher(),
            )

    def test_constructor_rejects_a_registry_without_a_grant_reader(self) -> None:
        """a changed grant is applied by reading it by key; without a reader only a whole reload is left."""
        with pytest.raises(ValueError, match="grant_reader"):
            LocalGrantAuthorizer(
                grant_loader=AsyncMock(return_value=[]), registry=CollectionRegistry(), generation_reader=_Watcher()
            )

    def test_a_non_positive_retry_interval_is_refused_before_anything_is_registered(self) -> None:
        registry = CollectionRegistry()
        with pytest.raises(ValueError, match="retry_interval_seconds must be positive"):
            LocalGrantAuthorizer(
                grant_loader=AsyncMock(return_value=[]),
                grant_reader=AsyncMock(side_effect=_no_grant),
                registry=registry,
                generation_reader=_Watcher(),
                retry_interval_seconds=0,
            )
        assert not registry.has_derived_caches(_GRANTS)


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
